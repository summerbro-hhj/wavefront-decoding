"""Batched Huginn-0125 (RDM) P/R/C primitives for exp4 (fixed batch, union calls).

Same computation as the single-sequence `HuginnBlocks` (P = embed*emb_scale +
2 prelude SandwichBlocks; R = adapter(cat[s, e]) + 4 core SandwichBlocks;
C = ln_f -> 2 coda SandwichBlocks -> ln_f -> lm_head), generalized to a fixed
batch of B sequences over one `BatchStaticKVCache`.

Unlike Ouro (P = embed, C = lm_head — attention-free), huginn's P and C carry
attention + cache traffic, so the batch scheduler drives all three phases
through the model-generic union interface:
    prelude_union(seq_ids, toks, cache)      — fills hidden(=seeded s0)/static/step
    advance_union(seq_ids, toks, cache)      — one recurrence for the union
    coda_argmax_union(seq_ids, toks, cache)  — greedy ids (one sync per call)

Slot layout (flat batch cache — huginn's native negative coda indices are
remapped to the tail):
    prelude layer i          -> slot i                      (0, 1)
    core (step r, layer l)   -> slot 2 + _slot_depth(r)*4 + l
    coda layer i             -> slot 2 + n_depth_slots*4 + i
With KV-sharing budget s the cache is ALLOCATED with only s depth slots —
n_slots drops from 2+4*T_total+2 (T=32: 132) to 2+4s+2 (s=1: 8), which is where
sharing buys batch size for this deep-recurrence model.

The grouped-R causal-varlen pack (same-slot tokens read the slot prefix once)
is applied in R (group key = (seq, depth-slot)) and in the coda (group key =
seq — all of a sequence's C tokens share the coda slot).

Initial state s0 follows the single-seq machinery exactly (position-seeded
trunc_normal(std)*emb_scale; prefill = one stream from state_seed), so B=1
batch runs are decision-path-identical to `HuginnBlocks` + `scheduler_emu`.
"""
from __future__ import annotations

import sys

import torch

from . import fa4_attention as fa
from .batch_kv_cache import BatchStaticKVCache
from .huginn_blocks import ensure_freqs_cis
from .state import ActiveToken

DEFAULT_MAX_SEQ_LEN = 2048


class HuginnBlocksBatch:
    def __init__(self, causal_lm, T_total: int, batch_size: int,
                 max_seq_len: int = DEFAULT_MAX_SEQ_LEN, kv_budget_s: int | None = None):
        self.model = causal_lm
        self.config = causal_lm.config
        self.T_total = T_total
        self.B = batch_size
        self.max_seq_len = max_seq_len
        self.kv_budget_s = kv_budget_s if kv_budget_s else None
        self.n_prelude = self.config.n_layers_in_prelude
        self.n_core = self.config.n_layers_in_recurrent_block
        self.n_coda = self.config.n_layers_in_coda
        self.device = next(causal_lm.parameters()).device
        self.dtype = next(causal_lm.parameters()).dtype
        self.n_kv_head = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim",
                                self.config.n_embd // self.config.num_attention_heads)
        self._apply_rope = sys.modules[type(causal_lm).__module__].apply_rotary_emb_complex_like
        ensure_freqs_cis(causal_lm, max_seq_len)
        self.group_r = True
        # q1 zero-pad routing (2026-09-15, mirrors OuroBlocksBatch.pad_q1):
        # pad single-query varlen reads to q_len=2 causal — same per-byte read
        # cost as the grouped causal path (sm86 tile-template parity; see
        # fa4_attention.attn_varlen_q1_padded). False = original path.
        self.pad_q1 = True
        # seeded s0 — identical recipe/seeding to ParcaeBlocks/HuginnBlocks.
        self.state_init = "random"
        self.state_seed = 0
        self._state_dim = self.config.n_embd
        self._state_std = float(self.config.init_values["std"])
        self._state_scale = float(causal_lm.emb_scale)

    # ------------------------------------------------------------------
    # slots
    # ------------------------------------------------------------------
    def _sharing(self) -> bool:
        return self.kv_budget_s is not None and self.kv_budget_s < self.T_total

    def _slot_depth(self, step: int) -> int:
        return (step % self.kv_budget_s) if self._sharing() else step

    def n_depth_slots(self) -> int:
        return self.kv_budget_s if self._sharing() else self.T_total

    def slot_prelude(self, i: int) -> int:
        return i

    def slot_core(self, step: int, layer: int) -> int:
        return self.n_prelude + self._slot_depth(step) * self.n_core + layer

    def slot_coda(self, i: int) -> int:
        return self.n_prelude + self.n_depth_slots() * self.n_core + i

    def n_slots(self) -> int:
        return self.slot_coda(self.n_coda - 1) + 1

    def create_cache(self) -> BatchStaticKVCache:
        return BatchStaticKVCache(
            n_slots=self.n_slots(), batch_size=self.B,
            n_kv_head=self.n_kv_head, head_dim=self.head_dim,
            max_seq_len=self.max_seq_len, dtype=self.dtype, device=self.device,
        )

    def cache_nbytes_estimate(self) -> int:
        elem = 2 if self.dtype in (torch.bfloat16, torch.float16) else 4
        return (self.n_slots() * self.B * self.max_seq_len
                * self.n_kv_head * self.head_dim * elem * 2)

    # ------------------------------------------------------------------
    # seeded s0 (mirrors ParcaeBlocks._init_state_* exactly)
    # ------------------------------------------------------------------
    def _trunc_normal_seeded(self, out32: torch.Tensor, seed: int) -> None:
        g = torch.Generator(device=self.device)
        g.manual_seed(seed & 0x7FFF_FFFF_FFFF)
        torch.nn.init.trunc_normal_(
            out32, mean=0.0, std=self._state_std,
            a=-3 * self._state_std, b=3 * self._state_std, generator=g,
        )

    def _init_state_tokens(self, positions: list[int]) -> torch.Tensor:
        n = len(positions)
        if self.state_init == "zero":
            return torch.zeros(n, self._state_dim, device=self.device, dtype=self.dtype)
        out = torch.empty(n, self._state_dim, device=self.device, dtype=torch.float32)
        for j, p in enumerate(positions):
            self._trunc_normal_seeded(out[j], self.state_seed * 1_000_003 + p + 1)
        return (out * self._state_scale).to(self.dtype)

    def _init_state_prefill(self, batch: int, length: int) -> torch.Tensor:
        """One (1, L, D) seeded draw SHARED by all sequences — keeps replicas
        symmetric and makes the B=1 batch bit-match the single-seq blocks'
        prefill draw (same seed, same stream)."""
        if self.state_init == "zero":
            return torch.zeros(batch, length, self._state_dim, device=self.device, dtype=self.dtype)
        out = torch.empty(1, length, self._state_dim, device=self.device, dtype=torch.float32)
        self._trunc_normal_seeded(out, self.state_seed * 1_000_003)
        return (out * self._state_scale).to(self.dtype).expand(batch, -1, -1).contiguous()

    # ------------------------------------------------------------------
    # shared layer pieces
    # ------------------------------------------------------------------
    def _qkv(self, attn, x, freqs):
        B, T, _ = x.shape
        q, k, v = attn.Wqkv(x).split(attn.chunks, dim=2)
        q = q.view(B, T, attn.n_head, attn.head_dim)
        k = k.view(B, T, attn.n_kv_heads, attn.head_dim)
        v = v.view(B, T, attn.n_kv_heads, attn.head_dim)
        if self.config.qk_bias:
            q_bias, k_bias = attn.qk_bias.split(1, dim=0)
            q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
        q, k = self._apply_rope(q, k, freqs_cis=freqs)
        return q, k, v

    def _sandwich_tail(self, block, x, att):
        x = block.norm_2(att + x)
        return block.norm_4(block.mlp(block.norm_3(x)) + x)

    def _block_dense(self, block, x, freqs, cache, slot):
        """Prefill: (B, L) dense causal SandwichBlock, writes [0, L) of `slot`."""
        B, T, C = x.shape
        h = block.norm_1(x)
        q, k, v = self._qkv(block.attn, h, freqs)
        cache.write_prefill(slot, k, v)
        out = fa.attn_dense(q, k, v, causal=True)
        return self._sandwich_tail(block, x, block.attn.proj(out.reshape(B, T, C)))

    def _freqs_at(self, positions: list[int]):
        return self.model.freqs_cis[:, torch.tensor(positions, device=self.device)]

    # ------------------------------------------------------------------
    # PREFILL — all B sequences (equal prompt length), hand-rolled batched
    # forward (native huginn prefill cannot target the flat batch cache).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        assert input_ids.dim() == 2 and input_ids.shape[0] == self.B, \
            f"prefill expects (B={self.B}, L) input_ids, got {tuple(input_ids.shape)}"
        model = self.model
        cache = self.create_cache()
        B, L = input_ids.shape
        freqs = model.freqs_cis[:, :L]
        x = model.transformer.wte(input_ids)
        if model.emb_scale != 1:
            x = x * model.emb_scale
        for i, block in enumerate(model.transformer.prelude):
            x = self._block_dense(block, x, freqs, cache, self.slot_prelude(i))
        e = x
        s = self._init_state_prefill(B, L)
        for r in range(self.T_total):
            s = model.transformer.adapter(torch.cat([s, e], dim=-1))
            for l, block in enumerate(model.transformer.core_block):
                s = self._block_dense(block, s, freqs, cache, self.slot_core(r, l))
        x = model.transformer.ln_f(s)
        for i, block in enumerate(model.transformer.coda):
            x = self._block_dense(block, x, freqs, cache, self.slot_coda(i))
        x = model.transformer.ln_f(x)
        last_logits = model.lm_head(x[:, -1]).float()          # (B, V)
        return cache, last_logits

    # ------------------------------------------------------------------
    # P — union prelude: embed -> 2 blocks (each token attends its own
    # sequence's prelude-slot prefix); fills hidden(=s0)/static_state/step.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prelude_union(self, seq_ids: list[int], toks: list[ActiveToken], cache) -> None:
        model = self.model
        n = len(toks)
        positions = [t.position for t in toks]
        ids = torch.tensor([[t.token_id for t in toks]], dtype=torch.long, device=self.device)
        x = model.transformer.wte(ids)
        if model.emb_scale != 1:
            x = x * model.emb_scale
        freqs = self._freqs_at(positions)
        cu_q = torch.arange(0, n + 1, dtype=torch.int32, device=self.device)
        for i, block in enumerate(model.transformer.prelude):
            slot = self.slot_prelude(i)
            h = block.norm_1(x)
            q, k, v = self._qkv(block.attn, h, freqs)
            slots = [slot] * n
            cache.write_tokens(slots, seq_ids, positions, k[0], v[0])
            k_pack, v_pack, cu_k, max_k = cache.gather_pack(slots, seq_ids, positions)
            if self.pad_q1:
                out = fa.attn_varlen_q1_padded(q[0], k_pack, v_pack, cu_k, max_k)
            else:
                out = fa.attn_varlen(q[0], k_pack, v_pack, cu_q, cu_k, 1, max_k)
            x = self._sandwich_tail(block, x, block.attn.proj(out.reshape(1, n, -1)))
        s0 = self._init_state_tokens(positions)
        for i, t in enumerate(toks):
            t.static_state = x[0, i]
            t.hidden = s0[i]
            t.step = 0

    # ------------------------------------------------------------------
    # grouped pack layout for a union call (hoisted; key = (seq, group_key)).
    # ------------------------------------------------------------------
    def _group_meta(self, keys: list[tuple[int, int]], positions: list[int]):
        by_key: dict[tuple[int, int], list[int]] = {}
        for i, kk in enumerate(keys):
            by_key.setdefault(kk, []).append(i)
        if all(len(v) == 1 for v in by_key.values()):
            return None
        g_bs, g_ds, g_pmax, q_lens = [], [], [], []
        rows = [0] * len(positions)
        base = 0
        for (b, d), idxs in by_key.items():
            idxs.sort(key=lambda i: positions[i])
            pmin, pmax = positions[idxs[0]], positions[idxs[-1]]
            span = pmax - pmin + 1
            g_bs.append(b); g_ds.append(d); g_pmax.append(pmax); q_lens.append(span)
            for i in idxs:
                rows[i] = base + positions[i] - pmin
            base += span
        identity = base == len(positions) and rows == list(range(len(positions)))
        row_t = None if identity else torch.tensor(rows, dtype=torch.long, device=self.device)
        cu_q = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=self.device)
        cu_q[1:] = torch.tensor(q_lens, dtype=torch.int32, device=self.device).cumsum(0)
        return dict(g_bs=g_bs, g_ds=g_ds, g_pmax=g_pmax, q_lens=q_lens,
                    total_q=base, identity=identity, row_t=row_t, cu_q=cu_q)

    def _union_attn(self, q, cache, slots, seq_ids, positions, cu_q1, grouped, g_slots):
        """One varlen attention over the union: grouped causal segments when
        `grouped` is set (g_slots = per-group physical slots for THIS layer),
        else the per-token pack. K/V must already be written."""
        if grouped is not None:
            k_pack, v_pack, cu_k, max_k = cache.gather_pack(
                g_slots, grouped["g_bs"], grouped["g_pmax"])
            if grouped["identity"]:
                q_pack = q[0]
            else:
                q_pack = q.new_zeros(grouped["total_q"], q.shape[2], q.shape[3])
                q_pack.index_copy_(0, grouped["row_t"], q[0])
            out = fa.attn_varlen(q_pack, k_pack, v_pack, grouped["cu_q"], cu_k,
                                 max(grouped["q_lens"]), max_k, causal=True)
            if not grouped["identity"]:
                out = out.index_select(0, grouped["row_t"])
            return out
        k_pack, v_pack, cu_k, max_k = cache.gather_pack(slots, seq_ids, positions)
        if self.pad_q1:
            return fa.attn_varlen_q1_padded(q[0], k_pack, v_pack, cu_k, max_k)
        return fa.attn_varlen(q[0], k_pack, v_pack, cu_q1, cu_k, 1, max_k)

    # ------------------------------------------------------------------
    # R — one recurrence for the union: adapter(cat[s, e]) -> 4 core blocks.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def advance_union(self, seq_ids: list[int], toks: list[ActiveToken], cache) -> None:
        model = self.model
        n = len(toks)
        positions = [t.position for t in toks]
        steps = [t.step for t in toks]
        x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)
        e = torch.stack([t.static_state for t in toks], dim=0).unsqueeze(0)
        freqs = self._freqs_at(positions)
        x = model.transformer.adapter(torch.cat([x, e], dim=-1))

        grouped = None
        if self.group_r:
            grouped = self._group_meta(
                [(seq_ids[i], self._slot_depth(steps[i])) for i in range(n)], positions)
        cu_q1 = torch.arange(0, n + 1, dtype=torch.int32, device=self.device)
        for layer_idx, block in enumerate(model.transformer.core_block):
            h = block.norm_1(x)
            q, k, v = self._qkv(block.attn, h, freqs)
            slots = [self.slot_core(steps[i], layer_idx) for i in range(n)]
            cache.write_tokens(slots, seq_ids, positions, k[0], v[0])
            g_slots = ([self.n_prelude + d * self.n_core + layer_idx for d in grouped["g_ds"]]
                       if grouped is not None else None)
            out = self._union_attn(q, cache, slots, seq_ids, positions, cu_q1, grouped, g_slots)
            x = self._sandwich_tail(block, x, block.attn.proj(out.reshape(1, n, -1)))
        for i, t in enumerate(toks):
            t.hidden = x[0, i]
            t.step += 1

    # ------------------------------------------------------------------
    # C — union coda: ln_f -> 2 blocks (per-seq causal group over the shared
    # coda slot) -> ln_f -> lm_head argmax (one sync).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def coda_argmax_union(self, seq_ids: list[int], toks: list[ActiveToken], cache) -> list[int]:
        model = self.model
        n = len(toks)
        positions = [t.position for t in toks]
        x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)
        x = model.transformer.ln_f(x)
        freqs = self._freqs_at(positions)
        # group key = sequence (all of a seq's C tokens share the coda slot;
        # depth key 0 keeps the (seq, key) tuple shape).
        grouped = self._group_meta([(b, 0) for b in seq_ids], positions) if self.group_r else None
        cu_q1 = torch.arange(0, n + 1, dtype=torch.int32, device=self.device)
        for i, block in enumerate(model.transformer.coda):
            slot = self.slot_coda(i)
            h = block.norm_1(x)
            q, k, v = self._qkv(block.attn, h, freqs)
            slots = [slot] * n
            cache.write_tokens(slots, seq_ids, positions, k[0], v[0])
            g_slots = [slot] * len(grouped["g_bs"]) if grouped is not None else None
            out = self._union_attn(q, cache, slots, seq_ids, positions, cu_q1, grouped, g_slots)
            x = self._sandwich_tail(block, x, block.attn.proj(out.reshape(1, n, -1)))
        x = model.transformer.ln_f(x)
        logits = model.lm_head(x).float()                      # (1, n, V)
        return logits[0].argmax(dim=-1).tolist()
