"""Batched Ouro P/R/C primitives for exp4 (fixed batch, union calls).

Same computation as `OuroBlocks` (P = embed, R = 48-layer ut-step + final norm,
C = bare lm_head), generalized to a fixed batch of B sequences sharing one
`BatchStaticKVCache`:

  * `prefill(input_ids (B, L))`  — all sequences at once (equal prompt lengths),
    dense causal attention, fills every (slot, b) prefix.
  * `advance_union(seq_ids, toks, cache)` — advance the UNION of tokens the
    schedulers need this tick (any mix of sequences / positions / ut-steps) by
    one ut-step: per layer, one vectorized cache write + one packed gather +
    ONE varlen attention call; MLP/norms run on the whole union in one shot.
  * `embed_union` / `lm_head_argmax_union` — trivial batched P / C.

KV-sharing across steps (exp3's `kv_budget_s`, round-robin `step % s`) is
supported natively: with a budget the cache is ALLOCATED with only s depth
slots, which is exactly where sharing buys batch size (exp4_plan.md §1).

Attention goes through the `fa4_attention` backend layer (torch-SDPA default),
so this file contains no kernel code. Separate from `ouro_blocks.py` per the
exp4 isolation rule (existing single-sequence code is not modified).
"""
from __future__ import annotations

import torch

from . import fa4_attention as fa
from .batch_kv_cache import BatchStaticKVCache
from .ouro_blocks import _apply_rope
from .state import ActiveToken

DEFAULT_MAX_SEQ_LEN = 2048


class OuroBlocksBatch:
    def __init__(self, causal_lm, T_total: int, batch_size: int,
                 max_seq_len: int = DEFAULT_MAX_SEQ_LEN, kv_budget_s: int | None = None):
        self.model = causal_lm            # OuroForCausalLM (.lm_head + .model)
        self.inner = causal_lm.model      # OuroModel
        self.config = causal_lm.config
        self.T_total = T_total
        self.B = batch_size
        self.n_layers = self.config.num_hidden_layers
        self.max_seq_len = max_seq_len
        # 0 / None -> off (one slot per depth = lossless exp2 behaviour)
        self.kv_budget_s = kv_budget_s if kv_budget_s else None
        self.device = next(causal_lm.parameters()).device
        self.dtype = next(causal_lm.parameters()).dtype
        self.n_kv_head = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim",
                                self.config.hidden_size // self.config.num_attention_heads)
        # R grouping (2026-07-13, mirrors OuroBlocks.group_r): tokens sharing one
        # (sequence, depth-slot) — KV-sharing waves, DtV verify sources — become
        # ONE bottom-right-causal varlen segment, so the shared slot prefix is
        # read once per group instead of once per token. Numerically identical
        # attention; all-singleton groups fall back to the per-token path.
        self.group_r = True
        # q1 zero-pad routing (2026-09-15): single-query varlen reads are
        # template-downgraded on sm86 (non-causal (128,32) tile, ~2x slower per
        # byte than the causal (64,64) tile the grouped path runs) — pad them
        # to q_len=2 causal so AR / per-token / grouped reads share the same
        # per-byte read cost (long-context fairness; see
        # fa4_attention.attn_varlen_q1_padded). False = original 0915-and-
        # earlier path.
        self.pad_q1 = True

    # ------------------------------------------------------------------
    def _sharing(self) -> bool:
        return self.kv_budget_s is not None and self.kv_budget_s < self.T_total

    def _slot_depth(self, step: int) -> int:
        return (step % self.kv_budget_s) if self._sharing() else step

    def n_depth_slots(self) -> int:
        return self.kv_budget_s if self._sharing() else self.T_total

    def slot_of(self, step: int, layer_idx: int) -> int:
        return self._slot_depth(step) * self.n_layers + layer_idx

    def create_cache(self) -> BatchStaticKVCache:
        return BatchStaticKVCache(
            n_slots=self.n_depth_slots() * self.n_layers,
            batch_size=self.B,
            n_kv_head=self.n_kv_head,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            device=self.device,
        )

    def cache_nbytes_estimate(self) -> int:
        elem = 2 if self.dtype in (torch.bfloat16, torch.float16) else 4
        return (self.n_depth_slots() * self.n_layers * self.B * self.max_seq_len
                * self.n_kv_head * self.head_dim * elem * 2)

    # ------------------------------------------------------------------
    def _rope(self, x_like: torch.Tensor, positions: list[int]):
        pos_ids = torch.tensor(positions, device=self.device, dtype=torch.long).unsqueeze(0)
        return self.inner.rotary_emb(x_like, pos_ids)      # cos/sin (1, n, Dh) — broadcasts over B

    def _compute_qkv(self, attn, x, cos, sin):
        """Mirrors OuroBlocks._compute_qkv (projections + Llama-style RoPE)."""
        B, T, _ = x.shape
        H = self.config.num_attention_heads
        Hkv = self.config.num_key_value_heads
        Dh = attn.head_dim
        q = attn.q_proj(x).view(B, T, H, Dh)
        k = attn.k_proj(x).view(B, T, Hkv, Dh)
        v = attn.v_proj(x).view(B, T, Hkv, Dh)
        q, k = _apply_rope(q, k, cos, sin)
        return q, k, v

    def _layer_mlp(self, layer, x):
        residual = x
        h = layer.post_attention_layernorm(x)
        h = layer.mlp(h)
        h = layer.post_attention_layernorm_2(h)
        return residual + h

    # ------------------------------------------------------------------
    # PREFILL — all B sequences (equal prompt length) through T_total ut-steps.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        assert input_ids.dim() == 2 and input_ids.shape[0] == self.B, \
            f"prefill expects (B={self.B}, L) input_ids, got {tuple(input_ids.shape)}"
        cache = self.create_cache()
        B, L = input_ids.shape
        x = self.inner.embed_tokens(input_ids)                       # (B, L, D)
        cos, sin = self._rope(x, list(range(L)))
        for ut in range(self.T_total):
            for layer_idx, layer in enumerate(self.inner.layers):
                residual = x
                h = layer.input_layernorm(x)
                q, k, v = self._compute_qkv(layer.self_attn, h, cos, sin)   # (B, L, ...)
                cache.write_prefill(self.slot_of(ut, layer_idx), k, v)
                out = fa.attn_dense(q, k, v, causal=True)                   # Tq == Tk
                attn_out = layer.self_attn.o_proj(out.reshape(B, L, -1))
                attn_out = layer.input_layernorm_2(attn_out)
                x = residual + attn_out
                x = self._layer_mlp(layer, x)
            x = self.inner.norm(x)                                          # per-ut final norm
        last_logits = self.model.lm_head(x[:, -1]).float()                  # (B, V)
        return cache, last_logits

    # ------------------------------------------------------------------
    # P — batched embedding of the union of new tokens.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def embed_union(self, token_ids: list[int]) -> torch.Tensor:
        ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        return self.inner.embed_tokens(ids)[0]                              # (n, D)

    @torch.no_grad()
    def prelude_union(self, seq_ids: list[int], toks: list[ActiveToken], cache) -> None:
        """Model-generic P interface (scheduler_batch): Ouro's prelude is a bare
        embedding lookup — no attention, no cache traffic. seq_ids/cache unused."""
        hid = self.embed_union([t.token_id for t in toks])
        for i, t in enumerate(toks):
            t.hidden = hid[i]
            t.static_state = None
            t.step = 0

    @torch.no_grad()
    def coda_argmax_union(self, seq_ids: list[int], toks: list[ActiveToken], cache) -> list[int]:
        """Model-generic C interface: Ouro's coda is a bare lm_head."""
        return self.lm_head_argmax_union(toks)

    # ------------------------------------------------------------------
    # R — advance the union of tokens by ONE ut-step (in place).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def advance_union(self, seq_ids: list[int], toks: list[ActiveToken], cache) -> None:
        n = len(toks)
        positions = [t.position for t in toks]
        steps = [t.step for t in toks]
        x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)       # (1, n, D)
        cos, sin = self._rope(x, positions)

        # ---- (seq, depth-slot) grouping — hoisted, identical every layer ----
        grouped = None
        if self.group_r:
            by_key: dict[tuple[int, int], list[int]] = {}
            for i in range(n):
                by_key.setdefault((seq_ids[i], self._slot_depth(steps[i])), []).append(i)
            if any(len(idxs) > 1 for idxs in by_key.values()):
                g_bs, g_ds, g_pmax, q_lens = [], [], [], []
                rows = [0] * n
                base = 0
                for (b, d), idxs in by_key.items():
                    idxs.sort(key=lambda i: positions[i])
                    pmin, pmax = positions[idxs[0]], positions[idxs[-1]]
                    span = pmax - pmin + 1
                    g_bs.append(b); g_ds.append(d); g_pmax.append(pmax); q_lens.append(span)
                    for i in idxs:
                        rows[i] = base + positions[i] - pmin
                    base += span
                identity = base == n and rows == list(range(n))
                row_t = None if identity else torch.tensor(rows, dtype=torch.long,
                                                           device=self.device)
                cu_q_g = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=self.device)
                cu_q_g[1:] = torch.tensor(q_lens, dtype=torch.int32,
                                          device=self.device).cumsum(0)
                grouped = (g_bs, g_ds, g_pmax, q_lens, base, identity, row_t, cu_q_g)

        cu_q1 = torch.arange(0, n + 1, dtype=torch.int32, device=self.device)
        for layer_idx, layer in enumerate(self.inner.layers):
            residual = x
            h = layer.input_layernorm(x)
            q, k, v = self._compute_qkv(layer.self_attn, h, cos, sin)       # (1, n, ...)
            slots = [self.slot_of(steps[i], layer_idx) for i in range(n)]
            cache.write_tokens(slots, seq_ids, positions, k[0], v[0])
            if grouped is not None:
                g_bs, g_ds, g_pmax, q_lens, total_q, identity, row_t, cu_q_g = grouped
                g_slots = [d * self.n_layers + layer_idx for d in g_ds]
                k_pack, v_pack, cu_k, max_k = cache.gather_pack(g_slots, g_bs, g_pmax)
                if identity:
                    q_pack = q[0]
                else:
                    q_pack = q.new_zeros(total_q, q.shape[2], q.shape[3])
                    q_pack.index_copy_(0, row_t, q[0])
                out = fa.attn_varlen(q_pack, k_pack, v_pack, cu_q_g, cu_k,
                                     max(q_lens), max_k, causal=True)
                if not identity:
                    out = out.index_select(0, row_t)                # drop pads, token order
            else:
                k_pack, v_pack, cu_k, max_k = cache.gather_pack(slots, seq_ids, positions)
                if self.pad_q1:
                    out = fa.attn_varlen_q1_padded(q[0], k_pack, v_pack, cu_k, max_k)
                else:
                    out = fa.attn_varlen(q[0], k_pack, v_pack, cu_q1, cu_k, 1, max_k)
            attn_out = layer.self_attn.o_proj(out.reshape(1, n, -1))
            attn_out = layer.input_layernorm_2(attn_out)
            x = residual + attn_out
            x = self._layer_mlp(layer, x)
        x = self.inner.norm(x)
        for i, t in enumerate(toks):
            t.hidden = x[0, i]
            t.step += 1

    # ------------------------------------------------------------------
    # C — bare lm_head over the union; returns greedy ids (one sync total).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def lm_head_argmax_union(self, toks: list[ActiveToken]) -> list[int]:
        x = torch.stack([t.hidden for t in toks], dim=0)                    # (n, D)
        logits = self.model.lm_head(x).float()                              # (n, V)
        return logits.argmax(dim=-1).tolist()
