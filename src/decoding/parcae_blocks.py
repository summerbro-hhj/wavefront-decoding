"""Parcae P/R/C implementation for wavefront SSD.

We borrow the released parcae model's weights but re-write the forward path so
that, inside a single R call, attention is split per-step (per cache slot)
while MLP / norm / adapter run on the mixed-step batch in one shot.

Attention backend: all the *decode-time* attention we issue (P prelude, R core,
C coda — single and multi-query) goes through `fa4_attention` (backend layer:
torch-SDPA default / FA4 / SDPA-ref — see its docstring). Only the one-shot
`prefill` keeps parcae's native path (it just has to populate the StaticKVCache;
its cost is amortised and identical for AR vs SSD). The AR baseline and the DtV
scheduler run on the same block primitives, so WFD / AR / DtV share the backend.

KV cache: `StaticKVCache` (kv_cache.py) — per-step-slot preallocated buffers,
position == buffer index. step_idx layout (matches parcae forward_for_generation):
    - prelude layer i           -> step_idx = i
    - core recurrence r layer l -> step_idx = n_prelude + r*n_core + l
    - coda layer i              -> step_idx = n_prelude + T_total*n_core + i
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

# Ensure the vendored parcae repo is importable before importing parcae_lm.
_PARCAE_REPO_DIR = Path(__file__).resolve().parents[2] / "external" / "parcae"
if _PARCAE_REPO_DIR.exists() and str(_PARCAE_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_PARCAE_REPO_DIR))

from . import fa4_attention as fa
from .blocks import RecursiveBlocks
from .kv_cache import create_static_parcae_cache
from .state import WavefrontState

DEFAULT_MAX_SEQ_LEN = 2048


def _create_parcae_cache(model, T_total: int, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
    """StaticKVCache sized for `model` at depth T_total. Replaces the old
    dict-based SortedKVCache (see kv_cache.py / fa4_backend_plan.md §4c)."""
    return create_static_parcae_cache(model, T_total, max_seq_len)


def _ve_for_prelude(model, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor | None:
    key = str(layer_idx)
    return model.value_embeds[key](input_ids) if key in model.value_embeds else None


def _ve_for_core(model, input_ids: torch.Tensor, layer_idx_in_core: int) -> torch.Tensor | None:
    n_prelude = len(model.transformer.prelude)
    key = str(n_prelude + layer_idx_in_core)
    return model.value_embeds[key](input_ids) if key in model.value_embeds else None


def _ve_for_coda(model, input_ids: torch.Tensor, layer_idx_in_coda: int) -> torch.Tensor | None:
    n_prelude = len(model.transformer.prelude)
    n_core = len(model.transformer.core_block)
    key = str(n_prelude + n_core + layer_idx_in_coda)
    return model.value_embeds[key](input_ids) if key in model.value_embeds else None


class ParcaeBlocks(RecursiveBlocks):
    """P / R / C primitives for a parcae checkpoint (FA4 attention)."""

    def __init__(self, model, T_total: int, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
        self.model = model
        self.config = model.config
        self.T_total = T_total
        self.max_seq_len = max_seq_len
        self.n_prelude = len(model.transformer.prelude)
        self.n_core = len(model.transformer.core_block)
        self.n_coda = len(model.transformer.coda)
        self.coda_base = self.n_prelude + T_total * self.n_core
        self.device = next(model.parameters()).device
        # exp3 KV-sharing budget s (None or >= T_total → off = exact exp2). Shared
        # by AR + SSD. Applies to the recurrent core depths only (prelude/coda
        # slots unchanged); varlen path, not mask_mod (see exp3_plan §latency).
        self.kv_budget_s = None
        # exp3 adaptive-total-T threshold ε (None = off = full depth = exact exp2).
        # parcae has no early_exit_gate, so the halt signal is latent convergence:
        # commit at the first core step whose hidden's relative-L2 step-to-step
        # change ‖Δh‖/‖h‖ < ε (see exp3_plan §"adaptive-total-T"). Common to AR+SSD.
        # Named separately from Ouro's `exit_threshold` (τ, a probability) since ε
        # is a relative-L2 residual — different scale.
        self.exit_hidden_eps = None
        # R grouping (2026-07-13): tokens sharing one physical core-KV slot
        # (same-step DtV verify sources; KV-sharing waves) become ONE causal
        # varlen segment — slot K/V read once per group instead of per token.
        # Numerically the same attention; all-distinct slots (lossless WFD)
        # fall back to the exact per-token path. False = old path. Mirrors
        # OuroBlocks.group_r.
        self.group_r = True
        # Initial recurrent state s0 (2026-07-14): "random" restores the
        # model-native like-init distribution (trunc_normal(std) * emb_scale;
        # the wrapper's config still says "zero" — we bypass
        # model.initialize_state here). Draws are SEEDED PER POSITION (prefill:
        # one stream from state_seed), so AR / WFD / DtV sample the identical
        # s0 for the same position — the lossless gates (AR ≡ WFD@T_draft=
        # T_total, deterministic rollback re-P) still hold. "zero" = previous
        # deterministic behaviour (pre-2026-07-14 results were measured with it).
        self.state_init = "random"
        self.state_seed = 0
        self._state_dim, self._state_std, self._state_scale = self._state_init_params(model)

    def _state_init_params(self, model) -> tuple[int, float, float]:
        """(dim, std, scale) of the model's like-init recipe (overridden by
        HuginnBlocks, whose config exposes the std differently)."""
        dim = getattr(self.config, "recurrent_embedding_dimension", None) or self.config.n_embd
        return dim, float(self.config.init.get_std("embedding")), float(model.emb_scale)

    def _sharing(self) -> bool:
        return self.kv_budget_s is not None and self.kv_budget_s < self.T_total

    def _slot(self, r: int) -> int:
        """Core recurrence index r -> recurrence-slot index. With budget s, depths
        collapse round-robin to s slots (`r % s`, last write wins); else identity."""
        return (r % self.kv_budget_s) if self._sharing() else r

    def _adaptive(self) -> bool:
        return self.exit_hidden_eps is not None

    # ------------------------------------------------------------------
    # Initial recurrent state s0 (shared by parcae and HuginnBlocks —
    # both models use the like-init recipe trunc_normal(std) * emb_scale).
    # ------------------------------------------------------------------
    def _trunc_normal_seeded(self, out32: torch.Tensor, seed: int) -> None:
        g = torch.Generator(device=self.device)
        g.manual_seed(seed & 0x7FFF_FFFF_FFFF)
        torch.nn.init.trunc_normal_(
            out32, mean=0.0, std=self._state_std,
            a=-3 * self._state_std, b=3 * self._state_std, generator=g,
        )

    def _init_state_tokens(self, positions: list[int]) -> torch.Tensor:
        """(len(positions), D) s0 draws — deterministic per (state_seed, position)."""
        dtype = next(self.model.parameters()).dtype
        n = len(positions)
        if self.state_init == "zero":
            return torch.zeros(n, self._state_dim, device=self.device, dtype=dtype)
        out = torch.empty(n, self._state_dim, device=self.device, dtype=torch.float32)
        for j, p in enumerate(positions):
            self._trunc_normal_seeded(out[j], self.state_seed * 1_000_003 + p + 1)
        return (out * self._state_scale).to(dtype)

    def _init_state_prefill(self, batch: int, length: int) -> torch.Tensor:
        """(batch, length, D) prompt s0 — one seeded stream (prompt positions are
        never re-drawn, so per-position seeding is unnecessary here)."""
        dtype = next(self.model.parameters()).dtype
        if self.state_init == "zero":
            return torch.zeros(batch, length, self._state_dim, device=self.device, dtype=dtype)
        out = torch.empty(batch, length, self._state_dim, device=self.device, dtype=torch.float32)
        self._trunc_normal_seeded(out, self.state_seed * 1_000_003)
        return (out * self._state_scale).to(dtype)

    def _freeze_kv(self, cache, pos: int, exit_step: int) -> None:
        """Early-exit core-KV pad (parcae). A token that halts at core-step
        `exit_step` (1..T_total-1) has written recurrence slots only for steps
        [0..exit_step-1]; replicate its last-computed step (exit_step-1) per-layer
        K/V into the skipped deeper step slots [exit_step..T_total-1] so later
        wavefront tokens reaching those depths still find this position's K/V
        (mirrors OuroBlocks._freeze_kv; prelude/coda slots untouched). Only
        meaningful with KV-sharing off (round-robin already keeps slots populated)."""
        last = exit_step - 1
        for layer_idx in range(self.n_core):
            src = self.n_prelude + last * self.n_core + layer_idx
            dst = [self.n_prelude + e * self.n_core + layer_idx
                   for e in range(exit_step, self.T_total)]
            cache.freeze_replicate(src, dst, pos)

    def ar_generate(self, prompt_ids, max_new_tokens, eos_token_id=None, sampler=None):
        """Greedy AR baseline (parcae). Uses the FA4 P/R/C-block sequential AR so
        AR and SSD share the same decode attention backend (apples-to-apples, like
        Ouro). The native-forward baseline is still available as
        `ar_baseline.ar_generate_parcae`. Lazy import avoids an import cycle."""
        from .ar_baseline import ar_generate_parcae_blocks
        return ar_generate_parcae_blocks(self, prompt_ids, max_new_tokens, eos_token_id, sampler=sampler)

    # ------------------------------------------------------------------
    # Unified low-level primitives (see RecursiveBlocks) — thin public wrappers
    # over parcae's internal P (prelude) / R (core step) / C (coda+head).
    # ------------------------------------------------------------------
    def prelude_forward(self, cache, token_id: int, position: int):
        return self._prelude_forward(cache, token_id, position)   # (hidden, static_state)

    def advance_one_step(self, toks: list, cache) -> None:
        self._advance_core(toks, cache)

    def coda_logits(self, state, toks: list):
        return self._coda_logits(state, toks)

    # ------------------------------------------------------------------
    # internal: FA4 logits head (coda output -> vocab logits)
    # ------------------------------------------------------------------
    def _head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.transformer.ln_f(x)
        logits = self.model.lm_head(x).float() * self.config.init.logit_scale
        if self.config.logit_softcap is not None:
            sc = self.config.logit_softcap
            logits = sc * torch.tanh(logits / sc)
        return logits

    # ------------------------------------------------------------------
    # PREFILL: native parcae forward over the whole prompt, populating every
    # StaticKVCache slot. (Native/SDPA — one-shot, amortised; AR pays the same.)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.Tensor,
        T_total: int,
        T_draft: int,
    ) -> tuple[WavefrontState, torch.Tensor]:
        assert T_total == self.T_total, "Prefill T_total must match the blocks' T_total"
        model = self.model
        cache = _create_parcae_cache(model, T_total, self.max_seq_len)
        seq_len = input_ids.shape[1]
        freqs_cis = model.freqs_cis[:, :seq_len]
        model._current_input_ids = input_ids

        # --- Prelude ---
        x = model.transformer.wte(input_ids)
        if model.emb_scale != 1:
            x = x * model.emb_scale
        for i, block in enumerate(model.transformer.prelude):
            ve = _ve_for_prelude(model, input_ids, i)
            x = block(x, freqs_cis, None, past_key_values=cache,
                      step_idx=torch.tensor(i, dtype=torch.long), ve=ve)
        if self.config.prelude_norm:
            x = model.transformer.ln_prelude(x)
        input_embeds = x

        # --- Recurrent core ---
        x = self._init_state_prefill(input_embeds.shape[0], seq_len)
        total_steps = torch.tensor(T_total, device=self.device)
        for r in range(T_total):
            x = model.core_block_forward(
                x, input_embeds, freqs_cis, None,
                step=torch.tensor(r, device=self.device),
                total_steps=total_steps,
                past_key_values=cache,
                step_idx_base=self.n_prelude + self._slot(r) * self.n_core,
            )

        # --- Coda + head ---
        x = model.transformer.C(x)
        for i, block in enumerate(model.transformer.coda):
            ve = _ve_for_coda(model, input_ids, i)
            x = block(x, freqs_cis, None, past_key_values=cache,
                      step_idx=torch.tensor(self.coda_base + i, dtype=torch.long), ve=ve)
        logits = self._head(x)
        last_token_logits = logits[0, -1, :]

        state = WavefrontState(
            active=[], committed=[], prefix_len=seq_len, kv_cache=cache,
            T_total=T_total, T_draft=T_draft, last_committed_position=seq_len - 1,
        )
        return state, last_token_logits

    # ------------------------------------------------------------------
    # FA4 attention for ONE transformer block (norm_1 -> attn(FA4) -> +res
    #                                          -> mlp(norm_2) -> +res).
    # Used by P (prelude) / R single (core) / C single (coda): Tq=1 query at
    # `position` attending its step slot's prefix [0, position] (causal).
    # ------------------------------------------------------------------
    def _block_fa4_single(self, block, x, freqs, ve, cache, step_idx, position):
        B, T, C = x.shape   # T == 1
        h = block.norm_1(x)
        q, k, v = fa.compute_qkv(block.attn, h, freqs, ve, self.config)   # (1,1,H,Dh)/(1,1,Hkv,Dh)
        cache.write_range(step_idx, position, k, v)
        K, V = cache.gather(step_idx, position)                          # (1, pos+1, Hkv, Dh)
        out = fa.attn_dense(q, K, V, causal=True)                        # (1, 1, H, Dh)
        attn_out = block.attn.c_proj(out.reshape(B, T, C))
        x = x + attn_out
        x = x + block.mlp(block.norm_2(x))
        return x

    # ------------------------------------------------------------------
    # P: process one new token through the prelude (FA4), append ActiveToken.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _prelude_forward(self, cache, token_id: int, position: int):
        """Run the prelude (FA4) for one new token at `position`, writing its
        prelude K/V to the cache. Returns (hidden0, static_state) for the
        recurrent core. Shared by P (static) and P_dyn (dynamic)."""
        model = self.model
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        freqs_cis = model.freqs_cis[:, position : position + 1]
        model._current_input_ids = input_ids
        x = model.transformer.wte(input_ids)
        if model.emb_scale != 1:
            x = x * model.emb_scale
        for i, block in enumerate(model.transformer.prelude):
            ve = _ve_for_prelude(model, input_ids, i)
            x = self._block_fa4_single(block, x, freqs_cis, ve, cache, step_idx=i, position=position)
        if self.config.prelude_norm:
            x = model.transformer.ln_prelude(x)
        static_state = x[0, 0]
        hidden0 = self._init_state_tokens([position])[0]
        return hidden0, static_state

    # ------------------------------------------------------------------
    # R core: advance step of alive tokens by 1. Attention via FA4 —
    # single token -> dense Tq=1, multiple -> one varlen-batched call
    # (each query attends only its own step slot — lossless, no KV sharing).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _advance_core(self, toks, cache) -> None:
        """Advance a list of alive ActiveTokens by ONE recurrent step, in place
        (updates each .hidden and .step += 1). Core attention via FA4: single
        token = dense Tq=1; multiple = one varlen-batched call (each query
        attends only its own step slot — lossless). Shared by R and R_dyn."""
        model = self.model
        core = model.transformer.core_block
        n = len(toks)
        positions = [t.position for t in toks]
        steps = [t.step for t in toks]
        if n == 1:
            x = toks[0].hidden.unsqueeze(0).unsqueeze(0)
            input_embeds = toks[0].static_state.unsqueeze(0).unsqueeze(0)
        else:
            x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)
            input_embeds = torch.stack([t.static_state for t in toks], dim=0).unsqueeze(0)
        p0 = positions[0]
        if positions[-1] - p0 + 1 == n:
            freqs_cis = model.freqs_cis[:, p0 : p0 + n]
        else:
            freqs_cis = model.freqs_cis[:, torch.tensor(positions, device=self.device)]
        input_ids_batch = torch.tensor([[t.token_id for t in toks]], dtype=torch.long, device=self.device)
        model._current_input_ids = input_ids_batch

        # Adapter (token-wise; mixed-step OK)
        x = model.transformer.adapter(x, input_embeds)

        if n == 1:
            s, pos = steps[0], positions[0]
            for layer_idx, block in enumerate(core):
                ve = _ve_for_core(model, input_ids_batch, layer_idx)
                step_idx = self.n_prelude + self._slot(s) * self.n_core + layer_idx
                x = self._block_fa4_single(block, x, freqs_cis, ve, cache, step_idx, pos)
        else:
            gctx = self._group_ctx(steps, positions) if self.group_r else None
            for layer_idx, block in enumerate(core):
                ve = _ve_for_core(model, input_ids_batch, layer_idx)
                h = block.norm_1(x)
                q, k, v = fa.compute_qkv(block.attn, h, freqs_cis, ve, self.config)  # (1,n,H/Hkv,Dh)
                out = self._core_attn_varlen(q, k, v, cache, steps, positions, layer_idx, gctx)
                _, _, C = x.shape
                attn_out = block.attn.c_proj(out.reshape(1, n, C))
                x = x + attn_out
                x = x + block.mlp(block.norm_2(x))

        # adaptive-total-T (parcae): latent fixed-point convergence. Compare each
        # token's new core hidden to its previous step's (relative-L2); flag
        # should_exit when ‖Δh‖/‖h_prev‖ < ε — but never at the final step (pre-inc
        # step == T_total-1; committed by the step==T_total rule anyway) and only
        # once a previous hidden exists (so the earliest early-exit is step 2).
        rels = None
        if self._adaptive():
            cur = x[0]                                                  # (n, D)
            prev_stack = torch.stack(
                [t.prev_hidden if t.prev_hidden is not None else cur[i]
                 for i, t in enumerate(toks)], dim=0)                   # (n, D)
            num = torch.linalg.vector_norm(cur - prev_stack, dim=1)
            den = torch.linalg.vector_norm(prev_stack, dim=1).clamp_min(1e-6)
            rels = (num / den).tolist()                                 # one sync/call

        for i, t in enumerate(toks):
            if self._adaptive():
                if t.prev_hidden is not None and t.step < self.T_total - 1 \
                        and rels[i] < self.exit_hidden_eps:
                    t.should_exit = True
                t.prev_hidden = x[0, i].clone()                         # d-vector, survives to next step
            t.hidden = x[0, i]
            t.step += 1
            # Freeze skipped deeper core-KV slots NOW (at halt), not at commit —
            # a halted token waits (frozen) until its whole earlier prefix commits,
            # and meanwhile shallower tokens may advance past its halt depth.
            if t.should_exit and not self._sharing():
                self._freeze_kv(cache, t.position, t.step)

    # ------------------------------------------------------------------
    # R core attention: write all K/V, then one varlen call. Mirrors
    # OuroBlocks._block_varlen (see its docstring): grouped path merges
    # same-slot tokens into one bottom-right-causal segment (slot K/V read
    # once per group; zero-pad rows for position gaps, outputs discarded);
    # per-token path is the original exp2/exp3 behaviour.
    # ------------------------------------------------------------------
    def _group_ctx(self, steps, positions):
        """Precompute the grouped-R pack layout for one advance (identical at
        every layer). Mirrors OuroBlocks._group_ctx; groups key on the core
        recurrence slot `_slot(step)`. None when all groups are singletons."""
        by_slot: dict[int, list[int]] = {}
        for i, s in enumerate(steps):
            by_slot.setdefault(self._slot(s), []).append(i)
        if all(len(idxs) == 1 for idxs in by_slot.values()):
            return None
        n = len(steps)
        dev = self.device
        groups, q_lens, k_lens = [], [], []
        rows = [0] * n
        base = 0
        for dslot, idxs in by_slot.items():
            idxs.sort(key=lambda i: positions[i])
            pmin, pmax = positions[idxs[0]], positions[idxs[-1]]
            span = pmax - pmin + 1
            whole_q = span == n and idxs == list(range(n))
            idxs_t = None if whole_q else torch.tensor(idxs, dtype=torch.long, device=dev)
            pad_rows_t = (None if span == len(idxs) else
                          torch.tensor([positions[i] - pmin for i in idxs],
                                       dtype=torch.long, device=dev))
            groups.append((dslot, pmax, span, whole_q, idxs_t, pad_rows_t))
            q_lens.append(span)
            k_lens.append(pmax + 1)
            for i in idxs:
                rows[i] = base + positions[i] - pmin
            base += span
        cu_q = torch.zeros(len(q_lens) + 1, dtype=torch.int32, device=dev)
        cu_q[1:] = torch.tensor(q_lens, dtype=torch.int32, device=dev).cumsum(0)
        cu_k = torch.zeros(len(k_lens) + 1, dtype=torch.int32, device=dev)
        cu_k[1:] = torch.tensor(k_lens, dtype=torch.int32, device=dev).cumsum(0)
        identity = base == n and rows == list(range(n))
        rows_t = None if identity else torch.tensor(rows, dtype=torch.long, device=dev)
        return dict(groups=groups, cu_q=cu_q, cu_k=cu_k,
                    max_q=max(q_lens), max_k=max(k_lens), rows_t=rows_t)

    def _core_attn_varlen(self, q, k, v, cache, steps, positions, layer_idx, gctx=None):
        n = q.shape[1]
        for i in range(n):
            step_idx = self.n_prelude + self._slot(steps[i]) * self.n_core + layer_idx
            cache.write_range(step_idx, positions[i], k[:, i : i + 1], v[:, i : i + 1])

        if gctx is not None:
            q_segs, k_segs, v_segs = [], [], []
            for dslot, pmax, span, whole_q, idxs_t, pad_rows_t in gctx["groups"]:
                step_idx = self.n_prelude + dslot * self.n_core + layer_idx
                Kg, Vg = cache.gather(step_idx, pmax)               # (1, pmax+1, Hkv, Dh)
                k_segs.append(Kg[0]); v_segs.append(Vg[0])
                if whole_q:
                    qseg = q[0]
                elif pad_rows_t is None:                            # contiguous positions
                    qseg = q[0].index_select(0, idxs_t)
                else:                                               # zero-pad the gaps
                    qseg = q.new_zeros(span, q.shape[2], q.shape[3])
                    qseg.index_copy_(0, pad_rows_t, q[0].index_select(0, idxs_t))
                q_segs.append(qseg)
            q_pack = q_segs[0] if len(q_segs) == 1 else torch.cat(q_segs, dim=0)
            k_pack = k_segs[0] if len(k_segs) == 1 else torch.cat(k_segs, dim=0)
            v_pack = v_segs[0] if len(v_segs) == 1 else torch.cat(v_segs, dim=0)
            out = fa.attn_varlen(q_pack, k_pack, v_pack, gctx["cu_q"], gctx["cu_k"],
                                 gctx["max_q"], gctx["max_k"], causal=True)
            if gctx["rows_t"] is not None:
                out = out.index_select(0, gctx["rows_t"])
            return out

        # per-token (original): each query gathers its own prefix copy
        K_list, V_list, klens = [], [], []
        for i in range(n):
            step_idx = self.n_prelude + self._slot(steps[i]) * self.n_core + layer_idx
            Ki, Vi = cache.gather(step_idx, positions[i])            # (1, pos_i+1, Hkv, Dh)
            K_list.append(Ki[0]); V_list.append(Vi[0]); klens.append(Ki.shape[1])
        q_pack = q[0]                                                # (n, H, Dh)
        k_pack = torch.cat(K_list, dim=0)                            # (sum klens, Hkv, Dh)
        v_pack = torch.cat(V_list, dim=0)
        cu_q = torch.arange(0, n + 1, device=self.device, dtype=torch.int32)
        cu_k = torch.zeros(n + 1, device=self.device, dtype=torch.int32)
        cu_k[1:] = torch.tensor(klens, device=self.device, dtype=torch.int32).cumsum(0)
        return fa.attn_varlen(q_pack, k_pack, v_pack, cu_q, cu_k, 1, max(klens))

    # ------------------------------------------------------------------
    # C core: coda forward + head for one or more tokens.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _coda_logits(self, state: WavefrontState, toks: list) -> list[torch.Tensor]:
        """Coda forward + head for a list of ActiveTokens → per-token logits in
        the SAME order as `toks`. nq==1: dense Tq=1 (query attends its coda-slot
        prefix). nq>1: disjoint multi-query — queries share the coda slots, each
        attends keys whose abs position <= its own (FA4 mask_mod); one coda pass
        instead of nq. Shared by C_multi (static) and C_multi_dyn (dynamic)."""
        model = self.model
        cache = state.kv_cache
        nq = len(toks)
        positions = [t.position for t in toks]
        input_ids = torch.tensor([[t.token_id for t in toks]], dtype=torch.long, device=self.device)
        model._current_input_ids = input_ids

        if nq == 1:
            pos = positions[0]
            freqs = model.freqs_cis[:, pos : pos + 1]
            x = toks[0].hidden.unsqueeze(0).unsqueeze(0)
            x = model.transformer.C(x)
            for i, block in enumerate(model.transformer.coda):
                ve = _ve_for_coda(model, input_ids, i)
                x = self._block_fa4_single(block, x, freqs, ve, cache, self.coda_base + i, pos)
            return [self._head(x)[0, 0]]

        # multi-query (disjoint positions, shared coda slot — per-query cutoff:
        # FA4 mask_mod / torch zero-pad-causal, see fa4_attention.attn_cutoff_multi)
        freqs = model.freqs_cis[:, torch.tensor(positions, device=self.device)]   # (1, nq, ...)
        upto = max(positions)
        x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)             # (1, nq, D)
        x = model.transformer.C(x)
        for i, block in enumerate(model.transformer.coda):
            ve = _ve_for_coda(model, input_ids, i)
            sidx = self.coda_base + i
            B, T, C = x.shape
            h = block.norm_1(x)
            q, k, v = fa.compute_qkv(block.attn, h, freqs, ve, self.config)       # (1,nq,...)
            cache.write_scatter(sidx, positions, k, v)
            K, V = cache.gather(sidx, upto)                                       # (1, upto+1, Hkv, Dh)
            out = fa.attn_cutoff_multi(q, K, V, positions)                        # (1, nq, H, Dh)
            attn_out = block.attn.c_proj(out.reshape(B, T, C))
            x = x + attn_out
            x = x + block.mlp(block.norm_2(x))
        logits = self._head(x)
        return [logits[0, i] for i in range(nq)]

    # ==================================================================
    # Dynamic Sync Scheduler block variants — see .claude/plans/scheduler.md.
    # Signature (state) -> (state, call_flag...). active_{p,r,c} carry tokens by
    # the block they next go through; compute reuses the static helpers above.
    # role: tokens enter as "draft"; R_dyn promotes to "verify" at a commit point.
    # ==================================================================
    @torch.no_grad()
    def P_dyn(self, state: WavefrontState):
        """Prelude every active_p token (FA4), fill its hidden/static_state, move
        it to active_r at step 0, clear active_p. Returns (state, call_r=True)."""
        cache = state.kv_cache
        for tok in state.active_p:
            hidden0, static_state = self._prelude_forward(cache, tok.token_id, tok.position)
            tok.hidden = hidden0
            tok.static_state = static_state
            tok.step = 0
            state.active_r.append(tok)
        state.active_p = []
        return state, True

    @torch.no_grad()
    def R_dyn(self, state: WavefrontState):
        """Advance active_r one step, then route policy-point tokens to active_c:
        commit point (step==T_total, or should_exit if a future parcae early-exit
        signal sets it) -> promote "verify" + MOVE; draft point (step==T_draft) ->
        keep "draft" + COPY. Commits the maximal contiguous commit-point prefix
        from the frontmost (mirrors OuroBlocks.R_dyn); parcae has no early-exit
        signal yet, so should_exit is always False and only the frontmost reaches
        T_total → the run is length ≤1, i.e. the original single-commit wave.
        Returns (state, call_c)."""
        cache = state.kv_cache
        toks = [t for t in state.active_r if t.is_alive and not t.should_exit]
        if toks:
            self._advance_core(toks, cache)
        T_draft, T_total = state.T_draft, state.T_total

        alive = sorted((t for t in state.active_r if t.is_alive), key=lambda t: t.position)
        committers = []
        for t in alive:
            if t.should_exit or t.step == T_total:
                committers.append(t)
            else:
                break
        committer_ids = {id(t) for t in committers}

        survivors, any_c = [], False
        for t in state.active_r:
            if not t.is_alive:
                continue
            if id(t) in committer_ids:             # commit point: promote + MOVE
                t.role = "verify"
                state.active_c.append(t)
                any_c = True
            else:
                survivors.append(t)
                if t.step == T_draft and not t.should_exit:   # draft point: COPY (step-T_draft hidden)
                    c = copy.copy(t)
                    c.role = "draft"
                    state.active_c.append(c)
                    any_c = True
        state.active_r = survivors
        return state, any_c

    @torch.no_grad()
    def C_multi_dyn(self, state: WavefrontState):
        """Coda over active_c WITHOUT emptying it (scheduler reads active_c for
        positions, then clears). Returns (state, n_verify, logits_per_pos): the
        verify tokens' logits (position-sorted) first, then the ≤1 draft's. Mirrors
        OuroBlocks.C_multi_dyn; parcae currently produces n_verify ≤ 1."""
        verify = sorted((t for t in state.active_c if t.role == "verify"),
                        key=lambda t: t.position)
        draft = [t for t in state.active_c if t.role == "draft"]
        assert len(draft) <= 1, "dyn wave invariant: <=1 draft per C"
        ordered = verify + draft
        if not ordered:
            return state, 0, []
        logits = self._coda_logits(state, ordered)
        return state, len(verify), logits
