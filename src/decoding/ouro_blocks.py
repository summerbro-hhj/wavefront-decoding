"""Ouro P/R/C implementation for wavefront SSD (attention via the fa4_attention
backend layer — torch-SDPA default / FA4 / SDPA-ref).

Ouro (`ByteDance/Ouro-*`) is a Universal-Transformer: `OuroModel.forward` loops
the *entire* 24-layer decoder stack `total_ut_steps` times, applying the final
RMSNorm after each loop and feeding that normed hidden into the next loop. The
step index `current_ut` enters the model in exactly ONE place — the attention
KV-cache slot index `current_ut * num_hidden_layers + layer_idx`
(`modeling_ouro.py::OuroAttention.forward`). The decoder-layer body (MLP, the
four RMSNorms, RoPE, projections) is step-INDEPENDENT: the same weights run every
loop. That is the same property parcae has, so the lossless mixed-step batch
argument holds (see `.claude/plans/exp2_plan.md` §"Near-Lossless") and Ouro maps
onto the same P/R/C abstraction as parcae:

    P (prelude) : embed_tokens(token)             — no transformer layers, no cache
    R (one call): the 24-layer stack + final norm — step = current_ut (0..T_total-1)
    C (coda)    : lm_head(hidden)                  — no attention, no cache

vs parcae's prelude=6-layer / core=6-layer / coda=6-layer+head. Differences that
make Ouro *simpler* than parcae here: no value-embeddings, no qk_norm, no adapter
re-injection (R input is just the previous R's normed output), and C is a bare
lm_head so the disjoint multi-query coda fusion collapses to one batched matmul.

KV cache: the same `StaticKVCache` (position == buffer index), with the slot
layout `step_idx = ut_step * num_hidden_layers + layer_idx` (total
`T_total * num_hidden_layers` slots, all written by R — P and C touch no cache).
Attention goes through `fa4_attention` exactly like parcae; only `compute_qkv`
differs (Ouro projections + Llama-style real RoPE instead of parcae's complex
RoPE + qk_norm), so it lives here rather than in the parcae-specific fa4 module.
"""
from __future__ import annotations

import copy

import torch

from . import fa4_attention as fa
from .blocks import RecursiveBlocks
from .kv_cache import StaticKVCache
from .state import WavefrontState

DEFAULT_MAX_SEQ_LEN = 2048


def create_static_ouro_cache(model, T_total: int, max_seq_len: int) -> StaticKVCache:
    """StaticKVCache sized for an Ouro model. Slots = T_total * num_hidden_layers
    (every slot written by R; P/C use no cache)."""
    cfg = model.config
    n_kv_head = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    return StaticKVCache(n_kv_head, head_dim, max_seq_len, dtype, device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    """q, k: (B, T, H, Dh); cos, sin: (B, T, Dh). Llama-style real RoPE, matching
    `modeling_ouro.apply_rotary_pos_emb` but for an (B, T, H, Dh) layout (FA4's
    preferred layout) — i.e. unsqueeze the head dim at axis 2 instead of 1."""
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class OuroBlocks(RecursiveBlocks):
    """P / R / C primitives for an Ouro (`OuroForCausalLM`) checkpoint (FA4)."""

    def __init__(self, causal_lm, T_total: int, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
        self.model = causal_lm            # OuroForCausalLM (has .lm_head + .model)
        self.inner = causal_lm.model      # OuroModel (embed_tokens / layers / norm / rotary_emb)
        self.config = causal_lm.config
        self.T_total = T_total
        self.n_layers = self.config.num_hidden_layers
        self.max_seq_len = max_seq_len
        self.device = next(causal_lm.parameters()).device
        # exp3 KV-sharing budget s (None or >= T_total → off = exact exp2 behavior).
        # Set on the blocks so BOTH the AR baseline and the SSD scheduler share it.
        self.kv_budget_s = None
        # exp3 adaptive-total-T threshold τ (None = off = full depth = exact exp2).
        # When set, R uses Ouro's built-in early_exit_gate: per ut-step the gate
        # sigmoid λ_j accumulates a halting PDF (PonderNet/ACT style); a token
        # commits at the first depth where the cumulative exit prob ≥ τ (lossy
        # verify — see exp3_plan). Common to AR + SSD.
        self.exit_threshold = None
        # R grouping (2026-07-13): tokens that share one physical KV slot
        # (same-step DtV verify sources; KV-sharing waves) become ONE causal
        # varlen segment instead of per-token duplicated-prefix segments —
        # slot K/V is read once per group instead of once per token. Numerically
        # the same attention (per-row cutoff == own position; writes all happen
        # before reads either way). When every slot is distinct (lossless WFD)
        # the code falls back to the exact per-token path. False = old path.
        self.group_r = True

    # ------------------------------------------------------------------
    # exp3 KV-sharing helpers (Q2: round-robin slot = step % s)
    # ------------------------------------------------------------------
    def _sharing(self) -> bool:
        return self.kv_budget_s is not None and self.kv_budget_s < self.T_total

    def _slot(self, step: int) -> int:
        """Recurrence depth -> depth-slot index. With KV-sharing budget s, depths
        collapse round-robin to s slots (`step % s`, last write wins); else
        identity (one slot per depth, exact exp2)."""
        return (step % self.kv_budget_s) if self._sharing() else step

    # ------------------------------------------------------------------
    # exp3 adaptive-total-T helpers (Ouro built-in early_exit_gate)
    # ------------------------------------------------------------------
    def _adaptive(self) -> bool:
        return self.exit_threshold is not None

    def _freeze_kv(self, cache, pos: int, exit_step: int) -> None:
        """Early-exit KV pad. A token that halts at `exit_step` (1..T_total-1) has
        written only the shallow depth-slots [0..exit_step-1]; replicate its
        last-computed depth (exit_step-1) K/V into the skipped deeper depth-slots
        [exit_step..T_total-1] for every layer, so later wavefront tokens that
        reach those depths still find this position's K/V (see kv_cache
        .freeze_replicate / exp3_plan §"adaptive-total-T"). Only meaningful with
        KV-sharing off (round-robin already keeps physical slots populated)."""
        last = exit_step - 1
        for layer_idx in range(self.n_layers):
            src = last * self.n_layers + layer_idx
            dst = [e * self.n_layers + layer_idx for e in range(exit_step, self.T_total)]
            cache.freeze_replicate(src, dst, pos)

    def ar_generate(self, prompt_ids, max_new_tokens, eos_token_id=None, sampler=None):
        """Greedy AR baseline (Ouro), dispatched to ar_baseline (lazy import to
        avoid an import cycle). Same contract as ParcaeBlocks.ar_generate."""
        from .ar_baseline import ar_generate_ouro
        return ar_generate_ouro(self, prompt_ids, max_new_tokens, eos_token_id, sampler=sampler)

    # ------------------------------------------------------------------
    # Unified low-level primitives (see RecursiveBlocks) — Ouro P = embed,
    # R = one ut-step, C = bare lm_head (no coda attention / no state).
    # ------------------------------------------------------------------
    def prelude_forward(self, cache, token_id: int, position: int):
        ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        return self.inner.embed_tokens(ids)[0, 0], None      # (hidden, static_state=None)

    def advance_one_step(self, toks: list, cache) -> None:
        self._advance_one_ut(toks, cache)

    def coda_logits(self, state, toks: list):
        return self._lm_head_logits(toks)                    # lm_head is state-free

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _rope(self, x_like: torch.Tensor, positions) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin (1, len(positions), Dh) for the given absolute positions."""
        if isinstance(positions, torch.Tensor):
            pos_ids = positions.to(self.device, torch.long).view(1, -1)
        else:
            pos_ids = torch.tensor(positions, device=self.device, dtype=torch.long).unsqueeze(0)
        return self.inner.rotary_emb(x_like, pos_ids)

    def _compute_qkv(self, attn, x, cos, sin):
        """x: (B, T, C) -> q (B, T, H, Dh), k/v (B, T, Hkv, Dh). Mirrors
        OuroAttention's projections + RoPE (no qk_norm / bias / value-embeds)."""
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
        """Sandwich-normed MLP half of an OuroDecoderLayer (token-wise)."""
        residual = x
        h = layer.post_attention_layernorm(x)
        h = layer.mlp(h)
        h = layer.post_attention_layernorm_2(h)
        return residual + h

    def _block_dense(self, layer, x, cos, sin, cache, step_idx, base_pos):
        """One OuroDecoderLayer with dense causal attention (prefill: Tq==Tk, or
        a decode Tq=1 query attending its slot prefix). Writes K/V to `step_idx`
        at positions [base_pos, base_pos+T). Sandwich norm around attn + mlp."""
        B, T, C = x.shape
        residual = x
        h = layer.input_layernorm(x)
        q, k, v = self._compute_qkv(layer.self_attn, h, cos, sin)
        cache.write_range(step_idx, base_pos, k, v)
        if T == 1:
            K, V = cache.gather(step_idx, base_pos)         # (1, base_pos+1, Hkv, Dh)
            out = fa.attn_dense(q, K, V, causal=True)       # Tq=1 full-prefix
        else:
            out = fa.attn_dense(q, k, v, causal=True)       # prefill Tq==Tk
        attn_out = layer.self_attn.o_proj(out.reshape(B, T, C))
        attn_out = layer.input_layernorm_2(attn_out)
        x = residual + attn_out
        return self._layer_mlp(layer, x)

    def _group_ctx(self, steps, positions):
        """Precompute the grouped-R pack layout for one advance (identical at
        every layer — only the slot's layer offset changes, so the index
        tensors are built ONCE per advance, not per layer). Tokens are grouped
        by depth-slot (`_slot(step)`); multi-token groups appear in the DtV
        parallel verify (same-step sources) and under KV-sharing. Returns None
        when every group is a singleton (lossless WFD) — callers then use the
        exact per-token path."""
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
            whole_q = span == n and idxs == list(range(n))       # single contiguous group
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

    def _block_varlen(self, layer, x, cos, sin, cache, steps, positions, layer_idx,
                      gctx=None):
        """One OuroDecoderLayer for n active tokens: write every token's K/V to
        its slot (`_slot(step)*n_layers+layer_idx`), then ONE varlen call.

        Grouped path (`gctx` from `_group_ctx`, hoisted per advance): tokens
        sharing a slot form ONE bottom-right-causal segment over that slot's
        prefix [0, max_pos] — the shared K/V is read once per group (query rows
        sit at their absolute positions; gaps get zero-pad rows whose outputs
        are discarded). Row r attends keys <= min_pos + r == its own position,
        i.e. exactly what the per-token path computes (writes precede reads in
        both; fp32 A/B: identical — see verify_group_r.py). Per-token path
        (gctx=None, and `group_r=False`): each token is its own q_len=1 segment
        gathering its own prefix copy — the original exp2/exp3 behaviour."""
        B, n, C = x.shape
        residual = x
        h = layer.input_layernorm(x)
        q, k, v = self._compute_qkv(layer.self_attn, h, cos, sin)   # (1, n, H/Hkv, Dh)
        for i in range(n):
            step_idx = self._slot(steps[i]) * self.n_layers + layer_idx
            cache.write_range(step_idx, positions[i], k[:, i : i + 1], v[:, i : i + 1])

        if gctx is not None:
            # ---- grouped: one causal segment per shared slot ----
            q_segs, k_segs, v_segs = [], [], []
            for dslot, pmax, span, whole_q, idxs_t, pad_rows_t in gctx["groups"]:
                step_idx = dslot * self.n_layers + layer_idx
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
                out = out.index_select(0, gctx["rows_t"])           # drop pads, token order
        else:
            # ---- per-token (original): each query gathers its own prefix ----
            K_list, V_list, klens = [], [], []
            for i in range(n):
                step_idx = self._slot(steps[i]) * self.n_layers + layer_idx
                Ki, Vi = cache.gather(step_idx, positions[i])
                K_list.append(Ki[0]); V_list.append(Vi[0]); klens.append(Ki.shape[1])
            q_pack = q[0]                                           # (n, H, Dh)
            k_pack = torch.cat(K_list, dim=0)                       # (sum klens, Hkv, Dh)
            v_pack = torch.cat(V_list, dim=0)
            cu_q = torch.arange(0, n + 1, device=self.device, dtype=torch.int32)
            cu_k = torch.zeros(n + 1, device=self.device, dtype=torch.int32)
            cu_k[1:] = torch.tensor(klens, device=self.device, dtype=torch.int32).cumsum(0)
            out = fa.attn_varlen(q_pack, k_pack, v_pack, cu_q, cu_k, 1, max(klens))

        attn_out = layer.self_attn.o_proj(out.reshape(1, n, C))
        attn_out = layer.input_layernorm_2(attn_out)
        x = residual + attn_out
        return self._layer_mlp(layer, x)

    # (A group-multiquery KV-sharing variant was prototyped here but removed:
    #  mask_mod was slower than varlen, esp. at long context — see exp3_plan
    #  . KV-sharing now rides the varlen path above.)

    # ------------------------------------------------------------------
    # PREFILL: full prompt through T_total ut-steps, filling every cache slot.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, T_total: int, T_draft: int):
        assert T_total == self.T_total, "Prefill T_total must match the blocks' T_total"
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        cache = create_static_ouro_cache(self.model, T_total, self.max_seq_len)
        seq_len = input_ids.shape[1]
        x = self.inner.embed_tokens(input_ids)                      # (1, L, D)
        cos, sin = self._rope(x, torch.arange(seq_len, device=self.device))
        for ut in range(T_total):
            for layer_idx, layer in enumerate(self.inner.layers):
                x = self._block_dense(layer, x, cos, sin, cache,
                                       step_idx=self._slot(ut) * self.n_layers + layer_idx, base_pos=0)
            x = self.inner.norm(x)                                  # per-ut-step final norm
        logits = self.model.lm_head(x).float()                      # (1, L, V)
        last_token_logits = logits[0, -1, :]

        state = WavefrontState(
            active=[], committed=[], prefix_len=seq_len, kv_cache=cache,
            T_total=T_total, T_draft=T_draft, last_committed_position=seq_len - 1,
        )
        return state, last_token_logits

    # ------------------------------------------------------------------
    # R core: advance alive tokens by ONE ut-step (24 layers + final norm).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _advance_one_ut(self, toks, cache) -> None:
        """Advance a list of alive ActiveTokens by one ut-step in place (updates
        each .hidden to the post-norm hidden and .step += 1). Single token = dense
        Tq=1; multiple = per-layer varlen batch (each query attends its own step
        slot — lossless)."""
        n = len(toks)
        positions = [t.position for t in toks]
        steps = [t.step for t in toks]
        if n == 1:
            x = toks[0].hidden.view(1, 1, -1)
        else:
            x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)   # (1, n, D)
        cos, sin = self._rope(x, positions)

        if n == 1:
            s, pos = steps[0], positions[0]
            for layer_idx, layer in enumerate(self.inner.layers):
                x = self._block_dense(layer, x, cos, sin, cache,
                                      step_idx=self._slot(s) * self.n_layers + layer_idx, base_pos=pos)
        else:
            # varlen handles both: KV-sharing off (_slot identity = exp2) and on
            # (slot = step % s). Same-slot tokens are merged into causal groups
            # (gctx, hoisted once per advance); all-distinct slots -> gctx=None
            # -> exact per-token path.
            gctx = self._group_ctx(steps, positions) if self.group_r else None
            for layer_idx, layer in enumerate(self.inner.layers):
                x = self._block_varlen(layer, x, cos, sin, cache, steps, positions,
                                       layer_idx, gctx)
        x = self.inner.norm(x)                                      # per-ut-step final norm

        # adaptive-total-T: Ouro's early_exit_gate on the post-norm hidden gives a
        # per-token halting prob λ for the ut-step just completed (= pre-increment
        # step s). Accumulate the halting PDF; flag should_exit when the cumulative
        # exit prob 1-∏(1-λ) first crosses τ (but never at the final step s ==
        # T_total-1 — that token is committed by the step==T_total rule anyway).
        lam = None
        if self._adaptive():
            gate = self.inner.early_exit_gate(x)                    # (1, n, 1)
            lam = torch.sigmoid(gate[0, :, 0].float())              # (n,)

        for i, t in enumerate(toks):
            if lam is not None and t.step < self.T_total - 1:
                t.exit_remaining *= (1.0 - float(lam[i]))
                if (1.0 - t.exit_remaining) >= self.exit_threshold:
                    t.should_exit = True
            t.hidden = x[0, i]
            t.step += 1
            # Freeze the skipped deeper KV slots NOW (at halt), not at commit:
            # a halted token waits (frozen) until it is the frontmost committer,
            # and meanwhile shallower tokens may advance PAST its halt depth and
            # attend its deeper slots — those must already hold its (frozen) K/V.
            if t.should_exit and not self._sharing():
                self._freeze_kv(cache, t.position, t.step)

    # ------------------------------------------------------------------
    # C coda: Ouro's coda is a bare lm_head (no attention / no cache), so the
    # disjoint multi-query fusion collapses to one batched matmul.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _lm_head_logits(self, toks: list) -> list[torch.Tensor]:
        x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)   # (1, nq, D)
        logits = self.model.lm_head(x).float()                         # (1, nq, V)
        return [logits[0, i] for i in range(len(toks))]

    # ==================================================================
    # Dynamic Sync Scheduler block variants (same contract as ParcaeBlocks;
    # see .claude/plans/scheduler.md). Tokens enter active_p as role "draft";
    # R_dyn promotes to "verify" at the commit point (step == T_total).
    # ==================================================================
    @torch.no_grad()
    def P_dyn(self, state: WavefrontState):
        """Prelude every active_p token. For Ouro the prelude is just the
        embedding lookup (no transformer layers, no cache, no static_state); move
        the token to active_r at step 0. Returns (state, call_r=True)."""
        for tok in state.active_p:
            ids = torch.tensor([[tok.token_id]], dtype=torch.long, device=self.device)
            tok.hidden = self.inner.embed_tokens(ids)[0, 0]
            tok.static_state = None
            tok.step = 0
            state.active_r.append(tok)
        state.active_p = []
        return state, True

    @torch.no_grad()
    def R_dyn(self, state: WavefrontState):
        """Advance active_r one ut-step, then route policy-point tokens to
        active_c: commit point -> promote "verify" + MOVE; draft point
        (step==T_draft) -> keep "draft" + COPY. Returns (state, call_c).

        Commit point = step==T_total (full depth) OR adaptive early-exit (Ouro
        gate cumulative ≥ τ, set in _advance_one_ut). With early-exit, commit
        depth varies per token, so we (a) do NOT advance already-halted tokens
        (their early-exit depth + hidden + KV are locked in), and (b) commit the
        MAXIMAL CONTIGUOUS run of commit-point tokens starting at the frontmost
        (lowest position) — i.e. a token commits once every earlier position is
        itself committed OR halted this tick. These go to active_c together and the
        scheduler verifies them as a position-ordered chain (≤1-verify is NOT
        required — that was only incidental to the no-early-exit wave). A later
        (higher-position) token may have advanced deeper than a frozen earlier one;
        its skipped KV was frozen at halt, so attending it is safe. When
        adaptive-total-T is off, should_exit is always False and only the frontmost
        ever reaches T_total → the run has length ≤1 → identical to the exp2 wave."""
        cache = state.kv_cache
        # Advance only in-flight tokens; halted (early-exit) ones are frozen until
        # their whole earlier prefix is ready to commit.
        toks = [t for t in state.active_r if t.is_alive and not t.should_exit]
        if toks:
            self._advance_one_ut(toks, cache)
        T_draft, T_total = state.T_draft, state.T_total

        # Maximal contiguous commit-point prefix from the frontmost.
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
                # draft point: COPY (only in-flight tokens; a halted token frozen
                # at T_draft must not re-draft every tick — guard on should_exit).
                if t.step == T_draft and not t.should_exit:
                    c = copy.copy(t)
                    c.role = "draft"
                    state.active_c.append(c)
                    any_c = True
        state.active_r = survivors
        return state, any_c

    @torch.no_grad()
    def C_multi_dyn(self, state: WavefrontState):
        """Coda (lm_head) over active_c WITHOUT emptying it (the scheduler reads
        active_c for positions, then clears). Returns (state, n_verify,
        logits_per_pos): logits for the verify tokens (position-sorted) come first,
        then the ≤1 draft. n_verify = number of verify tokens (>1 possible under
        adaptive-total-T when several tokens halt together; the scheduler verifies
        them as a position-ordered chain). The scheduler re-derives the same
        position order, so logits align."""
        verify = sorted((t for t in state.active_c if t.role == "verify"),
                        key=lambda t: t.position)
        draft = [t for t in state.active_c if t.role == "draft"]
        assert len(draft) <= 1, "dyn wave invariant: <=1 draft per C"
        ordered = verify + draft
        if not ordered:
            return state, 0, []
        logits = self._lm_head_logits(ordered)
        return state, len(verify), logits
