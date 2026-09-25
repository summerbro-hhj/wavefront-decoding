"""Huginn-0125 (RDM, arXiv:2502.05171) P/R/C blocks — subclass of ParcaeBlocks.

Huginn is the original 3.5B model of the parcae lineage; the recursive skeleton
is identical (prelude -> [adapter re-injection -> recurrent core] x r -> coda,
random like-init s0, same complex RoPE helper, and — crucially — the SAME cache
slot scheme: huginn's native `block_idx` counts prelude 0..1, then core
recurrence r layer l as `n_prelude + r*n_core + l` continuously, exactly our
StaticKVCache layout. Its coda uses the NEGATIVE indices -1, -2; we adopt those
so the native prefill and our decode share slots).

Differences from parcae, handled by overrides:
  * SandwichBlock:  x = norm_2(attn(norm_1 x) + x);  x = norm_4(mlp(norm_3 x) + x)
    (parcae is plain pre-norm) — 4 norms, residual then norm.
  * Attention: fused `Wqkv` (+ additive `qk_bias` applied BEFORE rotary),
    MHA 55 heads x 96; no value-embeds / qk_norm / clip_qkv / logit softcap /
    prelude_norm / logit_scale.
  * Coda path == huginn's `predict_from_latents`: ln_f -> coda(-1,-2) -> ln_f
    -> lm_head (ln_f "used twice").
  * Prefill is huginn's NATIVE forward (num_steps=T_total) writing into our
    StaticKVCache via its parcae-compatible `update`; the initial state is
    injected through the native `input_states=` hook (seeded draw), so prefill
    is deterministic and bit-native.

Inherited unchanged from ParcaeBlocks: the dynamic-scheduler blocks
(P_dyn/R_dyn/C_multi_dyn), the unified primitives, AR dispatch, the grouped-R
varlen pack (`_group_ctx`/`_core_attn_varlen` — n_prelude/n_core resolve to
2/4), KV-sharing (`kv_budget_s`) and adaptive-total-T (`exit_hidden_eps` —
huginn's own LatentDiffExitEvaluator is the same latent-convergence criterion),
and the seeded random-s0 machinery (`state_init`/`state_seed`).
"""
from __future__ import annotations

import sys

import torch

from . import fa4_attention as fa
from .kv_cache import StaticKVCache
from .parcae_blocks import DEFAULT_MAX_SEQ_LEN, ParcaeBlocks
from .state import WavefrontState


def ensure_freqs_cis(model, needed_len: int) -> None:
    """huginn precomputes its RoPE table only to config.block_size (4096);
    slicing `freqs_cis[:, :L]` beyond that SILENTLY truncates and the rope
    broadcast crashes (L vs 4096). Extend the table in place when a longer
    max_seq_len is requested. ⚠️ Positions beyond block_size are OUT OF
    DISTRIBUTION for the model (trained context 4096) — acceptable for exp4's
    emulated-acceptance walltime studies, NOT for quality experiments."""
    cur = model.freqs_cis.shape[1]
    if needed_len <= cur:
        return
    mod = sys.modules[type(model).__module__]
    head_dim = model.config.n_embd // model.config.num_attention_heads
    new = mod.precompute_freqs_cis(head_dim, needed_len, model.config.rope_base, 1)
    model.freqs_cis = new.to(device=model.freqs_cis.device, dtype=model.freqs_cis.dtype)
    print(f"[huginn] RoPE table extended {cur} -> {needed_len} "
          f"(beyond trained block_size={model.config.block_size}: OOD positions — "
          f"fine for exp4 emulation, avoid for quality runs)")


class HuginnBlocks(ParcaeBlocks):
    def __init__(self, model, T_total: int, max_seq_len: int = DEFAULT_MAX_SEQ_LEN):
        super().__init__(model, T_total, max_seq_len)   # n_prelude=2, n_core=4, n_coda=2
        # rope helper from the model's own dynamically-loaded module (identical
        # math to parcae's copy, but guaranteed consistent with this checkpoint).
        self._apply_rope = sys.modules[type(model).__module__].apply_rotary_emb_complex_like
        ensure_freqs_cis(model, max_seq_len)

    def _state_init_params(self, model) -> tuple[int, float, float]:
        return (model.config.n_embd, float(model.config.init_values["std"]),
                float(model.emb_scale))

    # ------------------------------------------------------------------
    # QKV — fused Wqkv + additive qk_bias (before rotary). No ve / qk_norm.
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

    # ------------------------------------------------------------------
    # One SandwichBlock for a single-token query (P / R single / C single).
    # `ve` is accepted for signature compatibility and ignored (no value embeds).
    # ------------------------------------------------------------------
    def _block_fa4_single(self, block, x, freqs, ve, cache, step_idx, position):
        B, T, C = x.shape   # T == 1
        h = block.norm_1(x)
        q, k, v = self._qkv(block.attn, h, freqs)
        cache.write_range(step_idx, position, k, v)
        K, V = cache.gather(step_idx, position)
        out = fa.attn_dense(q, K, V, causal=True)
        att = block.attn.proj(out.reshape(B, T, C))
        x = block.norm_2(att + x)
        x = block.norm_4(block.mlp(block.norm_3(x)) + x)
        return x

    # ------------------------------------------------------------------
    # PREFILL — huginn native forward into our StaticKVCache (slots match),
    # with the seeded s0 injected via the native `input_states` hook.
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, T_total: int, T_draft: int):
        assert T_total == self.T_total, "Prefill T_total must match the blocks' T_total"
        model = self.model
        dtype = next(model.parameters()).dtype
        head_dim = getattr(self.config, "head_dim",
                           self.config.n_embd // self.config.num_attention_heads)
        cache = StaticKVCache(
            n_kv_head=self.config.num_key_value_heads,
            head_dim=head_dim,
            max_seq_len=self.max_seq_len,
            dtype=dtype,
            device=self.device,
        )
        seq_len = input_ids.shape[1]
        s0 = self._init_state_prefill(input_ids.shape[0], seq_len)
        out = model(
            input_ids=input_ids,
            num_steps=T_total,       # plain int — native code len()-checks tensors
            past_key_values=cache,
            use_cache=True,
            input_states=s0,
        )
        last_token_logits = out.logits[0, -1, :].float()
        self._compact_prefill_kv_sharing(cache)

        state = WavefrontState(
            active=[], committed=[], prefix_len=seq_len, kv_cache=cache,
            T_total=T_total, T_draft=T_draft, last_committed_position=seq_len - 1,
        )
        return state, last_token_logits

    def _compact_prefill_kv_sharing(self, cache) -> None:
        """KV-sharing consistency for the NATIVE prefill (bugfix 2026-07-16).

        The native prefill writes FULL per-recurrence core slots (block_idx
        `2 + r*4 + l`) — correct: sharing must never alter the prefill
        computation (Ouro paper §5.4.2: sharing during prefill costs >10 GSM8K
        points). But our shared DECODE reads slots `2 + (r % s)*4 + l`, which
        after an unmapped prefill hold the FIRST recurrences' prompt K/V —
        exactly the paper's catastrophic "first-step reuse" (GSM8K 78.9→18.7).
        Fix: collapse the prompt's core K/V to the shared layout, keeping for
        each class c the DEEPEST recurrence r ≡ c (mod s) — s=1 == the papers'
        near-lossless "last-step reuse" — and free the remaining deep slots
        (this is also where the prompt-side memory saving is realised).
        parcae/Ouro/HuginnBlocksBatch prefills apply `_slot` at write time and
        need no compaction (same end state: last write per class == deepest)."""
        if not self._sharing():
            return
        s = self.kv_budget_s
        for c in range(s):
            r_star = max(r for r in range(self.T_total) if r % s == c)
            for l in range(self.n_core):
                src = self.n_prelude + r_star * self.n_core + l
                dst = self.n_prelude + c * self.n_core + l
                if src != dst and src in cache.K:
                    cache.K[dst] = cache.K.pop(src)
                    cache.V[dst] = cache.V.pop(src)
                    cache.seqlen[dst] = cache.seqlen.pop(src)
        lo = self.n_prelude + s * self.n_core
        hi = self.n_prelude + self.T_total * self.n_core
        for slot in [k for k in list(cache.K) if lo <= k < hi]:
            del cache.K[slot]
            del cache.V[slot]
            cache.seqlen.pop(slot, None)

    # ------------------------------------------------------------------
    # P — embed (x emb_scale) -> 2 prelude blocks; s0 is the seeded draw.
    # static_state = prelude output e (re-injected by the adapter every step).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _prelude_forward(self, cache, token_id: int, position: int):
        model = self.model
        ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        freqs = model.freqs_cis[:, position : position + 1]
        x = model.transformer.wte(ids)
        if model.emb_scale != 1:
            x = x * model.emb_scale
        for i, block in enumerate(model.transformer.prelude):
            x = self._block_fa4_single(block, x, freqs, None, cache, step_idx=i, position=position)
        static_state = x[0, 0]
        hidden0 = self._init_state_tokens([position])[0]
        return hidden0, static_state

    # ------------------------------------------------------------------
    # R — one recurrence: adapter(cat[s, e]) -> 4 core SandwichBlocks.
    # Slot formula inherited: n_prelude + _slot(step)*n_core + layer (native).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _advance_core(self, toks, cache) -> None:
        model = self.model
        core = model.transformer.core_block
        n = len(toks)
        positions = [t.position for t in toks]
        steps = [t.step for t in toks]
        if n == 1:
            x = toks[0].hidden.unsqueeze(0).unsqueeze(0)
            e = toks[0].static_state.unsqueeze(0).unsqueeze(0)
        else:
            x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)
            e = torch.stack([t.static_state for t in toks], dim=0).unsqueeze(0)
        p0 = positions[0]
        if positions[-1] - p0 + 1 == n:
            freqs = model.freqs_cis[:, p0 : p0 + n]
        else:
            freqs = model.freqs_cis[:, torch.tensor(positions, device=self.device)]

        x = model.transformer.adapter(torch.cat([x, e], dim=-1))   # concat injection

        if n == 1:
            s, pos = steps[0], positions[0]
            for layer_idx, block in enumerate(core):
                step_idx = self.n_prelude + self._slot(s) * self.n_core + layer_idx
                x = self._block_fa4_single(block, x, freqs, None, cache, step_idx, pos)
        else:
            gctx = self._group_ctx(steps, positions) if self.group_r else None
            for layer_idx, block in enumerate(core):
                h = block.norm_1(x)
                q, k, v = self._qkv(block.attn, h, freqs)
                out = self._core_attn_varlen(q, k, v, cache, steps, positions, layer_idx, gctx)
                _, _, C = x.shape
                att = block.attn.proj(out.reshape(1, n, C))
                x = block.norm_2(att + x)
                x = block.norm_4(block.mlp(block.norm_3(x)) + x)

        # adaptive-total-T: latent fixed-point convergence (same criterion as
        # huginn's own LatentDiffExitEvaluator; relative-L2 like parcae's).
        rels = None
        if self._adaptive():
            cur = x[0]
            prev_stack = torch.stack(
                [t.prev_hidden if t.prev_hidden is not None else cur[i]
                 for i, t in enumerate(toks)], dim=0)
            num = torch.linalg.vector_norm(cur - prev_stack, dim=1)
            den = torch.linalg.vector_norm(prev_stack, dim=1).clamp_min(1e-6)
            rels = (num / den).tolist()

        for i, t in enumerate(toks):
            if self._adaptive():
                if t.prev_hidden is not None and t.step < self.T_total - 1 \
                        and rels[i] < self.exit_hidden_eps:
                    t.should_exit = True
                t.prev_hidden = x[0, i].clone()
            t.hidden = x[0, i]
            t.step += 1
            if t.should_exit and not self._sharing():
                self._freeze_kv(cache, t.position, t.step)

    # ------------------------------------------------------------------
    # C — huginn's predict_from_latents: ln_f -> coda(-1,-2) -> ln_f -> head.
    # nq==1 dense; nq>1 disjoint multi-query (zero-pad causal / mask_mod).
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _coda_logits(self, state: WavefrontState, toks: list) -> list[torch.Tensor]:
        model = self.model
        cache = state.kv_cache
        nq = len(toks)
        positions = [t.position for t in toks]
        x = torch.stack([t.hidden for t in toks], dim=0).unsqueeze(0)   # (1, nq, D)
        x = model.transformer.ln_f(x)                                   # ln_f (enter coda)

        if nq == 1:
            pos = positions[0]
            freqs = model.freqs_cis[:, pos : pos + 1]
            for i, block in enumerate(model.transformer.coda):
                x = self._block_fa4_single(block, x, freqs, None, cache, -(i + 1), pos)
        else:
            freqs = model.freqs_cis[:, torch.tensor(positions, device=self.device)]
            upto = max(positions)
            for i, block in enumerate(model.transformer.coda):
                sidx = -(i + 1)
                B, T, C = x.shape
                h = block.norm_1(x)
                q, k, v = self._qkv(block.attn, h, freqs)
                cache.write_scatter(sidx, positions, k, v)
                K, V = cache.gather(sidx, upto)
                out = fa.attn_cutoff_multi(q, K, V, positions)
                att = block.attn.proj(out.reshape(B, T, C))
                x = block.norm_2(att + x)
                x = block.norm_4(block.mlp(block.norm_3(x)) + x)

        x = model.transformer.ln_f(x)                                   # ln_f (head)
        logits = model.lm_head(x).float()
        return [logits[0, i] for i in range(nq)]
