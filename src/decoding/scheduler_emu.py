"""B=1 emulated-acceptance schedulers for exp4 (Step 1 — semantics validation).

Verbatim copies of `scheduler.generate_wavefront_dynamic` and
`scheduler.generate_draft_then_verify` with ONE change: the accept decision
(`gold == draft`) is parameterized. With `emu=None` the real comparison runs —
these functions are then *exactly* the originals (the Step-1 equivalence gate).
With an `EmulatedAcceptance`, each attempt is decided by its Bernoulli(α) draw;
all tensor work (P/R/C, argmax, cache writes, rollback) still executes, so the
walltime profile matches the real pipeline (exp4_plan.md §3.a).

Kept in a separate file so the exp2/exp3 schedulers stay byte-identical
(user isolation rule). If `scheduler.py` changes, re-sync manually.

The batch (B>1) schedulers live in `scheduler_batch.py`; these B=1 variants use
the existing single-sequence blocks/state unchanged and serve as the reference
for the batch implementation's B=1 gates.
"""
from __future__ import annotations

import time

import torch

from .acceptance import greedy_sample
from .blocks import RecursiveBlocks
from .emu import EmulatedAcceptance
from .policy import SsdPolicy
from .scheduler import GenerationTrace, _finalize
from .state import ActiveToken


@torch.no_grad()
def generate_wavefront_emu(
    blocks: RecursiveBlocks,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    policy: SsdPolicy,
    emu: EmulatedAcceptance | None = None,
    eos_token_id: int | None = None,
) -> tuple[list[int], GenerationTrace]:
    """`generate_wavefront_dynamic` with a pluggable accept decision.
    emu=None -> identical to the original (equivalence gate)."""
    trace = GenerationTrace()
    device = prompt_ids.device

    t0 = time.perf_counter()
    state, last_prompt_logits = blocks.prefill(
        prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids,
        T_total=policy.T_total,
        T_draft=policy.T_draft,
    )

    first_token = greedy_sample(last_prompt_logits)
    state.committed.append(first_token)
    state.last_committed_position = state.prefix_len
    trace.tokens.append(first_token)
    if len(state.committed) >= max_new_tokens or (
        eos_token_id is not None and first_token == eos_token_id
    ):
        return _finalize(trace, state, t0, device, warmup_marker=None)

    state.active_p.append(ActiveToken(token_id=first_token, position=state.prefix_len, role="draft"))
    trace.n_p_calls += 1
    call_p = True

    warmup_marker: float | None = None
    steady_size = policy.steady_state_active_size

    while len(state.committed) < max_new_tokens:
        rejected = False
        stop = False
        logits_per_pos: list = []

        if call_p:
            state, _ = blocks.P_dyn(state)
            call_p = False

        state, call_c = blocks.R_dyn(state)
        trace.n_r_calls += 1

        n_verify = 0
        if call_c:
            state, n_verify, logits_per_pos = blocks.C_multi_dyn(state)
            trace.n_c_calls += 1
        draft_logits = logits_per_pos[n_verify:]

        if n_verify > 0:
            verifies = sorted((t for t in state.active_c if t.role == "verify"),
                              key=lambda t: t.position)
            for i, vtok in enumerate(verifies):
                gold = greedy_sample(logits_per_pos[i])
                vtok.is_alive = False
                next_pos = vtok.position + 1
                nxt = verifies[i + 1] if i + 1 < n_verify else state.find_active_prc(next_pos)

                if nxt is None:                               # boundary (nothing speculated yet)
                    committed_tok = gold
                    state.committed.append(gold)
                    state.last_committed_position = next_pos
                    trace.tokens.append(gold)
                    state.active_p.append(ActiveToken(token_id=gold, position=next_pos, role="draft"))
                    trace.n_p_calls += 1
                    call_p = True
                else:
                    # >>> the ONLY change vs scheduler.py: pluggable decision <<<
                    accept = (gold == nxt.token_id) if emu is None else emu.draw()
                    if accept:                                # ── ACCEPT ──
                        committed_tok = nxt.token_id
                        trace.n_drafts_proposed += 1
                        trace.n_drafts_accepted += 1
                        state.committed.append(nxt.token_id)
                        state.last_committed_position = next_pos
                        trace.tokens.append(nxt.token_id)
                    else:                                     # ── REJECT (cascade) ──
                        committed_tok = gold
                        trace.n_drafts_proposed += 1
                        trace.n_drafts_rejected += 1
                        trace.n_rollback_events += 1
                        rejected = True
                        state.committed.append(gold)
                        state.last_committed_position = next_pos
                        trace.tokens.append(gold)
                        state.rollback_prc(next_pos)
                        state.active_p.append(ActiveToken(token_id=gold, position=next_pos, role="draft"))
                        trace.n_p_calls += 1
                        call_p = True

                if (eos_token_id is not None and committed_tok == eos_token_id) or \
                   len(state.committed) >= max_new_tokens:
                    stop = True
                if stop or rejected or nxt is None:
                    break

        if stop:
            break

        if (not rejected) and draft_logits:
            draft_src = next((t for t in state.active_c if t.role == "draft"), None)
            if draft_src is not None:
                draft_token = greedy_sample(draft_logits[0])
                state.active_p.append(
                    ActiveToken(token_id=draft_token, position=draft_src.position + 1, role="draft")
                )
                trace.n_p_calls += 1
                call_p = True

        state.active_c = []

        if warmup_marker is None and len(state.active_r) >= steady_size:
            warmup_marker = time.perf_counter()

    return _finalize(trace, state, t0, device, warmup_marker)


@torch.no_grad()
def generate_dtv_emu(
    blocks: RecursiveBlocks,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    policy: SsdPolicy,
    draft_length: int,
    emu: EmulatedAcceptance | None = None,
    eos_token_id: int | None = None,
) -> tuple[list[int], GenerationTrace]:
    """`generate_draft_then_verify` with a pluggable accept decision.
    emu=None -> identical to the original (equivalence gate)."""
    trace = GenerationTrace()
    device = prompt_ids.device

    t0 = time.perf_counter()
    state, last_prompt_logits = blocks.prefill(
        prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids,
        T_total=policy.T_total,
        T_draft=policy.T_draft,
    )

    first_token = greedy_sample(last_prompt_logits)
    state.committed.append(first_token)
    state.last_committed_position = state.prefix_len
    trace.tokens.append(first_token)
    if len(state.committed) >= max_new_tokens or (
        eos_token_id is not None and first_token == eos_token_id
    ):
        return _finalize(trace, state, t0, device, warmup_marker=None)

    cache = state.kv_cache
    T_total, T_draft = policy.T_total, policy.T_draft
    p = state.prefix_len
    last_token = first_token
    warmup_marker: float | None = None

    while len(state.committed) < max_new_tokens:
        remaining = max_new_tokens - len(state.committed)
        K = min(draft_length, remaining)

        sources: list[ActiveToken] = []
        drafts: list[int] = []
        cur_tok, cur_pos = last_token, p
        for j in range(K + 1):
            hidden0, static = blocks.prelude_forward(cache, cur_tok, cur_pos)
            tok = ActiveToken(token_id=cur_tok, position=cur_pos, role="verify")
            tok.hidden, tok.static_state, tok.step = hidden0, static, 0
            for _ in range(T_draft):
                blocks.advance_one_step([tok], cache)
                trace.n_r_calls += 1
            trace.n_p_calls += 1
            sources.append(tok)
            nxt = greedy_sample(blocks.coda_logits(state, [tok])[0])
            trace.n_c_calls += 1
            if j < K:
                drafts.append(nxt)
            cur_tok, cur_pos = nxt, cur_pos + 1

        for _ in range(T_total - T_draft):
            blocks.advance_one_step(sources, cache)
            trace.n_r_calls += 1
        gold_logits = blocks.coda_logits(state, sources)
        trace.n_c_calls += 1

        stop = False
        rejected = False
        for i in range(K):
            gold = greedy_sample(gold_logits[i])
            trace.n_drafts_proposed += 1
            # >>> the ONLY change vs scheduler.py: pluggable decision <<<
            accept = (gold == drafts[i]) if emu is None else emu.draw()
            if accept:                                         # ── ACCEPT d_{i+1} ──
                trace.n_drafts_accepted += 1
                state.committed.append(drafts[i])
                state.last_committed_position = p + i + 1
                trace.tokens.append(drafts[i])
                if (eos_token_id is not None and drafts[i] == eos_token_id) or \
                   len(state.committed) >= max_new_tokens:
                    stop = True
                    break
            else:                                              # ── REJECT (cascade) ──
                trace.n_drafts_rejected += 1
                trace.n_rollback_events += 1
                rejected = True
                state.committed.append(gold)
                state.last_committed_position = p + i + 1
                trace.tokens.append(gold)
                cache.drop_positions_at_and_after(p + i + 1)
                last_token, p = gold, p + i + 1
                if (eos_token_id is not None and gold == eos_token_id) or \
                   len(state.committed) >= max_new_tokens:
                    stop = True
                break

        if not rejected and not stop:
            bonus = greedy_sample(gold_logits[K])
            state.committed.append(bonus)
            state.last_committed_position = p + K + 1
            trace.tokens.append(bonus)
            last_token, p = bonus, p + K + 1
            if (eos_token_id is not None and bonus == eos_token_id) or \
               len(state.committed) >= max_new_tokens:
                stop = True

        if warmup_marker is None:
            warmup_marker = time.perf_counter()
        if stop:
            break

    return _finalize(trace, state, t0, device, warmup_marker)
