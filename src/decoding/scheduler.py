"""Dynamic-sync wavefront SSD scheduler.

Flag-driven step pipeline: every iteration runs R once (a step tick); P / C /
settle fire only when a flag says so. Branching (which tokens hit a draft- or
commit-point) is decided inside `R_dyn`, so an adaptive policy can be dropped in
by changing only that check — the scheduler just routes blocks by flag and runs
the accept/reject settle. See `.claude/plans/scheduler.md` for the full design.

State: `WavefrontState.active_{p,r,c}` — tokens split by the block they next go
through. Tokens enter `active_p` as role="draft"; `R_dyn` promotes a token to
"verify" when it reaches a commit point (step == T_total, or an adaptive
early-exit / halt signal).

Accept logic (lossless under greedy, for the no-early-exit wave): when a token at
position N reaches its commit point, its logits predict position N+1. If a draft
sits at N+1 we accept iff argmax(logits) == draft.token_id; otherwise we commit
the corrected gold, invalidate the cache at/after N+1, and restart the wave.
Under `policy.mode == "temperature"` this becomes speculative sampling
(`acceptance.TokenSampler`): the draft at N+1 is x ~ q (shallow softmax at T, q
kept as `ActiveToken.draft_probs`), the verify at N computes p (full-depth
softmax at T) and accepts x iff r < p(x)/q(x) with r ~ U(0,1); on reject it
commits a sample of norm(max(0, p-q)). Tokens with nothing speculated (first
token, boundary gold, DtV bonus) are direct samples of p. The committed stream is
an exact sample of the full-depth model at T (see acceptance.py).

`R_dyn` may commit a whole contiguous run of commit-point tokens in one tick
(adaptive-total-T lets several halt together — see `exp3_plan.md`). The scheduler
verifies them as a position-ordered chain: each verify at P finalizes P+1 against
the next speculated token (the next verify in the run, or the leading-edge draft
for the last), accepting the longest matching prefix; the first mismatch commits
that gold and rolls back everything after it (cascade). A run of length 1 is
exactly the single-verify path above (always the case with no early-exit).

(The earlier fixed-cycle "static" scheduler `generate_wavefront` was removed
after this dynamic scheduler was validated equivalent to it — identical commit
sequence / acceptance / R·C call counts — at negligible walltime overhead.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .acceptance import TokenSampler
from .blocks import RecursiveBlocks
from .policy import SsdPolicy
from .state import ActiveToken, WavefrontState


@dataclass
class GenerationTrace:
    """Bookkeeping for benchmarking."""

    tokens: list[int] = field(default_factory=list)
    walltime_s: float = 0.0
    walltime_steady_s: float = 0.0    # post-warmup walltime
    n_r_calls: int = 0
    n_p_calls: int = 0
    n_c_calls: int = 0
    n_drafts_proposed: int = 0
    n_drafts_accepted: int = 0
    n_drafts_rejected: int = 0
    n_rollback_events: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.n_drafts_proposed == 0:
            return float("nan")
        return self.n_drafts_accepted / self.n_drafts_proposed


def _finalize(
    trace: GenerationTrace,
    state: WavefrontState,
    t0: float,
    device: torch.device,
    warmup_marker: float | None,
) -> tuple[list[int], GenerationTrace]:
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    trace.walltime_s = t1 - t0
    trace.walltime_steady_s = (t1 - warmup_marker) if warmup_marker is not None else trace.walltime_s
    return list(trace.tokens), trace


@torch.no_grad()
def generate_wavefront_dynamic(
    blocks: RecursiveBlocks,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    policy: SsdPolicy,
    eos_token_id: int | None = None,
) -> tuple[list[int], GenerationTrace]:
    """Dynamic-sync wavefront SSD generation (greedy or temperature sampling per
    `policy.mode`). Stops at max_new_tokens or when a committed (verify) token
    equals `eos_token_id` (if given). Returns (tokens, trace)."""
    trace = GenerationTrace()
    device = prompt_ids.device
    sampler: TokenSampler = policy.make_sampler()

    t0 = time.perf_counter()
    state, last_prompt_logits = blocks.prefill(
        prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids,
        T_total=policy.T_total,
        T_draft=policy.T_draft,
    )

    first_token = sampler(last_prompt_logits, state.prefix_len)
    state.committed.append(first_token)
    state.last_committed_position = state.prefix_len
    trace.tokens.append(first_token)
    if len(state.committed) >= max_new_tokens or (
        eos_token_id is not None and first_token == eos_token_id
    ):
        return _finalize(trace, state, t0, device, warmup_marker=None)

    # First active enters as a "draft"; R_dyn promotes it to verify at T_total.
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

        # R is always-on: one step tick per iteration.
        state, call_c = blocks.R_dyn(state)
        trace.n_r_calls += 1

        n_verify = 0
        if call_c:
            state, n_verify, logits_per_pos = blocks.C_multi_dyn(state)
            trace.n_c_calls += 1
        draft_logits = logits_per_pos[n_verify:]     # trailing entries are draft logits

        # Verify chain: R_dyn may commit a whole contiguous run of commit-point
        # tokens at once (adaptive-total-T lets several halt together). They are
        # position-sorted; each verify at position P finalizes P+1 by comparing its
        # gold to the token already speculated at P+1 — the NEXT verify in the chain
        # for all but the last, the leading-edge draft (in active_r) for the last.
        # Accept the longest matching prefix; the first reject commits the gold
        # there and rolls back everything after it (cascade), which drops the later
        # verifies + the leading draft. n_verify<=1 => exactly the old single-verify
        # path (lossless SSD / parcae, which never halt early).
        if n_verify > 0:
            verifies = sorted((t for t in state.active_c if t.role == "verify"),
                              key=lambda t: t.position)
            for i, vtok in enumerate(verifies):
                vtok.is_alive = False
                next_pos = vtok.position + 1
                nxt = verifies[i + 1] if i + 1 < n_verify else state.find_active_prc(next_pos)
                if nxt is None:
                    gold = sampler(logits_per_pos[i], next_pos)          # direct sample for next_pos
                    accepted = False
                else:                                                    # greedy: gold==draft; temperature: r<p/q
                    accepted, gold = sampler.verify(nxt.token_id, nxt.draft_probs, logits_per_pos[i], next_pos)

                if nxt is None:                               # boundary (nothing speculated yet)
                    committed_tok = gold
                    state.committed.append(gold)
                    state.last_committed_position = next_pos
                    trace.tokens.append(gold)
                    state.active_p.append(ActiveToken(token_id=gold, position=next_pos, role="draft"))
                    trace.n_p_calls += 1
                    call_p = True
                elif accepted:                                # ── ACCEPT ──
                    committed_tok = nxt.token_id
                    trace.n_drafts_proposed += 1
                    trace.n_drafts_accepted += 1
                    state.committed.append(nxt.token_id)
                    state.last_committed_position = next_pos
                    trace.tokens.append(nxt.token_id)
                else:                                         # ── REJECT (cascade) ──
                    committed_tok = gold
                    trace.n_drafts_proposed += 1
                    trace.n_drafts_rejected += 1
                    trace.n_rollback_events += 1
                    rejected = True
                    state.committed.append(gold)
                    state.last_committed_position = next_pos
                    trace.tokens.append(gold)
                    state.rollback_prc(next_pos)              # drops later verifies + leading draft
                    state.active_p.append(ActiveToken(token_id=gold, position=next_pos, role="draft"))
                    trace.n_p_calls += 1
                    call_p = True

                if (eos_token_id is not None and committed_tok == eos_token_id) or \
                   len(state.committed) >= max_new_tokens:
                    stop = True
                if stop or rejected or nxt is None:
                    break

        # A committed (verify) token hit EOS / max_new_tokens -> stop.
        if stop:
            break

        # New leading-edge draft (reject -> skip, matching the validated behaviour).
        if (not rejected) and draft_logits:
            draft_src = next((t for t in state.active_c if t.role == "draft"), None)
            if draft_src is not None:
                draft_token, draft_q = sampler.draft(draft_logits[0], draft_src.position + 1)
                state.active_p.append(
                    ActiveToken(token_id=draft_token, position=draft_src.position + 1, role="draft",
                                draft_probs=draft_q)
                )
                trace.n_p_calls += 1
                call_p = True

        state.active_c = []   # C_multi_dyn does not empty it; clear at cycle end.

        if warmup_marker is None and len(state.active_r) >= steady_size:
            warmup_marker = time.perf_counter()

    return _finalize(trace, state, t0, device, warmup_marker)


@torch.no_grad()
def generate_draft_then_verify(
    blocks: RecursiveBlocks,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    policy: SsdPolicy,
    draft_length: int,
    eos_token_id: int | None = None,
) -> tuple[list[int], GenerationTrace]:
    """Draft-then-verify self-speculative decoding (greedy or speculative sampling
    per `policy.mode`), exploiting the
    recursive model's *nested* structure so the draft computation is REUSED by
    verify (not thrown away). Unlike the wavefront scheduler (diagonal, one
    scheduler), this is the classic draft-then-verify loop:

      DRAFT  — forward `draft_length`+1 source positions AUTOREGRESSIVELY at the
               shallow depth `T_draft` (P + R x T_draft + C). Each source saves
               its depth-T_draft hidden and core-KV slots [0..T_draft-1]. The
               shallow logits emit the `draft_length` speculative tokens.
      VERIFY — CONTINUE all sources from depth T_draft to full T_total IN PARALLEL
               (R x (T_total - T_draft) + C), reusing the saved depth-T_draft
               hidden as input and writing core-KV slots [T_draft..T_total-1]. So
               the draft's [0..T_draft-1] recurrence is the prefix of verify's
               full recurrence — nested reuse, nothing recomputed.
      ACCEPT — source p+i's full-depth (gold) logit predicts p+i+1; accept the
               longest prefix where gold == draft (greedy). First mismatch commits
               the corrected gold and rolls back the rest (cascade). If ALL accept,
               the last source's gold gives a free bonus token (standard spec
               decoding). Loop until max_new_tokens / EOS.

    Verify runs to the full depth T_total, so the committed sequence equals greedy
    AR (lossless, modulo bf16 drift) — adaptive-depth verify is a future lossy
    extension. Assumes the adaptive-total-T levers are off (it sets depth
    explicitly). Stops at max_new_tokens or a committed EOS. Returns (tokens, trace)."""
    trace = GenerationTrace()
    device = prompt_ids.device
    sampler: TokenSampler = policy.make_sampler()

    t0 = time.perf_counter()
    state, last_prompt_logits = blocks.prefill(
        prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids,
        T_total=policy.T_total,
        T_draft=policy.T_draft,
    )

    first_token = sampler(last_prompt_logits, state.prefix_len)
    state.committed.append(first_token)
    state.last_committed_position = state.prefix_len
    trace.tokens.append(first_token)
    if len(state.committed) >= max_new_tokens or (
        eos_token_id is not None and first_token == eos_token_id
    ):
        return _finalize(trace, state, t0, device, warmup_marker=None)

    cache = state.kv_cache
    T_total, T_draft = policy.T_total, policy.T_draft
    p = state.prefix_len          # position of the last committed token (not yet forwarded)
    last_token = first_token
    warmup_marker: float | None = None

    while len(state.committed) < max_new_tokens:
        remaining = max_new_tokens - len(state.committed)
        K = min(draft_length, remaining)     # number of speculative drafts this round

        # ---- DRAFT: shallow AR over sources p..p+K (K+1 forwards at depth T_draft).
        # Each source saves its depth-T_draft hidden + core-KV[0..T_draft-1] (for
        # verify reuse); the shallow logits emit drafts d_1..d_K at p+1..p+K. ----
        sources: list[ActiveToken] = []
        drafts: list[int] = []
        draft_qs: list = []                  # q per draft (None under greedy)
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
            nxt, nxt_q = sampler.draft(blocks.coda_logits(state, [tok])[0], cur_pos + 1)
            trace.n_c_calls += 1
            if j < K:
                drafts.append(nxt)           # d_{j+1} at position p+j+1
                draft_qs.append(nxt_q)
            cur_tok, cur_pos = nxt, cur_pos + 1

        # ---- VERIFY: continue all sources T_draft -> T_total in parallel, reusing
        # the saved hidden + KV[0..T_draft-1], writing KV[T_draft..T_total-1]. ----
        for _ in range(T_total - T_draft):
            blocks.advance_one_step(sources, cache)
            trace.n_r_calls += 1
        gold_logits = blocks.coda_logits(state, sources)   # source p+i predicts p+i+1
        trace.n_c_calls += 1

        # ---- ACCEPT / REJECT cascade (greedy: gold==draft; temperature: r<p/q) ----
        stop = False
        rejected = False
        for i in range(K):
            accepted, gold = sampler.verify(drafts[i], draft_qs[i], gold_logits[i], p + i + 1)
            trace.n_drafts_proposed += 1
            if accepted:                                       # ── ACCEPT d_{i+1} ──
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
                cache.drop_positions_at_and_after(p + i + 1)   # invalidate wrong speculative KV
                last_token, p = gold, p + i + 1
                if (eos_token_id is not None and gold == eos_token_id) or \
                   len(state.committed) >= max_new_tokens:
                    stop = True
                break

        if not rejected and not stop:
            # All K accepted -> bonus token from the last source's full-depth logit.
            bonus = sampler(gold_logits[K], p + K + 1)
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
