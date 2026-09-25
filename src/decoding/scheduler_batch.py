"""Fixed-batch emulated-acceptance schedulers for exp4: AR / WFD / DtV.

Batch semantics (exp4_plan.md §3.e, user spec):
  * FIXED batch — a finished sequence idles inside the batch (no refill, no
    continuous batching); the run ends when every sequence is done.
  * A block (P / R / C) call fires when ANY live sequence needs it, processing
    the union of everything needed this tick.
  * WFD keeps one independent wave per sequence; a reject rolls back only the
    owning sequence. DtV runs one round state machine per sequence; sequences
    naturally desync (a drafting sequence contributes 1 token to R per tick, a
    verifying one contributes K+1).
  * The accept decision comes from each sequence's `EmulatedAcceptance`
    (Bernoulli(α)); with emu=None the real `gold == draft` compare runs. All
    tensor work (P/R/C forwards, argmax, cache writes, rollback) is executed
    either way — only the decision is emulated.
  * exp4 uses dummy tokens: no EOS (pass dummy prompts, run to max_new_tokens).

Per-sequence `GenerationTrace` counters follow the B=1 schedulers' semantics
(`n_r_calls` = ticks the sequence participated in, etc.), so the B=1 gates in
verify_batch.py can compare them 1:1 against `scheduler_emu`.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import torch

from .batch_state import SeqState
from .emu import EmulatedAcceptance
from .policy import SsdPolicy
from .state import ActiveToken


@dataclass
class BatchTrace:
    walltime_s: float = 0.0
    prefill_s: float = 0.0      # prompt prefill + first-token argmax (see below)
    n_r_calls: int = 0          # global union R calls
    n_p_calls: int = 0
    n_c_calls: int = 0
    r_union_total: int = 0      # sum of union sizes over R calls
    r_union_max: int = 0
    peak_alloc_gb: float = 0.0
    peak_reserved_gb: float = 0.0
    cache_gb: float = 0.0
    per_seq: list = field(default_factory=list)   # GenerationTrace per sequence

    @property
    def n_tokens(self) -> int:
        return sum(len(t.tokens) for t in self.per_seq)

    @property
    def tokens_per_s(self) -> float:
        return self.n_tokens / self.walltime_s if self.walltime_s > 0 else float("nan")

    @property
    def walltime_decode_s(self) -> float:
        """Walltime with the prefill phase removed — the decode loop only. The
        prefill cost is O(prompt_len) and identical across schedulers, so it
        dilutes AR-vs-SSD comparisons at large --prefill-len; this is the
        scheduler-sensitive half."""
        return max(self.walltime_s - self.prefill_s, 0.0)

    @property
    def decode_tokens_per_s(self) -> float:
        d = self.walltime_decode_s
        return self.n_tokens / d if d > 0 else float("nan")

    @property
    def r_union_mean(self) -> float:
        return self.r_union_total / self.n_r_calls if self.n_r_calls else 0.0

    @property
    def acceptance_rate(self) -> float:
        prop = sum(t.n_drafts_proposed for t in self.per_seq)
        acc = sum(t.n_drafts_accepted for t in self.per_seq)
        return acc / prop if prop else float("nan")


# ---------------------------------------------------------------------------
@dataclass
class SharedPrefill:
    """One prompt prefill reused by several scheduler runs (exp4 `--skip-prefill`).

    Sound because of the `position == buffer index` invariant: prefill writes
    only positions [0, L) (`write_prefill`) and every decode write lands at
    position >= L (`write_tokens`), so the prompt region is immutable once
    written. The only decode-mutated state is the logical `seqlen` bookkeeping,
    which `restore()` rewinds. `first_tokens` is likewise reusable — it is the
    argmax of the prompt's last-position logits, identical for every scheduler.
    """

    cache: object
    first_tokens: list[int]
    prefill_s: float
    _seqlen0: list = field(default_factory=list)

    def restore(self) -> None:
        """Rewind logical lengths to the post-prefill state before the next run.
        K/V needs no restore: the prompt region was never overwritten, and stale
        decode data beyond the rewound length is never gathered — the same
        argument BatchStaticKVCache already makes for rollback."""
        self.cache.seqlen = [row[:] for row in self._seqlen0]


@torch.no_grad()
def prefill_once(blocks, prompt_ids) -> SharedPrefill:
    """Run the prompt prefill a single time so a whole (s, prefill, B) cell of
    the sweep can share it. Timed the same way a scheduler run times its own
    prefill, so the borrowed `prefill_s` is directly substitutable."""
    t0 = time.perf_counter()
    cache, last_logits = blocks.prefill(prompt_ids)
    first = last_logits.argmax(dim=-1).tolist()
    if prompt_ids.device.type == "cuda":
        torch.cuda.synchronize()
    return SharedPrefill(cache=cache, first_tokens=first,
                         prefill_s=time.perf_counter() - t0,
                         _seqlen0=[row[:] for row in cache.seqlen])


def _prefill_and_first(blocks, prompt_ids, emus, max_new_tokens, bt, t0,
                       prefilled: SharedPrefill | None = None):
    """Prefill the prompt and commit the first (greedy) token, recording the
    phase cost in `bt.prefill_s` so the decode loop can be timed separately.
    Returns `(cache, seqs, t0)` — t0 rebased when a prefill is borrowed.

    The `.tolist()` below already syncs, but we synchronize explicitly so the
    split stays correct if that ever becomes a device-side op. One sync per run
    (not per tick) — negligible next to the prefill itself.

    With `prefilled` the prefill is SKIPPED and its measured cost is charged to
    this run instead, so `walltime_s` stays comparable across schedulers while
    `walltime_decode_s` still measures only the loop that follows.

    Note the first token is committed HERE, so `n_tokens` counts one token the
    decode loop did not produce; it cancels in any scheduler-vs-scheduler ratio.
    """
    if prefilled is not None:
        prefilled.restore()
        cache, first = prefilled.cache, prefilled.first_tokens
        bt.prefill_s = prefilled.prefill_s
        t0 = time.perf_counter() - prefilled.prefill_s
    else:
        cache, last_logits = blocks.prefill(prompt_ids)
        first = last_logits.argmax(dim=-1).tolist()
        if prompt_ids.device.type == "cuda":
            torch.cuda.synchronize()
        bt.prefill_s = time.perf_counter() - t0
    L = prompt_ids.shape[1]
    seqs: list[SeqState] = []
    for b in range(prompt_ids.shape[0]):
        s = SeqState(idx=b, prefix_len=L, emu=(emus[b] if emus is not None else None))
        s.commit(first[b])
        if len(s.committed) >= max_new_tokens:
            s.done = True
        seqs.append(s)
    return cache, seqs, t0


def _finalize_batch(bt: BatchTrace, seqs, t0, device, cache) -> BatchTrace:
    if device.type == "cuda":
        torch.cuda.synchronize()
    bt.walltime_s = time.perf_counter() - t0
    for s in seqs:
        s.trace.walltime_s = bt.walltime_s
    bt.per_seq = [s.trace for s in seqs]
    if device.type == "cuda":
        bt.peak_alloc_gb = torch.cuda.max_memory_allocated() / 1e9
        bt.peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
    bt.cache_gb = cache.nbytes / 1e9
    return bt


def _record_r(bt: BatchTrace, n_union: int) -> None:
    bt.n_r_calls += 1
    bt.r_union_total += n_union
    bt.r_union_max = max(bt.r_union_max, n_union)


# ===========================================================================
# AR — lockstep batched baseline (α-independent).
# ===========================================================================
@torch.no_grad()
def ar_generate_batch(blocks, prompt_ids: torch.Tensor, max_new_tokens: int,
                      prefilled: SharedPrefill | None = None) -> BatchTrace:
    device = prompt_ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    bt = BatchTrace()
    t0 = time.perf_counter()
    cache, seqs, t0 = _prefill_and_first(blocks, prompt_ids, None, max_new_tokens,
                                         bt, t0, prefilled)
    T_total = blocks.T_total

    while True:
        act = [s for s in seqs if not s.done]
        if not act:
            break
        toks: list[ActiveToken] = []
        for s in act:
            pos = s.prefix_len + len(s.committed) - 1
            t = ActiveToken(token_id=s.committed[-1], position=pos, role="verify")
            toks.append(t)
            s.trace.n_p_calls += 1
        seq_ids = [s.idx for s in act]
        blocks.prelude_union(seq_ids, toks, cache)
        bt.n_p_calls += 1
        for _ in range(T_total):
            blocks.advance_union(seq_ids, toks, cache)
            _record_r(bt, len(toks))
            for s in act:
                s.trace.n_r_calls += 1
        out_ids = blocks.coda_argmax_union(seq_ids, toks, cache)
        bt.n_c_calls += 1
        for s, nid in zip(act, out_ids):
            s.trace.n_c_calls += 1
            s.commit(nid)
            if len(s.committed) >= max_new_tokens:
                s.done = True
    return _finalize_batch(bt, seqs, t0, device, cache)


# ===========================================================================
# WFD — one independent wave per sequence, union P/R/C per tick.
# ===========================================================================
@torch.no_grad()
def generate_wavefront_batch(
    blocks,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    policy: SsdPolicy,
    emus: list[EmulatedAcceptance | None],
    prefilled: SharedPrefill | None = None,
) -> BatchTrace:
    device = prompt_ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    bt = BatchTrace()
    t0 = time.perf_counter()
    cache, seqs, t0 = _prefill_and_first(blocks, prompt_ids, emus, max_new_tokens,
                                         bt, t0, prefilled)
    T_total, T_draft = policy.T_total, policy.T_draft

    for s in seqs:
        if s.done:
            continue
        s.active_p.append(ActiveToken(token_id=s.committed[-1], position=s.prefix_len, role="draft"))
        s.trace.n_p_calls += 1

    ticks, max_ticks = 0, max_new_tokens * (T_total + 4) + 100
    while True:
        alive = [s for s in seqs if not s.done]
        if not alive:
            break
        ticks += 1
        if ticks > max_ticks:
            raise RuntimeError(f"wfd batch exceeded tick guard ({max_ticks}) — scheduler bug?")

        # ---- P: union of queued tokens -----------------------------------
        p_pairs = [(s, t) for s in alive for t in s.active_p]
        if p_pairs:
            blocks.prelude_union([s.idx for s, _ in p_pairs],
                                 [t for _, t in p_pairs], cache)
            for s, t in p_pairs:
                s.active_r.append(t)
            for s in alive:
                s.active_p = []
            bt.n_p_calls += 1

        # ---- R: one step tick for the union of live waves ----------------
        r_pairs = [(s, t) for s in alive for t in s.active_r if t.is_alive]
        if r_pairs:
            blocks.advance_union([s.idx for s, _ in r_pairs], [t for _, t in r_pairs], cache)
            _record_r(bt, len(r_pairs))
            for s in alive:
                if any(t.is_alive for t in s.active_r):
                    s.trace.n_r_calls += 1

        # ---- route commit/draft points (R_dyn logic, adaptive levers off) -
        for s in alive:
            survivors = []
            for t in s.active_r:
                if not t.is_alive:
                    continue
                if t.step == T_total:
                    t.role = "verify"
                    s.active_c.append(t)
                else:
                    survivors.append(t)
                    if t.step == T_draft:
                        c = copy.copy(t)
                        c.role = "draft"
                        s.active_c.append(c)
            s.active_r = survivors

        # ---- C: one lm_head over the union of active_c -------------------
        c_meta = []            # (seq, verifies_sorted, drafts)
        c_toks: list[ActiveToken] = []
        c_seq_ids: list[int] = []
        for s in alive:
            if not s.active_c:
                continue
            vs = sorted((t for t in s.active_c if t.role == "verify"), key=lambda t: t.position)
            ds = [t for t in s.active_c if t.role == "draft"]
            c_meta.append((s, vs, ds))
            c_toks += vs + ds
            c_seq_ids += [s.idx] * (len(vs) + len(ds))
        gold_of: dict[int, int] = {}
        if c_toks:
            out_ids = blocks.coda_argmax_union(c_seq_ids, c_toks, cache)
            bt.n_c_calls += 1
            gold_of = {id(t): nid for t, nid in zip(c_toks, out_ids)}
            for s, _, _ in c_meta:
                s.trace.n_c_calls += 1

        # ---- per-sequence settle + new leading draft ---------------------
        for s, vs, ds in c_meta:
            rejected = False
            boundary = False
            for i, vtok in enumerate(vs):
                gold = gold_of[id(vtok)]
                vtok.is_alive = False
                next_pos = vtok.position + 1
                nxt = vs[i + 1] if i + 1 < len(vs) else s.find_active(next_pos)

                if nxt is None:                            # boundary — nothing speculated
                    s.commit(gold)
                    s.active_p.append(ActiveToken(gold, next_pos, "draft"))
                    s.trace.n_p_calls += 1
                    boundary = True
                else:
                    accept = s.emu.draw() if s.emu is not None else (gold == nxt.token_id)
                    s.trace.n_drafts_proposed += 1
                    if accept:                             # ── ACCEPT ──
                        s.trace.n_drafts_accepted += 1
                        s.commit(nxt.token_id)
                    else:                                  # ── REJECT (cascade) ──
                        s.trace.n_drafts_rejected += 1
                        s.trace.n_rollback_events += 1
                        rejected = True
                        s.commit(gold)
                        s.rollback(next_pos, cache)
                        s.active_p.append(ActiveToken(gold, next_pos, "draft"))
                        s.trace.n_p_calls += 1

                if len(s.committed) >= max_new_tokens:
                    s.done = True
                if s.done or rejected or boundary:
                    break

            if s.done:
                s.clear_working_state()
                continue
            if not rejected and ds:                        # new leading-edge draft
                dsrc = ds[0]
                s.active_p.append(ActiveToken(gold_of[id(dsrc)], dsrc.position + 1, "draft"))
                s.trace.n_p_calls += 1
            s.active_c = []

    return _finalize_batch(bt, seqs, t0, device, cache)


# ===========================================================================
# DtV — one round state machine per sequence, union P/R/C per tick.
# ===========================================================================
def _dtv_start_round(s: SeqState, max_new_tokens: int, draft_length: int) -> None:
    remaining = max_new_tokens - len(s.committed)
    s.K = min(draft_length, remaining)
    s.j = 0
    s.sources = []
    s.drafts = []
    s.cur_tok, s.cur_pos = s.last_token, s.p
    s.phase = "need_p"


@torch.no_grad()
def generate_dtv_batch(
    blocks,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    policy: SsdPolicy,
    draft_length: int,
    emus: list[EmulatedAcceptance | None],
    prefilled: SharedPrefill | None = None,
) -> BatchTrace:
    device = prompt_ids.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    bt = BatchTrace()
    t0 = time.perf_counter()
    cache, seqs, t0 = _prefill_and_first(blocks, prompt_ids, emus, max_new_tokens,
                                         bt, t0, prefilled)
    T_total, T_draft = policy.T_total, policy.T_draft

    for s in seqs:
        if s.done:
            continue
        s.last_token, s.p = s.committed[-1], s.prefix_len
        _dtv_start_round(s, max_new_tokens, draft_length)

    ticks = 0
    max_ticks = max_new_tokens * ((draft_length + 1) * max(T_draft, 1) + T_total + 4) + 100
    while True:
        alive = [s for s in seqs if not s.done]
        if not alive:
            break
        ticks += 1
        if ticks > max_ticks:
            raise RuntimeError(f"dtv batch exceeded tick guard ({max_ticks}) — scheduler bug?")

        # ---- P: sequences whose next source needs its embedding ----------
        pneed = [s for s in alive if s.phase == "need_p"]
        if pneed:
            new_toks = []
            for s in pneed:
                t = ActiveToken(token_id=s.cur_tok, position=s.cur_pos, role="verify")
                new_toks.append(t)
                s.sources.append(t)
                s.trace.n_p_calls += 1
                s.phase = "draft"
            blocks.prelude_union([s.idx for s in pneed], new_toks, cache)
            bt.n_p_calls += 1

        # ---- R: drafting seqs contribute their current source; verifying
        #         seqs contribute all K+1 sources (until full depth). --------
        r_pairs: list[tuple[SeqState, ActiveToken]] = []
        participated: list[SeqState] = []
        for s in alive:
            if s.phase == "draft":
                r_pairs.append((s, s.sources[-1]))
                participated.append(s)
            elif s.phase == "verify" and s.sources[0].step < T_total:
                r_pairs += [(s, t) for t in s.sources]
                participated.append(s)
        if r_pairs:
            blocks.advance_union([s.idx for s, _ in r_pairs], [t for _, t in r_pairs], cache)
            _record_r(bt, len(r_pairs))
            for s in participated:
                s.trace.n_r_calls += 1

        # ---- C: draft-C (source hit T_draft) and verify-C (hit T_total) ---
        c_entries = []          # (seq, kind, toks)
        for s in alive:
            if s.phase == "draft" and s.sources[-1].step == T_draft:
                c_entries.append((s, "draft", [s.sources[-1]]))
            elif s.phase == "verify" and s.sources[0].step == T_total:
                c_entries.append((s, "verify", list(s.sources)))
        if not c_entries:
            continue
        flat = [t for _, _, toks in c_entries for t in toks]
        flat_seq_ids = [s.idx for s, _, toks in c_entries for _ in toks]
        out_ids = blocks.coda_argmax_union(flat_seq_ids, flat, cache)
        bt.n_c_calls += 1

        k0 = 0
        for s, kind, toks in c_entries:
            ids = out_ids[k0 : k0 + len(toks)]
            k0 += len(toks)
            s.trace.n_c_calls += 1

            if kind == "draft":
                nxt = ids[0]
                if s.j < s.K:
                    s.drafts.append(nxt)                   # d_{j+1} at cur_pos+1
                s.cur_tok, s.cur_pos = nxt, s.cur_pos + 1
                s.j += 1
                s.phase = "need_p" if s.j <= s.K else "verify"
                continue

            # kind == "verify": accept/reject cascade (emulated decision)
            golds = ids                                    # source p+i predicts p+i+1
            rejected = False
            for i in range(s.K):
                s.trace.n_drafts_proposed += 1
                accept = s.emu.draw() if s.emu is not None else (golds[i] == s.drafts[i])
                if accept:                                 # ── ACCEPT d_{i+1} ──
                    s.trace.n_drafts_accepted += 1
                    s.commit(s.drafts[i])
                    if len(s.committed) >= max_new_tokens:
                        s.done = True
                        break
                else:                                      # ── REJECT (cascade) ──
                    s.trace.n_drafts_rejected += 1
                    s.trace.n_rollback_events += 1
                    rejected = True
                    s.commit(golds[i])
                    cache.drop_positions_at_and_after(s.idx, s.p + i + 1)
                    s.last_token, s.p = golds[i], s.p + i + 1
                    if len(s.committed) >= max_new_tokens:
                        s.done = True
                    break
            if not rejected and not s.done:
                bonus = golds[s.K]                         # all accepted -> bonus token
                s.commit(bonus)
                s.last_token, s.p = bonus, s.p + s.K + 1
                if len(s.committed) >= max_new_tokens:
                    s.done = True
            if s.done:
                s.clear_working_state()
            else:
                _dtv_start_round(s, max_new_tokens, draft_length)

    return _finalize_batch(bt, seqs, t0, device, cache)
