"""Per-sequence state for the exp4 fixed-batch schedulers.

`SeqState` carries one sequence's generation bookkeeping inside a fixed batch:
committed tokens, its `EmulatedAcceptance` stream, its `GenerationTrace`, and
the scheduler-specific working state — WFD's active_{p,r,c} token lists or
DtV's round state machine. The batch schedulers (`scheduler_batch.py`) hold a
list of these plus one shared `BatchStaticKVCache`; block calls take the union
of what every live sequence needs this tick.

Kept separate from `state.WavefrontState` (exp2/exp3 single-sequence state) per
the exp4 isolation rule — existing files are not modified. `ActiveToken` is
reused as-is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .emu import EmulatedAcceptance
from .scheduler import GenerationTrace
from .state import ActiveToken


@dataclass
class SeqState:
    idx: int                                  # batch index b (cache row)
    prefix_len: int
    emu: Optional[EmulatedAcceptance] = None  # None -> real gold==draft compare
    trace: GenerationTrace = field(default_factory=GenerationTrace)
    committed: list[int] = field(default_factory=list)
    done: bool = False

    # --- WFD (wavefront) working state ---
    active_p: list[ActiveToken] = field(default_factory=list)
    active_r: list[ActiveToken] = field(default_factory=list)
    active_c: list[ActiveToken] = field(default_factory=list)

    # --- DtV round state machine ---
    # phase: "need_p" (next source needs embed) | "draft" (current source running
    # 0..T_draft) | "verify" (all K+1 sources running T_draft..T_total)
    phase: str = "need_p"
    K: int = 0                                # drafts this round
    j: int = 0                                # current source index (0..K)
    sources: list[ActiveToken] = field(default_factory=list)
    drafts: list[int] = field(default_factory=list)
    cur_tok: int = 0                          # token id being forwarded as source j
    cur_pos: int = 0                          # its absolute position
    last_token: int = 0                       # last committed token (round seed)
    p: int = 0                                # its absolute position

    # ------------------------------------------------------------------
    def commit(self, token_id: int) -> None:
        self.committed.append(token_id)
        self.trace.tokens.append(token_id)

    def find_active(self, position: int) -> Optional[ActiveToken]:
        for lst in (self.active_r, self.active_p, self.active_c):
            for t in lst:
                if t.is_alive and t.position == position:
                    return t
        return None

    def rollback(self, position: int, cache) -> None:
        """Drop this sequence's speculative state at/after `position` (WFD
        reject). Other sequences are untouched."""
        self.active_p = [t for t in self.active_p if t.position < position]
        self.active_r = [t for t in self.active_r if t.position < position]
        self.active_c = [t for t in self.active_c if t.position < position]
        cache.drop_positions_at_and_after(self.idx, position)

    def clear_working_state(self) -> None:
        """Called when the sequence finishes: stop contributing to block calls."""
        self.active_p, self.active_r, self.active_c = [], [], []
        self.sources, self.drafts = [], []
        self.phase = "idle"
