"""Dataclasses describing wavefront SSD state.

Lifetime:
  WavefrontState  — held by the scheduler across the whole generation run.
  ActiveToken     — one entry per token currently being processed in the
                    recurrent wave. Lives until it reaches step == T_total
                    (then it's committed or rejected and removed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch


Role = Literal["verify", "draft"]


@dataclass
class ActiveToken:
    """One token currently progressing through the recurrent core.

    Field order lets the dynamic scheduler create a token with just
    (token_id, position, role); `hidden`/`static_state` are filled by P (the
    prelude forward). The static scheduler still creates it via kwargs.

    Role semantics (dynamic scheduler): a token enters as "draft"; R promotes it
    to "verify" when it reaches a commit point (step == T_total) — i.e. role
    marks "when this token is C'd at its current step, is the sample final
    (verify) or speculative (draft)". (The static scheduler also flips
    draft->verify on accept.)
    """

    token_id: int
    position: int           # absolute sequence position (0-indexed, incl. prefix)
    role: Role
    step: int = 0           # number of R calls performed on this token so far
    hidden: Optional[torch.Tensor] = None   # (D,) current recurrent-core hidden; filled by P
    static_state: Optional[torch.Tensor] = None
                            # parcae: P's output (prelude+ln_prelude+initialize_state target),
                            #         re-injected into R every iteration via adapter(x, input_embeds).
    is_alive: bool = True   # rollback / commit flag

    # --- speculative-sampling metadata (policy.mode == "temperature") ---
    # The draft distribution q = softmax(shallow_logits / T) this token was
    # SAMPLED from, kept so stochastic verification can accept w.p. min(1, p/q)
    # and sample the residual norm(max(0, p-q)) on reject (Leviathan et al. 2023,
    # arXiv:2211.17192). None for greedy drafts and for committed tokens that
    # merely seed the wave (first token / boundary gold) — those are never verified.
    draft_probs: Optional[torch.Tensor] = None

    # --- adaptive-total-T (exp3 step 3) scratch ---
    # should_exit is set by R at an early-exit / halt point (commit early at the
    # current depth). The per-model signal state lives alongside it:
    #   - Ouro: exit_remaining = running ∏_j (1-λ_j) of the early_exit_gate sigmoid
    #     λ_j; cumulative exit prob = 1 - exit_remaining, halt when it crosses τ.
    #   - parcae: prev_hidden = previous R-step's core hidden; halt when the
    #     relative-L2 step-to-step change ‖h_t-h_{t-1}‖/‖h_{t-1}‖ falls below ε
    #     (latent fixed-point convergence — no lm_head, cheap).
    # All inert when adaptive-total-T is off (exit_remaining=1.0, prev_hidden=None,
    # should_exit=False) — exact exp2 behaviour.
    exit_remaining: float = 1.0
    should_exit: bool = False
    prev_hidden: Optional[torch.Tensor] = None


@dataclass
class WavefrontState:
    """Top-level mutable state for a single generation."""

    active: list[ActiveToken] = field(default_factory=list)   # static scheduler
    committed: list[int] = field(default_factory=list)
    prefix_len: int = 0                       # number of tokens (prompt incl.) already cached
    kv_cache: object = None                   # opaque per-model KVCache wrapper
    T_total: int = 8                          # full-depth step count
    T_draft: int = 2                          # draft window (R calls between draft samples)
    last_committed_position: int = -1         # position of last committed token

    # Dynamic scheduler: `active` split by which block a token next goes through.
    # active_p -> waiting for P (prelude),  active_r -> in the R (recurrent) wave,
    # active_c -> queued for C (coda) this iteration. Empty for the static scheduler.
    active_p: list[ActiveToken] = field(default_factory=list)
    active_r: list[ActiveToken] = field(default_factory=list)
    active_c: list[ActiveToken] = field(default_factory=list)

    def alive(self) -> list[ActiveToken]:
        return [t for t in self.active if t.is_alive]

    def drop_dead(self) -> None:
        self.active = self.alive()

    # --- dynamic-scheduler helpers ---
    def find_active_prc(self, position: int) -> Optional[ActiveToken]:
        """Find an alive token at `position` across active_r / active_p / active_c."""
        for lst in (self.active_r, self.active_p, self.active_c):
            for t in lst:
                if t.is_alive and t.position == position:
                    return t
        return None

    def rollback_prc(self, position: int) -> None:
        """Drop every token at position >= `position` from active_p/r/c and
        invalidate the cache slots there (reject / rollback)."""
        self.active_p = [t for t in self.active_p if t.position < position]
        self.active_r = [t for t in self.active_r if t.position < position]
        self.active_c = [t for t in self.active_c if t.position < position]
        if self.kv_cache is not None and hasattr(self.kv_cache, "drop_positions_at_and_after"):
            self.kv_cache.drop_positions_at_and_after(position)
