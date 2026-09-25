"""Abstract `RecursiveBlocks` interface — what each model must implement.

The dynamic-sync scheduler (`scheduler.generate_wavefront_dynamic`) talks only
to this interface, so the same scheduler runs for parcae and Ouro once both
implement it. Blocks borrow the model's weights but re-implement the forward so
attention can be split per step (and routed through FA4).

Dynamic interface: each block takes the whole `WavefrontState` (carrying
`active_{p,r,c}`) and returns it plus a flag telling the scheduler which block
to run next. Tokens live in `active_p` (waiting for prelude), `active_r` (in the
recurrent wave), or `active_c` (queued for coda this iteration).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .state import ActiveToken, WavefrontState


class RecursiveBlocks(ABC):
    """P / R / C primitives for a recursive LM (dynamic-sync scheduler)."""

    T_total: int
    """The model's natural full-recursion depth (parcae mean_recurrence, Ouro
    total_ut_steps). The scheduler uses it as the verify target."""

    @abstractmethod
    def prefill(
        self,
        input_ids: torch.Tensor,
        T_total: int,
        T_draft: int,
    ) -> tuple[WavefrontState, torch.Tensor]:
        """Standard forward over the prompt at depth T_total, filling every
        cache slot so later R/C calls can attend over the prefix.

        Returns (state, last_token_logits): state has empty active_{p,r,c} and a
        populated kv_cache / prefix_len; last_token_logits is (V,) at the last
        prompt position (full depth).
        """

    @abstractmethod
    def P_dyn(self, state: WavefrontState) -> tuple[WavefrontState, bool]:
        """Prelude every token in active_p (computing its initial recurrent
        hidden / static_state), move it to active_r at step 0, and clear
        active_p. Returns (state, call_r=True)."""

    @abstractmethod
    def R_dyn(self, state: WavefrontState) -> tuple[WavefrontState, bool]:
        """Advance every alive active_r token by one recurrent step, then route
        policy-point tokens to active_c. Commit point = step == T_total OR an
        adaptive early-exit / halt signal (`should_exit`); the maximal contiguous
        commit-point prefix from the frontmost is promoted to role "verify" and
        MOVED (so several may commit in one tick under adaptive-total-T; without
        it, only the frontmost at T_total → exactly one). Draft point (e.g.
        step == T_draft) -> keep role "draft" and COPY (the original keeps
        stepping). Attention is split per step (FA4). Returns (state, call_c)."""

    @abstractmethod
    def C_multi_dyn(self, state: WavefrontState):
        """Coda forward over active_c WITHOUT emptying it (the scheduler reads
        active_c for positions, then clears it). Returns (state, n_verify,
        logits_per_pos): logits for the verify tokens (position-sorted) come
        first, then the <=1 draft's. n_verify = number of verify tokens; the
        scheduler re-derives the same position order and processes them as a
        verify chain (n_verify<=1 with no early-exit)."""

    @abstractmethod
    def ar_generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        sampler=None,
    ) -> tuple[list[int], float]:
        """Autoregressive baseline with this model's weights (one token at a time,
        full depth = T_total), returning (token_ids, walltime_s). `sampler` is an
        `acceptance.TokenSampler` (None = greedy); pass `policy.make_sampler()` so
        AR and SSD emit the same greedy / temperature token per position. This is
        the reference wavefront SSD is benchmarked against; each model dispatches
        to its own AR loop (parcae / Ouro) so the benchmark stays model-agnostic."""

    # ------------------------------------------------------------------
    # Low-level primitives (model-agnostic). The dynamic scheduler drives the
    # WFD wave through P_dyn/R_dyn/C_multi_dyn; these expose the same underlying
    # steps WITHOUT the WFD routing, so an alternative scheduler (e.g.
    # `generate_draft_then_verify`) can compose them directly. Each is a thin
    # public wrapper over the model's existing internal method.
    # ------------------------------------------------------------------
    @abstractmethod
    def prelude_forward(self, cache, token_id: int, position: int):
        """Prelude a single new token at `position` (step 0). Returns
        (hidden, static_state): the initial recurrent hidden and any per-token
        state R re-injects each step (parcae adapter target; None for Ouro)."""

    @abstractmethod
    def advance_one_step(self, toks: list, cache) -> None:
        """Advance a batch of alive ActiveTokens by ONE recurrence step in place
        (updates each .hidden and .step += 1). Attention split per step (FA4);
        each token uses its own .step slot, so a mixed-step batch is fine."""

    @abstractmethod
    def coda_logits(self, state: WavefrontState, toks: list) -> list[torch.Tensor]:
        """Coda + head over a batch of ActiveTokens (using each token's current
        .hidden, whatever depth that is) → per-token logits in the SAME order.
        Depth-agnostic: pass depth-T_draft hidden for a draft logit, depth-T_total
        for a full-depth verify logit."""

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def find_active(self, state: WavefrontState, position: int) -> ActiveToken | None:
        """First alive token at `position` across active_r / active_p / active_c."""
        return state.find_active_prc(position)
