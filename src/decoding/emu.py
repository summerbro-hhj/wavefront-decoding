"""Emulated acceptance for exp4 (see .claude/plans/exp4_plan.md §3.a).

Injects a target acceptance rate α into the schedulers: every draft-verify
comparison ("attempt") is replaced by an independent Bernoulli(α) draw from a
seeded stream. Everything else — P/R/C compute, argmax, cache writes, rollback —
runs exactly as in the real schedulers, so walltime stays faithful; only the
accept/reject *decision* is emulated.

Fairness (random tape): each sequence gets its own stream (seed = base_seed +
seq index). Re-using the same seed across scheduler runs (WFD vs DtV) makes
them consume the same i.i.d. Bernoulli stream per attempt — statistically the
same acceptance conditions even though the attempt structure differs.
"""
from __future__ import annotations

import random


class EmulatedAcceptance:
    """Bernoulli(α) accept/reject stream with bookkeeping."""

    def __init__(self, rate: float, seed: int = 0):
        assert 0.0 <= rate <= 1.0, f"acceptance rate must be in [0,1], got {rate}"
        self.rate = float(rate)
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_draws = 0
        self.n_accepts = 0

    def draw(self) -> bool:
        """Decide the n-th draft attempt. Called once per proposed draft."""
        self.n_draws += 1
        ok = self.rng.random() < self.rate
        self.n_accepts += int(ok)
        return ok

    @property
    def empirical_rate(self) -> float:
        return self.n_accepts / self.n_draws if self.n_draws else float("nan")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"EmulatedAcceptance(rate={self.rate}, seed={self.seed}, "
                f"draws={self.n_draws}, empirical={self.empirical_rate:.3f})")
