"""Wavefront SSD policy: hyperparameters that govern the cycle pattern.

For first-impl this is a plain dataclass with fixed (T_total, T_draft). Future
adaptive policy will subclass this with an `on_step()` hook that can read the
WavefrontState and return a different T_draft per cycle / per token.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .acceptance import TokenSampler


@dataclass
class SsdPolicy:
    T_total: int = 8       # verify depth (the full-recursion step count)
    T_draft: int = 2       # number of R calls per draft cycle (must divide T_total cleanly for steady-state)
    # "greedy": argmax everywhere (lossless top-1). "temperature": speculative
    # sampling at temperature T — draft x ~ q (shallow softmax), accept iff
    # r < p(x)/q(x), reject -> sample norm(max(0,p-q)); AR baseline samples p
    # directly. Position-seeded noise, reproducible per sample_seed (acceptance.py).
    mode: Literal["greedy", "temperature"] = "greedy"
    temperature: float = 1.0             # used only when mode == "temperature" (must be > 0)
    sample_seed: int = 0                 # seeds the position-wise sampling noise

    def __post_init__(self) -> None:
        if self.T_draft < 1 or self.T_total < self.T_draft:
            raise ValueError(
                f"Invalid policy: T_total={self.T_total}, T_draft={self.T_draft}. "
                "Require 1 <= T_draft <= T_total."
            )
        if self.mode not in ("greedy", "temperature"):
            raise ValueError(f"Unknown sampling mode: {self.mode!r} (greedy | temperature).")
        if self.mode == "temperature" and not self.temperature > 0.0:
            raise ValueError(f"mode='temperature' requires temperature > 0, got {self.temperature}.")
        if self.T_total % self.T_draft != 0:
            # Not strictly required for correctness, but it lets active-set stay at a clean
            # T_total/T_draft size at steady state. Warn if violated.
            import warnings

            warnings.warn(
                f"T_total ({self.T_total}) is not a multiple of T_draft ({self.T_draft}); "
                "active wave size will fluctuate.",
                stacklevel=2,
            )

    @property
    def steady_state_active_size(self) -> int:
        return self.T_total // self.T_draft

    def make_sampler(self) -> "TokenSampler":
        """The token sampler every decode path (AR baseline, WFD, DtV) must share
        so they emit the same token for the same (position, logits)."""
        from .acceptance import TokenSampler
        if self.mode == "greedy":
            return TokenSampler(temperature=0.0, seed=self.sample_seed)
        return TokenSampler(temperature=self.temperature, seed=self.sample_seed)
