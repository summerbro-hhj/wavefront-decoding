"""Unified interface for recursive / looped LLMs.

Concrete subclasses wrap a specific architecture (Loopformer, parcae) and expose
a uniform `forward(input_ids, num_iterations, return_intermediates)` signature
so downstream experiment code does not branch on model type.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class RecursiveOutput:
    """Result of a forward pass on a recursive LM.

    final_logits: (B, T, V) logits computed at the final iteration `num_iterations`.
    intermediate_logits: list of (B, T, V) logits, one per iteration index 1..num_iterations-1,
        or None if not requested. intermediate_logits[k] corresponds to iteration k+1
        (i.e. index 0 -> after iteration 1, index n-2 -> after iteration n-1).
    """

    final_logits: torch.Tensor
    intermediate_logits: list[torch.Tensor] | None = None


class RecursiveLM(ABC):
    """Abstract wrapper around a recursive / looped language model."""

    def __init__(self, model, tokenizer, max_iterations: int, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.max_iterations = max_iterations
        self.device = device

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        num_iterations: int | None = None,
        return_intermediates: bool = False,
        steps: list[float] | None = None,
    ) -> RecursiveOutput:
        """Run a single forward pass.

        Args:
            input_ids: (B, T) token IDs on self.device.
            num_iterations: target depth `n`. Defaults to self.max_iterations.
                Ignored if `steps` is provided.
            return_intermediates: if True, also return logits computed after each
                intermediate iteration k=1..n-1.
            steps: optional explicit per-iteration schedule.
                - LoopformerWrapper: list of dt floats; each entry is the dt fed
                  to the time embedder at that iteration. `n = len(steps)`.
                  Allows nonuniform scheduling (the released checkpoint was
                  trained with uniform dt = 1 / max_iterations).
                - ParcaeWrapper: parcae's recurrent core has no dt concept; only
                  `len(steps)` is used to set the iteration count, the values
                  themselves are ignored. A warning is printed if values look
                  meaningful.
        """
        raise NotImplementedError

    def encode(self, text: str | list[str], **kwargs) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        return self.tokenizer(text, return_tensors="pt", padding=True, **kwargs).input_ids.to(
            self.device
        )

    def decode(self, ids: torch.Tensor) -> list[str]:
        return [self.tokenizer.decode(row, skip_special_tokens=False) for row in ids]
