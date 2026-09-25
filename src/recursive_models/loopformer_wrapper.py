"""Loopformer wrapper exposing intermediate-iteration logits.

The HuggingFace checkpoint `armenjeddi/LoopFormer-3block-8iterations` ships with
a `modeling_loopformer.py` whose `GPT.forward()` iterates a shared block stack
M times. We do NOT modify that file; instead, this wrapper re-implements the
inner iteration loop so that the logits computed after each iteration k=1..n
are returned in one forward pass (sharing all preceding compute).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import RecursiveLM, RecursiveOutput


def _loop_with_intermediates(
    gpt,
    idx: torch.Tensor,
    steps: list[float],
    return_intermediates: bool,
):
    """Re-implementation of `GPT.forward` from modeling_loopformer.py that can
    additionally return the logits after each intermediate iteration.

    `gpt` is the inner `GPT` module (model.gpt). `steps` is a list of dt floats
    of length `num_iterations`; each entry is 1 / num_iterations under the
    uniform schedule the released checkpoints were trained with.
    """
    device = idx.device
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = gpt.transformer.wte(idx)
    pos_emb = gpt.transformer.wpe(pos)
    x = gpt.transformer.drop(tok_emb + pos_emb)

    intermediate_logits: list[torch.Tensor] = [] if return_intermediates else None

    ti = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    n = len(steps)
    for i, dt in enumerate(steps):
        dt_base = torch.ones_like(ti) * dt
        te = gpt.time_embedder(ti)
        dte = gpt.dt_embedder(dt_base)
        c = te + dte
        x = gpt.transformer.h(x, c)
        ti = ti + dt

        # For all iterations *except the last*, optionally compute logits.
        if return_intermediates and i < n - 1:
            x_iter = gpt.transformer.norm_f(x)
            intermediate_logits.append(gpt.lm_head(x_iter))

    x = gpt.transformer.norm_f(x)
    final_logits = gpt.lm_head(x)

    return final_logits, intermediate_logits


class LoopformerWrapper(RecursiveLM):
    """Wrap `armenjeddi/LoopFormer-*` HF checkpoint."""

    DEFAULT_MODEL_ID = "armenjeddi/LoopFormer-3block-8iterations"
    DEFAULT_MAX_ITERATIONS = 8

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        max_iterations: int | None = None,
    ) -> "LoopformerWrapper":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, dtype=dtype
        ).to(device)
        model.eval()

        if max_iterations is None:
            # Convention in upstream repo: "<arch>-<k>block-<n>iterations".
            # We default to 8 when not inferable.
            if "iterations" in model_id.lower():
                try:
                    tag = model_id.lower().split("-")[-1]
                    max_iterations = int(tag.replace("iterations", ""))
                except Exception:
                    max_iterations = cls.DEFAULT_MAX_ITERATIONS
            else:
                max_iterations = cls.DEFAULT_MAX_ITERATIONS

        return cls(model, tok, max_iterations, torch.device(device))

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        num_iterations: int | None = None,
        return_intermediates: bool = False,
        steps: list[float] | None = None,
    ) -> RecursiveOutput:
        if steps is not None:
            if len(steps) < 1:
                raise ValueError("steps must contain at least one dt value")
            steps_list = [float(x) for x in steps]
        else:
            n = num_iterations if num_iterations is not None else self.max_iterations
            if n < 1:
                raise ValueError(f"num_iterations must be >= 1, got {n}")
            # The released checkpoints were trained with uniform dt = 1 / max_iterations.
            # We keep the same dt magnitude when requesting smaller n so that the time
            # embedder sees in-distribution inputs at each step.
            dt = 1.0 / self.max_iterations
            steps_list = [dt] * n

        # model.gpt is the inner GPT module (see modeling_loopformer.py).
        inner = self.model.gpt
        final_logits, intermediates = _loop_with_intermediates(
            inner, input_ids, steps=steps_list, return_intermediates=return_intermediates
        )
        return RecursiveOutput(final_logits=final_logits, intermediate_logits=intermediates)
