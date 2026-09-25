"""HellaSwag 4-way multiple-choice scoring for recursive LMs.

For each example {ctx, endings[4], label}, we score every candidate ending by
summing log p(token_t | token_<t) for the tokens that belong to the ending
(under the final-iteration logits of the recursive LM). The unnormalized score
gives `acc`; the byte-length-normalized score gives `acc_norm`, which is the
standard HellaSwag metric (length-bias correction).

We DO NOT mutate the LM wrapper — we pass `num_iterations` / `steps` straight
through `RecursiveLM.forward`, so the same scoring code works for both
LoopformerWrapper and ParcaeWrapper.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ChoiceScore:
    raw: list[float]            # sum log-prob per candidate
    norm: list[float]           # byte-length-normalized per candidate
    num_tokens: list[int]       # # tokens scored for each candidate (debug)
    predicted_raw: int
    predicted_norm: int


def _per_token_logprobs(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """logits: (1, T, V), target_ids: (1, T). Returns (1, T-1) log p(target_t | ..._{<t})."""
    # Cast to float32 for stability.
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = target_ids[:, 1:]
    # gather: (1, T-1, V) -> (1, T-1)
    return logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _safe_encode(lm, text: str) -> torch.Tensor:
    """Return (1, T) long input_ids, same shape contract for both wrappers."""
    return lm.encode(text)


def score_example(
    lm,
    ctx: str,
    endings: list[str],
    *,
    num_iterations: int | None = None,
    steps: list[float] | None = None,
    max_length: int = 1024,
) -> ChoiceScore:
    """Score 4 (or N) candidate endings under one recursive LM forward each."""
    raw_scores: list[float] = []
    norm_scores: list[float] = []
    n_tok: list[int] = []

    ctx_ids = _safe_encode(lm, ctx)            # (1, L_ctx)
    L_ctx = ctx_ids.shape[1]

    for ending in endings:
        full_text = f"{ctx} {ending}"
        full_ids = _safe_encode(lm, full_text)  # (1, L_full)
        # Truncate. Keep the ending portion intact when possible by truncating
        # from the left (we still need at least one ending token to score).
        if full_ids.shape[1] > max_length:
            full_ids = full_ids[:, -max_length:]
            # After truncation, L_ctx prefix may no longer be present; fall back
            # to "everything after the first token is ending" — extremely rare
            # for HellaSwag (ctx is short), but keeps the code robust.
            ending_start = 1
        else:
            ending_start = L_ctx

        if full_ids.shape[1] < 2 or ending_start >= full_ids.shape[1]:
            # Pathological case: ending is empty. Skip with -inf score.
            raw_scores.append(float("-inf"))
            norm_scores.append(float("-inf"))
            n_tok.append(0)
            continue

        out = lm.forward(
            full_ids,
            num_iterations=num_iterations,
            steps=steps,
            return_intermediates=False,
        )
        logits = out.final_logits  # (1, T, V)
        token_logps = _per_token_logprobs(logits, full_ids)  # (1, T-1)

        # Positions in token_logps correspond to predicting token at index t+1
        # from context up to t. So a token at index `i` in full_ids is scored
        # by token_logps[:, i-1]. The ending tokens are at indices
        # [ending_start, T). Scored entries are [ending_start-1, T-1).
        scored = token_logps[0, ending_start - 1 : full_ids.shape[1] - 1]
        if scored.numel() == 0:
            raw_scores.append(float("-inf"))
            norm_scores.append(float("-inf"))
            n_tok.append(0)
            continue

        raw = scored.sum().item()
        # Byte-length normalization (lm-eval-harness HellaSwag acc_norm).
        # Use the leading-space-included ending text so length matches what was scored.
        ending_text = " " + ending if not ending.startswith(" ") else ending
        byte_len = max(1, len(ending_text.encode("utf-8")))
        norm = raw / byte_len

        raw_scores.append(raw)
        norm_scores.append(norm)
        n_tok.append(int(scored.numel()))

    predicted_raw = int(max(range(len(raw_scores)), key=lambda i: raw_scores[i]))
    predicted_norm = int(max(range(len(norm_scores)), key=lambda i: norm_scores[i]))
    return ChoiceScore(
        raw=raw_scores,
        norm=norm_scores,
        num_tokens=n_tok,
        predicted_raw=predicted_raw,
        predicted_norm=predicted_norm,
    )


def aggregate_accuracy(
    per_example: list[tuple[ChoiceScore, int]],
) -> dict:
    """per_example: list of (ChoiceScore, gold_label)."""
    n = len(per_example)
    if n == 0:
        return {"n": 0}
    correct_raw = sum(1 for s, g in per_example if s.predicted_raw == g)
    correct_norm = sum(1 for s, g in per_example if s.predicted_norm == g)
    return {
        "n": n,
        "acc": correct_raw / n,
        "acc_norm": correct_norm / n,
    }
