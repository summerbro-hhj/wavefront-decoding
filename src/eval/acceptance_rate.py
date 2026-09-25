"""Experiment 1 metrics.

Given the final-iteration logits (T = n) and the intermediate-iteration logits
(T = 1..n-1) for a batch of inputs, compute:

  - top-1 acceptance:  1[argmax(final) == argmax(intermediate)]
  - top-k overlap:     |topk(final) ∩ topk(intermediate)| / k
  - softmax KL:        KL( softmax(final) || softmax(intermediate) )  (in nats)
  - reverse KL:        KL( softmax(intermediate) || softmax(final) )
  - entropies for diagnostic purposes

Everything is computed per (batch, position) and the caller decides how to
aggregate. We pass logits in float32 for KL stability.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class IterationMetrics:
    """Per-token metrics for one intermediate iteration vs. the final.

    Shapes are (B, T) where T is sequence length minus any masked positions.
    """

    top1_match: torch.Tensor  # bool
    topk_overlap: torch.Tensor  # int (0..k)
    kl_final_to_iter: torch.Tensor  # float
    kl_iter_to_final: torch.Tensor  # float
    entropy_iter: torch.Tensor  # float
    iter_index: int  # 1-indexed iteration count (k in T=k)


def _safe_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    # Cast to float32 for numerical stability when logits are bf16/fp16.
    return F.log_softmax(logits.float(), dim=-1)


def compute_acceptance_metrics(
    final_logits: torch.Tensor,
    intermediate_logits: list[torch.Tensor],
    top_k: int = 5,
    position_mask: torch.Tensor | None = None,
) -> tuple[list[IterationMetrics], torch.Tensor]:
    """Compute per-position metrics for each intermediate iteration.

    Args:
        final_logits: (B, T, V) logits at iteration n.
        intermediate_logits: list of (B, T, V) logits at iterations 1..n-1.
            The k-th element (0-indexed) corresponds to iteration k+1.
        top_k: size of the top-k overlap set.
        position_mask: optional (B, T) bool mask; if given, metrics at masked-out
            positions are returned as NaN/0 but still kept for downstream
            aggregation. The mask itself is also returned so the caller knows
            which positions to count.

    Returns:
        list of IterationMetrics (length n-1), plus the entropy of the final
        distribution at every position (used as a diagnostic).
    """
    if final_logits.dim() != 3:
        raise ValueError(f"final_logits must be (B,T,V), got {tuple(final_logits.shape)}")

    log_p_final = _safe_log_softmax(final_logits)
    p_final = log_p_final.exp()
    final_top1 = final_logits.argmax(dim=-1)  # (B, T)
    final_topk = final_logits.topk(top_k, dim=-1).indices  # (B, T, K)
    entropy_final = -(p_final * log_p_final).sum(dim=-1)  # (B, T)

    metrics: list[IterationMetrics] = []
    for k, iter_logits in enumerate(intermediate_logits):
        if iter_logits.shape != final_logits.shape:
            raise ValueError(
                f"intermediate_logits[{k}] shape {tuple(iter_logits.shape)} "
                f"!= final_logits shape {tuple(final_logits.shape)}"
            )

        log_q = _safe_log_softmax(iter_logits)
        q = log_q.exp()

        iter_top1 = iter_logits.argmax(dim=-1)  # (B, T)
        iter_topk = iter_logits.topk(top_k, dim=-1).indices  # (B, T, K)

        top1_match = iter_top1 == final_top1  # (B, T) bool

        # Top-K overlap: count tokens that appear in both top-k sets.
        # Reshape to (B, T, K, 1) vs (B, T, 1, K) and any-match along last dim.
        overlap_matrix = (final_topk.unsqueeze(-1) == iter_topk.unsqueeze(-2)).any(dim=-2)
        topk_overlap = overlap_matrix.sum(dim=-1)  # (B, T) int

        # KL(p || q) = sum p * (log p - log q)
        kl_final_to_iter = (p_final * (log_p_final - log_q)).sum(dim=-1)
        kl_iter_to_final = (q * (log_q - log_p_final)).sum(dim=-1)
        entropy_iter = -(q * log_q).sum(dim=-1)

        if position_mask is not None:
            mask = position_mask.bool()
            zero = torch.zeros_like(kl_final_to_iter)
            nan = torch.full_like(kl_final_to_iter, float("nan"))
            kl_final_to_iter = torch.where(mask, kl_final_to_iter, nan)
            kl_iter_to_final = torch.where(mask, kl_iter_to_final, nan)
            entropy_iter = torch.where(mask, entropy_iter, nan)
            top1_match = torch.where(mask, top1_match, torch.zeros_like(top1_match))
            topk_overlap = torch.where(mask, topk_overlap, torch.zeros_like(topk_overlap))

        metrics.append(
            IterationMetrics(
                top1_match=top1_match,
                topk_overlap=topk_overlap,
                kl_final_to_iter=kl_final_to_iter,
                kl_iter_to_final=kl_iter_to_final,
                entropy_iter=entropy_iter,
                iter_index=k + 1,
            )
        )

    return metrics, entropy_final


def aggregate_metrics(
    all_metrics: list[list[IterationMetrics]],
    masks: list[torch.Tensor] | None = None,
    top_k: int = 5,
) -> dict:
    """Aggregate per-sample metrics across the whole dataset.

    Args:
        all_metrics: list (over samples) of list (over iterations) of IterationMetrics.
        masks: optional list of (B, T) position masks corresponding to each sample's
            valid positions.
        top_k: top-k value used during compute_acceptance_metrics; used only to scale
            the topk_overlap fraction.

    Returns:
        dict {iter_index -> {metric_name -> scalar}}.
    """
    if not all_metrics:
        return {}
    n_iter = len(all_metrics[0])
    out: dict = {}

    for it in range(n_iter):
        top1_hits = 0
        topk_overlap_sum = 0
        kl_pq_sum = 0.0
        kl_qp_sum = 0.0
        entropy_sum = 0.0
        count = 0

        for sample_idx, sample_iters in enumerate(all_metrics):
            m = sample_iters[it]
            mask = masks[sample_idx] if masks else None
            if mask is None:
                mask = torch.ones_like(m.top1_match, dtype=torch.bool)
            mask = mask.bool()
            n = mask.sum().item()
            if n == 0:
                continue
            top1_hits += m.top1_match.bool()[mask].sum().item()
            topk_overlap_sum += m.topk_overlap[mask].sum().item()
            kl_pq_sum += m.kl_final_to_iter[mask].sum().item()
            kl_qp_sum += m.kl_iter_to_final[mask].sum().item()
            entropy_sum += m.entropy_iter[mask].sum().item()
            count += n

        if count == 0:
            continue
        out[it + 1] = {
            "top1_acceptance": top1_hits / count,
            "topk_overlap_frac": topk_overlap_sum / (count * top_k),
            "kl_final_to_iter_mean": kl_pq_sum / count,
            "kl_iter_to_final_mean": kl_qp_sum / count,
            "entropy_iter_mean": entropy_sum / count,
            "n_positions": count,
        }
    return out
