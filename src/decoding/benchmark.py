"""Per-prompt AR vs wavefront-SSD walltime + acceptance benchmark.

This is the metric layer exp2 uses to report results — wraps the per-model AR
baseline (`blocks.ar_generate`, which dispatches to the FA4 P/R/C-block sequential
AR: `ar_generate_parcae_blocks` / `ar_generate_ouro`) and `generate_wavefront_dynamic`,
producing a single `BenchmarkMetrics` per prompt. AR and SSD thus share the same
decode attention backend (FA4), so the speedup isolates the wavefront benefit.

Lossless caveat: bf16 cublas kernels are batch-shape-dependent (token-wise
ops like MLP differ at ~1e-4 between (1,1,D) and (1,T,D) inputs), so AR vs
SSD token sequences can diverge despite algorithmically identical forward
paths. We report `match_rate` rather than treating mismatch as failure. See
[.claude/plans/exp2_plan.md] §"bf16 numerical drift" and
[.claude/plans/exp2_implementation_log.md] §"Bug 5".
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from .blocks import RecursiveBlocks
from .policy import SsdPolicy
from .scheduler import generate_draft_then_verify, generate_wavefront_dynamic


@dataclass
class BenchmarkMetrics:
    # Walltime
    ar_walltime_s: float
    ssd_walltime_s: float
    ssd_walltime_steady_s: float
    speedup: float
    speedup_steady: float
    # Throughput (speedup IS the throughput ratio ssd_tps/ar_tps, not walltime ratio)
    n_tokens_ar: int
    n_tokens_ssd: int
    ar_tokens_per_s: float
    ssd_tokens_per_s: float
    ar_ms_per_token: float
    ssd_ms_per_token: float
    # Wavefront internals
    n_r_calls_ssd: int
    n_c_calls_ssd: int
    n_p_calls_ssd: int
    accepted_drafts: int
    rejected_drafts: int
    proposed_drafts: int
    acceptance_rate: float
    n_rollback_events: int
    # Near-lossless: AR vs SSD token-by-token match rate (bf16 drift can lower this)
    n_match: int
    match_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def benchmark_one(
    blocks: RecursiveBlocks,
    prompt_ids: torch.Tensor,
    policy: SsdPolicy,
    max_new_tokens: int,
    *,
    n_warmup: int = 1,
    wrapper,
    eos_token_id: int | None = None,
    scheduler: str = "wfd",
    draft_length: int = 4,
) -> BenchmarkMetrics:
    """Run AR and self-spec SSD on one prompt, return metrics.

    n_warmup: number of throwaway runs to warm up cublas / CUDA caching
              allocator before timed runs. Set to 0 to disable.
    eos_token_id: if given, both AR and SSD stop when it is generated.
    scheduler: "wfd" (wavefront, diagonal) or "dtv" (draft-then-verify, nested
               reuse). Both compared against the same AR baseline.
    draft_length: speculation length for the "dtv" scheduler (ignored by "wfd").
    """
    device = next(blocks.model.parameters()).device
    # Same sampler for AR and SSD (greedy, or position-seeded temperature) so the
    # two decode paths draw the identical token per position -> match_rate stays
    # a valid lossless check under temperature sampling too.
    sampler = policy.make_sampler()

    def _run_ar(prompt, mnt):
        return blocks.ar_generate(prompt, max_new_tokens=mnt, eos_token_id=eos_token_id, sampler=sampler)

    def _run_ssd(prompt, mnt):
        if scheduler == "dtv":
            return generate_draft_then_verify(
                blocks, prompt, max_new_tokens=mnt, policy=policy,
                draft_length=draft_length, eos_token_id=eos_token_id)
        return generate_wavefront_dynamic(
            blocks, prompt, max_new_tokens=mnt, policy=policy, eos_token_id=eos_token_id)

    # Warm-up: short AR + SSD run, untimed. AR baseline dispatches per model
    # (parcae / Ouro) via blocks.ar_generate, keeping the benchmark model-agnostic.
    for _ in range(n_warmup):
        _ = _run_ar(prompt_ids.squeeze(0), 4)
        _ = _run_ssd(prompt_ids.squeeze(0), 4)

    # AR baseline
    if device.type == "cuda":
        torch.cuda.synchronize()
    ar_gen, ar_walltime = _run_ar(prompt_ids.squeeze(0), max_new_tokens)

    # Self-spec SSD (wavefront or draft-then-verify)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ssd_gen, trace = _run_ssd(prompt_ids.squeeze(0), max_new_tokens)



    n_tok_ar = len(ar_gen)
    n_tok_ssd = len(ssd_gen)
    n_compare = min(n_tok_ar, n_tok_ssd)
    n_match = sum(1 for i in range(n_compare) if ar_gen[i] == ssd_gen[i])
    match_rate = (n_match / n_compare) if n_compare > 0 else float("nan")

    # Speedup = THROUGHPUT ratio (tokens/s), not walltime ratio: with EOS, AR and
    # SSD can emit different token counts, so per-token time is the fair compare.
    ar_tps = n_tok_ar / ar_walltime if ar_walltime > 0 else float("nan")
    ssd_tps = n_tok_ssd / trace.walltime_s if trace.walltime_s > 0 else float("nan")
    ssd_tps_steady = n_tok_ssd / trace.walltime_steady_s if trace.walltime_steady_s > 0 else float("nan")
    speedup = ssd_tps / ar_tps if ar_tps > 0 else float("nan")
    speedup_steady = ssd_tps_steady / ar_tps if ar_tps > 0 else float("nan")

    return BenchmarkMetrics(
        ar_walltime_s=ar_walltime,
        ssd_walltime_s=trace.walltime_s,
        ssd_walltime_steady_s=trace.walltime_steady_s,
        speedup=speedup,
        speedup_steady=speedup_steady,
        n_tokens_ar=n_tok_ar,
        n_tokens_ssd=n_tok_ssd,
        ar_tokens_per_s=ar_tps,
        ssd_tokens_per_s=ssd_tps,
        ar_ms_per_token=(ar_walltime / n_tok_ar * 1e3) if n_tok_ar > 0 else float("nan"),
        ssd_ms_per_token=(trace.walltime_s / n_tok_ssd * 1e3) if n_tok_ssd > 0 else float("nan"),
        n_r_calls_ssd=trace.n_r_calls,
        n_c_calls_ssd=trace.n_c_calls,
        n_p_calls_ssd=trace.n_p_calls,
        accepted_drafts=trace.n_drafts_accepted,
        rejected_drafts=trace.n_drafts_rejected,
        proposed_drafts=trace.n_drafts_proposed,
        acceptance_rate=trace.acceptance_rate,
        n_rollback_events=trace.n_rollback_events,
        n_match=n_match,
        match_rate=match_rate,
    )


def aggregate(metrics_list: list[BenchmarkMetrics]) -> dict:
    """Per-prompt → average. Walltimes summed, rates micro-averaged over tokens."""
    if not metrics_list:
        return {}
    total_ar = sum(m.ar_walltime_s for m in metrics_list)
    total_ssd = sum(m.ssd_walltime_s for m in metrics_list)
    total_ssd_steady = sum(m.ssd_walltime_steady_s for m in metrics_list)
    total_n_ar = sum(m.n_tokens_ar for m in metrics_list)
    total_n_ssd = sum(m.n_tokens_ssd for m in metrics_list)
    total_accepted = sum(m.accepted_drafts for m in metrics_list)
    total_proposed = sum(m.proposed_drafts for m in metrics_list)
    total_match = sum(m.n_match for m in metrics_list)
    total_compare = sum(min(m.n_tokens_ar, m.n_tokens_ssd) for m in metrics_list)

    return {
        "n_prompts": len(metrics_list),
        "ar_walltime_s_total": total_ar,
        "ssd_walltime_s_total": total_ssd,
        "ssd_walltime_steady_s_total": total_ssd_steady,
        # speedup = throughput ratio (tokens/s) — fair when EOS makes token counts differ.
        "speedup": ((total_n_ssd / total_ssd) / (total_n_ar / total_ar))
                   if (total_ssd > 0 and total_ar > 0 and total_n_ar > 0) else float("nan"),
        "speedup_steady": ((total_n_ssd / total_ssd_steady) / (total_n_ar / total_ar))
                          if (total_ssd_steady > 0 and total_ar > 0 and total_n_ar > 0) else float("nan"),
        "ar_tokens_per_s": total_n_ar / total_ar if total_ar > 0 else float("nan"),
        "ssd_tokens_per_s": total_n_ssd / total_ssd if total_ssd > 0 else float("nan"),
        "ar_ms_per_token": (total_ar / total_n_ar * 1e3) if total_n_ar > 0 else float("nan"),
        "ssd_ms_per_token": (total_ssd / total_n_ssd * 1e3) if total_n_ssd > 0 else float("nan"),
        "acceptance_rate": (total_accepted / total_proposed) if total_proposed > 0 else float("nan"),
        "match_rate": (total_match / total_compare) if total_compare > 0 else float("nan"),
        "n_tokens_ar_total": total_n_ar,
        "n_tokens_ssd_total": total_n_ssd,
        "accepted_drafts_total": total_accepted,
        "proposed_drafts_total": total_proposed,
        "rollback_events_total": sum(m.n_rollback_events for m in metrics_list),
    }
