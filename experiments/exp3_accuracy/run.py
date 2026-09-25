"""Experiment 3 entry point — generative math accuracy × wavefront-SSD latency.

Step 1: the test environment. Loads gsm8k / MATH-500 with lm-eval-harness-style
few-shot prompts, runs AR and/or wavefront-SSD generation with the SAME model
weights (reusing exp2's blocks + FA4 decode), extracts the final answer, scores
exact-match accuracy, and reports accuracy alongside throughput/speedup.

The lossy levers (KV-sharing budget s, adaptive total-T τ/ε, adaptive draft-T)
arrive in steps 2–4; this runner is the harness they will plug into. Sweeps are
driven by run.sh env vars (no auto best-search in code).

Sampling: `--sampling greedy` (default) or `--sampling temperature --temperature T`
(speculative sampling: draft x ~ q, accept iff r < p(x)/q(x), residual on
reject; AR samples p directly. SSD output is an exact sample of the full-depth
model at T — equal to AR in distribution, not token-wise; see acceptance.py).

Usage:
    PYTHONPATH=. python experiments/exp3_accuracy/run.py \
        --model parcae-1.3b --dataset gsm8k \
        --T-total 8 --T-draft 2 --limit 20 --decode both \
        --output results/exp3/parcae-1.3b_gsm8k_T8_d2.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

# Reuse exp2's model registry + block builder (exp3 shares the decode stack).
from experiments.exp2_spec_bench.run import (MODEL_REGISTRY, build_blocks, format_prompt,
                                             is_chat_model, resolve_prompt_format)
from src.decoding.policy import SsdPolicy
from src.decoding.scheduler import generate_wavefront_dynamic

from experiments.exp3_accuracy import tasks as ds


def _acc_extras(store: dict, extras: dict) -> None:
    """Accumulate boolean secondary metrics (lm-eval: gsm8k flexible-extract,
    math500 math_verify); None (metric unavailable) is skipped."""
    for k, v in extras.items():
        if isinstance(v, bool):
            store[k] = store.get(k, 0) + int(v)


def _detok(wrapper, tokens: list[int], skip: bool = True) -> str:
    return wrapper.tokenizer.decode(tokens, skip_special_tokens=skip)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--dataset", required=True, choices=["gsm8k", "math500"])
    p.add_argument("--limit", type=int, default=0, help="First N problems (0 = all).")
    p.add_argument("--num-fewshot", type=int, default=None, help="Default 8 (gsm8k) / 4 (math500).")
    p.add_argument("--T-total", type=int, default=8, help="Full recursion depth (parcae 8, Ouro 4).")
    p.add_argument("--T-draft", type=int, default=2)
    p.add_argument("--kv-budget-s", type=int, default=0,
                   help="exp3 R-block KV-sharing budget s (0 or >=T_total = off/exp2-lossless). "
                        "Applied to BOTH AR and SSD. s<T_total: lossy, depths share slots round-robin.")
    p.add_argument("--exit-threshold", type=float, default=None,
                   help="exp3 adaptive-total-T threshold τ (OURO early_exit_gate). Unset = off "
                        "(full depth = exp2-lossless). A token commits at the first depth where the "
                        "cumulative early-exit prob ≥ τ (lossy verify). Applied to BOTH AR and SSD.")
    p.add_argument("--exit-hidden-eps", type=float, default=None,
                   help="exp3 adaptive-total-T threshold ε (PARCAE latent convergence). Unset = off "
                        "(full depth). A token commits at the first core step whose hidden's relative-L2 "
                        "step-to-step change ‖Δh‖/‖h‖ < ε (lossy verify). Applied to BOTH AR and SSD.")
    p.add_argument("--decode", default="both", choices=["both", "ar", "ssd"])
    p.add_argument("--sampling", default="greedy", choices=["greedy", "temperature"],
                   help="'greedy' (argmax everywhere, default) or 'temperature': speculative sampling "
                        "at T (Leviathan et al. 2023) — draft x ~ q=softmax(shallow/T), accept iff "
                        "r < p(x)/q(x), reject -> sample norm(max(0,p-q)); AR samples p directly. "
                        "SSD output ~ AR output in distribution (not token-wise: match_rate is "
                        "uninformative under temperature). Position-seeded noise, reproducible per --seed.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature T (> 0); used only with --sampling temperature.")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-prompt-tokens", type=int, default=1024)
    p.add_argument("--prompt-format", default="auto", choices=["auto", "raw", "chat"],
                   help="How prompts are fed to the model: 'raw' = plain text (base models); 'chat' = "
                        "wrap in the tokenizer's chat template as a single user turn with "
                        "add_generation_prompt + enable_thinking (Ouro-*-Thinking SFT models); "
                        "'auto' (default) = chat for *-thinking models, raw otherwise.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--state-init", default="random", choices=["random", "zero"],
                   help="Initial recurrent state s0 (parcae/huginn): 'random' = model-native "
                        "like-init, position-seeded (default); 'zero' = legacy deterministic.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds torch, the recurrent s0 draw and (temperature) the sampling noise.")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    if args.sampling == "temperature" and not args.temperature > 0:
        raise ValueError(f"--temperature must be > 0 with --sampling temperature (got {args.temperature}).")
    torch.manual_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    do_ar = args.decode in ("both", "ar")
    do_ssd = args.decode in ("both", "ssd")

    blocks, wrapper = build_blocks(args.model, dtype=dtype, device=args.device,
                                   T_total=args.T_total,
                                   state_init=args.state_init, state_seed=args.seed)
    blocks.kv_budget_s = args.kv_budget_s or None    # common to AR + SSD (None = off)
    blocks.exit_threshold = args.exit_threshold      # Ouro early-exit gate τ (None = off)
    blocks.exit_hidden_eps = args.exit_hidden_eps    # parcae/huginn latent-convergence ε (None = off)
    # Guard against silently-ignored levers: τ is Ouro's built-in gate; parcae /
    # huginn have no gate and use the latent-convergence ε instead (and vice versa).
    from src.decoding.ouro_blocks import OuroBlocks as _OuroBlocks
    if args.exit_threshold is not None and not isinstance(blocks, _OuroBlocks):
        print(f"[exp3] WARNING: --exit-threshold(τ)={args.exit_threshold} is IGNORED by "
              f"{type(blocks).__name__} (no early-exit gate) — use --exit-hidden-eps(ε) instead.")
    if args.exit_hidden_eps is not None and isinstance(blocks, _OuroBlocks):
        print(f"[exp3] WARNING: --exit-hidden-eps(ε)={args.exit_hidden_eps} is IGNORED by "
              f"OuroBlocks — use --exit-threshold(τ) instead.")
    eos_token_id = getattr(wrapper, "eos_token_id", None)
    policy = SsdPolicy(T_total=args.T_total, T_draft=args.T_draft, mode=args.sampling,
                       temperature=args.temperature, sample_seed=args.seed)
    sampler = policy.make_sampler()     # shared by AR + SSD (same token per position)

    prompt_format = resolve_prompt_format(args.model, args.prompt_format)
    if is_chat_model(args.model) and prompt_format == "raw":
        print(f"[exp3] WARNING: {args.model} is a chat/thinking model but --prompt-format raw was requested.")

    rows = ds.load_dataset_prompts(args.dataset, args.num_fewshot, args.limit)
    # 'chat': the lm-eval few-shot prompt text becomes the single user turn
    # (set --num-fewshot 0 for a zero-shot question). The few-shot exemplars /
    # gold extraction are unchanged, so the scoring regexes still assume the
    # base-model answer formats ("The answer is N." / Minerva "Final Answer").
    for row in rows:
        row["prompt"] = format_prompt(wrapper, row["prompt"], prompt_format)
    print(f"[exp3] {args.model} | {args.dataset} | {len(rows)} problems | "
          f"T_total={args.T_total} T_draft={args.T_draft} kv_s={blocks.kv_budget_s} "
          f"exit_tau={blocks.exit_threshold} exit_eps={blocks.exit_hidden_eps} "
          f"decode={args.decode} eos={eos_token_id} sampling={sampler} prompt_format={prompt_format}")

    # Warm-up (untimed) so the first real sample isn't penalised.
    if rows and args.warmup:
        warm = wrapper.encode(rows[0]["prompt"])[:, : args.max_prompt_tokens].squeeze(0)
        for _ in range(args.warmup):
            if do_ar:
                blocks.ar_generate(warm, max_new_tokens=4, sampler=sampler)
            if do_ssd:
                generate_wavefront_dynamic(blocks, warm, max_new_tokens=4, policy=policy)

    n = 0
    correct_ar = correct_ssd = 0
    extra_ar: dict = {}
    extra_ssd: dict = {}
    tok_ar = tok_ssd = 0
    wall_ar = wall_ssd = 0.0
    n_acc = n_prop = n_match = n_cmp = 0
    rcalls_ssd = 0
    per_sample = []

    for i, row in enumerate(rows):
        ids = wrapper.encode(row["prompt"])
        if ids.shape[1] > args.max_prompt_tokens:
            ids = ids[:, : args.max_prompt_tokens]
            if prompt_format == "chat":
                print(f"[exp3] WARNING: sample {i} chat prompt right-truncated to {args.max_prompt_tokens} "
                      f"tokens (drops the assistant/<think> tail) — raise --max-prompt-tokens.")
        ids1 = ids.squeeze(0).to(args.device)
        prompt_list = ids1.tolist()
        rec = {"idx": i, "gold": row["gold"]}

        gen_ar = gen_ssd = None
        if do_ar:
            torch.cuda.synchronize() if args.device == "cuda" else None
            gen_ar, t_ar = blocks.ar_generate(ids1, max_new_tokens=args.max_new_tokens,
                                              eos_token_id=eos_token_id, sampler=sampler)
            txt = _detok(wrapper, gen_ar)
            ok, pred, extra = ds.score(args.dataset, txt, row["gold"], row["stop"], row.get("meta"))
            correct_ar += int(ok); tok_ar += len(gen_ar); wall_ar += t_ar
            _acc_extras(extra_ar, extra)
            # generation_* = model's continuation; full_* = prompt+continuation
            # (the full decoded token sequence as text), special tokens kept for faithfulness.
            rec.update(pred_ar=pred, correct_ar=ok, n_tokens_ar=len(gen_ar),
                       generation_ar=_detok(wrapper, gen_ar, skip=False),
                       full_ar=_detok(wrapper, prompt_list + gen_ar, skip=False),
                       **{f"{k}_ar": v for k, v in extra.items()})
        if do_ssd:
            torch.cuda.synchronize() if args.device == "cuda" else None
            gen_ssd, trace = generate_wavefront_dynamic(
                blocks, ids1, max_new_tokens=args.max_new_tokens, policy=policy, eos_token_id=eos_token_id)
            txt = _detok(wrapper, gen_ssd)
            ok, pred, extra = ds.score(args.dataset, txt, row["gold"], row["stop"], row.get("meta"))
            correct_ssd += int(ok); tok_ssd += len(gen_ssd); wall_ssd += trace.walltime_s
            _acc_extras(extra_ssd, extra)
            n_acc += trace.n_drafts_accepted; n_prop += trace.n_drafts_proposed
            rcalls_ssd += trace.n_r_calls
            # r_calls_per_token = mean commit depth proxy (adaptive-total-T lever:
            # early-exit lowers it below T_total -> fewer R calls -> latency).
            rec.update(pred_ssd=pred, correct_ssd=ok, n_tokens_ssd=len(gen_ssd),
                       **{f"{k}_ssd": v for k, v in extra.items()},
                       r_calls_ssd=trace.n_r_calls,
                       r_calls_per_token_ssd=(trace.n_r_calls / len(gen_ssd)) if gen_ssd else None,
                       generation_ssd=_detok(wrapper, gen_ssd, skip=False),
                       full_ssd=_detok(wrapper, prompt_list + gen_ssd, skip=False))
        if gen_ar is not None and gen_ssd is not None:
            c = min(len(gen_ar), len(gen_ssd))
            n_match += sum(1 for k in range(c) if gen_ar[k] == gen_ssd[k]); n_cmp += c
        n += 1
        per_sample.append(rec)
        if (i + 1) % max(1, len(rows) // 10) == 0 or (i + 1) == len(rows):
            acc_a = correct_ar / n if do_ar else float("nan")
            acc_s = correct_ssd / n if do_ssd else float("nan")
            print(f"  [{i+1}/{len(rows)}] acc_ar={acc_a:.3f} acc_ssd={acc_s:.3f}")

    ar_tps = tok_ar / wall_ar if wall_ar > 0 else float("nan")
    ssd_tps = tok_ssd / wall_ssd if wall_ssd > 0 else float("nan")
    agg = {
        "n": n,
        "accuracy_ar": (correct_ar / n) if (do_ar and n) else None,
        "accuracy_ssd": (correct_ssd / n) if (do_ssd and n) else None,
        "ar_tokens_per_s": ar_tps if do_ar else None,
        "ssd_tokens_per_s": ssd_tps if do_ssd else None,
        "speedup": (ssd_tps / ar_tps) if (do_ar and do_ssd and ar_tps > 0) else None,
        "acceptance_rate": (n_acc / n_prop) if n_prop else None,
        "match_rate": (n_match / n_cmp) if n_cmp else None,
        "n_tokens_ar": tok_ar, "n_tokens_ssd": tok_ssd,
        "ar_walltime_s": wall_ar, "ssd_walltime_s": wall_ssd,
        # adaptive-total-T: R calls / committed token (≈ mean commit depth). Off
        # (full depth) ≈ T_total/acceptance-driven; early-exit drives it down.
        "ssd_r_calls_per_token": (rcalls_ssd / tok_ssd) if (do_ssd and tok_ssd) else None,
        # secondary lm-eval metrics: gsm8k correct_flex (flexible-extract),
        # math500 math_verify. Primary accuracy_* = strict-match / minerva EM.
        **{f"{k}_ar": (c / n) for k, c in extra_ar.items()},
        **{f"{k}_ssd": (c / n) for k, c in extra_ssd.items()},
    }
    print(f"[exp3] DONE  acc_ar={agg['accuracy_ar']}  acc_ssd={agg['accuracy_ssd']}  "
          f"speedup={agg['speedup']}  acceptance={agg['acceptance_rate']}  match={agg['match_rate']}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump({"config": vars(args) | {"prompt_format_resolved": prompt_format},
                   "aggregate": agg, "per_sample": per_sample}, f, indent=2)
        print(f"[exp3] wrote {out}")


if __name__ == "__main__":
    main()
