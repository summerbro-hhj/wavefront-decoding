"""Experiment 2 entry point — wavefront self-speculative decoding benchmark.

Measures wall-clock and acceptance metrics of wavefront SSD against the
standard AR baseline using identical model weights. Supports parcae
(`ParcaeBlocks`) and Ouro (`OuroBlocks`); both implement the same
`RecursiveBlocks` interface so the scheduler / benchmark are model-agnostic.

Natural recursion depth differs per family — set --T-total accordingly:
    parcae : mean_recurrence = 8   (use --T-total 8)
    Ouro   : total_ut_steps  = 4   (use --T-total 4)

Sampling: `--sampling greedy` (default, lossless top-1) or
`--sampling temperature --temperature T` (speculative sampling: draft x ~ q,
accept iff r < p(x)/q(x), residual on reject; AR samples p directly. SSD output
is an exact sample of the full-depth model at T, equal to AR in distribution but
not token-wise — see acceptance.py).

Usage:
    PYTHONPATH=. python experiments/exp2_spec_bench/run.py \
        --model parcae-770m \
        --corpus spec_bench \
        --index-start 81 --index-end 560 \
        --T-total 8 --T-draft 2 \
        --max-new-tokens 256 \
        --output results/exp2/parcae-770m_specbench_T8_d2.json

    # Ouro (T_total = total_ut_steps = 4):
    PYTHONPATH=. python experiments/exp2_spec_bench/run.py \
        --model ouro-1.4b --corpus spec_bench --T-total 4 --T-draft 1,2 ...

For T_draft we accept a comma-separated list (e.g. "1,2,4,8") to sweep in one run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.request import urlretrieve

import torch

from src.recursive_models import OuroWrapper, ParcaeWrapper
from src.recursive_models.huginn_wrapper import HuginnWrapper
from src.decoding.benchmark import BenchmarkMetrics, aggregate, benchmark_one
from src.decoding.huginn_blocks import HuginnBlocks
from src.decoding.ouro_blocks import OuroBlocks
from src.decoding.parcae_blocks import ParcaeBlocks
from src.decoding.policy import SsdPolicy


# name -> (wrapper class, HF repo id, blocks class)
MODEL_REGISTRY = {
    "parcae-140m": (ParcaeWrapper, "SandyResearch/parcae-140m", ParcaeBlocks),
    "parcae-370m": (ParcaeWrapper, "SandyResearch/parcae-370m", ParcaeBlocks),
    "parcae-770m": (ParcaeWrapper, "SandyResearch/parcae-770m", ParcaeBlocks),
    "parcae-1.3b": (ParcaeWrapper, "SandyResearch/parcae-1.3b", ParcaeBlocks),
    "ouro-1.4b": (OuroWrapper, "ByteDance/Ouro-1.4B", OuroBlocks),
    "ouro-2.6b": (OuroWrapper, "ByteDance/Ouro-2.6B", OuroBlocks),
    # Reasoning SFT variants: identical architecture/config to the base Ouro
    # checkpoints (only weights + bos/eos ids differ; eos = <|im_end|> = 2), so
    # they reuse OuroWrapper/OuroBlocks unchanged. They are chat models — feed
    # prompts through the chat template (--prompt-format auto/chat) and expect
    # a <think>...</think> block before the answer (needs a large max_new_tokens).
    "ouro-1.4b-thinking": (OuroWrapper, "ByteDance/Ouro-1.4B-Thinking", OuroBlocks),
    "ouro-2.6b-thinking": (OuroWrapper, "ByteDance/Ouro-2.6B-Thinking", OuroBlocks),
    # RDM (recurrent-depth model, arXiv:2502.05171) — parcae's 3.5B original.
    # Natural depth mean_recurrence = 32 (use --T-total up to 32).
    "huginn-3.5b": (HuginnWrapper, "tomg-group-umd/huginn-0125", HuginnBlocks),
}


# Bundled debug prompts so dry-runs do not require downloading any dataset.
DEBUG_PROMPTS = [
    "Once upon a time, in a small village nestled between two mountains,",
    "The Fibonacci sequence is defined as",
    "Yesterday I went to the grocery store and bought",
    "Artificial intelligence is",
    "In the beginning, God created the heavens and the earth.",
    "To be, or not to be, that is the question:",
    "The quick brown fox jumps over the lazy dog.",
    "Hello world! This is a test of",
]


_SPEC_BENCH_URL = "https://raw.githubusercontent.com/hemingkx/Spec-Bench/main/data/spec_bench/question.jsonl"


def load_corpus(name: str, index_start: int, index_end: int, cache_dir: Path) -> list[tuple[str, str]]:
    """Return list of (prompt_text, category) tuples.

    For spec_bench, select rows whose `question_id` is in
    [max(81, index_start), min(561, index_end)] (the jsonl spans question_id
    81..560), so the work can be split across GPUs by index range.
    """
    if name == "debug":
        return [(p, "debug") for p in DEBUG_PROMPTS]

    if name == "spec_bench":
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / "spec_bench_question.jsonl"
        if not path.exists():
            print(f"[exp2] downloading SpecBench question.jsonl → {path}")
            urlretrieve(_SPEC_BENCH_URL, path)
        lo, hi = max(81, index_start), min(561, index_end)
        rows = []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                # SpecBench format: {"question_id": ..., "category": ..., "turns": ["...prompt..."]}
                qid = obj.get("question_id")
                if qid is None or not (lo <= qid <= hi):
                    continue
                turns = obj.get("turns") or []
                if not turns:
                    continue
                text = turns[0].strip()
                if not text:
                    continue
                rows.append((text, obj.get("category", "unk")))
        return rows

    raise ValueError(f"Unknown corpus: {name}")


def is_chat_model(name: str) -> bool:
    """Registry names of instruction/reasoning-tuned checkpoints (chat template expected)."""
    return name.endswith("-thinking")


def resolve_prompt_format(name: str, prompt_format: str) -> str:
    """'auto' -> 'chat' for chat models, 'raw' otherwise; explicit values pass through."""
    if prompt_format == "auto":
        return "chat" if is_chat_model(name) else "raw"
    return prompt_format


def format_prompt(wrapper, text: str, prompt_format: str) -> str:
    """Return the string actually encoded for the model. 'raw' = text as-is.
    'chat' = tokenizer.apply_chat_template([{user: text}], add_generation_prompt=True,
    enable_thinking=True) — for Ouro-*-Thinking this yields
    '<|im_start|>system...<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n'."""
    if prompt_format == "raw":
        return text
    if prompt_format != "chat":
        raise ValueError(f"Unknown prompt_format: {prompt_format}")
    tok = wrapper.tokenizer
    if getattr(tok, "chat_template", None) is None:
        raise ValueError("--prompt-format chat requested but the tokenizer has no chat_template.")
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def build_blocks(name: str, dtype: torch.dtype, device: str, T_total: int,
                 state_init: str = "random", state_seed: int = 0):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    wrapper_cls, model_id, blocks_cls = MODEL_REGISTRY[name]
    print(f"[exp2] Loading {name} from {model_id} (dtype={dtype}, device={device})")
    wrapper = wrapper_cls.from_pretrained(model_id, dtype=dtype, device=device)
    # Warn if T_total exceeds the model's trained recursion depth (out-of-distribution):
    # parcae -> mean_recurrence (8), Ouro -> total_ut_steps (4), huginn -> mean_recurrence (32).
    natural = getattr(wrapper.model.config, "total_ut_steps", None) or getattr(
        wrapper.model.config, "mean_recurrence", None
    )
    if natural is not None and T_total > natural:
        print(f"[exp2] WARNING: T_total={T_total} exceeds {name}'s trained depth ({natural}); "
              f"recursion is out-of-distribution. Use --T-total {natural} for in-distribution results.")
    blocks = blocks_cls(wrapper.model, T_total=T_total)
    # Initial recurrent state s0 (parcae / huginn; Ouro has no such concept):
    # "random" = model-native like-init with position-seeded draws (lossless
    # gates preserved); "zero" = pre-2026-07-14 deterministic behaviour.
    if hasattr(blocks, "state_init"):
        blocks.state_init = state_init
        blocks.state_seed = state_seed
        print(f"[exp2] state_init={state_init} (seed={state_seed})")
    return blocks, wrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--corpus", default="debug", choices=["debug", "spec_bench"])
    p.add_argument("--index-start", type=int, default=81,
                   help="Lower question_id bound (clamped to >= 81). For splitting work across GPUs.")
    p.add_argument("--index-end", type=int, default=560,
                   help="Upper question_id bound (clamped to <= 561). SpecBench question_id spans 81..560.")
    p.add_argument("--T-total", type=int, default=8)
    p.add_argument("--T-draft", type=str, default="2",
                   help="Comma-separated list of T_draft values to sweep, e.g. '1,2,4,8'.")
    p.add_argument("--scheduler", default="wfd", choices=["wfd", "dtv"],
                   help="SSD scheduler: 'wfd' (wavefront, diagonal — default) or 'dtv' "
                        "(draft-then-verify, nested reuse). Both compared vs the same AR baseline.")
    p.add_argument("--draft-length", type=int, default=4,
                   help="dtv speculation length (# draft tokens per round; ignored by wfd).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--max-prompt-tokens", type=int, default=512,
                   help="Truncate prompt to this many tokens (long SpecBench prompts).")
    p.add_argument("--prompt-format", default="auto", choices=["auto", "raw", "chat"],
                   help="How prompts are fed to the model: 'raw' = plain text (base models); 'chat' = "
                        "wrap in the tokenizer's chat template as a single user turn with "
                        "add_generation_prompt + enable_thinking (Ouro-*-Thinking SFT models); "
                        "'auto' (default) = chat for *-thinking models, raw otherwise.")
    p.add_argument("--state-init", default="random", choices=["random", "zero"],
                   help="Initial recurrent state s0 (parcae/huginn): 'random' = model-native "
                        "like-init, position-seeded (default); 'zero' = legacy deterministic.")
    p.add_argument("--sampling", default="greedy", choices=["greedy", "temperature"],
                   help="'greedy' (argmax everywhere, default) or 'temperature': speculative sampling "
                        "at T (Leviathan et al. 2023) — draft x ~ q=softmax(shallow/T), accept iff "
                        "r < p(x)/q(x), reject -> sample norm(max(0,p-q)); AR samples p directly. "
                        "SSD output ~ AR output in distribution (not token-wise: match_rate is "
                        "uninformative under temperature). Position-seeded noise, reproducible per --seed.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature T (> 0); used only with --sampling temperature.")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0,
                   help="Seeds torch, the recurrent s0 draw and (temperature) the sampling noise.")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--warmup", type=int, default=1, help="Warm-up runs per prompt before timed measurements.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    if args.sampling == "temperature" and not args.temperature > 0:
        raise ValueError(f"--temperature must be > 0 with --sampling temperature (got {args.temperature}).")
    T_drafts = [int(x.strip()) for x in args.T_draft.split(",") if x.strip()]
    for d in T_drafts:
        if d < 1 or d > args.T_total:
            raise ValueError(f"Invalid T_draft={d}; must satisfy 1 <= T_draft <= T_total ({args.T_total}).")

    blocks, wrapper = build_blocks(args.model, dtype=dtype, device=args.device,
                                   T_total=args.T_total,
                                   state_init=args.state_init, state_seed=args.seed)
    eos_token_id = getattr(wrapper, "eos_token_id", None)
    cache_dir = Path("data") / "exp2"
    prompts = load_corpus(args.corpus, args.index_start, args.index_end, cache_dir)
    print(f"[exp2] Loaded {len(prompts)} prompts from corpus={args.corpus} "
          f"(index {args.index_start}..{args.index_end}, eos_token_id={eos_token_id})")
    _samp = "greedy" if args.sampling == "greedy" else f"speculative sampling T={args.temperature} (seed={args.seed})"
    print(f"[exp2] sampling={_samp}" + ("" if args.sampling == "greedy" else
          "  [AR and SSD are independent samples of the same distribution -> match_rate is uninformative]"))

    prompt_format = resolve_prompt_format(args.model, args.prompt_format)
    print(f"[exp2] prompt_format={prompt_format}" + (
        "" if prompt_format == "raw" else "  (chat template, single user turn, enable_thinking=True)"))
    if is_chat_model(args.model) and prompt_format == "raw":
        print(f"[exp2] WARNING: {args.model} is a chat/thinking model but --prompt-format raw was requested.")

    # Encode all prompts up front so encode time isn't charged to either baseline.
    encoded: list[tuple[torch.Tensor, str]] = []
    n_trunc = 0
    for text, cat in prompts:
        ids = wrapper.encode(format_prompt(wrapper, text, prompt_format))
        if ids.shape[1] > args.max_prompt_tokens:
            ids = ids[:, : args.max_prompt_tokens]
            n_trunc += 1
        if ids.shape[1] < 2:
            continue
        encoded.append((ids, cat))
    print(f"[exp2] {len(encoded)} prompts after length filtering")
    if n_trunc and prompt_format == "chat":
        print(f"[exp2] WARNING: {n_trunc} chat-formatted prompts were right-truncated to "
              f"--max-prompt-tokens {args.max_prompt_tokens}, which drops the "
              f"'<|im_start|>assistant\\n<think>' tail — raise --max-prompt-tokens.")

    all_results: dict = {
        "config": vars(args) | {"T_drafts": T_drafts, "prompt_format_resolved": prompt_format},
        "per_T_draft": {},
    }

    for T_draft in T_drafts:
        policy = SsdPolicy(T_total=args.T_total, T_draft=T_draft, mode=args.sampling,
                           temperature=args.temperature, sample_seed=args.seed)
        _sched = f"{args.scheduler}" + (f" (draft_length={args.draft_length})" if args.scheduler == "dtv" else "")
        print(f"\n[exp2] === T_total={args.T_total}  T_draft={T_draft}  scheduler={_sched} ===")

        per_prompt: list[BenchmarkMetrics] = []
        per_category: dict[str, list[BenchmarkMetrics]] = {}
        t0 = time.perf_counter()
        for i, (ids, cat) in enumerate(encoded):
            metrics = benchmark_one(
                blocks, ids, policy,
                max_new_tokens=args.max_new_tokens,
                n_warmup=args.warmup,
                wrapper=wrapper,
                eos_token_id=eos_token_id,
                scheduler=args.scheduler,
                draft_length=args.draft_length,
            )
            per_prompt.append(metrics)
            per_category.setdefault(cat, []).append(metrics)
            if (i + 1) % max(1, len(encoded) // 10) == 0:
                # GPU mem trend: alloc = live tensors (rising => real leak),
                # resv = reserved pool (rising with flat alloc => allocator
                # fragmentation/high-water, not a leak). See exp2 memory notes.
                if args.device == "cuda":
                    _al = torch.cuda.memory_allocated() / 1e9
                    _rv = torch.cuda.memory_reserved() / 1e9
                    _mem = f"  mem={_al:.1f}/{_rv:.1f}G(alloc/resv)"
                else:
                    _mem = ""
                print(
                    f"  [{i+1}/{len(encoded)}] cat={cat:8s}  "
                    f"acc={metrics.acceptance_rate:.3f}  "
                    f"match={metrics.match_rate:.3f}  "
                    f"speedup={metrics.speedup:.2f}x  "
                    f"steady={metrics.speedup_steady:.2f}x" + _mem
                )

        elapsed = time.perf_counter() - t0
        agg = aggregate(per_prompt)
        agg_by_cat = {cat: aggregate(ms) for cat, ms in per_category.items()}
        print(
            f"[exp2] T_draft={T_draft} done in {elapsed:.1f}s  |  "
            f"speedup={agg['speedup']:.2f}x  "
            f"speedup_steady={agg['speedup_steady']:.2f}x  "
            f"acc={agg['acceptance_rate']:.3f}  "
            f"match={agg['match_rate']:.3f}"
        )

        all_results["per_T_draft"][str(T_draft)] = {
            "elapsed_s": elapsed,
            "aggregate": agg,
            "by_category": agg_by_cat,
            "per_prompt": [m.to_dict() for m in per_prompt],
        }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[exp2] wrote {out_path}")


if __name__ == "__main__":
    main()
