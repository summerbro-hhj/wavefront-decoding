"""exp4 runner — acceptance-emulation × batch × KV-sharing × prefill sweep (AR / WFD / DtV).

For every (kv_budget_s, prefill_len, batch_size): run the batched AR baseline
once (α-independent), then for every α run WFD (each --wfd-T-drafts) and DtV
(each --dtv-draft-lengths) with per-sequence Bernoulli(α) accept streams. The same
per-sequence seeds are reused across schedulers (fair random tape). Dummy
random-token prompts; no EOS; every sequence commits exactly --decode-len
tokens. See .claude/plans/exp4_plan.md.

Pick a FREE gpu first (nvidia-smi), e.g.:
  CUDA_VISIBLE_DEVICES=2 bash experiments/exp4_emu/run.sh
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.decoding.emu import EmulatedAcceptance  # noqa: E402
from src.decoding.huginn_blocks_batch import HuginnBlocksBatch  # noqa: E402
from src.decoding.ouro_blocks_batch import OuroBlocksBatch  # noqa: E402
from src.decoding.policy import SsdPolicy  # noqa: E402
from src.decoding.scheduler_batch import (  # noqa: E402
    ar_generate_batch, generate_dtv_batch, generate_wavefront_batch, prefill_once)
from src.recursive_models.huginn_wrapper import HuginnWrapper  # noqa: E402
from src.recursive_models.ouro_wrapper import OuroWrapper  # noqa: E402

# name -> (wrapper cls, HF repo id, batch blocks cls)
MODELS = {
    "ouro-1.4b": (OuroWrapper, "ByteDance/Ouro-1.4B", OuroBlocksBatch),
    "ouro-2.6b": (OuroWrapper, "ByteDance/Ouro-2.6B", OuroBlocksBatch),
    # RDM 3.5B — natural depth mean_recurrence=32 (set --T-total accordingly;
    # sharing shrinks slots 2+4T+2 -> 2+4s+2, e.g. T=32: 132 -> 8 at s=1).
    "huginn-3.5b": (HuginnWrapper, "tomg-group-umd/huginn-0125", HuginnBlocksBatch),
}


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def record(bt, ar_tps: float | None, ar_decode_tps: float | None, **kw) -> dict:
    return dict(
        **kw,
        tokens=bt.n_tokens,
        walltime_s=round(bt.walltime_s, 3),
        walltime_decode=round(bt.walltime_decode_s, 3),
        tokens_per_s=round(bt.tokens_per_s, 2),
        speedup_vs_ar=(round(bt.tokens_per_s / ar_tps, 3) if ar_tps else None),
        speedup_decode_vs_ar=(round(bt.decode_tokens_per_s / ar_decode_tps, 3)
                              if ar_decode_tps else None),
        acceptance=(round(bt.acceptance_rate, 4) if bt.acceptance_rate == bt.acceptance_rate else None),
        n_r_calls=bt.n_r_calls, n_c_calls=bt.n_c_calls, n_p_calls=bt.n_p_calls,
        r_union_mean=round(bt.r_union_mean, 2), r_union_max=bt.r_union_max,
        cache_gb=round(bt.cache_gb, 2),
        peak_alloc_gb=round(bt.peak_alloc_gb, 2),
        peak_reserved_gb=round(bt.peak_reserved_gb, 2),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ouro-2.6b", choices=list(MODELS))
    p.add_argument("--schedulers", default="ar,wfd,dtv")
    p.add_argument("--alphas", default="0.6,0.8,0.9,0.95,1.0")
    p.add_argument("--batch-sizes", default="1,4,8")
    p.add_argument("--kv-budget-s", default="0", help="comma list; 0 = sharing off")
    p.add_argument("--T-total", type=int, default=4, help="ouro 4, huginn up to 32")
    p.add_argument("--state-init", default="random", choices=["random", "zero"],
                   help="Initial recurrent state s0 (huginn only; ouro has none).")
    p.add_argument("--wfd-T-drafts", default="1,2")
    p.add_argument("--dtv-T-draft", type=int, default=1)
    p.add_argument("--dtv-draft-lengths", default="4,8")
    p.add_argument("--prefill-len", default="1024",
                   help="comma list of prompt lengths (sweep dim, like batch/kv-budget)")
    p.add_argument("--decode-len", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=0,
                   help="0 = per-prefill prefill+decode+margin; nonzero = fixed override")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--skip-prefill", action="store_true",
                   help="Prefill once per (s, prefill, B) cell and share it across "
                        "AR/WFD/DtV instead of re-prefilling per run. Every run "
                        "reports that one measured prefill as its prefill_s, so "
                        "walltime_s stays comparable while only the decode loop "
                        "is actually re-executed. Sound: prefill writes positions "
                        "[0, L) and decode only writes >= L.")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    scheds = [s.strip() for s in args.schedulers.split(",")]
    alphas = parse_floats(args.alphas)
    batches = parse_ints(args.batch_sizes)
    kv_list = parse_ints(args.kv_budget_s)
    prefills = parse_ints(args.prefill_len)
    wfd_tds = parse_ints(args.wfd_T_drafts)
    dtv_dls = parse_ints(args.dtv_draft_lengths)
    # WFD's leading draft edge runs up to T_total positions ahead of the last
    # commit (DtV: draft_length+1) — reserve that margin beyond prefill+decode.
    margin = max(args.T_total, (max(dtv_dls) if dtv_dls else 0) + 2) + 8

    torch.manual_seed(args.seed)
    wrapper_cls, model_id, batch_cls = MODELS[args.model]
    wrapper = wrapper_cls.from_pretrained(model_id, dtype=torch.bfloat16, device="cuda")
    model = wrapper.model
    vocab = model.config.vocab_size
    print(f"[exp4] {args.model} T_total={args.T_total} prefill={prefills} "
          f"decode={args.decode_len} margin={margin} schedulers={scheds}")

    def emus_for(alpha: float, B: int):
        # same per-sequence seeds across schedulers -> fair random tape
        return [EmulatedAcceptance(alpha, seed=args.seed * 100003 + 1000 + b) for b in range(B)]

    results: list[dict] = []
    for s_budget in kv_list:
      for P in prefills:
        for B in batches:
            max_seq = args.max_seq_len or (P + args.decode_len + margin)
            base = dict(kv_budget_s=s_budget, prefill_len=P, batch_size=B, max_seq=max_seq)
            blocks = batch_cls(model, T_total=args.T_total, batch_size=B,
                               max_seq_len=max_seq,
                               kv_budget_s=(s_budget or None))
            if hasattr(blocks, "state_init"):
                blocks.state_init = args.state_init
                blocks.state_seed = args.seed
            est = blocks.cache_nbytes_estimate() / 1e9
            print(f"\n[exp4] === s={s_budget or 'off'}  p={P}  B={B}  (KV cache ≈ {est:.1f} GB) ===")
            g = torch.Generator().manual_seed(args.seed)
            prompts = torch.randint(0, vocab, (B, P), generator=g).to("cuda")
            pf = None
            try:
                if args.warmup:
                    ar_generate_batch(blocks, prompts, min(8, args.decode_len))
                if args.skip_prefill:
                    # After the warmup, so the one measured prefill is as warm as
                    # the per-run prefill it replaces. Every run below borrows it.
                    pf = prefill_once(blocks, prompts)
                    print(f"  [shared prefill {pf.prefill_s:.2f}s — reused by "
                          f"{'/'.join(scheds)}]")
                ar_tps = ar_decode_tps = None
                if "ar" in scheds:
                    bt = ar_generate_batch(blocks, prompts, args.decode_len, pf)
                    ar_tps, ar_decode_tps = bt.tokens_per_s, bt.decode_tokens_per_s
                    r = record(bt, None, None, scheduler="ar", alpha=None, **base)
                    results.append(r)
                    print(f"  ar                      | {bt.tokens_per_s:8.1f} tok/s  "
                          f"({bt.decode_tokens_per_s:.1f} decode-only)  "
                          f"r_union={bt.r_union_mean:.1f}  peak={bt.peak_reserved_gb:.1f}G")
                for alpha in alphas:
                    if "wfd" in scheds:
                        for td in wfd_tds:
                            bt = generate_wavefront_batch(
                                blocks, prompts, args.decode_len,
                                SsdPolicy(T_total=args.T_total, T_draft=td),
                                emus_for(alpha, B), pf)
                            r = record(bt, ar_tps, ar_decode_tps, scheduler="wfd",
                                       alpha=alpha, T_draft=td, **base)
                            results.append(r)
                            print(f"  wfd  a={alpha:4.2f} Td={td}     | {bt.tokens_per_s:8.1f} tok/s  "
                                  f"x{r['speedup_vs_ar'] or float('nan'):5.2f}  "
                                  f"(decode x{r['speedup_decode_vs_ar'] or float('nan'):5.2f})  "
                                  f"acc={bt.acceptance_rate:.3f}  r_union={bt.r_union_mean:.1f}")
                    if "dtv" in scheds:
                        for dl in dtv_dls:
                            bt = generate_dtv_batch(
                                blocks, prompts, args.decode_len,
                                SsdPolicy(T_total=args.T_total, T_draft=args.dtv_T_draft),
                                dl, emus_for(alpha, B), pf)
                            r = record(bt, ar_tps, ar_decode_tps, scheduler="dtv",
                                       alpha=alpha, draft_length=dl,
                                       T_draft=args.dtv_T_draft, **base)
                            results.append(r)
                            print(f"  dtv  a={alpha:4.2f} dl={dl}     | {bt.tokens_per_s:8.1f} tok/s  "
                                  f"x{r['speedup_vs_ar'] or float('nan'):5.2f}  "
                                  f"(decode x{r['speedup_decode_vs_ar'] or float('nan'):5.2f})  "
                                  f"acc={bt.acceptance_rate:.3f}  r_union={bt.r_union_mean:.1f}")
            except torch.OutOfMemoryError:
                print(f"  OOM at s={s_budget or 'off'} p={P} B={B} — "
                      f"skipping larger batches for this (s, prefill)")
                results.append(dict(scheduler="OOM", alpha=None, **base))
                del blocks, pf
                gc.collect()
                torch.cuda.empty_cache()
                break
            del blocks, pf          # pf holds the shared KV cache — drop it first
            gc.collect()
            torch.cuda.empty_cache()

    ptag = "-".join(str(x) for x in prefills)
    out = args.output or f"results/exp4/{args.model}_p{ptag}d{args.decode_len}_{time.strftime('%m%d_%H%M')}.json"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(config=vars(args) | dict(margin=margin), results=results)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[exp4] wrote {out_path}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
