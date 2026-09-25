"""exp5 runner — steady-state decode speed: eager vs CUDA-graph (AR/WFD/DtV).

Forced schedules (no accept/reject, no batch scheduling; alpha=1 equivalent,
B=1) on a dummy random prompt, decode-only timing. See
.claude/plans/exp5_plan.md. Pick a FREE GPU first (nvidia-smi):

    CUDA_VISIBLE_DEVICES=<free> bash experiments/exp5_graph/run.sh
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

from src.decoding.acceptance import greedy_sample  # noqa: E402
from src.decoding.forced_schedule import (  # noqa: E402
    forced_ar_eager, forced_ar_graph, forced_dtv_eager, forced_dtv_graph,
    forced_wfd_eager, forced_wfd_graph)
from src.decoding.graph_blocks import HuginnGraphAdapter, OuroGraphAdapter  # noqa: E402
from src.decoding.huginn_blocks import HuginnBlocks  # noqa: E402
from src.decoding.ouro_blocks import OuroBlocks  # noqa: E402
from src.recursive_models import OuroWrapper  # noqa: E402
from src.recursive_models.huginn_wrapper import HuginnWrapper  # noqa: E402

MODELS = {
    "ouro-2.6b": (OuroWrapper, "ByteDance/Ouro-2.6B", OuroBlocks, OuroGraphAdapter),
    "ouro-1.4b": (OuroWrapper, "ByteDance/Ouro-1.4B", OuroBlocks, OuroGraphAdapter),
    "huginn-3.5b": (HuginnWrapper, "tomg-group-umd/huginn-0125", HuginnBlocks,
                    HuginnGraphAdapter),
}


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="ouro-2.6b", choices=list(MODELS))
    p.add_argument("--modes", default="ar,wfd,dtv")
    p.add_argument("--engines", default="eager,graph")
    p.add_argument("--prefill-len", type=int, default=1024)
    p.add_argument("--decode-len", type=int, default=512)
    p.add_argument("--T-total", type=int, default=None, help="default: ouro 4, huginn 32")
    p.add_argument("--wfd-T-draft", type=int, default=None, help="default: ouro 1, huginn 4")
    p.add_argument("--dtv-T-draft", type=int, default=None, help="default: same as wfd")
    p.add_argument("--dtv-gamma", type=int, default=8)
    p.add_argument("--kv-budget-s", default="0", help="comma list; 0 = sharing off")
    p.add_argument("--state-init", default="random", choices=["random", "zero"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    name = args.model
    is_huginn = name.startswith("huginn")
    T_total = args.T_total or (32 if is_huginn else 4)
    wfd_td = args.wfd_T_draft or (4 if is_huginn else 1)
    dtv_td = args.dtv_T_draft or wfd_td
    modes = [m.strip() for m in args.modes.split(",")]
    engines = [e.strip() for e in args.engines.split(",")]
    kv_list = parse_ints(args.kv_budget_s)
    P, D = args.prefill_len, args.decode_len
    max_seq = P + D + 96

    torch.manual_seed(args.seed)
    wrapper_cls, model_id, blocks_cls, adapter_cls = MODELS[name]
    print(f"[exp5] loading {name} ({model_id}) ...", flush=True)
    wrapper = wrapper_cls.from_pretrained(model_id, dtype=torch.bfloat16, device="cuda")
    model = wrapper.model
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(args.seed)
    prompt = torch.randint(0, vocab, (1, P), generator=g).to("cuda")
    print(f"[exp5] {name} T={T_total} wfd_Td={wfd_td} dtv_Td={dtv_td} gamma={args.dtv_gamma} "
          f"prefill={P} decode={D} max_seq={max_seq} s={kv_list} "
          f"modes={modes} engines={engines}", flush=True)

    results = []
    for s_budget in kv_list:
        blocks = blocks_cls(model, T_total=T_total, max_seq_len=max_seq)
        blocks.kv_budget_s = s_budget or None
        if hasattr(blocks, "state_init"):
            blocks.state_init = args.state_init
            blocks.state_seed = args.seed
        ar_tps = {e: None for e in engines}
        for engine in engines:
            for mode in modes:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                state, last_logits = blocks.prefill(prompt, T_total, T_total)
                torch.cuda.synchronize()
                t_prefill = time.perf_counter() - t0
                first_id = greedy_sample(last_logits)

                adapter = adapter_cls(blocks, state.kv_cache) if engine == "graph" else None
                try:
                    if mode == "ar":
                        if engine == "eager":
                            r = forced_ar_eager(blocks, state, first_id, D, T_total)
                        else:
                            r = forced_ar_graph(adapter, state, first_id, D, T_total)
                        r["r_steps_per_token"] = T_total
                    elif mode == "wfd":
                        if engine == "eager":
                            r = forced_wfd_eager(blocks, state, first_id, D, T_total, wfd_td)
                        else:
                            grouped = blocks._sharing()
                            r = forced_wfd_graph(adapter, blocks, state, first_id, D,
                                                 T_total, wfd_td, grouped=grouped)
                        r["r_steps_per_token"] = wfd_td
                    elif mode == "dtv":
                        if engine == "eager":
                            r = forced_dtv_eager(blocks, state, first_id, D,
                                                 T_total, dtv_td, args.dtv_gamma)
                        else:
                            r = forced_dtv_graph(adapter, state, first_id, D,
                                                 T_total, dtv_td, args.dtv_gamma)
                        gp1 = args.dtv_gamma + 1
                        r["r_steps_per_token"] = (gp1 * dtv_td + (T_total - dtv_td)) / gp1
                    else:
                        raise ValueError(mode)
                except torch.OutOfMemoryError:
                    print(f"  OOM: s={s_budget} {engine}/{mode}", flush=True)
                    results.append(dict(model=name, kv_budget_s=s_budget,
                                        engine=engine, mode=mode, oom=True))
                    continue
                finally:
                    del adapter
                    gc.collect()
                    torch.cuda.empty_cache()

                rec = dict(model=name, kv_budget_s=s_budget, engine=engine, mode=mode,
                           T_total=T_total,
                           T_draft=(wfd_td if mode == "wfd" else dtv_td if mode == "dtv" else None),
                           gamma=(args.dtv_gamma if mode == "dtv" else None),
                           prefill_len=P, decode_len=D,
                           prefill_s=round(t_prefill, 3),
                           **{k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in r.items()})
                rec["ms_per_token"] = round(1000.0 * r["decode_s"] / r["n_tokens"], 3)
                if mode == "ar":
                    ar_tps[engine] = r["tok_s"]
                rec["speedup_vs_ar"] = (round(r["tok_s"] / ar_tps[engine], 3)
                                        if ar_tps[engine] else None)
                results.append(rec)
                extra = (f"  capture={r.get('capture_s', 0):.1f}s x{r.get('n_graphs', 0)}"
                         if engine == "graph" else "")
                print(f"  s={s_budget or 'off'} {engine:5s} {mode:3s} | "
                      f"{r['tok_s']:8.2f} tok/s  ({rec['ms_per_token']:7.2f} ms/tok)  "
                      f"x{rec['speedup_vs_ar'] or float('nan'):5.2f}{extra}", flush=True)
        del blocks
        gc.collect()
        torch.cuda.empty_cache()

    out = args.output or (f"results/exp5/{name}_p{P}d{D}_{time.strftime('%m%d_%H%M')}.json")
    out_path = ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(config=vars(args) | dict(
        T_total=T_total, wfd_T_draft=wfd_td, dtv_T_draft=dtv_td, max_seq=max_seq),
        results=results), indent=2))
    print(f"[exp5] wrote {out_path}  ({len(results)} runs)")


if __name__ == "__main__":
    main()
