"""exp4 Step-1 gates (exp4_plan.md §4-1): B=1 emulation semantics.

  1. emu=None  -> `scheduler_emu` functions are EXACTLY the originals
     (same tokens, same counters) — validates the copied scheduler code.
  2. α=1.0     -> zero rejects, exactly max_new_tokens commits.
  3. α∈{0.5,0.8} -> empirical acceptance converges to α.

Run on a FREE gpu:
  CUDA_VISIBLE_DEVICES=<free> python experiments/exp4_emu/verify_step1.py --model ouro-2.6b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.decoding.emu import EmulatedAcceptance  # noqa: E402
from src.decoding.ouro_blocks import OuroBlocks  # noqa: E402
from src.decoding.policy import SsdPolicy  # noqa: E402
from src.decoding.scheduler import generate_draft_then_verify, generate_wavefront_dynamic  # noqa: E402
from src.decoding.scheduler_emu import generate_dtv_emu, generate_wavefront_emu  # noqa: E402
from src.recursive_models.ouro_wrapper import OuroWrapper  # noqa: E402

MODELS = {"ouro-1.4b": "ByteDance/Ouro-1.4B", "ouro-2.6b": "ByteDance/Ouro-2.6B"}
FAILED: list[str] = []


def counters(tr) -> dict:
    return dict(n_tok=len(tr.tokens), r=tr.n_r_calls, c=tr.n_c_calls, p=tr.n_p_calls,
                prop=tr.n_drafts_proposed, acc=tr.n_drafts_accepted,
                rej=tr.n_drafts_rejected, rb=tr.n_rollback_events)


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILED.append(name)
    print(f"  {'OK ' if ok else 'FAIL'} {name}  {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ouro-2.6b", choices=list(MODELS))
    ap.add_argument("--prefill-len", type=int, default=256)
    args = ap.parse_args()

    torch.manual_seed(0)
    wrapper = OuroWrapper.from_pretrained(MODELS[args.model], dtype=torch.bfloat16, device="cuda")
    T_total = 4
    blocks = OuroBlocks(wrapper.model, T_total=T_total)
    vocab = wrapper.model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, vocab, (args.prefill_len,), generator=g).to("cuda")

    # ---- Gate 1: emu=None ≡ original (tokens + counters exact) -----------
    print("\n== Gate 1: emu=None equivalence ==")
    for T_draft in (1, 2):
        pol = SsdPolicy(T_total=T_total, T_draft=T_draft)
        tok0, tr0 = generate_wavefront_dynamic(blocks, prompt, 48, pol)
        tok1, tr1 = generate_wavefront_emu(blocks, prompt, 48, pol, emu=None)
        check(f"wfd T_draft={T_draft} emu=None == original",
              tok0 == tok1 and counters(tr0) == counters(tr1),
              f"{counters(tr0)} vs {counters(tr1)}" if counters(tr0) != counters(tr1) else "")
    pol = SsdPolicy(T_total=T_total, T_draft=1)
    tok0, tr0 = generate_draft_then_verify(blocks, prompt, 48, pol, draft_length=4)
    tok1, tr1 = generate_dtv_emu(blocks, prompt, 48, pol, draft_length=4, emu=None)
    check("dtv dl=4 emu=None == original",
          tok0 == tok1 and counters(tr0) == counters(tr1),
          f"{counters(tr0)} vs {counters(tr1)}" if counters(tr0) != counters(tr1) else "")

    # ---- Gate 2: α=1.0 — no rejects, exact commit count ------------------
    print("\n== Gate 2: α=1.0 ==")
    pol = SsdPolicy(T_total=T_total, T_draft=1)
    _, tr = generate_wavefront_emu(blocks, prompt, 48, pol, emu=EmulatedAcceptance(1.0, 0))
    check("wfd α=1.0: rejects=0, commits=48",
          tr.n_drafts_rejected == 0 and len(tr.tokens) == 48, str(counters(tr)))
    _, tr = generate_dtv_emu(blocks, prompt, 48, pol, draft_length=4, emu=EmulatedAcceptance(1.0, 0))
    check("dtv α=1.0: rejects=0, commits=48",
          tr.n_drafts_rejected == 0 and len(tr.tokens) == 48, str(counters(tr)))

    # ---- Gate 3: empirical acceptance -> α --------------------------------
    print("\n== Gate 3: empirical acceptance ==")
    for a in (0.5, 0.8):
        emu = EmulatedAcceptance(a, seed=7)
        _, tr = generate_wavefront_emu(blocks, prompt, 384, SsdPolicy(T_total=T_total, T_draft=1), emu=emu)
        check(f"wfd α={a}: empirical={emu.empirical_rate:.3f} (draws={emu.n_draws})",
              abs(emu.empirical_rate - a) < 0.06)
        emu = EmulatedAcceptance(a, seed=7)
        _, tr = generate_dtv_emu(blocks, prompt, 384, SsdPolicy(T_total=T_total, T_draft=1),
                                 draft_length=4, emu=emu)
        check(f"dtv α={a}: empirical={emu.empirical_rate:.3f} (draws={emu.n_draws})",
              abs(emu.empirical_rate - a) < 0.06)

    print(f"\n{'STEP1 PASS' if not FAILED else 'STEP1 FAIL: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
