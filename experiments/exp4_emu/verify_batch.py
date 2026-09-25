"""exp4 Step-2/3/4 gates (exp4_plan.md §4-2..4): batch blocks + schedulers.

  A. Replica symmetry — B identical sequences (same emu seed) produce identical
     per-sequence commits/counters (WFD and AR).
  B. B=1 batch == B=1 `scheduler_emu` on the old single-seq blocks: per-seq
     counters exactly equal at α=1.0 AND α=0.8 (same Bernoulli stream ->
     identical decision path; token ids may differ across implementations).
  C. Cross-sequence isolation — B=2 with α=(0.0, 1.0): the α=1.0 sequence's
     counters equal its solo run; rejects of seq0 never touch seq1.
  D. KV-sharing — s=1 vs off: identical counters, cache memory shrinks by the
     model's slot ratio (ouro: 1/T_total; huginn: (2+4s+2)/(2+4T+2)).
  E. AR sanity — R calls per token == T_total.
  F. Batched prefill parity — B replicas' last-prompt logits vs the single-seq
     blocks' prefill (huginn single-seq prefill is the NATIVE forward, so this
     cross-checks the hand-rolled batched prefill).

Supports --model ouro-2.6b (T_total=4) and huginn-3.5b (T_total=8 for gates;
natural depth 32 — gate semantics are depth-agnostic).

Run on a FREE gpu:
  CUDA_VISIBLE_DEVICES=<free> python experiments/exp4_emu/verify_batch.py --model huginn-3.5b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.decoding.emu import EmulatedAcceptance  # noqa: E402
from src.decoding.policy import SsdPolicy  # noqa: E402
from src.decoding.scheduler_batch import (  # noqa: E402
    ar_generate_batch, generate_dtv_batch, generate_wavefront_batch)
from src.decoding.scheduler_emu import generate_dtv_emu, generate_wavefront_emu  # noqa: E402

L, N, MAX_SEQ = 128, 40, 512
FAILED: list[str] = []


def counters(tr) -> dict:
    return dict(n_tok=len(tr.tokens), r=tr.n_r_calls, c=tr.n_c_calls, p=tr.n_p_calls,
                prop=tr.n_drafts_proposed, acc=tr.n_drafts_accepted,
                rej=tr.n_drafts_rejected, rb=tr.n_rollback_events)


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILED.append(name)
    print(f"  {'OK ' if ok else 'FAIL'} {name}  {detail}")


def load(model_name: str):
    if model_name.startswith("ouro"):
        from src.decoding.ouro_blocks import OuroBlocks
        from src.decoding.ouro_blocks_batch import OuroBlocksBatch
        from src.recursive_models.ouro_wrapper import OuroWrapper
        rid = {"ouro-2.6b": "ByteDance/Ouro-2.6B", "ouro-1.4b": "ByteDance/Ouro-1.4B"}[model_name]
        wrapper = OuroWrapper.from_pretrained(rid, dtype=torch.bfloat16, device="cuda")
        return wrapper.model, OuroBlocks, OuroBlocksBatch, 4
    if model_name == "huginn-3.5b":
        from src.decoding.huginn_blocks import HuginnBlocks
        from src.decoding.huginn_blocks_batch import HuginnBlocksBatch
        from src.recursive_models.huginn_wrapper import HuginnWrapper
        wrapper = HuginnWrapper.from_pretrained(dtype=torch.bfloat16, device="cuda")
        return wrapper.model, HuginnBlocks, HuginnBlocksBatch, 8
    raise ValueError(model_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ouro-2.6b",
                    choices=["ouro-1.4b", "ouro-2.6b", "huginn-3.5b"])
    args = ap.parse_args()

    torch.manual_seed(0)
    model, old_cls, batch_cls, T = load(args.model)
    vocab = model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    base = torch.randint(0, vocab, (1, L), generator=g).to("cuda")
    print(f"model={args.model} T_total={T} (gates)")

    def batch_blocks(B, s=None):
        return batch_cls(model, T_total=T, batch_size=B, max_seq_len=MAX_SEQ, kv_budget_s=s)

    # ---- A. replica symmetry ----------------------------------------------
    print("\n== A. replica symmetry (B=4 identical seqs) ==")
    B = 4
    prompts = base.repeat(B, 1)
    emus = [EmulatedAcceptance(1.0, seed=11) for _ in range(B)]
    bt = generate_wavefront_batch(batch_blocks(B), prompts, N,
                                  SsdPolicy(T_total=T, T_draft=1), emus)
    toks = [t.tokens for t in bt.per_seq]
    check("wfd: identical commits across replicas",
          all(t == toks[0] for t in toks), f"lens={[len(t) for t in toks]}")
    cs = [counters(t) for t in bt.per_seq]
    check("wfd: identical counters across replicas", all(c == cs[0] for c in cs), str(cs[0]))

    bt = ar_generate_batch(batch_blocks(B), prompts, N)
    toks = [t.tokens for t in bt.per_seq]
    check("ar: identical commits across replicas", all(t == toks[0] for t in toks))

    # ---- B. B=1 batch == B=1 scheduler_emu (old blocks) --------------------
    print("\n== B. B=1 batch vs scheduler_emu counters ==")
    old_blocks = old_cls(model, T_total=T, max_seq_len=MAX_SEQ)
    for a in (1.0, 0.8):
        for T_draft in (1, 2):
            pol = SsdPolicy(T_total=T, T_draft=T_draft)
            _, tr_ref = generate_wavefront_emu(old_blocks, base[0], N, pol,
                                               emu=EmulatedAcceptance(a, seed=23))
            bt = generate_wavefront_batch(batch_blocks(1), base, N, pol,
                                          [EmulatedAcceptance(a, seed=23)])
            check(f"wfd α={a} T_draft={T_draft}: counters equal",
                  counters(tr_ref) == counters(bt.per_seq[0]),
                  f"ref={counters(tr_ref)} batch={counters(bt.per_seq[0])}")
        pol = SsdPolicy(T_total=T, T_draft=1)
        _, tr_ref = generate_dtv_emu(old_blocks, base[0], N, pol, draft_length=4,
                                     emu=EmulatedAcceptance(a, seed=29))
        bt = generate_dtv_batch(batch_blocks(1), base, N, pol, 4,
                                [EmulatedAcceptance(a, seed=29)])
        check(f"dtv α={a} dl=4: counters equal",
              counters(tr_ref) == counters(bt.per_seq[0]),
              f"ref={counters(tr_ref)} batch={counters(bt.per_seq[0])}")

    # ---- C. cross-sequence isolation ---------------------------------------
    print("\n== C. cross-sequence isolation (α=0.0 next to α=1.0) ==")
    pol = SsdPolicy(T_total=T, T_draft=1)
    bt_solo = generate_wavefront_batch(batch_blocks(1), base, N, pol,
                                       [EmulatedAcceptance(1.0, seed=31)])
    bt_mix = generate_wavefront_batch(batch_blocks(2), base.repeat(2, 1), N, pol,
                                      [EmulatedAcceptance(0.0, seed=99),
                                       EmulatedAcceptance(1.0, seed=31)])
    check("seq1 (α=1.0) counters unaffected by seq0's rejects",
          counters(bt_solo.per_seq[0]) == counters(bt_mix.per_seq[1]),
          f"solo={counters(bt_solo.per_seq[0])} mix={counters(bt_mix.per_seq[1])}")
    tr0 = bt_mix.per_seq[0]
    check("seq0 (α=0.0): all proposals rejected",
          tr0.n_drafts_proposed > 0 and tr0.n_drafts_accepted == 0
          and len(tr0.tokens) == N)

    # ---- D. KV-sharing ------------------------------------------------------
    print("\n== D. KV-sharing (s=1 vs off) ==")
    pol = SsdPolicy(T_total=T, T_draft=1)
    emus2 = lambda: [EmulatedAcceptance(1.0, seed=41), EmulatedAcceptance(1.0, seed=42)]
    bt_off = generate_wavefront_batch(batch_blocks(2, s=None), base.repeat(2, 1), N, pol, emus2())
    bt_s1 = generate_wavefront_batch(batch_blocks(2, s=1), base.repeat(2, 1), N, pol, emus2())
    check("counters identical with sharing on",
          [counters(t) for t in bt_off.per_seq] == [counters(t) for t in bt_s1.per_seq])
    exp_ratio = (batch_blocks(2, s=1).cache_nbytes_estimate()
                 / batch_blocks(2, s=None).cache_nbytes_estimate())
    ratio = bt_s1.cache_gb / bt_off.cache_gb
    check(f"cache memory ratio ≈ {exp_ratio:.3f} (got {ratio:.3f})",
          abs(ratio - exp_ratio) < 0.01,
          f"off={bt_off.cache_gb:.2f}GB s1={bt_s1.cache_gb:.2f}GB")

    # ---- E. AR sanity --------------------------------------------------------
    print("\n== E. AR sanity ==")
    bt = ar_generate_batch(batch_blocks(1), base, N)
    tr = bt.per_seq[0]
    check(f"AR: r_calls == (N-1)*T_total = {(N-1)*T}",
          tr.n_r_calls == (N - 1) * T, f"got {tr.n_r_calls}")
    check(f"AR: commits == {N}", len(tr.tokens) == N)

    # ---- F. batched prefill parity vs single-seq blocks ---------------------
    print("\n== F. batched prefill parity ==")
    cache_b, logits_b = batch_blocks(2).prefill(base.repeat(2, 1))
    _, logits_s = old_blocks.prefill(base, T, 1)
    rel = (logits_b[0] - logits_s).abs().max().item() / logits_s.abs().max().item()
    same_argmax = int(logits_b[0].argmax()) == int(logits_s.argmax())
    check(f"batch prefill ≈ single-seq prefill (rel={rel:.3e}, argmax_eq={same_argmax})",
          rel < 5e-2)
    del cache_b

    print(f"\n{'BATCH GATES PASS' if not FAILED else 'BATCH GATES FAIL: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
