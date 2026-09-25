"""exp5 verification gates (.claude/plans/exp5_plan.md §4).

Gate 1 (Step 0) — varlen length-as-data under CUDA graph:
    Can `aten::_flash_attention_forward` be captured with cu_seqlens as DEVICE
    tensors so that ONE graph serves a GROWING KV length (content of cu_k
    updated between replays, grid sized for max_seq at capture)? This is the
    load-bearing mechanism for every exp5 graph (AR/WFD/DtV). Two shapes:
      (a) Tq=1, causal=False, full-prefix   — AR / lossless-WFD per-token segment
      (b) Tq=nq, causal=True (bottom-right) — grouped segment (KV-sharing wave,
                                              DtV same-step verify)
    PASS = replay output matches an eager SDPA reference recomputed at each
    length (bf16 tolerance), AND a cross-length check proves the new length
    actually took effect (not a stale capture).

Later gates (2-4: forced-schedule fidelity, graph parity, commit invariants)
are appended below as they are implemented.

Run on an EMPTY GPU:  CUDA_VISIBLE_DEVICES=<free> python verify_graph.py
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

TOL = 3e-2          # bf16 flash-vs-sdpa kernel disagreement bound (random-normal data)
DISTINGUISH = 1e-3  # outputs at different lengths must differ at least this much


def _capture(fn):
    """Warm up on a side stream (cublas/allocator), then capture fn() once."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            out = fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out


def gate1_varlen_length_as_data() -> bool:
    dev = torch.device("cuda")
    torch.manual_seed(0)
    H, Hkv, Dh, MAX = 16, 16, 128, 2048
    K = torch.randn(MAX, Hkv, Dh, device=dev, dtype=torch.bfloat16)
    V = torch.randn(MAX, Hkv, Dh, device=dev, dtype=torch.bfloat16)
    ok = True

    # ---- (a) Tq=1 full-prefix (causal=False over a pre-cut segment) ----
    q = torch.randn(1, H, Dh, device=dev, dtype=torch.bfloat16)
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device=dev)
    cu_k = torch.tensor([0, 600], dtype=torch.int32, device=dev)

    def fa1():
        return torch.ops.aten._flash_attention_forward(
            q, K, V, cu_q, cu_k, 1, MAX, 0.0, False, False)[0]

    def ref1(n: int):
        qt = q.view(1, 1, H, Dh).transpose(1, 2)
        kt = K[:n].unsqueeze(0).transpose(1, 2)
        vt = V[:n].unsqueeze(0).transpose(1, 2)
        o = F.scaled_dot_product_attention(qt, kt, vt)
        return o.transpose(1, 2).reshape(1, H, Dh)

    d0 = (fa1() - ref1(600)).abs().max().item()
    print(f"[gate1a] eager varlen(cu_k=600, max_k={MAX}) vs sdpa@600: max|d|={d0:.3e}")
    ok &= d0 < TOL

    g1, out1 = _capture(fa1)
    for n in (600, 900, 1500, 2048):
        cu_k[1] = n
        g1.replay()
        torch.cuda.synchronize()
        d = (out1 - ref1(n)).abs().max().item()
        print(f"[gate1a] replay @len={n}: max|graph - sdpa@{n}| = {d:.3e}")
        ok &= d < TOL
    cu_k[1] = 900
    g1.replay(); torch.cuda.synchronize()
    d_x = (out1 - ref1(600)).abs().max().item()
    print(f"[gate1a] cross: graph@900 vs sdpa@600 = {d_x:.3e} (must be > {DISTINGUISH})")
    ok &= d_x > DISTINGUISH

    # ---- (b) multi-row bottom-right causal segment (grouped-R semantics) ----
    nq = 8
    q2 = torch.randn(nq, H, Dh, device=dev, dtype=torch.bfloat16)
    cu_q2 = torch.tensor([0, nq], dtype=torch.int32, device=dev)
    cu_k2 = torch.tensor([0, 600], dtype=torch.int32, device=dev)

    def fa2():
        return torch.ops.aten._flash_attention_forward(
            q2, K, V, cu_q2, cu_k2, nq, MAX, 0.0, True, False)[0]

    def ref2(n: int):
        # bottom-right causal: row r attends keys j <= n - nq + r
        qt = q2.unsqueeze(0).transpose(1, 2)
        kt = K[:n].unsqueeze(0).transpose(1, 2)
        vt = V[:n].unsqueeze(0).transpose(1, 2)
        ri = torch.arange(nq, device=q2.device).unsqueeze(1)
        cj = torch.arange(n, device=q2.device).unsqueeze(0)
        mask = cj <= (ri + (n - nq))
        o = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask[None, None])
        return o.transpose(1, 2).reshape(nq, H, Dh)

    d0 = (fa2() - ref2(600)).abs().max().item()
    print(f"[gate1b] eager causal varlen(cu_k=600) vs sdpa-mask@600: max|d|={d0:.3e}")
    ok &= d0 < TOL

    g2, out2 = _capture(fa2)
    for n in (600, 900, 1500, 2048):
        cu_k2[1] = n
        g2.replay()
        torch.cuda.synchronize()
        d = (out2 - ref2(n)).abs().max().item()
        print(f"[gate1b] replay @len={n}: max|graph - sdpa@{n}| = {d:.3e}")
        ok &= d < TOL
    cu_k2[1] = 900
    g2.replay(); torch.cuda.synchronize()
    d_x = (out2 - ref2(600)).abs().max().item()
    print(f"[gate1b] cross: graph@900 vs sdpa@600 = {d_x:.3e} (must be > {DISTINGUISH})")
    ok &= d_x > DISTINGUISH

    # ---- in-graph position bump + index_copy_ write (AR graph mechanics) ----
    pos_t = torch.tensor([600], dtype=torch.long, device=dev)
    buf = torch.zeros(1, MAX, Hkv, Dh, device=dev, dtype=torch.bfloat16)
    newk = torch.randn(1, 1, Hkv, Dh, device=dev, dtype=torch.bfloat16)

    def write_bump():
        buf.index_copy_(1, pos_t, newk)
        pos_t.add_(1)

    g3, _ = _capture(write_bump)
    pos_t.fill_(700)
    buf.zero_()
    for _ in range(3):
        g3.replay()
    torch.cuda.synchronize()
    written = [int(p) for p in (buf.abs().sum(dim=(0, 2, 3)) > 0).nonzero().flatten()]
    print(f"[gate1c] index_copy_ + in-graph pos bump wrote positions {written} "
          f"(expect [700, 701, 702]), pos_t={int(pos_t)}")
    ok &= written == [700, 701, 702] and int(pos_t) == 703

    print(f"[gate1] {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Gates 2-4 — need a real checkpoint (run on an EMPTY GPU).
#   gate2: forced-schedule fidelity (eager) vs real AR / emu-alpha=1 schedulers
#   gate3: graph-vs-eager committed-token parity (bf16 kernel-swap tolerance)
#   gate4: commit-count invariants (DtV rounds)
# ---------------------------------------------------------------------------
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load(name: str, T_total: int, max_seq: int):
    from src.decoding.huginn_blocks import HuginnBlocks
    from src.decoding.ouro_blocks import OuroBlocks
    from src.recursive_models import OuroWrapper
    from src.recursive_models.huginn_wrapper import HuginnWrapper
    if name.startswith("huginn"):
        w = HuginnWrapper.from_pretrained("tomg-group-umd/huginn-0125",
                                          dtype=torch.bfloat16, device="cuda")
        b = HuginnBlocks(w.model, T_total=T_total, max_seq_len=max_seq)
    else:
        repo = "ByteDance/Ouro-2.6B" if "2.6" in name else "ByteDance/Ouro-1.4B"
        w = OuroWrapper.from_pretrained(repo, dtype=torch.bfloat16, device="cuda")
        b = OuroBlocks(w.model, T_total=T_total, max_seq_len=max_seq)
    return w, b


def _prefill(blocks, prompt, T_total):
    from src.decoding.acceptance import greedy_sample
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    state, last_logits = blocks.prefill(prompt, T_total, T_total)
    torch.cuda.synchronize()
    return state, greedy_sample(last_logits), time.perf_counter() - t0


def _band(label: str, got: float, ref: float, warn: float, fail: float) -> bool:
    ratio = got / ref if ref else float("nan")
    flag = "OK" if abs(ratio - 1) <= warn else ("WARN" if abs(ratio - 1) <= fail else "FAIL")
    print(f"[gate2] {label}: forced={got:8.2f} tok/s  ref={ref:8.2f} tok/s  "
          f"ratio={ratio:5.3f}  {flag}")
    return abs(ratio - 1) <= fail


def gate2_eager_fidelity(name: str, T_total: int, td: int, gamma: int,
                         prefill_len: int, decode: int) -> bool:
    from src.decoding.emu import EmulatedAcceptance
    from src.decoding.forced_schedule import (forced_ar_eager, forced_dtv_eager,
                                              forced_wfd_eager)
    from src.decoding.policy import SsdPolicy
    from src.decoding.scheduler_emu import generate_dtv_emu, generate_wavefront_emu

    max_seq = prefill_len + decode + 96
    _w, blocks = _load(name, T_total, max_seq)
    vocab = blocks.model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, vocab, (1, prefill_len), generator=g).to("cuda")
    policy = SsdPolicy(T_total=T_total, T_draft=td)
    ok = True

    # AR vs blocks.ar_generate (decode-only: subtract a measured prefill)
    state, first_id, t_pf = _prefill(blocks, prompt, T_total)
    r = forced_ar_eager(blocks, state, first_id, decode, T_total, warmup=4)
    _gen, el = blocks.ar_generate(prompt.squeeze(0), max_new_tokens=decode,
                                  eos_token_id=None)
    ok &= _band("AR  vs ar_generate    ", r["tok_s"], decode / (el - t_pf), 0.05, 0.15)

    # WFD vs emu(alpha=1) wavefront
    state, first_id, t_pf = _prefill(blocks, prompt, T_total)
    r = forced_wfd_eager(blocks, state, first_id, decode, T_total, td, warmup_commits=2)
    toks, trace = generate_wavefront_emu(blocks, prompt.squeeze(0), decode, policy,
                                         emu=EmulatedAcceptance(1.0, 0), eos_token_id=None)
    ok &= _band("WFD vs emu(alpha=1)   ", r["tok_s"],
                len(toks) / (trace.walltime_s - t_pf), 0.10, 0.25)

    # DtV vs emu(alpha=1) draft-then-verify
    state, first_id, t_pf = _prefill(blocks, prompt, T_total)
    r = forced_dtv_eager(blocks, state, first_id, decode, T_total, td, gamma,
                         warmup_rounds=1)
    toks, trace = generate_dtv_emu(blocks, prompt.squeeze(0), decode, policy,
                                   draft_length=gamma,
                                   emu=EmulatedAcceptance(1.0, 0), eos_token_id=None)
    ok &= _band("DtV vs emu(alpha=1)   ", r["tok_s"],
                len(toks) / (trace.walltime_s - t_pf), 0.10, 0.25)
    print(f"[gate2] {'PASS' if ok else 'FAIL'}")
    return ok


def gate3_graph_parity(name: str, T_total: int, td: int, gamma: int,
                       prefill_len: int, decode: int, kv_s: int = 0) -> bool:
    from src.decoding.forced_schedule import (forced_ar_eager, forced_ar_graph,
                                              forced_dtv_eager, forced_dtv_graph,
                                              forced_wfd_eager, forced_wfd_graph)
    from src.decoding.graph_blocks import HuginnGraphAdapter, OuroGraphAdapter

    max_seq = prefill_len + decode + 96
    _w, blocks = _load(name, T_total, max_seq)
    blocks.kv_budget_s = kv_s or None
    if kv_s:
        print(f"[gate3] KV-sharing s={kv_s} (grouped wave path)")
    adapter_cls = HuginnGraphAdapter if name.startswith("huginn") else OuroGraphAdapter
    vocab = blocks.model.config.vocab_size
    g = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, vocab, (1, prefill_len), generator=g).to("cuda")
    ok = True

    def match(a, b):
        n = min(len(a), len(b))
        return sum(x == y for x, y in zip(a[:n], b[:n])) / max(n, 1)

    # AR parity
    ids_e, ids_g = [], []
    state, first_id, _ = _prefill(blocks, prompt, T_total)
    forced_ar_eager(blocks, state, first_id, decode, T_total, warmup=0, collect=ids_e)
    state, first_id, _ = _prefill(blocks, prompt, T_total)
    forced_ar_graph(adapter_cls(blocks, state.kv_cache), state, first_id, decode,
                    T_total, warmup=0, collect=ids_g)
    m = match(ids_e, ids_g)
    print(f"[gate3] AR  eager-vs-graph token match: {m:.3f} ({decode} tok)")
    ok &= m >= 0.8

    # WFD parity (committed tokens)
    ids_e, ids_g = [], []
    state, first_id, _ = _prefill(blocks, prompt, T_total)
    forced_wfd_eager(blocks, state, first_id, decode, T_total, td,
                     warmup_commits=0, collect=ids_e)
    state, first_id, _ = _prefill(blocks, prompt, T_total)
    forced_wfd_graph(adapter_cls(blocks, state.kv_cache), blocks, state, first_id,
                     decode, T_total, td, grouped=blocks._sharing(),
                     warmup_commits=0, collect=ids_g)
    m = match(ids_e, ids_g)
    print(f"[gate3] WFD eager-vs-graph commit match: {m:.3f} ({decode} tok)")
    ok &= m >= 0.8

    # gate4: DtV round/commit invariants (both engines)
    import math
    state, first_id, _ = _prefill(blocks, prompt, T_total)
    r_e = forced_dtv_eager(blocks, state, first_id, decode, T_total, td, gamma,
                           warmup_rounds=0)
    state, first_id, _ = _prefill(blocks, prompt, T_total)
    r_g = forced_dtv_graph(adapter_cls(blocks, state.kv_cache), state, first_id,
                           decode, T_total, td, gamma, warmup_rounds=0)
    expect = math.ceil(decode / (gamma + 1)) * (gamma + 1)
    print(f"[gate4] DtV commits: eager={r_e['n_tokens']} graph={r_g['n_tokens']} "
          f"expected={expect}")
    ok &= r_e["n_tokens"] == expect == r_g["n_tokens"]
    print(f"[gate3/4] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="1", help="comma list: 1,2,3 (3 includes 4)")
    ap.add_argument("--model", default="ouro-2.6b")
    ap.add_argument("--T-total", type=int, default=None)
    ap.add_argument("--T-draft", type=int, default=None)
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--prefill-len", type=int, default=512)
    ap.add_argument("--decode-len", type=int, default=128)
    ap.add_argument("--kv-budget-s", type=int, default=0, help="gate3: sharing budget (0=off)")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a GPU (pick an EMPTY one)"
    hug = args.model.startswith("huginn")
    T = args.T_total or (32 if hug else 4)
    td = args.T_draft or (4 if hug else 1)
    gates = [g.strip() for g in args.gates.split(",")]
    passed = True
    if "1" in gates:
        passed &= gate1_varlen_length_as_data()
    if "2" in gates:
        passed &= gate2_eager_fidelity(args.model, T, td, args.gamma,
                                       args.prefill_len, args.decode_len)
    if "3" in gates:
        passed &= gate3_graph_parity(args.model, T, td, args.gamma,
                                     args.prefill_len, min(args.decode_len, 64),
                                     kv_s=args.kv_budget_s)
    sys.exit(0 if passed else 1)
