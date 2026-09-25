"""Attention backend layer for our P/R/C decode code (torch-SDPA / FA4 / SDPA-ref).

This module is the *only* place that talks to an attention kernel. Our
re-written parcae/Ouro forwards (`parcae_blocks.py` / `ouro_blocks.py`) compute
Q/K/V from the model's projection weights and then call the helpers here; the
AR baseline and the DtV scheduler run on the same block primitives, so all
three modes (WFD / AR / DtV) share whatever backend is selected here.

Three attention shapes occur in wavefront SSD, all served by FA4:

  * dense causal          — prefill / P prelude (Tq == Tk, standard causal).
  * single-/full-prefix   — decode-time Tq=1 attending to a slot's whole prefix
                            (expressed as causal=True; FA aligns the lone query
                            to the last key so it sees everything <= its pos).
  * disjoint multi-query  — C-fusing: nq queries at disjoint absolute positions
                            sharing one coda slot, each attending keys whose
                            absolute position <= the query's. Expressed with a
                            FlexAttention-style `mask_mod` (`n_idx <= qpos[m]`),
                            relying on the StaticKVCache invariant that a key's
                            absolute position equals its buffer index.
  * varlen-batched R      — the n active tokens (each at its own step slot,
                            different K) batched into one varlen call.

Three backends, switch with `set_backend("torch" | "fa4" | "sdpa")`:

  * "torch" (DEFAULT since 2026-07-13) — torch SDPA routed through fused
    kernels: Tq=1 -> no-mask SDPA-flash; Tq==Tk causal -> is_causal; Tq<Tk
    causal -> `causal_lower_right` bias (bottom-right, flash-eligible); the
    disjoint multi-query coda -> ZERO-PAD CAUSAL trick (one query row per
    position in [min(qpos), Tk-1], bottom-right causal, pad rows discarded —
    identical semantics to the per-query cutoff mask but flash-eligible);
    varlen -> `aten._flash_attention_forward` (the exact kernel behind SDPA's
    flash backend; per-sequence SDPA loop fallback if the private op changes).
    On A6000 (sm_86) this matches or beats FA4 on every decode shape — see
    .claude/plans/environment_and_troubleshooting.md.
  * "fa4"  — FlashAttention-4 (CuTeDSL) fused kernels (B200-optimal).
  * "sdpa" — slow explicit-mask / loop reference twins (parity ground truth).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

# Ensure the vendored parcae repo is importable (the wrapper does this too, but
# this module may be imported first, e.g. in isolated tests).
_PARCAE_REPO_DIR = Path(__file__).resolve().parents[2] / "external" / "parcae"
if _PARCAE_REPO_DIR.exists() and str(_PARCAE_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_PARCAE_REPO_DIR))

from parcae_lm.modules.mixer import norm as _qk_norm
from parcae_lm.modules.utils import apply_rotary_emb_complex_like


# ---------------------------------------------------------------------------
# Backend availability + selection
# ---------------------------------------------------------------------------
def _detect_fa4():
    """Return the flash_attn.cute module if FA4 is importable on a supported
    arch, else None.

    FA4 (this CuTeDSL build) ships forward kernels for Ampere (sm_8x,
    `FlashAttentionForwardSm80`), Hopper (sm_90), Blackwell (sm_10x/sm_11x), and
    sm_12x — `interface.py` asserts `arch//10 in {8,9,10,11,12}` and routes
    arch 8.x to the Sm80 kernel with mask_mod support. Our decode path uses only
    dense-causal / varlen / mask_mod, all implemented by the Sm80 kernel (only a
    USER-provided score_mod is blocked on sm_8x, which we never pass). So FA4 is
    enabled on Ampere here.

    ⚠️ sm_8x / sm_12x are functional but upstream-tagged "should tune" and less
    battle-tested than Hopper/Blackwell — on a NEW card verify FA4-vs-SDPA parity
    and speed (compare via `set_backend("sdpa")`), and fall back with
    `set_backend("sdpa")` if a kernel misbehaves (this import-guard only catches
    import failures, not runtime kernel errors)."""
    if not torch.cuda.is_available():
        return None
    major = torch.cuda.get_device_capability()[0]
    if major not in (8, 9, 10, 11, 12):
        return None
    try:
        import flash_attn.cute as fc  # noqa
        return fc
    except Exception:
        return None


_FA4 = _detect_fa4()
HAS_FA4 = _FA4 is not None
# Default backend: "torch" (SDPA routed through fused kernels). On A6000 (sm_86)
# it matches/beats FA4 on every decode shape; FA4 stays available via set_backend.
_BACKEND = "torch"

# The CuTeDSL mask_mod is built lazily (importing cutlass at module load is
# heavy and only needed for the multi-query coda path).
_CODA_MASK_MOD = None


def _unwrap(out):
    """FA4 entry points may return (out, lse, ...) tuples; we only want out."""
    return out[0] if isinstance(out, (tuple, list)) else out


def set_backend(name: str) -> None:
    """'torch' (default), 'fa4' (CuTeDSL, when available), or 'sdpa' (slow
    reference twins). Used by parity tests and backend comparisons."""
    global _BACKEND
    assert name in ("torch", "fa4", "sdpa")
    if name == "fa4" and not HAS_FA4:
        raise RuntimeError("FA4 backend requested but flash_attn.cute is unavailable.")
    _BACKEND = name


def backend() -> str:
    return _BACKEND


# ---------------------------------------------------------------------------
# Q/K/V projection — faithful copy of parcae CausalSelfAttention.forward
# (external/parcae/parcae_lm/modules/mixer.py:60-80). We reproduce it instead
# of calling block.attn so the kernel afterwards is *ours* (FA4), while the AR
# baseline keeps calling the native block.attn (SDPA).
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_qkv(attn, x, freqs, ve, config):
    """x: (B, T, C) -> q (B, T, H, Dh), k/v (B, T, Hkv, Dh). Mirrors parcae."""
    B, T, _ = x.shape
    H, Hkv, Dh = attn.n_head, attn.n_kv_head, attn.head_dim
    q = attn.c_q(x).view(B, T, H, Dh)
    k = attn.c_k(x).view(B, T, Hkv, Dh)
    v = attn.c_v(x).view(B, T, Hkv, Dh)
    if ve is not None and attn.ve_gate is not None:
        ve_v = ve.view(B, T, Hkv, Dh)
        gate = 2 * torch.sigmoid(attn.ve_gate(x[..., : attn.ve_gate_channels]))
        v = v + gate.unsqueeze(-1) * ve_v
    if config.clip_qkv is not None:
        c = config.clip_qkv
        q = q.clamp(min=-c, max=c)
        k = k.clamp(min=-c, max=c)
        v = v.clamp(min=-c, max=c)
    if config.qk_bias:
        q_bias, k_bias = attn.qk_bias.split(1, dim=0)
        q = (q + q_bias).to(q.dtype)
        k = (k + k_bias).to(q.dtype)
    if config.rope_settings.use_rope:
        q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs)
    if config.qk_norm:
        q, k = _qk_norm(q), _qk_norm(k)
    return q, k, v


# ---------------------------------------------------------------------------
# CuTeDSL mask_mod for the disjoint multi-query coda.
# Template: tests/cute/mask_mod_definitions.py::cute_global_causal_window_mask
# (indexes a per-query aux tensor by the query index inside the kernel).
# Key abs-position == FA n_idx because StaticKVCache uses position==buffer-index
# and we pass K = slot[0 : max_query_pos+1].
# ---------------------------------------------------------------------------
def _build_coda_mask_mod():
    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def coda_cutoff_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
        # aux_tensors[0]: qpos (Tq,) int32 — absolute position of each query.
        # keep key n_idx for query m_idx iff key_abs_pos(n_idx) <= query_abs_pos.
        qpos = aux_tensors[0]
        m_frag = cute.make_rmem_tensor(1, cutlass.Int32)
        m_frag.store(m_idx)
        qp_frag = cute.make_rmem_tensor(1, cutlass.Int32)
        qp_frag[0] = qpos[m_frag[0]]
        return n_idx <= qp_frag.load()

    return coda_cutoff_mask


def _coda_mask_mod():
    global _CODA_MASK_MOD
    if _CODA_MASK_MOD is None:
        _CODA_MASK_MOD = _build_coda_mask_mod()
    return _CODA_MASK_MOD


# ---------------------------------------------------------------------------
# Dense / single-query  (Tq==Tk causal, or Tq=1 full-prefix via causal=True)
# q: (B, Tq, H, Dh)   k,v: (B, Tk, Hkv, Dh)   ->  (B, Tq, H, Dh)
# ---------------------------------------------------------------------------
def attn_dense(q, k, v, *, causal: bool):
    if _BACKEND == "fa4":
        return _unwrap(_FA4.flash_attn_func(q, k, v, causal=causal))
    if _BACKEND == "torch":
        return _attn_dense_torch(q, k, v, causal=causal)
    return _attn_dense_sdpa(q, k, v, causal=causal)


def _attn_dense_torch(q, k, v, *, causal: bool):
    """SDPA routed to fused kernels. Tq=1 needs no mask at all (a lone
    bottom-right-causal query attends the whole prefix); Tq==Tk uses is_causal;
    Tq<Tk uses the lower-right CausalBias (torch's plain is_causal is top-left
    aligned = wrong for decode, and an explicit mask would disable flash)."""
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.shape[2] != k.shape[2]
    Tq, Tk = q.shape[1], k.shape[1]
    if not causal or Tq == 1:
        out = F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=enable_gqa)
    elif Tq == Tk:
        out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, enable_gqa=enable_gqa)
    else:
        out = F.scaled_dot_product_attention(
            qt, kt, vt, attn_mask=causal_lower_right(Tq, Tk), enable_gqa=enable_gqa
        )
    return out.transpose(1, 2)


def _attn_dense_sdpa(q, k, v, *, causal: bool):
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.shape[2] != k.shape[2]
    Tq, Tk = q.shape[1], k.shape[1]
    if causal and Tq != Tk:
        # Match FA "bottom-right" causal (query i attends key j <= i + (Tk-Tq)).
        # PyTorch is_causal aligns top-left, which is wrong when Tq<Tk (decode:
        # a single query must see the whole prefix, not just key 0).
        ri = torch.arange(Tq, device=q.device).unsqueeze(1)
        cj = torch.arange(Tk, device=q.device).unsqueeze(0)
        mask = cj <= (ri + (Tk - Tq))
        out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask[None, None], enable_gqa=enable_gqa)
    else:
        out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal, enable_gqa=enable_gqa)
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Disjoint multi-query coda.
# q: (1, nq, H, Dh)   k,v: (1, Tk, Hkv, Dh)   qpos: (nq,) int32 abs positions
# (Tk == max(qpos)+1; key abs pos == its index).  ->  (1, nq, H, Dh)
# ---------------------------------------------------------------------------
def attn_cutoff_multi(q, k, v, qpos):
    """qpos: list[int] (preferred — avoids a device sync in the torch path) or a
    1-D tensor of the queries' absolute positions. Contract: Tk == max(qpos)+1."""
    if _BACKEND == "fa4":
        qpos_i32 = torch.as_tensor(qpos, device=q.device, dtype=torch.int32)
        return _unwrap(_FA4.flash_attn_func(q, k, v, mask_mod=_coda_mask_mod(), aux_tensors=[qpos_i32]))
    if _BACKEND == "torch":
        return _attn_cutoff_multi_torch(q, k, v, qpos)
    return _attn_cutoff_multi_sdpa(q, k, v, qpos)


def _attn_cutoff_multi_torch(q, k, v, qpos):
    """Zero-pad causal trick. Place each query at the row of its absolute
    position within [pmin, Tk-1]; a bottom-right causal block over kv[:Tk] then
    lets row j attend exactly the keys <= pmin+j — the per-query cutoff — with
    the flash kernel (an explicit bool mask would force the slow backends).
    Pad rows are zero queries whose outputs are discarded (attention rows are
    independent); no pad K/V is written anywhere. Contiguous ascending
    positions (e.g. the DtV verify chain) need no padding at all."""
    pos = qpos if isinstance(qpos, list) else qpos.tolist()
    B, nq, H, Dh = q.shape
    Tk = k.shape[1]
    assert Tk == max(pos) + 1, "attn_cutoff_multi expects K sliced to max(qpos)+1"
    pmin = min(pos)
    span = Tk - pmin                       # rows for positions pmin .. Tk-1
    rows = [p - pmin for p in pos]
    contiguous = span == nq and rows == list(range(nq))
    if contiguous:
        qpad = q
    else:
        qpad = q.new_zeros(B, span, H, Dh)
        qpad[:, rows] = q
    qt, kt, vt = qpad.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.shape[2] != k.shape[2]
    out = F.scaled_dot_product_attention(
        qt, kt, vt, attn_mask=causal_lower_right(span, Tk), enable_gqa=enable_gqa
    )
    out = out.transpose(1, 2)
    return out if contiguous else out[:, rows]


def _attn_cutoff_multi_sdpa(q, k, v, qpos):
    Tk = k.shape[1]
    kpos = torch.arange(Tk, device=q.device)              # key abs pos == index
    qpos_t = torch.as_tensor(qpos, device=q.device)
    mask = kpos.unsqueeze(0) <= qpos_t.to(kpos.dtype).unsqueeze(1)   # (nq, Tk)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    enable_gqa = q.shape[2] != k.shape[2]
    out = F.scaled_dot_product_attention(
        qt, kt, vt, attn_mask=mask[None, None], enable_gqa=enable_gqa
    )
    return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Varlen-batched R: n queries (1 token each) packed; each attends its own slot's
# pre-sliced K/V (positions <= its pos), so causal=False over each sub-seq.
# q: (total_q=n, H, Dh)   k,v: (total_k, Hkv, Dh)
# cu_q: (n+1,) int32     cu_k: (n+1,) int32
# returns (n, H, Dh)
# ---------------------------------------------------------------------------
def attn_varlen(q, k, v, cu_q, cu_k, max_q, max_k, causal: bool = False):
    """causal=False: every query row attends its whole segment (the original
    per-token R pack, q_len==1 per segment). causal=True: bottom-right causal
    PER SEGMENT (FA2/FA4 varlen convention) — row r of a (q_len, k_len) segment
    attends keys <= k_len - q_len + r. Used by the grouped R pack, where tokens
    sharing one KV slot become one multi-row segment whose rows sit at their
    absolute positions (zero-padded rows for gaps, outputs discarded)."""
    if _BACKEND == "fa4":
        return _unwrap(_FA4.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=max_q, max_seqlen_k=max_k,
            causal=causal,
        ))
    if _BACKEND == "torch":
        return _attn_varlen_torch(q, k, v, cu_q, cu_k, max_q, max_k, causal)
    return _attn_varlen_sdpa(q, k, v, cu_q, cu_k, causal)


def attn_varlen_q1_padded(q, k, v, cu_k, max_k):
    """n single-query segments, each padded to q_len=2 (row duplicated; row 1
    kept) and issued as bottom-right-causal varlen — numerically the same
    attention as the q_len=1 causal=False pack (row 1 attends its whole
    segment), but it dodges TWO template downgrades that hit single-query
    varlen reads: flash resets is_causal when max_seqlen_q==1, and the sm86/89
    hdim128 NON-causal config is (kBlockM,kBlockN)=(128,32) vs causal (64,64).
    The kernel is per-k-block-iteration latency-bound at decode occupancy
    (grid = batch x heads CTAs, no split-KV in the varlen entry), so halving
    the iteration count halves the read time: 165 -> 332 GB/s measured at
    L=64k on A6000 (grouped causal reads run 357 GB/s). This aligns the
    per-byte read cost of AR / per-token-WFD / grouped reads for the exp4
    long-context sweep (kernel notes 2026-09-15; exp4_plan.md).

    q: (n, H, Dh), one row per segment; k/v/cu_k/max_k exactly as the q_len=1
    per-token pack. Row 0 of each segment attends k_len-1 keys and is
    discarded (k_len >= 2 assumed — decode positions follow a non-empty
    prefix)."""
    n = q.shape[0]
    q2 = torch.repeat_interleave(q, 2, dim=0)
    cu_q2 = torch.arange(0, 2 * n + 1, 2, dtype=torch.int32, device=q.device)
    out = attn_varlen(q2, k, v, cu_q2, cu_k, 2, max_k, causal=True)
    return out[1::2]


_ATEN_VARLEN_BROKEN = False


def _attn_varlen_torch(q, k, v, cu_q, cu_k, max_q, max_k, causal: bool = False):
    """torch-internal FA2 varlen kernel (`aten._flash_attention_forward` — the
    exact kernel SDPA's flash backend runs) called directly with the packed
    layout our call sites already build. FA2 varlen causal is bottom-right per
    segment. It is a private op: if a torch upgrade changes its signature, warn
    once and fall back to the per-sequence SDPA loop (same math, slower)."""
    global _ATEN_VARLEN_BROKEN
    if not _ATEN_VARLEN_BROKEN:
        try:
            return torch.ops.aten._flash_attention_forward(
                q, k, v, cu_q, cu_k, max_q, max_k, 0.0, causal, False
            )[0]
        except Exception as ex:  # signature drift / kernel unavailable
            _ATEN_VARLEN_BROKEN = True
            warnings.warn(
                f"aten._flash_attention_forward unavailable ({type(ex).__name__}: {ex}); "
                "falling back to the per-sequence SDPA loop for varlen attention."
            )
    return _attn_varlen_sdpa(q, k, v, cu_q, cu_k, causal)


def _attn_varlen_sdpa(q, k, v, cu_q, cu_k, causal: bool = False):
    """Reference: loop each sub-sequence through the dense SDPA twin (which
    builds the bottom-right mask explicitly when causal and q_len < k_len)."""
    n = cu_q.numel() - 1
    cu_q_l = cu_q.tolist()
    cu_k_l = cu_k.tolist()
    outs = []
    for i in range(n):
        qi = q[cu_q_l[i] : cu_q_l[i + 1]].unsqueeze(0)   # (1, sq, H, Dh)
        ki = k[cu_k_l[i] : cu_k_l[i + 1]].unsqueeze(0)   # (1, sk, Hkv, Dh)
        vi = v[cu_k_l[i] : cu_k_l[i + 1]].unsqueeze(0)
        outs.append(_attn_dense_sdpa(qi, ki, vi, causal=causal)[0])   # (sq, H, Dh)
    return torch.cat(outs, dim=0)
