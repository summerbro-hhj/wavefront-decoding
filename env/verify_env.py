"""Environment verification for 2026_ssd_recursive.

Run inside the ssd conda env:
    /home/<user>/.conda/envs/ssd/bin/python env/verify_env.py

Designed to be GPU-portable: it reports the GPU/arch and SDPA backend support
rather than hard-asserting a specific compute capability, so it works on both
the previous GB10 (sm_121) box and the current B200 (sm_100) box. The only
hard requirements are "CUDA is available" and "the core modeling stack imports".
"""

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

print("=== Versions ===")
import transformers, accelerate, datasets, numpy, tokenizers, safetensors

print(f"torch:        {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"accelerate:   {accelerate.__version__}")
print(f"datasets:     {datasets.__version__}")
print(f"numpy:        {numpy.__version__}")
print(f"tokenizers:   {tokenizers.__version__}")
print(f"safetensors:  {safetensors.__version__}")
try:
    import wandb
    print(f"wandb:        {wandb.__version__}")
except ImportError:
    print("wandb:        (not installed — optional)")
print()

print("=== CUDA / GPU ===")
assert torch.cuda.is_available(), "CUDA not available"
arch_list = torch.cuda.get_arch_list()
cap = torch.cuda.get_device_capability(0)
sm = f"sm_{cap[0]}{cap[1]}"
print(f"cuda version:   {torch.version.cuda}")
print(f"arch list:      {arch_list}")
print(f"device:         {torch.cuda.get_device_name(0)}")
print(f"capability:     {cap}  ({sm})")
# GPU-agnostic runnability check (works for sm_80 A100, sm_90 H100, sm_100 B200,
# sm_120 RTX PRO 6000, sm_121 GB10, ...). The real requirement is that torch's
# build has SASS/PTX that can run this device: an exact arch match (native), or
# a same-major arch (e.g. sm_120 runs an sm_121 device via PTX forward-compat).
def _sm_major(a: str) -> int:
    digits = "".join(ch for ch in a.split("_", 1)[-1] if ch.isdigit())  # 'sm_90a' -> 90
    return int(digits) // 10 if digits else -1

build_sms = [a for a in arch_list if a.startswith("sm_")]
if sm in arch_list:
    print(f"  OK: {sm} is natively in torch's arch list.")
elif any(_sm_major(a) == cap[0] for a in build_sms):
    print(f"  OK: {sm} runs via PTX/forward-compat "
          f"(same-major arch present in {arch_list}).")
else:
    print(f"  WARNING: torch build {arch_list} has no arch for {sm}; "
          f"reinstall torch from a cuXXX index that targets {sm}.")
print()

# Wavefront-SSD decode attention backend (src/decoding/fa4_attention.py).
# DEFAULT = "torch" (2026-07-13): SDPA routed through fused kernels — q1/dense
# -> flash (no mask), Tq<Tk causal -> causal_lower_right bias, multi-query coda
# -> zero-pad causal trick, varlen -> aten._flash_attention_forward. FA4
# (CuTeDSL) stays available via set_backend("fa4") on arch major {8,9,10,11,12}
# (B200-optimal; on A6000 the torch backend matches/beats it on every shape).
print("=== Decode attention backend (wavefront SSD P/R/C) ===")
print('  default: "torch" — SDPA fused (flash / causal_lower_right / aten varlen).')
try:
    from torch.nn.attention.bias import causal_lower_right  # noqa: F401
    _ = torch.ops.aten._flash_attention_forward
    print("  torch-backend prerequisites OK (causal_lower_right + aten._flash_attention_forward).")
except Exception as e:
    print(f"  WARNING: torch-backend prerequisite missing ({type(e).__name__}: {e}) — "
          f"varlen falls back to the per-sequence SDPA loop (slower, same math).")
if cap[0] in (8, 9, 10, 11, 12):
    try:
        import flash_attn.cute  # noqa: F401
        print(f"  FA4 (CuTeDSL) also available via set_backend('fa4') — {sm} supported.")
    except Exception as e:
        print(f"  FA4 not importable ({type(e).__name__}) — torch default unaffected.")
else:
    print(f"  FA4 has no kernel for {sm} — torch default unaffected.")
print()

print("=== SDPA backends — is_causal path (bf16, B=2 H=8 S=512 D=64) ===")
q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
for bk in [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.MATH,
]:
    try:
        with sdpa_kernel(bk):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  {bk.name:22s}: OK  shape={tuple(out.shape)}")
    except Exception as e:
        print(f"  {bk.name:22s}: unavailable  ({type(e).__name__}: {str(e)[:80]})")
print()

# Custom (explicit bool) attention mask is the path wavefront SSD's multi-query
# C/R need (per-query causal cutoff). On the SDPA fallback (no FA) this drops to
# the MATH backend and is ~2-3x slower than is_causal (see exp2_optimize.md);
# FlashAttention-4 (CuTeDSL) on sm_100/sm_90 should serve it from a fused kernel. This probe
# shows which backends accept an explicit mask on the current box.
print("=== SDPA backends — explicit bool mask path (the FA4 / multi-query case) ===")
mask = torch.ones(2, 1, 512, 512, dtype=torch.bool, device="cuda").tril()
for bk in [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.MATH,
]:
    try:
        with sdpa_kernel(bk):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        print(f"  {bk.name:22s}: OK  shape={tuple(out.shape)}")
    except Exception as e:
        print(f"  {bk.name:22s}: unavailable  ({type(e).__name__}: {str(e)[:80]})")
print()

print("=== transformers AutoModel/AutoTokenizer import ===")
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401

print("OK")
print()
print("ALL CHECKS PASSED")
