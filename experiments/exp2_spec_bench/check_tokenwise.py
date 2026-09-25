"""Verify that parcae's core_block layers are bit-identical token-wise.

If norm_1 / mlp / norm_2 / adapter are all element-wise on the last dim, then
processing (x, y) as a (1, 2, D) batch and processing them as two separate
(1, 1, D) calls should produce identical results. If not, we've found the bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.recursive_models.parcae_wrapper import ParcaeWrapper


@torch.no_grad()
def main():
    wrapper = ParcaeWrapper.from_pretrained(
        model_id="SandyResearch/parcae-770m",
        dtype=torch.bfloat16,
        device="cuda",
        state_init="zero",
    )
    model = wrapper.model
    device = next(model.parameters()).device
    D = model.config.n_embd
    g = torch.Generator(device=device).manual_seed(0)
    x_pos10 = torch.randn(1, 1, D, dtype=torch.bfloat16, device=device, generator=g)
    x_pos11 = torch.randn(1, 1, D, dtype=torch.bfloat16, device=device, generator=g)
    in_emb10 = torch.randn(1, 1, D, dtype=torch.bfloat16, device=device, generator=g)
    in_emb11 = torch.randn(1, 1, D, dtype=torch.bfloat16, device=device, generator=g)

    # 1. adapter test
    x_single10 = model.transformer.adapter(x_pos10, in_emb10)
    x_single11 = model.transformer.adapter(x_pos11, in_emb11)
    x_mixed = model.transformer.adapter(
        torch.cat([x_pos10, x_pos11], dim=1),
        torch.cat([in_emb10, in_emb11], dim=1),
    )
    d0 = (x_single10 - x_mixed[:, :1, :]).abs().max().item()
    d1 = (x_single11 - x_mixed[:, 1:, :]).abs().max().item()
    print(f"adapter token-wise diff:  token0={d0:.4e}  token1={d1:.4e}")

    # 2. norm_1 (block 0) test
    block = model.transformer.core_block[0]
    n_single10 = block.norm_1(x_pos10)
    n_single11 = block.norm_1(x_pos11)
    n_mixed = block.norm_1(torch.cat([x_pos10, x_pos11], dim=1))
    d0 = (n_single10 - n_mixed[:, :1, :]).abs().max().item()
    d1 = (n_single11 - n_mixed[:, 1:, :]).abs().max().item()
    print(f"norm_1 token-wise diff:   token0={d0:.4e}  token1={d1:.4e}")

    # 3. mlp test
    m_single10 = block.mlp(x_pos10)
    m_single11 = block.mlp(x_pos11)
    m_mixed = block.mlp(torch.cat([x_pos10, x_pos11], dim=1))
    d0 = (m_single10 - m_mixed[:, :1, :]).abs().max().item()
    d1 = (m_single11 - m_mixed[:, 1:, :]).abs().max().item()
    print(f"mlp token-wise diff:      token0={d0:.4e}  token1={d1:.4e}")

    # 4. norm_2 test
    n2_single10 = block.norm_2(x_pos10)
    n2_single11 = block.norm_2(x_pos11)
    n2_mixed = block.norm_2(torch.cat([x_pos10, x_pos11], dim=1))
    d0 = (n2_single10 - n2_mixed[:, :1, :]).abs().max().item()
    d1 = (n2_single11 - n2_mixed[:, 1:, :]).abs().max().item()
    print(f"norm_2 token-wise diff:   token0={d0:.4e}  token1={d1:.4e}")


if __name__ == "__main__":
    main()
