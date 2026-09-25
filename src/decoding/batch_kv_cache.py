"""Batched static KV cache for exp4 (fixed batch, per-sequence slots).

One preallocated pair of tensors

    K, V : (n_slots, B, max_seq_len, Hkv, Dh)

with the same `position == buffer index` invariant as `StaticKVCache`, extended
by a batch dim. All slots are allocated up front (Ouro's R writes every slot),
so construction fail-fasts on OOM and `nbytes` reports the real footprint —
this is where KV-sharing (fewer depth slots) buys batch size.

Union operations are vectorized (no per-token python cat):
  * `write_tokens`  — scatter n tokens at (slot, b, pos) triples in one
    index_copy_ on the flattened buffer.
  * `gather_pack`   — pack n variable-length prefixes [(slot, b, 0..upto)] into
    contiguous (total, Hkv, Dh) K/V + int32 cu_seqlens for the varlen kernel,
    via repeat_interleave/arange index math + one index_select (2-3 small
    kernels + the unavoidable copy; no host loop over tokens).

Rollback is per-sequence: `drop_positions_at_and_after(b, pos)` only truncates
sequence b's logical lengths — other sequences' buffers are untouched.
"""
from __future__ import annotations

import torch


class BatchStaticKVCache:
    def __init__(self, n_slots: int, batch_size: int, n_kv_head: int, head_dim: int,
                 max_seq_len: int, dtype: torch.dtype, device: torch.device):
        self.n_slots = n_slots
        self.B = batch_size
        self.S = max_seq_len
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.device = device
        shape = (n_slots, batch_size, max_seq_len, n_kv_head, head_dim)
        self.K = torch.zeros(shape, dtype=dtype, device=device)
        self.V = torch.zeros(shape, dtype=dtype, device=device)
        self._flatK = self.K.view(-1, n_kv_head, head_dim)   # (n_slots*B*S, Hkv, Dh)
        self._flatV = self.V.view(-1, n_kv_head, head_dim)
        # logical per-(slot, seq) lengths — bookkeeping / asserts only (reads are
        # always [0 .. caller-provided upto], mirroring StaticKVCache).
        self.seqlen = [[0] * batch_size for _ in range(n_slots)]

    @property
    def nbytes(self) -> int:
        return (self.K.numel() + self.V.numel()) * self.K.element_size()

    # ------------------------------------------------------------------
    def write_prefill(self, slot: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v: (B, L, Hkv, Dh) — prompt K/V for ALL sequences at positions [0, L)."""
        L = k.shape[1]
        assert L <= self.S
        self.K[slot][:, :L] = k
        self.V[slot][:, :L] = v
        row = self.seqlen[slot]
        for b in range(self.B):
            row[b] = max(row[b], L)

    def write_tokens(self, slots: list[int], seq_ids: list[int], positions: list[int],
                     k: torch.Tensor, v: torch.Tensor) -> None:
        """Scatter n tokens: k, v (n, Hkv, Dh) -> (slots[i], seq_ids[i], positions[i])."""
        flat = [(sl * self.B + b) * self.S + p
                for sl, b, p in zip(slots, seq_ids, positions)]
        idx = torch.tensor(flat, dtype=torch.long, device=self.device)
        self._flatK.index_copy_(0, idx, k)
        self._flatV.index_copy_(0, idx, v)
        for sl, b, p in zip(slots, seq_ids, positions):
            if p >= self.S:
                raise IndexError(f"position {p} exceeds max_seq_len={self.S}")
            if p + 1 > self.seqlen[sl][b]:
                self.seqlen[sl][b] = p + 1

    def gather_pack(self, slots: list[int], seq_ids: list[int], uptos: list[int]):
        """Pack n prefixes [(slot, b, 0..upto)] -> (k_pack, v_pack, cu_k, max_k).
        k_pack, v_pack: (sum(upto_i+1), Hkv, Dh) contiguous; cu_k: (n+1,) int32."""
        lengths = [u + 1 for u in uptos]
        starts = [(sl * self.B + b) * self.S
                  for sl, b in zip(slots, seq_ids)]
        cu_host = [0]
        for ln in lengths:
            cu_host.append(cu_host[-1] + ln)
        total = cu_host[-1]

        len_t = torch.tensor(lengths, dtype=torch.long, device=self.device)
        start_t = torch.tensor(starts, dtype=torch.long, device=self.device)
        cu_prev = torch.tensor(cu_host[:-1], dtype=torch.long, device=self.device)
        # flat index of element j of segment i = starts[i] + (j - cu[i])
        offs = torch.repeat_interleave(start_t - cu_prev, len_t)
        flat = offs + torch.arange(total, dtype=torch.long, device=self.device)

        k_pack = self._flatK.index_select(0, flat)
        v_pack = self._flatV.index_select(0, flat)
        cu_k = torch.tensor(cu_host, dtype=torch.int32, device=self.device)
        return k_pack, v_pack, cu_k, max(lengths)

    # ------------------------------------------------------------------
    def drop_positions_at_and_after(self, b: int, position: int) -> None:
        """Per-sequence rollback: truncate sequence b's logical lengths. Buffers
        are not zeroed (stale data beyond the new length is never gathered — the
        same argument as StaticKVCache); other sequences untouched."""
        for sl in range(self.n_slots):
            if self.seqlen[sl][b] > position:
                self.seqlen[sl][b] = position
