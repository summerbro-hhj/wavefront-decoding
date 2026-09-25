"""StaticCache-based KV cache for wavefront SSD (FA4-friendly).

This replaces the earlier dict-based `SortedKVCache`. Motivation: FA4 (like
FA3) wants contiguous `(B, S, H, D)` K/V tensors, whereas the dict cache stored
`{step_idx: {position: vec}}` and re-stacked on every read — exactly the
mismatch flagged in `.claude/plans/fa4_backend_plan.md` §4(c).

Design — one preallocated buffer per parcae `step_idx` slot, with the invariant

        buffer index  ==  absolute sequence position.

Slot step_idx layout matches parcae's `forward_for_generation`:
    prelude layer i            -> step_idx = i
    core recurrence r layer l  -> step_idx = n_prelude + r*n_core + l
    coda layer i               -> step_idx = n_prelude + T_total*n_core + i

Every absolute position passes through a given slot at most once and in
increasing order (prefill writes 0..L-1; generation appends; rollback
truncates the tail; C(verify) may *overwrite* a C(draft) entry at the same
position). So a slot's valid region is always the contiguous prefix
`[0, seqlen[step_idx])`, which a static buffer + a length counter captures
exactly — no sorting, no windowing, no future-position leakage.

Two interfaces over the same buffers:
  * `update(k, v, step_idx)` — parcae-compatible (uses `_seen_tokens`, returns
    the `(B, Hkv, S, D)` prefix). The native parcae `block.attn` SDPA path
    (AR baseline) calls this unchanged.
  * `write_range`/`write_scatter` + `gather` — contiguous `(1, S, Hkv, D)` for
    our FA4 blocks.
  * `gather_merged` — built but UNUSED hook for a future R KV-sharing (Step 3)
    experiment (merges several step slots into one K/V for true cross-step
    multi-query). The current lossless R never calls it.
"""
from __future__ import annotations

import torch


class StaticKVCache:
    """Per-step-slot static KV cache; position == buffer index."""

    def __init__(self, n_kv_head: int, head_dim: int, max_seq_len: int,
                 dtype: torch.dtype, device: torch.device):
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        self.device = device
        # lazily-allocated per-slot buffers (1, max_seq_len, n_kv_head, head_dim)
        self.K: dict[int, torch.Tensor] = {}
        self.V: dict[int, torch.Tensor] = {}
        self.seqlen: dict[int, int] = {}
        self._seen_tokens = 0   # parcae-compat counter (bumped on step_idx==0)

    # ------------------------------------------------------------------
    def _ensure_slot(self, step_idx: int) -> None:
        if step_idx not in self.K:
            shape = (1, self.max_seq_len, self.n_kv_head, self.head_dim)
            self.K[step_idx] = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.V[step_idx] = torch.zeros(shape, dtype=self.dtype, device=self.device)
            self.seqlen[step_idx] = 0

    def _bump_seqlen(self, step_idx: int, end: int) -> None:
        if end > self.seqlen[step_idx]:
            if end > self.max_seq_len:
                raise IndexError(
                    f"StaticKVCache slot {step_idx}: position {end-1} exceeds "
                    f"max_seq_len={self.max_seq_len}. Increase max_seq_len."
                )
            self.seqlen[step_idx] = end

    # ------------------------------------------------------------------
    # parcae-compatible interface (native SDPA path / AR baseline)
    # ------------------------------------------------------------------
    def update(self, key_states, value_states, step_idx_tensor, lookup_strategy=None):
        """key_states / value_states: (B, Hkv, T_new, D). Returns the
        causal prefix (B, Hkv, S, D). Positions derived from `_seen_tokens`,
        matching parcae's ParcaeDynamicCache semantics."""
        step_idx = int(step_idx_tensor)
        T_new = key_states.shape[-2]
        if step_idx == 0:
            self._seen_tokens += T_new
        base = self._seen_tokens - T_new
        self._ensure_slot(step_idx)
        # store as (1, T, Hkv, D)
        self.K[step_idx][:, base : base + T_new] = key_states.transpose(1, 2)
        self.V[step_idx][:, base : base + T_new] = value_states.transpose(1, 2)
        self._bump_seqlen(step_idx, base + T_new)
        S = self.seqlen[step_idx]
        # return (B, Hkv, S, D)
        return (self.K[step_idx][:, :S].transpose(1, 2),
                self.V[step_idx][:, :S].transpose(1, 2))

    # ------------------------------------------------------------------
    # FA4-friendly interface (our re-written parcae blocks)
    # ------------------------------------------------------------------
    def write_range(self, step_idx: int, base_pos: int, k, v) -> None:
        """Write a contiguous run. k, v: (1, T, Hkv, D) -> positions
        [base_pos, base_pos+T). Used by prefill (T=L) and append (T=1)."""
        self._ensure_slot(step_idx)
        T = k.shape[1]
        self.K[step_idx][:, base_pos : base_pos + T] = k
        self.V[step_idx][:, base_pos : base_pos + T] = v
        self._bump_seqlen(step_idx, base_pos + T)

    def write_scatter(self, step_idx: int, positions: list[int], k, v) -> None:
        """Write nq queries at disjoint absolute positions. k, v: (1, nq, Hkv, D).
        Used by the multi-query coda (verify overwrites an earlier draft slot,
        draft appends)."""
        self._ensure_slot(step_idx)
        for i, p in enumerate(positions):
            self.K[step_idx][:, p : p + 1] = k[:, i : i + 1]
            self.V[step_idx][:, p : p + 1] = v[:, i : i + 1]
            self._bump_seqlen(step_idx, p + 1)

    def gather(self, step_idx: int, upto_pos: int):
        """Return contiguous (1, upto_pos+1, Hkv, D) K/V — positions [0, upto_pos].
        Key abs-position == its index, which the FA4 coda mask_mod relies on."""
        self._ensure_slot(step_idx)
        S = upto_pos + 1
        return self.K[step_idx][:, :S], self.V[step_idx][:, :S]

    def freeze_replicate(self, src_slot: int, dst_slots: list[int], pos: int) -> None:
        """Copy position `pos`'s K/V from `src_slot` into each of `dst_slots`
        (adaptive-total-T early-exit KV pad). When a token halts at depth d it has
        only written its shallow slots [0..d-1]; later wavefront tokens that DO
        reach a deeper depth e would otherwise find this position's slot-e KV
        empty. Replicating the last-computed depth's K/V into the skipped deeper
        slots fills that gap — justified by the early-exit premise that the
        representation has converged (deep K/V ≈ the depth it stopped at), so this
        is a *consistent* lossy approximation rather than an arbitrary fill.
        Bumps each dst slot's seqlen so the padded position joins its prefix."""
        self._ensure_slot(src_slot)
        k = self.K[src_slot][:, pos : pos + 1]
        v = self.V[src_slot][:, pos : pos + 1]
        for d in dst_slots:
            self._ensure_slot(d)
            self.K[d][:, pos : pos + 1] = k
            self.V[d][:, pos : pos + 1] = v
            self._bump_seqlen(d, pos + 1)

    def gather_merged(self, step_indices: list[int], upto_pos: int):
        """UNUSED future hook (Step 3 R KV-sharing). Concatenate several step
        slots' prefixes [0, upto_pos] into one (1, sum_S, Hkv, D) K/V for a
        true cross-step multi-query attention. The current lossless R does NOT
        call this; kept so the KV-sharing experiment needs no cache change."""
        Ks, Vs = [], []
        for s in step_indices:
            self._ensure_slot(s)
            S = upto_pos + 1
            Ks.append(self.K[s][:, :S])
            Vs.append(self.V[s][:, :S])
        return torch.cat(Ks, dim=1), torch.cat(Vs, dim=1)

    # ------------------------------------------------------------------
    # rollback
    # ------------------------------------------------------------------
    def drop_positions_at_and_after(self, position: int) -> None:
        """Invalidate every position >= `position` in every slot (reject /
        rollback). With position==index this is just truncating each slot's
        length; stale buffer entries are overwritten on the next write."""
        for s in list(self.seqlen.keys()):
            if self.seqlen[s] > position:
                self.seqlen[s] = position
        self._seen_tokens = position

    def reset(self) -> None:
        self.K.clear()
        self.V.clear()
        self.seqlen.clear()
        self._seen_tokens = 0

    def get_seq_length(self) -> int:
        return self._seen_tokens


def create_static_parcae_cache(model, T_total: int, max_seq_len: int) -> StaticKVCache:
    """Build a StaticKVCache sized for a parcae model at depth T_total."""
    cfg = model.config
    n_kv_head = cfg.num_key_value_heads
    head_dim = cfg.n_embd // cfg.num_attention_heads
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    return StaticKVCache(n_kv_head, head_dim, max_seq_len, dtype, device)
