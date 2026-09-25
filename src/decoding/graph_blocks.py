"""exp5 CUDA-graph engine — graph-safe R primitives for Ouro / Huginn.

See .claude/plans/exp5_plan.md. Design (§2.b, §3):

  * The KV length is passed to attention as DEVICE DATA: every R attention is a
    varlen call (`aten::_flash_attention_forward`) whose K/V argument is the
    slot's FULL max_seq buffer and whose cu_seqlens_k is a device int32 tensor
    updated between (or inside) replays; the grid is sized once for max_seq at
    capture (verified: experiments/exp5_graph/verify_graph.py gate 1).
  * KV writes are `index_copy_` with device position tensors (data, not shape).
  * RoPE is a precomputed table indexed in-graph by the device position tensor.
  * P (embed / prelude) and C (lm_head / coda) stay EAGER (§7-O3/O4); only R is
    captured. Graph units expose static buffers (x, pos, cu_k, [e]) that the
    forced scheduler reads/writes eagerly between replays.
  * Existing exp2/3/4 files are NOT modified; this module only reads model
    weights and the StaticKVCache buffers created by `blocks.prefill`.

Graph units per (model, mode):
    AR   — one unrolled graph: R x T_total, n=1, in-graph pos/cu_k bump.
    WFD  — T_draft phase graphs: one R-tick, n=W (wave rank 0 = newest/
           shallowest/highest position); commit-tick roll/bump is eager.
    DtV  — draft graph (R x T_draft, n=1, in-graph bump) + verify graph
           (R x (T_total-T_draft), n=gamma+1, same-step rows -> ONE causal
           grouped segment per layer, positions ascending by row).

Numerical note: the varlen kernel is the same FA2 kernel family the eager path
uses, but per-segment packing differs from the eager dense path -> bf16-level
divergence only (same class as exp2 §"bf16 drift").
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

_ATEN_FA = torch.ops.aten._flash_attention_forward


def _fa_varlen(q, K, V, cu_q, cu_k, max_q, max_k, causal):
    """q: (total_q, H, Dh); K/V: (max_seq, Hkv, Dh) full buffers; lengths live
    in cu_k's DATA. Returns (total_q, H, Dh)."""
    return _ATEN_FA(q, K, V, cu_q, cu_k, max_q, max_k, 0.0, causal, False)[0]


# ---------------------------------------------------------------------------
# Graph unit: one captured graph + the static buffers it reads/writes.
# ---------------------------------------------------------------------------
@dataclass
class GraphUnit:
    n: int
    x: torch.Tensor                    # (1, n, D) hidden IO buffer
    pos: torch.Tensor                  # (n,) int64 absolute positions
    cu_q1: torch.Tensor                # (2,) [0, 1] (per-row segments)
    cu_qn: torch.Tensor                # (2,) [0, n] (grouped segment)
    cu_k_rows: torch.Tensor            # (n, 2) int32 per-row [0, pos_i+1]
    cu_k_group: torch.Tensor           # (2,) int32 [0, max_pos+1]
    e: torch.Tensor | None = None      # (1, n, D) huginn static_state buffer
    graph: torch.cuda.CUDAGraph | None = None
    graphs: list = field(default_factory=list)   # phase graphs (WFD)
    capture_s: float = 0.0
    # Any tensor a captured graph reads by POINTER must stay referenced for the
    # graph's lifetime (graphs hold no Python refs) — e.g. the grouped-path row
    # permutation. Dropping it lets the allocator reuse the block -> garbage
    # indices -> device-side asserts on replay.
    group_order: torch.Tensor | None = None

    def set_positions(self, positions: list[int]) -> None:
        """Eager: load absolute positions (row order!) into the device buffers."""
        p = torch.as_tensor(positions, dtype=torch.long, device=self.pos.device)
        self.pos.copy_(p)
        self.cu_k_rows[:, 1].copy_((p + 1).to(torch.int32))
        self.cu_k_group[1] = int(max(positions)) + 1

    def bump_positions(self, by: int = 1) -> None:
        """Eager position advance (WFD commit tick / DtV round end)."""
        self.pos.add_(by)
        self.cu_k_rows[:, 1].add_(by)
        self.cu_k_group[1] += by


def _capture(fn, side_warmup: int = 3) -> tuple[torch.cuda.CUDAGraph, float]:
    """Warm up fn on a side stream, then capture. Returns (graph, seconds)."""
    t0 = time.perf_counter()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(side_warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    return g, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Ouro adapter
# ---------------------------------------------------------------------------
class OuroGraphAdapter:
    """Graph-safe Ouro R (one ut-step = 48 layers + final norm). P = embed,
    C = lm_head (both eager, device-id capable -> zero per-token host syncs)."""

    def __init__(self, blocks, cache):
        self.blocks = blocks
        self.cache = cache
        self.inner = blocks.inner
        self.config = blocks.config
        self.n_layers = blocks.n_layers
        self.device = blocks.device
        self.D = self.config.hidden_size
        self.max_seq = blocks.max_seq_len
        dt = next(blocks.model.parameters()).dtype
        # RoPE table for all positions (cos/sin: (max_seq, Dh))
        x_like = torch.zeros(1, 1, self.D, dtype=dt, device=self.device)
        pos_all = torch.arange(self.max_seq, device=self.device).view(1, -1)
        cos, sin = self.inner.rotary_emb(x_like, pos_all)
        self.cos_tab = cos[0].contiguous()
        self.sin_tab = sin[0].contiguous()

    # -- eager P / C (device token ids -> no .item()) --
    def p_embed(self, ids_t: torch.Tensor) -> torch.Tensor:
        """ids_t: (1, k) long on device -> (1, k, D)."""
        return self.inner.embed_tokens(ids_t)

    def c_logits(self, h: torch.Tensor) -> torch.Tensor:
        """h: (k, D) -> (k, V) float logits (eager lm_head)."""
        return self.blocks.model.lm_head(h).float()

    def prefill(self, prompt_ids: torch.Tensor, T_total: int):
        return self.blocks.prefill(prompt_ids, T_total, T_total)

    # -- unit construction --
    def make_unit(self, n: int) -> GraphUnit:
        dev, dt = self.device, next(self.blocks.model.parameters()).dtype
        return GraphUnit(
            n=n,
            x=torch.zeros(1, n, self.D, dtype=dt, device=dev),
            pos=torch.zeros(n, dtype=torch.long, device=dev),
            cu_q1=torch.tensor([0, 1], dtype=torch.int32, device=dev),
            cu_qn=torch.tensor([0, n], dtype=torch.int32, device=dev),
            cu_k_rows=torch.zeros(n, 2, dtype=torch.int32, device=dev),
            cu_k_group=torch.zeros(2, dtype=torch.int32, device=dev),
        )

    def _slot(self, step: int, layer_idx: int) -> int:
        return self.blocks._slot(step) * self.n_layers + layer_idx

    # -- graph-safe one R-step (steps baked; positions/lengths = data) --
    def r_step(self, unit: GraphUnit, x, steps: list[int], grouped: bool,
               group_order: torch.Tensor | None = None):
        """x: (1, n, D) -> (1, n, D). `steps[i]` is row i's recurrence step
        (baked into slot indices). grouped=True: all rows share one physical
        slot -> ONE bottom-right causal varlen segment (rows must be ascending
        by position; pass `group_order` to permute if needed)."""
        n = unit.n
        cos = self.cos_tab.index_select(0, unit.pos).unsqueeze(0)   # (1, n, Dh)
        sin = self.sin_tab.index_select(0, unit.pos).unsqueeze(0)
        H = self.config.num_attention_heads
        Hkv = self.config.num_key_value_heads
        for layer_idx, layer in enumerate(self.inner.layers):
            residual = x
            h = layer.input_layernorm(x)
            attn = layer.self_attn
            Dh = attn.head_dim
            q = attn.q_proj(h).view(1, n, H, Dh)
            k = attn.k_proj(h).view(1, n, Hkv, Dh)
            v = attn.v_proj(h).view(1, n, Hkv, Dh)
            q, k = _ouro_rope(q, k, cos, sin)
            # writes (all rows) BEFORE reads — same order as the eager path
            for i in range(n):
                slot = self._slot(steps[i], layer_idx)
                Kb, Vb = self.cache.K[slot][0], self.cache.V[slot][0]
                Kb.index_copy_(0, unit.pos[i : i + 1], k[0, i : i + 1])
                Vb.index_copy_(0, unit.pos[i : i + 1], v[0, i : i + 1])
            if grouped:
                slot = self._slot(steps[0], layer_idx)
                Kb, Vb = self.cache.K[slot][0], self.cache.V[slot][0]
                qp = q[0] if group_order is None else q[0].index_select(0, group_order)
                out = _fa_varlen(qp, Kb, Vb, unit.cu_qn, unit.cu_k_group,
                                 n, self.max_seq, causal=True)
                if group_order is not None:
                    out = out.index_select(0, group_order)
                out = out.unsqueeze(0)
            else:
                outs = []
                for i in range(n):
                    slot = self._slot(steps[i], layer_idx)
                    Kb, Vb = self.cache.K[slot][0], self.cache.V[slot][0]
                    outs.append(_fa_varlen(q[0, i : i + 1], Kb, Vb, unit.cu_q1,
                                           unit.cu_k_rows[i], 1, self.max_seq,
                                           causal=False))
                out = (outs[0] if n == 1 else torch.cat(outs, dim=0)).unsqueeze(0)
            attn_out = attn.o_proj(out.reshape(1, n, self.D))
            attn_out = layer.input_layernorm_2(attn_out)
            x = residual + attn_out
            # sandwich MLP half
            residual = x
            h = layer.post_attention_layernorm(x)
            h = layer.mlp(h)
            h = layer.post_attention_layernorm_2(h)
            x = residual + h
        return self.inner.norm(x)


def _ouro_rope(q, k, cos, sin):
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)

    def rot(t):
        h1, h2 = t[..., : t.shape[-1] // 2], t[..., t.shape[-1] // 2 :]
        return torch.cat((-h2, h1), dim=-1)

    return (q * cos) + (rot(q) * sin), (k * cos) + (rot(k) * sin)


# ---------------------------------------------------------------------------
# Huginn adapter
# ---------------------------------------------------------------------------
class HuginnGraphAdapter:
    """Graph-safe Huginn R (one recurrence = adapter(cat[x, e]) + 4 core
    SandwichBlocks). P (prelude, 2 attn layers + seeded s0) and C (coda,
    2 attn layers + ln_f x2 + head) stay eager via the existing HuginnBlocks
    internals — they touch only the prelude/coda slots, never the core slots
    the graphs use. Slot formula: n_prelude + _slot(r)*n_core + layer."""

    def __init__(self, blocks, cache):
        self.blocks = blocks
        self.cache = cache
        self.model = blocks.model
        self.config = blocks.config
        self.device = blocks.device
        self.D = self.config.n_embd
        self.max_seq = blocks.max_seq_len
        self.n_core = blocks.n_core
        self.n_prelude = blocks.n_prelude
        self._rope = blocks._apply_rope           # complex-like helper
        # freqs table: (1, L, ...) complex — indexed along dim 1 in-graph.
        self.freqs_tab = self.model.freqs_cis

    # -- eager P: prelude with a DEVICE token id (no .item()) --
    def p_prelude(self, ids_t: torch.Tensor, position: int):
        """ids_t: (1,1) long. Returns (hidden0 (D,), static_state e (D,)) —
        mirrors HuginnBlocks._prelude_forward with device ids."""
        model = self.model
        freqs = model.freqs_cis[:, position : position + 1]
        x = model.transformer.wte(ids_t)
        if model.emb_scale != 1:
            x = x * model.emb_scale
        for i, block in enumerate(model.transformer.prelude):
            x = self.blocks._block_fa4_single(block, x, freqs, None, self.cache,
                                              step_idx=i, position=position)
        static_state = x[0, 0]
        hidden0 = self.blocks._init_state_tokens([position])[0]
        return hidden0, static_state

    # -- eager C: coda logits for k rows (positions are Python ints) --
    def c_logits(self, hiddens: list[torch.Tensor], positions: list[int]) -> list[torch.Tensor]:
        from .state import ActiveToken
        toks = []
        for h, p in zip(hiddens, positions):
            t = ActiveToken(token_id=0, position=p, role="verify")
            t.hidden = h
            toks.append(t)
        state = _ShimState(self.cache)
        return self.blocks._coda_logits(state, toks)

    def prefill(self, prompt_ids: torch.Tensor, T_total: int):
        return self.blocks.prefill(prompt_ids, T_total, T_total)

    def make_unit(self, n: int) -> GraphUnit:
        dev, dt = self.device, next(self.model.parameters()).dtype
        return GraphUnit(
            n=n,
            x=torch.zeros(1, n, self.D, dtype=dt, device=dev),
            pos=torch.zeros(n, dtype=torch.long, device=dev),
            cu_q1=torch.tensor([0, 1], dtype=torch.int32, device=dev),
            cu_qn=torch.tensor([0, n], dtype=torch.int32, device=dev),
            cu_k_rows=torch.zeros(n, 2, dtype=torch.int32, device=dev),
            cu_k_group=torch.zeros(2, dtype=torch.int32, device=dev),
            e=torch.zeros(1, n, self.D, dtype=dt, device=dev),
        )

    def _slot(self, step: int, layer_idx: int) -> int:
        return self.n_prelude + self.blocks._slot(step) * self.n_core + layer_idx

    def r_step(self, unit: GraphUnit, x, steps: list[int], grouped: bool,
               group_order: torch.Tensor | None = None):
        model = self.model
        core = model.transformer.core_block
        n = unit.n
        freqs = self.freqs_tab.index_select(1, unit.pos)          # (1, n, ...)
        x = model.transformer.adapter(torch.cat([x, unit.e], dim=-1))
        for layer_idx, block in enumerate(core):
            h = block.norm_1(x)
            attn = block.attn
            q, k, v = attn.Wqkv(h).split(attn.chunks, dim=2)
            q = q.view(1, n, attn.n_head, attn.head_dim)
            k = k.view(1, n, attn.n_kv_heads, attn.head_dim)
            v = v.view(1, n, attn.n_kv_heads, attn.head_dim)
            if self.config.qk_bias:
                q_bias, k_bias = attn.qk_bias.split(1, dim=0)
                q, k = (q + q_bias).to(q.dtype), (k + k_bias).to(q.dtype)
            q, k = self._rope(q, k, freqs_cis=freqs)
            for i in range(n):
                slot = self._slot(steps[i], layer_idx)
                Kb, Vb = self.cache.K[slot][0], self.cache.V[slot][0]
                Kb.index_copy_(0, unit.pos[i : i + 1], k[0, i : i + 1])
                Vb.index_copy_(0, unit.pos[i : i + 1], v[0, i : i + 1])
            if grouped:
                slot = self._slot(steps[0], layer_idx)
                Kb, Vb = self.cache.K[slot][0], self.cache.V[slot][0]
                qp = q[0] if group_order is None else q[0].index_select(0, group_order)
                out = _fa_varlen(qp, Kb, Vb, unit.cu_qn, unit.cu_k_group,
                                 n, self.max_seq, causal=True)
                if group_order is not None:
                    out = out.index_select(0, group_order)
                out = out.unsqueeze(0)
            else:
                outs = []
                for i in range(n):
                    slot = self._slot(steps[i], layer_idx)
                    Kb, Vb = self.cache.K[slot][0], self.cache.V[slot][0]
                    outs.append(_fa_varlen(q[0, i : i + 1], Kb, Vb, unit.cu_q1,
                                           unit.cu_k_rows[i], 1, self.max_seq,
                                           causal=False))
                out = (outs[0] if n == 1 else torch.cat(outs, dim=0)).unsqueeze(0)
            att = attn.proj(out.reshape(1, n, self.D))
            x = block.norm_2(att + x)
            x = block.norm_4(block.mlp(block.norm_3(x)) + x)
        return x


class _ShimState:
    """Minimal WavefrontState stand-in for HuginnBlocks._coda_logits (which
    only reads .kv_cache)."""

    def __init__(self, cache):
        self.kv_cache = cache


# ---------------------------------------------------------------------------
# Capture builders (mode-level)
# ---------------------------------------------------------------------------
def build_ar_graph(adapter, unit: GraphUnit, T_total: int) -> None:
    """One unrolled graph: R x T_total at n=1, then copy back to unit.x and
    bump pos/cu_k in-graph (one replay == one token's R work)."""

    def fn():
        x = unit.x
        for t in range(T_total):
            x = adapter.r_step(unit, x, [t], grouped=False)
        unit.x.copy_(x)
        unit.pos.add_(1)
        unit.cu_k_rows[:, 1].add_(1)
        unit.cu_k_group[1] += 1

    unit.graph, unit.capture_s = _capture(fn)


def build_wfd_graphs(adapter, unit: GraphUnit, T_total: int, T_draft: int,
                     grouped: bool) -> None:
    """T_draft phase graphs over the same unit buffers. Row i (0 = newest,
    highest position) advances step i*T_draft + p -> +1 at phase p. `grouped`
    = every row maps to the same physical slot (KV-sharing s=1): one causal
    segment, rows ascending by position = REVERSED row order."""
    n = unit.n
    # keep the permutation ALIVE on the unit — the graphs read it by pointer
    unit.group_order = (torch.arange(n - 1, -1, -1, device=unit.x.device)
                        if (grouped and n > 1) else None)

    unit.graphs = []
    total_s = 0.0
    for p in range(T_draft):
        steps = [i * T_draft + p for i in range(n)]

        def fn(steps=steps):
            out = adapter.r_step(unit, unit.x, steps, grouped=grouped,
                                 group_order=unit.group_order)
            unit.x.copy_(out)

        g, sec = _capture(fn)
        unit.graphs.append(g)
        total_s += sec
    unit.capture_s = total_s


def build_dtv_graphs(adapter, draft_unit: GraphUnit, verify_unit: GraphUnit,
                     T_total: int, T_draft: int, grouped_verify: bool = True) -> None:
    """Draft: R x T_draft at n=1 with in-graph bump (one replay per source).
    Verify: R x (T_total - T_draft) at n=gamma+1; all rows share the step ->
    same slot -> grouped causal segment (rows ascending by position: row i =
    source i). With KV sharing the slot mapping changes but rows still share
    one slot per layer -> same grouped path."""

    def fn_draft():
        x = draft_unit.x
        for t in range(T_draft):
            x = adapter.r_step(draft_unit, x, [t], grouped=False)
        draft_unit.x.copy_(x)
        draft_unit.pos.add_(1)
        draft_unit.cu_k_rows[:, 1].add_(1)
        draft_unit.cu_k_group[1] += 1

    draft_unit.graph, draft_unit.capture_s = _capture(fn_draft)

    nv = verify_unit.n

    def fn_verify():
        x = verify_unit.x
        for t in range(T_draft, T_total):
            x = adapter.r_step(verify_unit, x, [t] * nv, grouped=grouped_verify)
        verify_unit.x.copy_(x)

    verify_unit.graph, verify_unit.capture_s = _capture(fn_verify)
