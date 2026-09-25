"""exp5 forced-schedule steady-state decode loops (eager & CUDA-graph engines).

No accept/reject decisions, no batch scheduling: AR / WFD / DtV replay their
STEADY-STATE schedule (alpha=1 equivalent) on a dummy prompt for exactly
`decode_len` committed tokens. All tensor work (P/R/C forwards, argmax, cache
writes) is real — only the decisions are removed — so walltime is faithful
(same philosophy as exp4's emulated acceptance, taken to alpha=1 and W fixed).

Eager engines mirror the real schedulers' code paths (ActiveToken machinery,
`greedy_sample`'s per-commit `.item()` sync). Graph engines keep token ids on
device and never sync inside the timed loop (that IS the optimized regime).

See .claude/plans/exp5_plan.md §2-3; graph mechanics in graph_blocks.py.
"""
from __future__ import annotations

import copy as _copy
import time

import torch

from .acceptance import greedy_sample
from .graph_blocks import (GraphUnit, build_ar_graph, build_dtv_graphs,
                           build_wfd_graphs)
from .state import ActiveToken


def _sync():
    torch.cuda.synchronize()


def _mk_tok(token_id: int, position: int, hidden, static_state) -> ActiveToken:
    t = ActiveToken(token_id=token_id, position=position, role="draft")
    t.hidden, t.static_state, t.step = hidden, static_state, 0
    return t


# ===========================================================================
# Wave warm-up (shared by both WFD engines): fill the diagonal to width W.
# Token j enters at position P0+j when token j-1 reaches step T_draft; after
# (W-1)*T_draft ticks token j sits at step (W-1-j)*T_draft — phase-0 start.
# ===========================================================================
@torch.no_grad()
def build_steady_wave(blocks, state, first_id: int, T_total: int, T_draft: int):
    W = T_total // T_draft
    assert W * T_draft == T_total, "exp5 assumes T_draft | T_total"
    cache = state.kv_cache
    P0 = state.prefix_len
    ids = [first_id]
    h, ss = blocks.prelude_forward(cache, first_id, P0)
    toks = [_mk_tok(first_id, P0, h, ss)]
    for _tick in range((W - 1) * T_draft):
        blocks.advance_one_step(toks, cache)
        newest = toks[-1]
        if newest.step == T_draft:                      # draft point -> next enters
            logits = blocks.coda_logits(state, [_copy.copy(newest)])[0]
            nid = greedy_sample(logits)
            ids.append(nid)
            pos = P0 + len(ids) - 1
            h, ss = blocks.prelude_forward(cache, nid, pos)
            toks.append(_mk_tok(nid, pos, h, ss))
    # sanity: steps are (W-1)*d, (W-2)*d, ..., 0 in entry order
    assert [t.step for t in toks] == [(W - 1 - j) * T_draft for j in range(W)]
    return toks


# ===========================================================================
# Eager engines
# ===========================================================================
@torch.no_grad()
def forced_ar_eager(blocks, state, first_id: int, decode_len: int,
                    T_total: int, warmup: int = 8,
                    collect: list | None = None) -> dict:
    cache = state.kv_cache

    def _one(next_id: int, pos: int) -> int:
        h, ss = blocks.prelude_forward(cache, next_id, pos)
        tok = _mk_tok(next_id, pos, h, ss)
        for _ in range(T_total):
            blocks.advance_one_step([tok], cache)
        return greedy_sample(blocks.coda_logits(state, [tok])[0])

    nid, pos = first_id, state.prefix_len
    for _ in range(warmup):
        nid = _one(nid, pos)
        pos += 1
    _sync()
    t0 = time.perf_counter()
    for _ in range(decode_len):
        nid = _one(nid, pos)
        pos += 1
        if collect is not None:
            collect.append(nid)
    _sync()
    dt = time.perf_counter() - t0
    return dict(decode_s=dt, n_tokens=decode_len, tok_s=decode_len / dt)


@torch.no_grad()
def forced_wfd_eager(blocks, state, first_id: int, decode_len: int,
                     T_total: int, T_draft: int, warmup_commits: int = 4,
                     collect: list | None = None) -> dict:
    cache = state.kv_cache
    toks = build_steady_wave(blocks, state, first_id, T_total, T_draft)

    def _commit_cycle():
        for _ in range(T_draft):
            blocks.advance_one_step(toks, cache)
        deep = max(toks, key=lambda t: t.step)          # step == T_total
        newest = min(toks, key=lambda t: t.step)        # step == T_draft
        logits = blocks.coda_logits(state, [deep, _copy.copy(newest)])
        commit_id = greedy_sample(logits[0])            # committed (bookkeeping)
        if collect is not None:
            collect.append(commit_id)
        new_id = greedy_sample(logits[1])               # leading-edge draft
        toks.remove(deep)
        new_pos = max(t.position for t in toks) + 1
        h, ss = blocks.prelude_forward(cache, new_id, new_pos)
        toks.append(_mk_tok(new_id, new_pos, h, ss))

    for _ in range(warmup_commits):
        _commit_cycle()
    _sync()
    t0 = time.perf_counter()
    for _ in range(decode_len):
        _commit_cycle()
    _sync()
    dt = time.perf_counter() - t0
    return dict(decode_s=dt, n_tokens=decode_len, tok_s=decode_len / dt)


@torch.no_grad()
def forced_dtv_eager(blocks, state, first_id: int, decode_len: int,
                     T_total: int, T_draft: int, gamma: int,
                     warmup_rounds: int = 1) -> dict:
    cache = state.kv_cache
    ctx = dict(last_id=first_id, p=state.prefix_len)

    def _round() -> int:
        cur, cpos = ctx["last_id"], ctx["p"]
        sources: list[ActiveToken] = []
        for _j in range(gamma + 1):
            h, ss = blocks.prelude_forward(cache, cur, cpos)
            tok = _mk_tok(cur, cpos, h, ss)
            for _ in range(T_draft):
                blocks.advance_one_step([tok], cache)
            sources.append(tok)
            cur = greedy_sample(blocks.coda_logits(state, [tok])[0])
            cpos += 1
        for _ in range(T_total - T_draft):
            blocks.advance_one_step(sources, cache)
        gold = blocks.coda_logits(state, sources)
        ctx["last_id"] = greedy_sample(gold[gamma])     # bonus == next round base
        ctx["p"] += gamma + 1
        return gamma + 1                                # all accepted + bonus

    for _ in range(warmup_rounds):
        _round()
    _sync()
    t0 = time.perf_counter()
    committed = 0
    while committed < decode_len:
        committed += _round()
    _sync()
    dt = time.perf_counter() - t0
    return dict(decode_s=dt, n_tokens=committed, tok_s=committed / dt)


# ===========================================================================
# Graph engines
# ===========================================================================
def _is_huginn(adapter) -> bool:
    return adapter.__class__.__name__.startswith("Huginn")


@torch.no_grad()
def forced_ar_graph(adapter, state, first_id: int, decode_len: int,
                    T_total: int, warmup: int = 8,
                    collect: list | None = None) -> dict:
    dev = adapter.device
    unit = adapter.make_unit(1)
    P0 = state.prefix_len
    unit.set_positions([P0])
    build_ar_graph(adapter, unit, T_total)              # capture (pollutes pos)
    unit.set_positions([P0])                            # reset to start
    ids_t = torch.tensor([[first_id]], dtype=torch.long, device=dev)
    huginn = _is_huginn(adapter)
    ctx = dict(pos=P0)

    def _one():
        pos = ctx["pos"]
        if huginn:
            h0, e = adapter.p_prelude(ids_t, pos)
            unit.x[0, 0].copy_(h0)
            unit.e[0, 0].copy_(e)
            unit.graph.replay()
            logits = adapter.c_logits([unit.x[0, 0]], [pos])[0]
        else:
            unit.x.copy_(adapter.p_embed(ids_t))
            unit.graph.replay()
            logits = adapter.c_logits(unit.x[0])[0]
        ids_t.copy_(logits.argmax(dim=-1).view(1, 1))   # stays on device
        ctx["pos"] = pos + 1
        if collect is not None:                          # gate use only (syncs!)
            collect.append(int(ids_t.item()))

    for _ in range(warmup):
        _one()
    _sync()
    t0 = time.perf_counter()
    for _ in range(decode_len):
        _one()
    _sync()
    dt = time.perf_counter() - t0
    return dict(decode_s=dt, n_tokens=decode_len, tok_s=decode_len / dt,
                capture_s=unit.capture_s, n_graphs=1)


@torch.no_grad()
def forced_wfd_graph(adapter, blocks, state, first_id: int, decode_len: int,
                     T_total: int, T_draft: int, grouped: bool,
                     warmup_commits: int = 4, collect: list | None = None) -> dict:
    """`grouped` = KV-sharing collapses all wave rows onto one physical slot
    (Huginn s=1) -> one causal segment per layer; else per-row varlen."""
    dev = adapter.device
    W = T_total // T_draft
    toks = build_steady_wave(blocks, state, first_id, T_total, T_draft)
    # row i = newest-first: toks[W-1-i] (step i*T_draft, position P0+W-1-i)
    rows = [toks[W - 1 - i] for i in range(W)]
    snap_x = torch.stack([t.hidden for t in rows]).unsqueeze(0).clone()
    snap_e = (torch.stack([t.static_state for t in rows]).unsqueeze(0).clone()
              if rows[0].static_state is not None else None)
    positions = [t.position for t in rows]

    unit = adapter.make_unit(W)
    unit.set_positions(positions)
    unit.x.copy_(snap_x)
    if unit.e is not None:
        unit.e.copy_(snap_e)
    build_wfd_graphs(adapter, unit, T_total, T_draft, grouped=grouped)
    # reset after capture warm-up pollution
    unit.set_positions(positions)
    unit.x.copy_(snap_x)
    if unit.e is not None:
        unit.e.copy_(snap_e)

    huginn = _is_huginn(adapter)
    ids_t = torch.zeros(1, 1, dtype=torch.long, device=dev)
    ctx = dict(pmax=positions[0])

    def _commit_cycle():
        for g in unit.graphs:
            g.replay()
        pmax = ctx["pmax"]
        p_deep = pmax - (W - 1)
        if huginn:
            logits = adapter.c_logits([unit.x[0, W - 1], unit.x[0, 0]],
                                      [p_deep, pmax])
            ids_t.copy_(logits[1].argmax(dim=-1).view(1, 1))
            h0, e0 = adapter.p_prelude(ids_t, pmax + 1)
            unit.e.copy_(torch.roll(unit.e, 1, dims=1))
            unit.e[0, 0].copy_(e0)
        else:
            logits = adapter.c_logits(torch.stack([unit.x[0, W - 1], unit.x[0, 0]]))
            ids_t.copy_(logits[1].argmax(dim=-1).view(1, 1))
            h0 = adapter.p_embed(ids_t)[0, 0]
        if collect is not None:                          # gate use only (syncs!)
            commit_logits = logits[0] if isinstance(logits, list) else logits[0]
            collect.append(int(commit_logits.argmax(dim=-1).item()))
        unit.x.copy_(torch.roll(unit.x, 1, dims=1))
        unit.x[0, 0].copy_(h0)
        unit.bump_positions(1)
        ctx["pmax"] = pmax + 1

    for _ in range(warmup_commits):
        _commit_cycle()
    _sync()
    t0 = time.perf_counter()
    for _ in range(decode_len):
        _commit_cycle()
    _sync()
    dt = time.perf_counter() - t0
    return dict(decode_s=dt, n_tokens=decode_len, tok_s=decode_len / dt,
                capture_s=unit.capture_s, n_graphs=len(unit.graphs))


@torch.no_grad()
def forced_dtv_graph(adapter, state, first_id: int, decode_len: int,
                     T_total: int, T_draft: int, gamma: int,
                     warmup_rounds: int = 1) -> dict:
    dev = adapter.device
    P0 = state.prefix_len
    draft_unit = adapter.make_unit(1)
    verify_unit = adapter.make_unit(gamma + 1)
    draft_unit.set_positions([P0])
    verify_unit.set_positions(list(range(P0, P0 + gamma + 1)))
    build_dtv_graphs(adapter, draft_unit, verify_unit, T_total, T_draft)
    draft_unit.set_positions([P0])
    verify_unit.set_positions(list(range(P0, P0 + gamma + 1)))

    huginn = _is_huginn(adapter)
    ids_t = torch.tensor([[first_id]], dtype=torch.long, device=dev)
    ctx = dict(p=P0)

    def _round() -> int:
        p = ctx["p"]
        for j in range(gamma + 1):
            if huginn:
                h0, e0 = adapter.p_prelude(ids_t, p + j)
                draft_unit.x[0, 0].copy_(h0)
                draft_unit.e[0, 0].copy_(e0)
                verify_unit.e[0, j].copy_(e0)
            else:
                draft_unit.x.copy_(adapter.p_embed(ids_t))
            draft_unit.graph.replay()                   # in-graph pos += 1
            verify_unit.x[0, j].copy_(draft_unit.x[0, 0])
            if huginn:
                logits = adapter.c_logits([draft_unit.x[0, 0]], [p + j])[0]
            else:
                logits = adapter.c_logits(draft_unit.x[0])[0]
            ids_t.copy_(logits.argmax(dim=-1).view(1, 1))
        verify_unit.graph.replay()
        if huginn:
            gold = adapter.c_logits(
                [verify_unit.x[0, j] for j in range(gamma + 1)],
                [p + j for j in range(gamma + 1)])
            ids_t.copy_(gold[gamma].argmax(dim=-1).view(1, 1))
        else:
            gold = adapter.c_logits(verify_unit.x[0])
            ids_t.copy_(gold[gamma].argmax(dim=-1).view(1, 1))
        verify_unit.bump_positions(gamma + 1)
        ctx["p"] = p + gamma + 1
        return gamma + 1

    for _ in range(warmup_rounds):
        _round()
    _sync()
    t0 = time.perf_counter()
    committed = 0
    while committed < decode_len:
        committed += _round()
    _sync()
    dt = time.perf_counter() - t0
    return dict(decode_s=dt, n_tokens=committed, tok_s=committed / dt,
                capture_s=draft_unit.capture_s + verify_unit.capture_s, n_graphs=2)
