"""Standard autoregressive baseline using the same model weights.

`sampler` (acceptance.TokenSampler, default greedy) picks each next token; pass
`policy.make_sampler()` so AR uses the same greedy / position-seeded temperature
draw as the SSD schedulers (then AR ≡ SSD token-for-token, modulo bf16 drift).

This is the reference we compare wavefront SSD against for both correctness
(lossless: same output token sequence) and walltime.

Implementation: one full T_total forward per generated token, with KV cache
shared across steps. Mirrors parcae's `forward_for_generation`/`generate` so we
know wavefront-SSD output should match this byte-for-byte under greedy.
"""
from __future__ import annotations

import time

import torch

from .acceptance import GREEDY, TokenSampler
from .parcae_blocks import (
    ParcaeBlocks,
    _create_parcae_cache,
    _ve_for_coda,
    _ve_for_core,
    _ve_for_prelude,
)


@torch.no_grad()
def _parcae_step(model, blocks: ParcaeBlocks, input_ids: torch.Tensor, cache, position_start: int) -> torch.Tensor:
    """One AR step: forward `input_ids` (shape (1, T_new)) at `position_start`,
    returning the last-position logits of shape (V,).
    """
    config = blocks.config
    T_new = input_ids.shape[1]
    freqs_cis = model.freqs_cis[:, position_start : position_start + T_new]
    model._current_input_ids = input_ids

    x = model.transformer.wte(input_ids)
    if model.emb_scale != 1:
        x = x * model.emb_scale
    for i, block in enumerate(model.transformer.prelude):
        ve = _ve_for_prelude(model, input_ids, i)
        x = block(
            x, freqs_cis, None,
            past_key_values=cache,
            step_idx=torch.tensor(i, dtype=torch.long),
            ve=ve,
        )
    if config.prelude_norm:
        x = model.transformer.ln_prelude(x)
    input_embeds = x

    x = model.initialize_state(input_embeds)
    total_steps = torch.tensor(blocks.T_total, device=x.device)
    for r in range(blocks.T_total):
        x = model.core_block_forward(
            x, input_embeds, freqs_cis, None,
            step=torch.tensor(r, device=x.device),
            total_steps=total_steps,
            past_key_values=cache,
            step_idx_base=blocks.n_prelude + r * blocks.n_core,
        )

    x = model.transformer.C(x)
    for i, block in enumerate(model.transformer.coda):
        ve = _ve_for_coda(model, input_ids, i)
        x = block(
            x, freqs_cis, None,
            past_key_values=cache,
            step_idx=torch.tensor(blocks.coda_base + i, dtype=torch.long),
            ve=ve,
        )
    x = model.transformer.ln_f(x)

    logits = model.lm_head(x).float() * config.init.logit_scale
    if config.logit_softcap is not None:
        sc = config.logit_softcap
        logits = sc * torch.tanh(logits / sc)
    return logits[0, -1, :]


@torch.no_grad()
def ar_generate_parcae(
    blocks: ParcaeBlocks,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    sampler: TokenSampler | None = None,
) -> tuple[list[int], float]:
    """Greedy AR generation using parcae's NATIVE forward (its own flash backend),
    NOT our FA4 P/R/C blocks. Retained as the "no-framework, real model speed"
    reference, but `ParcaeBlocks.ar_generate` now defaults to
    `ar_generate_parcae_blocks` (FA4, same backend as SSD) so the benchmark is
    apples-to-apples. Stops at max_new_tokens or `eos_token_id`. Returns
    (generated_token_ids, walltime_s)."""
    model = blocks.model
    device = next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)

    cache = _create_parcae_cache(model, blocks.T_total, blocks.max_seq_len)

    sampler = sampler or GREEDY

    t0 = time.perf_counter()

    # Prefill: full prompt at position 0
    last_logits = _parcae_step(model, blocks, prompt_ids, cache, position_start=0)
    generated: list[int] = []
    cur_pos = prompt_ids.shape[1]
    next_id = sampler(last_logits, cur_pos)        # token for position cur_pos
    generated.append(next_id)

    hit_eos = eos_token_id is not None and next_id == eos_token_id
    while len(generated) < max_new_tokens and not hit_eos:
        cur_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
        last_logits = _parcae_step(model, blocks, cur_input, cache, position_start=cur_pos)
        next_id = sampler(last_logits, cur_pos + 1)
        generated.append(next_id)
        cur_pos += 1
        hit_eos = eos_token_id is not None and next_id == eos_token_id

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return generated, elapsed


@torch.no_grad()
def ar_generate_parcae_blocks(
    blocks,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    sampler: TokenSampler | None = None,
) -> tuple[list[int], float]:
    """Greedy AR baseline built on `ParcaeBlocks` (the same FA4 P/R/C primitives
    the wavefront SSD uses), run sequentially at full depth: one token at a time,
    P (prelude) -> R x T_total (core) -> C (coda+head). This shares the decode
    attention backend (FA4) with SSD, so the speedup at T_draft=T_total is ~1.0x
    and other T_draft values isolate the pure wavefront benefit — mirroring how
    `ar_generate_ouro` works, for an apples-to-apples cross-model comparison.

    Contrast `ar_generate_parcae` (above), which runs parcae's NATIVE forward
    (its own flash backend); that one is retained as the "no-framework, real
    model speed" reference but is no longer the benchmark default. Prefill is
    identical (`blocks.prefill`, native, amortised) for both AR and SSD, so only
    the decode path differs. Stops at max_new_tokens or `eos_token_id`."""
    from .state import ActiveToken

    device = blocks.device
    prompt_ids = prompt_ids.to(device)
    if prompt_ids.dim() == 1:                       # ParcaeBlocks.prefill expects (1, L)
        prompt_ids = prompt_ids.unsqueeze(0)
    T_total = blocks.T_total
    sampler = sampler or GREEDY

    t0 = time.perf_counter()
    state, last_logits = blocks.prefill(prompt_ids, T_total, T_total)
    cache = state.kv_cache
    generated: list[int] = []
    next_id = sampler(last_logits, state.prefix_len)   # token for position prefix_len
    generated.append(next_id)

    cur_pos = state.prefix_len
    hit_eos = eos_token_id is not None and next_id == eos_token_id
    while len(generated) < max_new_tokens and not hit_eos:
        # One token: prelude -> core until commit -> coda+head. Commit at full
        # depth, or earlier if adaptive-total-T (blocks.exit_hidden_eps) fires —
        # the SAME option the SSD wave uses. _advance_core sets should_exit and
        # freezes the skipped deeper core-KV slots at halt.
        hidden0, static_state = blocks._prelude_forward(cache, next_id, cur_pos)
        tok = ActiveToken(token_id=next_id, position=cur_pos, role="verify")
        tok.hidden = hidden0
        tok.static_state = static_state
        tok.step = 0
        for _ in range(T_total):
            blocks._advance_core([tok], cache)
            if tok.should_exit:
                break
        last_logits = blocks._coda_logits(state, [tok])[0]
        next_id = sampler(last_logits, cur_pos + 1)
        generated.append(next_id)
        cur_pos += 1
        hit_eos = eos_token_id is not None and next_id == eos_token_id

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return generated, elapsed


@torch.no_grad()
def ar_generate_ouro(
    blocks,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    sampler: TokenSampler | None = None,
) -> tuple[list[int], float]:
    """Greedy AR baseline for Ouro, built on `OuroBlocks` (the same FA4 forward
    the wavefront SSD uses), run sequentially at full depth: one token at a time,
    `T_total` R calls (ut-steps) per token, then lm_head. This is the "no
    speculation" reference — sharing the kernel with SSD isolates the wavefront
    benefit (vs the stock `OuroForCausalLM` forward, which is incompatible with
    the installed transformers and never used here). Stops at max_new_tokens or
    when `eos_token_id` is generated.
    """
    from .state import ActiveToken

    device = blocks.device
    prompt_ids = prompt_ids.to(device)
    T_total = blocks.T_total
    sampler = sampler or GREEDY

    t0 = time.perf_counter()
    state, last_logits = blocks.prefill(prompt_ids, T_total, T_total)
    cache = state.kv_cache
    generated: list[int] = []
    next_id = sampler(last_logits, state.prefix_len)   # token for position prefix_len
    generated.append(next_id)

    cur_pos = state.prefix_len
    hit_eos = eos_token_id is not None and next_id == eos_token_id
    while len(generated) < max_new_tokens and not hit_eos:
        # One token: P (embed) -> R until commit -> C (lm_head). Commit at full
        # depth, or earlier if adaptive-total-T (blocks.exit_threshold) fires —
        # the SAME option the SSD wave uses. _advance_one_ut sets should_exit and
        # freezes the skipped deeper KV slots at halt, so the next token can
        # attend this position at any depth.
        tok = ActiveToken(token_id=next_id, position=cur_pos, role="verify")
        ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        tok.hidden = blocks.inner.embed_tokens(ids)[0, 0]
        tok.step = 0
        for _ in range(T_total):
            blocks._advance_one_ut([tok], cache)
            if tok.should_exit:
                break
        last_logits = blocks._lm_head_logits([tok])[0]
        next_id = sampler(last_logits, cur_pos + 1)
        generated.append(next_id)
        cur_pos += 1
        hit_eos = eos_token_id is not None and next_id == eos_token_id

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return generated, elapsed
