"""Parcae wrapper exposing intermediate-iteration logits.

Two things differ from `parcae_lm.from_pretrained`:

1. We override `state_init="zero"` to avoid bf16 `torch.nn.init.trunc_normal_`
   (which calls erfinv) being JIT-compiled by NVRTC at forward time. The cu128
   NVRTC on sm_121 (GB10 / DGX Spark) refuses sm_121 as a target and the
   forward crashes. Setting state_init to a deterministic zero tensor also
   removes a source of randomness for acceptance-rate experiments.

2. We re-implement `Parcae.forward` so that the recurrent loop runs in the
   wrapper itself, and at every iteration we apply the same post-recurrent
   stack (C -> coda -> ln_f -> lm_head + optional logit_softcap) the model
   would have applied at the final step. That gives a logits tensor per
   intermediate iteration k=1..n-1, sharing all prior compute.

The wrapper does NOT touch the parcae_lm source tree; everything is monkey-
free, just an external forward.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from .base import RecursiveLM, RecursiveOutput

# Make `external/parcae` importable without pip-installing parcae (its
# `numpy<2.0` pin would conflict with the rest of our env).
_PARCAE_REPO_DIR = Path(__file__).resolve().parents[2] / "external" / "parcae"
if _PARCAE_REPO_DIR.exists() and str(_PARCAE_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_PARCAE_REPO_DIR))


def _load_parcae_pretrained(
    repo_id: str,
    *,
    dtype: torch.dtype,
    device: str | torch.device,
    state_init: str = "zero",
):
    """Re-implementation of `parcae_lm.from_pretrained` that lets us patch the
    config before model construction."""
    from huggingface_hub import hf_hub_download
    from parcae_lm.models.parcae.config import ParcaeConfig

    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("rope_settings", None)  # mirrors parcae_lm's own from_pretrained
    cfg["state_init"] = state_init

    config = ParcaeConfig(**cfg)
    # The pretrained weights loaded below overwrite every parameter, so the
    # model's default init is wasted compute. parcae inits big Linears with an
    # orthogonal QR (`trunc_orthogonal_` -> torch.linalg.qr), which over the
    # ~hundreds of layers here takes minutes on CPU. Temporarily swap it for a
    # cheap trunc_normal_ fill during construction; load_state_dict then
    # restores the real weights (final model is identical). External parcae
    # source is untouched (module-level monkeypatch, restored in finally).
    import parcae_lm.utils.init as _pinit

    _orig_trunc_ortho = _pinit.trunc_orthogonal_

    def _fast_trunc_ortho(tensor, gain: float = 1.0):
        with torch.no_grad():
            torch.nn.init.trunc_normal_(tensor, mean=0.0, std=0.02)
            tensor.mul_(gain)
        return tensor

    _pinit.trunc_orthogonal_ = _fast_trunc_ortho
    try:
        model = config.construct_model()
    finally:
        _pinit.trunc_orthogonal_ = _orig_trunc_ortho

    weights_path = hf_hub_download(repo_id, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # We don't expect any keys to differ; surface it if we do.
        print(
            f"[parcae] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected"
        )

    model = model.to(dtype=dtype).to(device).eval()
    return model


# Canonical parcae special-token marker -> the `Tokenizer` attr that names its role.
# The `SandyResearch/parcae-tokenizer` repo ships ONLY `tokenizer.json` (no
# `tokenizer_config.json` / `special_tokens_map.json`), so HF AutoTokenizer never
# learns which special token plays the bos/eos/pad *role* and reports them all as
# None — even though `<|bos|>`=0, `<|eos|>`=1, `<|pad|>`=2 sit in the vocab. We
# resolve the roles from the marker names so EOS-stop (and bos/pad) work.
_PARCAE_SPECIAL_MARKERS = {"bos_id": "<|bos|>", "eos_id": "<|eos|>", "pad_id": "<|pad|>"}


def _backfill_special_token_roles(tokenizer) -> None:
    """Fill in eos_id/bos_id/pad_id on a parcae `Tokenizer` when HF left them None.

    Looks each role's marker token up in the vocab via the underlying HF processor;
    only fills an attr that is currently None (so a tokenizer that *does* declare
    these roles is untouched). In-memory only — never touches the repo files."""
    processor = getattr(tokenizer, "processor", None)
    if processor is None:
        return
    unk = getattr(processor, "unk_token_id", None)
    for attr, marker in _PARCAE_SPECIAL_MARKERS.items():
        if getattr(tokenizer, attr, None) is not None:
            continue
        try:
            tid = processor.convert_tokens_to_ids(marker)
        except Exception:
            tid = None
        if tid is not None and tid != unk:
            setattr(tokenizer, attr, tid)


@torch.no_grad()
def _parcae_forward_with_intermediates(
    model,
    input_ids: torch.Tensor,
    num_iterations: int,
    return_intermediates: bool,
):
    """Run the prelude, then loop the recurrent core `num_iterations` times,
    applying the full post-recurrent head at each step if asked.
    """
    cfg = model.config
    freqs_cis = model.freqs_cis[:, : input_ids.shape[1]]

    input_embeds = model.transformer.wte(input_ids)
    if model.emb_scale != 1:
        input_embeds = input_embeds * model.emb_scale

    # Some blocks read this attribute to fetch value-embeddings for `input_ids`.
    model._current_input_ids = input_ids

    # Prelude (non-recurrent layers).
    for i, block in enumerate(model.transformer.prelude):
        ve = (
            model.value_embeds[str(i)](input_ids) if str(i) in model.value_embeds else None
        )
        input_embeds = block(input_embeds, freqs_cis, None, ve=ve)
    if cfg.prelude_norm:
        input_embeds = model.transformer.ln_prelude(input_embeds)

    # Recurrent state.
    x = model.initialize_state(input_embeds)
    total_steps = torch.tensor(num_iterations, device=input_embeds.device)
    coda_ve_offset = cfg.n_layers_in_prelude + cfg.n_layers_in_recurrent_block

    def _apply_head(h: torch.Tensor) -> torch.Tensor:
        """Run coda + head on a recurrent-core output `h`."""
        h2 = model.transformer.C(h)
        for j, block in enumerate(model.transformer.coda):
            ve_idx = str(coda_ve_offset + j)
            ve = (
                model.value_embeds[ve_idx](input_ids)
                if ve_idx in model.value_embeds
                else None
            )
            h2 = block(h2, freqs_cis, None, ve=ve)
        h2 = model.transformer.ln_f(h2)
        logits = model.lm_head(h2).float() * cfg.init.logit_scale
        if cfg.logit_softcap is not None:
            softcap = cfg.logit_softcap
            logits = softcap * torch.tanh(logits / softcap)
        return logits

    intermediate_logits: list[torch.Tensor] | None = (
        [] if return_intermediates else None
    )
    for step in range(num_iterations):
        xk = x
        step_t = torch.tensor(step, device=input_embeds.device)
        x = model.update_recurrent_state(xk, input_embeds, freqs_cis, None, step_t, total_steps)
        if return_intermediates and step < num_iterations - 1:
            intermediate_logits.append(_apply_head(x))

    final_logits = _apply_head(x)
    return final_logits, intermediate_logits


class ParcaeWrapper(RecursiveLM):
    """Wrap `SandyResearch/parcae-*` checkpoints."""

    DEFAULT_MODEL_ID = "SandyResearch/parcae-770m"
    DEFAULT_TOKENIZER_ID = "SandyResearch/parcae-tokenizer"
    # parcae models are trained with `mean_recurrence` (8 for the released
    # checkpoints) but the architecture supports any positive int at inference.
    DEFAULT_MAX_ITERATIONS = 8

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        tokenizer_id: str = DEFAULT_TOKENIZER_ID,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        max_iterations: int | None = None,
        state_init: str = "zero",
    ) -> "ParcaeWrapper":
        # Side-effect: registers external/parcae on sys.path.
        from parcae_lm.tokenizer import Tokenizer

        model = _load_parcae_pretrained(
            model_id, dtype=dtype, device=device, state_init=state_init
        )
        tokenizer = Tokenizer.from_pretrained(tokenizer_id)
        _backfill_special_token_roles(tokenizer)

        # Prefer the released mean_recurrence as the natural max_iterations.
        if max_iterations is None:
            max_iterations = int(getattr(model.config, "mean_recurrence", cls.DEFAULT_MAX_ITERATIONS))

        return cls(model, tokenizer, max_iterations, torch.device(device))

    def encode(self, text: str | list[str], **kwargs) -> torch.Tensor:
        # parcae's Tokenizer returns int32 tensor without a batch dim; HF expects long.
        if isinstance(text, str):
            ids = self.tokenizer.encode(text, return_tensors=True).to(self.device).long()
            return ids.unsqueeze(0)
        encoded = [
            self.tokenizer.encode(t, return_tensors=True).to(self.device).long() for t in text
        ]
        # Left-pad to common length (acceptance experiments treat samples independently).
        max_len = max(t.shape[0] for t in encoded)
        padded = torch.zeros(len(encoded), max_len, dtype=torch.long, device=self.device)
        for i, t in enumerate(encoded):
            padded[i, : t.shape[0]] = t
        return padded

    def decode(self, ids: torch.Tensor) -> list[str]:
        return [self.tokenizer.decode(row) for row in ids]

    @property
    def eos_token_id(self) -> int | None:
        """EOS token id used to stop generation (parcae `<|eos|>` = 1).

        `eos_id` is populated by `_backfill_special_token_roles` at load time
        because the parcae-tokenizer repo omits the HF role-mapping files (see
        that helper); without the backfill this would be None despite `<|eos|>`
        being in the vocab."""
        return getattr(self.tokenizer, "eos_id", None)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        num_iterations: int | None = None,
        return_intermediates: bool = False,
        steps: list[float] | None = None,
    ) -> RecursiveOutput:
        if steps is not None:
            if len(steps) < 1:
                raise ValueError("steps must contain at least one entry")
            # parcae has no per-step dt; only the iteration count matters.
            # Warn if the caller appears to have provided meaningful dt values
            # (i.e. a non-uniform list), since they will be silently ignored.
            if len(set(steps)) > 1:
                import warnings

                warnings.warn(
                    "ParcaeWrapper.forward: `steps` values are ignored — parcae's "
                    "recurrent core has no dt concept. Only len(steps) is used as "
                    "the iteration count.",
                    stacklevel=2,
                )
            n = len(steps)
        else:
            n = num_iterations if num_iterations is not None else self.max_iterations
            if n < 1:
                raise ValueError(f"num_iterations must be >= 1, got {n}")

        final_logits, intermediates = _parcae_forward_with_intermediates(
            self.model,
            input_ids,
            num_iterations=n,
            return_intermediates=return_intermediates,
        )
        return RecursiveOutput(final_logits=final_logits, intermediate_logits=intermediates)
