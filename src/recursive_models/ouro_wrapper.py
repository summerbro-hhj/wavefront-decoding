"""Ouro wrapper exposing intermediate-iteration logits.

The HuggingFace checkpoint `ByteDance/Ouro-1.4B` ships with a `modeling_ouro.py`
whose `OuroModel.forward()` iterates a shared decoder-layer stack `total_ut_steps`
times (a Loopformer-style "loop the entire body" architecture, not a parcae-style
"loop only the middle" architecture). Conveniently, the inner OuroModel already
returns `hidden_states_list` (one tensor per ut_step) alongside its primary
output — so to obtain per-iteration logits we just apply `lm_head` to each
entry of that list. No source-tree modifications needed.

We verified that `lm_head(hidden_states_list[-1])` is bit-identical to
`OuroForCausalLM(...).logits`, so the wrapper's `final_logits` matches the
stock model's output exactly.
"""
from __future__ import annotations

import torch

from .base import RecursiveLM, RecursiveOutput


def _ensure_default_rope_init() -> None:
    """Ouro's bundled modeling code (transformers 4.55-era) looks up
    `ROPE_INIT_FUNCTIONS["default"]`, but transformers 5.x dropped that key
    (only the scaled variants remain). Register the canonical no-scaling RoPE
    init (`inv_freq = 1/base**(arange(0,dim,2)/dim)`, attention_factor=1.0) under
    "default" if it's missing, so `OuroRotaryEmbedding.__init__` works. Idempotent
    and only adds a missing standard key — does not alter existing entries."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    if "default" in ROPE_INIT_FUNCTIONS:
        return

    def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        partial = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device).float() / dim)
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters


def _patch_ouro_rotary_class(model_id: str) -> None:
    """transformers 5.x `_init_weights` re-initialises RotaryEmbedding buffers via
    `module.compute_default_rope_parameters` (a method on the module) when
    rope_type == "default". Ouro's bundled `OuroRotaryEmbedding` predates that and
    lacks the method, so weight-init crashes. Add the canonical implementation to
    the class if missing. Patches the dynamically-loaded class object only; the
    repo's modeling_ouro.py file is untouched."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    try:
        Rot = get_class_from_dynamic_module("modeling_ouro.OuroRotaryEmbedding", model_id)
    except Exception:
        return
    if hasattr(Rot, "compute_default_rope_parameters"):
        return

    def compute_default_rope_parameters(self, config, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        partial = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device).float() / dim)
        )
        return inv_freq, 1.0

    Rot.compute_default_rope_parameters = compute_default_rope_parameters


@torch.no_grad()
def _ouro_forward_with_intermediates(
    causal_lm,
    input_ids: torch.Tensor,
    num_iterations: int,
    return_intermediates: bool,
):
    """Run OuroModel for `num_iterations` ut_steps, returning per-iter logits.

    Implementation: temporarily override `model.total_ut_steps` so the inner
    loop runs exactly `num_iterations` times. `OuroModel.forward` already
    builds `hidden_states_list`; we then apply `lm_head` to each entry.
    """
    inner = causal_lm.model
    saved_steps = inner.total_ut_steps
    try:
        inner.total_ut_steps = num_iterations
        outputs, hidden_states_list, _gate_list = inner(
            input_ids=input_ids,
            use_cache=False,
        )
    finally:
        inner.total_ut_steps = saved_steps

    if len(hidden_states_list) != num_iterations:
        raise RuntimeError(
            f"Ouro forward returned {len(hidden_states_list)} hidden states "
            f"but requested {num_iterations} iterations"
        )

    lm_head = causal_lm.lm_head
    final_logits = lm_head(hidden_states_list[-1])
    intermediates: list[torch.Tensor] | None = None
    if return_intermediates:
        intermediates = [lm_head(h) for h in hidden_states_list[:-1]]
    return final_logits, intermediates


class OuroWrapper(RecursiveLM):
    """Wrap `ByteDance/Ouro-*` HF checkpoints."""

    DEFAULT_MODEL_ID = "ByteDance/Ouro-1.4B"
    DEFAULT_MAX_ITERATIONS = 4  # Ouro-1.4B trained with total_ut_steps=4

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        max_iterations: int | None = None,
        attn_implementation: str | None = "sdpa",
    ) -> "OuroWrapper":
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        _ensure_default_rope_init()
        _patch_ouro_rotary_class(model_id)
        tok = AutoTokenizer.from_pretrained(model_id)
        # Ouro's tokenizer ships without a pad_token, which breaks the base
        # encode()'s `padding=True`. Fall back to eos_token (or the first
        # special token if eos is also missing) so encoding multiple strings
        # works. For single-string inputs the override below avoids padding
        # entirely.
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token or tok.bos_token or tok.unk_token

        # Ouro's bundled modeling code targets transformers 4.55 and reads
        # `config.pad_token_id` in OuroModel.__init__, but the checkpoint's
        # config.json omits it and transformers 5.x no longer auto-creates that
        # attribute on the config — so loading raises AttributeError. Patch it to
        # None (→ nn.Embedding padding_idx=None, i.e. normal embedding for every
        # token, including bos/eos=0). In-memory config only; repo files untouched.
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if not hasattr(config, "pad_token_id"):
            config.pad_token_id = None

        # Default to SDPA for the stock forward (the AR baseline path): robust and
        # avoids touching the FA2 Python API of the editable FA4 `flash_attn`
        # install. Our wavefront SSD never uses the model's own attention — it
        # recomputes Q/K/V and calls FA4 directly (see OuroBlocks).
        model_kwargs = {"trust_remote_code": True, "dtype": dtype, "config": config}
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(device)
        model.eval()

        if max_iterations is None:
            max_iterations = int(
                getattr(model.config, "total_ut_steps", cls.DEFAULT_MAX_ITERATIONS)
            )

        return cls(model, tok, max_iterations, torch.device(device))

    def encode(self, text: str | list[str], **kwargs) -> torch.Tensor:
        # Override base.encode to avoid padding when only a single string is
        # passed (the typical experiment-1/2 path). For a list, defer to the
        # base implementation which now has a working pad_token.
        if isinstance(text, str):
            ids = self.tokenizer(text, return_tensors="pt", **kwargs).input_ids
            return ids.to(self.device)
        return super().encode(text, **kwargs)

    @property
    def eos_token_id(self) -> int | None:
        """EOS token id used to stop generation. Ouro's HF tokenizer / config
        both define it (`<|eos|>` = 0); prefer the tokenizer, fall back to config."""
        eid = getattr(self.tokenizer, "eos_token_id", None)
        if eid is not None:
            return eid
        return getattr(self.model.config, "eos_token_id", None)

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
            # Ouro has no per-step dt: only the iteration count matters.
            if len(set(steps)) > 1:
                import warnings

                warnings.warn(
                    "OuroWrapper.forward: `steps` values are ignored — Ouro's "
                    "recurrent core has no dt concept. Only len(steps) is used "
                    "as the iteration count.",
                    stacklevel=2,
                )
            n = len(steps)
        else:
            n = num_iterations if num_iterations is not None else self.max_iterations
            if n < 1:
                raise ValueError(f"num_iterations must be >= 1, got {n}")

        final_logits, intermediates = _ouro_forward_with_intermediates(
            self.model,
            input_ids,
            num_iterations=n,
            return_intermediates=return_intermediates,
        )
        return RecursiveOutput(final_logits=final_logits, intermediate_logits=intermediates)
