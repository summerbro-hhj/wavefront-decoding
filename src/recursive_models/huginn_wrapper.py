"""Huginn-0125 wrapper (RDM — recurrent-depth model, arXiv:2502.05171).

`tomg-group-umd/huginn-0125` is the 3.5B original of the parcae lineage:
prelude(2) -> [adapter(cat) -> core_block(4)] x r -> ln_f -> coda(2) -> ln_f ->
lm_head, trained with mean_recurrence=32 and a RANDOM initial state s0
(`state_init="like-init"`: trunc_normal(std) * emb_scale). Same complex-valued
RoPE helper as parcae; SandwichBlock (4 norms) instead of parcae's pre-norm
block; fused Wqkv + additive qk_bias; MHA 55 heads x 96.

This wrapper is decode-experiment-scoped (exp2/exp3): it provides the duck-typed
interface the runners use — `from_pretrained(model_id, dtype, device)`,
`.model`, `.encode(text)`, `.eos_token_id`. (exp0/exp1's RecursiveLM.forward
with intermediate logits is not implemented yet; Huginn's
`predict_from_latents` makes that straightforward if needed.)
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_ID = "tomg-group-umd/huginn-0125"


class HuginnWrapper:
    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @staticmethod
    def _patch_tied_weights_keys(model_id: str) -> None:
        """huginn's bundled code declares `_tied_weights_keys = ["lm_head.weight"]`
        (transformers 4.x list convention); 5.x expects a {tied: source} dict and
        crashes in post_init otherwise. Patch the dynamically-loaded class object
        only (the repo file is untouched); lm_head is tied to transformer.wte."""
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        try:
            raven = get_class_from_dynamic_module(
                "raven_modeling_minimal.RavenForCausalLM", model_id
            )
        except Exception:
            return
        if isinstance(getattr(raven, "_tied_weights_keys", None), list):
            raven._tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        **_: object,
    ) -> "HuginnWrapper":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        cls._patch_tied_weights_keys(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=dtype
        )
        model.to(device)
        model.eval()
        return cls(model, tokenizer, device)

    def encode(self, text: str, **kwargs) -> torch.Tensor:
        """(1, L) input_ids on the model device."""
        ids = self.tokenizer(text, return_tensors="pt", **kwargs).input_ids
        return ids.to(self.device)

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self.model.config, "eos_token_id", None)
