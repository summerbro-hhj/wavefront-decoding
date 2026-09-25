"""Token sampling + acceptance: comparing a draft token vs the verify logits.

Two sampling modes, selected by `SsdPolicy.mode`:

  greedy       token = argmax(logits). Draft = shallow argmax; acceptance = top-1
               match with the full-depth argmax (lossless).
  temperature  speculative sampling (Leviathan et al. 2023, arXiv:2211.17192).
               Draft x ~ q = softmax(shallow_logits / T) (its q is kept on the
               ActiveToken as `draft_probs`). Verify with p = softmax(full_logits
               / T): draw r ~ U(0,1) and ACCEPT iff r < p(x) / q(x); on reject,
               commit a sample from the residual norm(max(0, p - q)). Committed
               tokens with nothing speculated (first token, boundary after a
               rollback, DtV bonus) are direct samples of p. The committed stream
               is an exact sample of the full-depth model at temperature T, and
               P(accept) = sum_x min(p(x), q(x)) — the maximum any acceptance rule
               can reach while staying exact (vs sum_x p(x) q(x) for exact-match
               with a sampled draft, or p(argmax q) with a greedy draft).

Noise scheme: every random draw is a deterministic function of (sample_seed,
sequence position, salt), regenerated on demand — draft Gumbel noise (salt 0),
the verification uniform r (salt 1) and the residual-sampling Gumbel noise
(salt 2) are mutually independent, and independent across positions. Given the
prefix (committed from strictly earlier positions), the draws for position k are
fresh i.i.d. noise, so exactness holds; a position is verified at most once
(reject at k commits k and restarts the wave after it), so no draw is reused
against a different (p, q). Direct samples (AR baseline, boundary, bonus) use
the salt-0 Gumbel: AR and SSD are therefore two INDEPENDENT samples of the same
distribution under temperature, and `match_rate` is not a lossless check there
(it is under greedy). Runs are reproducible per --seed. This mirrors how
`ParcaeBlocks` draws the recurrent s0 position-seeded.
"""
from __future__ import annotations

import torch


# ----------------------------------------------------------------------------
# greedy
# ----------------------------------------------------------------------------
def greedy_sample(logits: torch.Tensor) -> int:
    """Return argmax token id from a (V,) or (1, V) logit tensor."""
    if logits.dim() > 1:
        logits = logits.reshape(-1)
    return int(logits.argmax(dim=-1).item())


def greedy_accept(draft_token: int, verify_logits: torch.Tensor) -> tuple[bool, int]:
    """Top-1 greedy acceptance.

    Returns:
        (accepted, committed_token):
          accepted=True  -> draft equals argmax(verify_logits); commit draft_token.
          accepted=False -> commit argmax(verify_logits) instead (lossless correction).
    """
    target_token = greedy_sample(verify_logits)
    if draft_token == target_token:
        return True, target_token
    return False, target_token


# ----------------------------------------------------------------------------
# temperature / speculative sampling
# ----------------------------------------------------------------------------
_SEED_MASK = 0x7FFF_FFFF_FFFF
_EPS = 1e-20
SALT_DRAFT, SALT_UNIFORM, SALT_RESIDUAL = 0, 1, 2


def _position_seed(seed: int, position: int, salt: int = SALT_DRAFT) -> int:
    # Distinct mixing from the s0 recipe (seed*1_000_003 + p + 1) so sampling
    # noise and the recurrent-state draw never share a stream; `salt` separates
    # the independent draws needed at one position.
    return (seed * 2_654_435_761 + position * 40_503 + salt * 97_003 + 0x5A17) & _SEED_MASK


_GEN_CACHE: dict[str, torch.Generator] = {}


def _generator_for(device: torch.device) -> torch.Generator:
    key = str(device)
    g = _GEN_CACHE.get(key)
    if g is None:
        g = torch.Generator(device=device)
        _GEN_CACHE[key] = g
    return g


def position_gumbel(vocab_size: int, position: int, seed: int, device: torch.device,
                    salt: int = SALT_DRAFT) -> torch.Tensor:
    """(V,) float32 standard-Gumbel noise, deterministic in (seed, position, salt)."""
    g = _generator_for(device)
    g.manual_seed(_position_seed(seed, position, salt))
    u = torch.rand(vocab_size, generator=g, device=device, dtype=torch.float32)
    u = u.clamp_(_EPS, 1.0 - 1e-7)
    return -torch.log(-torch.log(u))


def position_uniform(position: int, seed: int, device: torch.device) -> torch.Tensor:
    """() float32 r ~ U(0,1) on `device`, deterministic in (seed, position); the
    verification coin (salt 1, independent of the draft noise)."""
    g = _generator_for(device)
    g.manual_seed(_position_seed(seed, position, SALT_UNIFORM))
    return torch.rand((), generator=g, device=device, dtype=torch.float32)


def _flat(logits: torch.Tensor) -> torch.Tensor:
    return logits.reshape(-1) if logits.dim() > 1 else logits


def temperature_sample(logits: torch.Tensor, temperature: float, position: int, seed: int = 0) -> int:
    """Direct sample of softmax(logits / temperature) for sequence `position`
    (Gumbel-max, position-seeded). Used for committed tokens with nothing
    speculated: AR next token, first token, boundary gold, DtV bonus.
    temperature <= 0 -> greedy."""
    logits = _flat(logits)
    if temperature <= 0.0:
        return int(logits.argmax(dim=-1).item())
    noise = position_gumbel(logits.shape[-1], position, seed, logits.device)
    return int((logits.float() / temperature + noise).argmax(dim=-1).item())


def temperature_draft(logits: torch.Tensor, temperature: float, position: int, seed: int = 0,
                      ) -> tuple[int, torch.Tensor]:
    """Draft x ~ q = softmax(shallow logits / T) for `position`; returns (x, q) so
    q can ride on the ActiveToken for stochastic verification."""
    logits = _flat(logits).float() / temperature
    q = torch.softmax(logits, dim=-1)
    noise = position_gumbel(logits.shape[-1], position, seed, logits.device)
    return int((logits + noise).argmax(dim=-1).item()), q


def temperature_accept(
    draft_token: int,
    draft_probs: torch.Tensor,
    verify_logits: torch.Tensor,
    temperature: float,
    position: int,
    seed: int = 0,
) -> tuple[bool, int]:
    """Stochastic verification (speculative sampling): with p = softmax(verify /
    T) and q = draft_probs, accept the draft iff r < p(x)/q(x), r ~ U(0,1);
    otherwise commit a sample of norm(max(0, p - q)).

    Returns:
        (accepted, committed_token):
          accepted=True  -> commit draft_token.
          accepted=False -> commit the residual sample (distribution-lossless
                            correction: marginally the committed token ~ p).
    """
    p = torch.softmax(_flat(verify_logits).float() / temperature, dim=-1)
    q = draft_probs
    r = position_uniform(position, seed, p.device)
    if bool((r * q[draft_token] < p[draft_token]).item()):      # r < p/q  (one sync)
        return True, draft_token
    residual = (p - q).clamp_(min=0.0)
    total = residual.sum()
    if not bool((total > 0).item()):                             # p == q numerically
        residual = p
    noise = position_gumbel(p.shape[-1], position, seed, p.device, salt=SALT_RESIDUAL)
    return False, int((torch.log(residual) + noise).argmax(dim=-1).item())


# ----------------------------------------------------------------------------
# unified callable used by the schedulers / AR baseline
# ----------------------------------------------------------------------------
class TokenSampler:
    """One object per generation run, shared by AR and SSD.

      sampler(logits, position) -> token         committed token with nothing
                                                  speculated (AR next token, first
                                                  token, boundary gold, DtV bonus)
      sampler.draft(logits, position)
          -> (token, q | None)                   speculative draft; q rides on the
                                                  ActiveToken as `draft_probs`
      sampler.verify(x, q, verify_logits, position)
          -> (accepted, committed_token)          accept/reject x against the
                                                  full-depth logits

    temperature <= 0 is greedy (argmax / top-1 match, q=None; the default and the
    exp2/exp3 baseline); otherwise speculative sampling as in the module docstring.
    `position` is the sequence index the token occupies (one past the position
    whose logits are given)."""

    def __init__(self, temperature: float = 0.0, seed: int = 0):
        self.temperature = float(temperature)
        self.seed = int(seed)

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0

    def __call__(self, logits: torch.Tensor, position: int) -> int:
        if self.is_greedy:
            return greedy_sample(logits)
        return temperature_sample(logits, self.temperature, position, self.seed)

    def draft(self, logits: torch.Tensor, position: int) -> tuple[int, torch.Tensor | None]:
        if self.is_greedy:
            return greedy_sample(logits), None
        return temperature_draft(logits, self.temperature, position, self.seed)

    def verify(self, draft_token: int, draft_probs: torch.Tensor | None,
               verify_logits: torch.Tensor, position: int) -> tuple[bool, int]:
        if self.is_greedy:
            return greedy_accept(draft_token, verify_logits)
        if draft_probs is None:
            raise ValueError("temperature verify needs the draft's q (ActiveToken.draft_probs); "
                             "the draft was not produced by sampler.draft().")
        return temperature_accept(draft_token, draft_probs, verify_logits,
                                  self.temperature, position, self.seed)

    def __repr__(self) -> str:
        return "TokenSampler(greedy)" if self.is_greedy else \
            f"TokenSampler(speculative, temperature={self.temperature}, seed={self.seed})"


GREEDY = TokenSampler()
