"""Greedy and temperature -> exact top-k -> nucleus sampling, no global RNG."""

from __future__ import annotations

import torch
from torch import Tensor

from llm_scratch.inference.config import GenerationConfig


def _validate(logits: Tensor, config: GenerationConfig) -> None:
    if not isinstance(config, GenerationConfig):
        raise ValueError("config must be GenerationConfig")
    if (
        not isinstance(logits, Tensor)
        or logits.ndim != 2
        or 0 in logits.shape
        or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        or logits.layout != torch.strided
        or logits.device.type not in ("cpu", "cuda")
    ):
        raise ValueError("logits must be floating nonempty [batch, vocab] on CPU/CUDA")
    config.validate_vocab(logits.shape[1])
    # -inf may represent an already excluded candidate; every row needs a finite one.
    if (
        torch.isnan(logits).any()
        or torch.isposinf(logits).any()
        or not torch.isfinite(logits).any(dim=-1).all()
    ):
        raise ValueError("logits need a finite candidate per row, with no NaN/+inf")


@torch.no_grad()
def filter_logits(logits: Tensor, config: GenerationConfig) -> Tensor:
    """Return scaled FP32/FP64 logits with excluded candidates set to -inf.

    Stable sort breaks score ties by lower token ID. Top-k keeps exactly k IDs
    (or all with k=0). Nucleus is the smallest prefix reaching top_p *after*
    top-k renormalization, including the first candidate crossing the threshold.
    At least one finite candidate always survives. Inputs are never modified.
    """
    _validate(logits, config)
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    # Python scalars outside FP32's range must not become zero/inf during division.
    promote = (
        config.temperature < torch.finfo(dtype).tiny or config.temperature > torch.finfo(dtype).max
    )
    values = logits.to(torch.float64 if promote else dtype)
    if config.temperature >= 1:
        values = values / config.temperature
        values = values - values.max(dim=-1, keepdim=True).values
    else:
        # Subtract before division so a tiny temperature cannot overflow the max.
        values = (values - values.max(dim=-1, keepdim=True).values) / config.temperature
    values = values.to(dtype)
    if config.top_k or config.top_p < 1:
        # Positive scaling is monotone. Rank original scores so extreme temperatures
        # cannot manufacture ties by rounding every scaled score to zero or -inf.
        indices = torch.argsort(logits, descending=True, stable=True, dim=-1)
        ordered = values.gather(-1, indices)
        if config.top_k:
            ordered[:, config.top_k :] = -torch.inf
        if config.top_p < 1:
            probs = ordered.softmax(dim=-1)
            before = torch.cat((torch.zeros_like(probs[:, :1]), probs.cumsum(-1)[:, :-1]), -1)
            remove = before >= config.top_p
            remove[:, 0] = False  # Even top_p below FP32's range keeps the best candidate.
            ordered = ordered.masked_fill(remove, -torch.inf)
        values = torch.full_like(values, -torch.inf).scatter(-1, indices, ordered)
    return values


@torch.no_grad()
def sample_next_token(
    logits: Tensor,
    config: GenerationConfig,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Return int64 [batch]. Stochastic mode requires an explicit same-device RNG.

    Greedy uses raw argmax (lowest token ID on ties), ignores sampling filters and
    consumes no RNG. Invalid settings/logits are rejected in either mode.
    """
    _validate(logits, config)
    if not config.do_sample:
        return logits.argmax(dim=-1)
    if not isinstance(generator, torch.Generator):
        raise ValueError("Sampling requires an explicit generator on the logits device")
    rng_device = generator.device
    if rng_device.type == "cuda" and rng_device.index is None:
        rng_device = torch.device("cuda", torch.cuda.current_device())
    if rng_device != logits.device:
        raise ValueError("Sampling requires an explicit generator on the logits device")
    probabilities = filter_logits(logits, config).softmax(dim=-1)
    return torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
