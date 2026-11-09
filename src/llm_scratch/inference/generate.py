"""Autoregressive inference with uncached reference and request-local static KV cache."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.sampling import sample_next_token
from llm_scratch.model.transformer import CausalLM


@dataclass(frozen=True)
class GenerationOutput:
    sequences: Tensor
    attention_mask: Tensor
    generated_ids: Tensor
    generated_mask: Tensor
    generated_lengths: Tensor
    finish_reasons: tuple[str, ...]
    seed: int


@torch.no_grad()
def generate(
    model: CausalLM,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    *,
    config: GenerationConfig | None = None,
    seed: int = 42,
    use_cache: bool = False,
    position_offset: int = 0,
    on_token: Callable[[int, Tensor, Tensor], None] | None = None,
) -> GenerationOutput:
    """Generate without changing model mode or global RNG.

    All masks/positions follow physical columns, including padding. Initial readout
    selects each row's last valid prompt token; new tokens append after prompt width.
    EOS is included, padding after EOS is masked out, and finished rows are not sampled.
    Prompt EOS does not finish a request. No implicit BOS or control-token suppression.
    Reject width + budget > context upfront; never truncate or silently reduce budget.
    Cached mode prefills once then forwards only the last sampled token per step.
    position_offset shifts every physical RoPE position; it never compacts padding.
    Cache is private to this call and not returned (last sampled token is not fed).
    Optional on_token(step, IDs, valid_rows) observes completed generation steps;
    borrowed tensors are read-only. Callback overhead belongs to the caller's path.
    """
    if not isinstance(model, CausalLM):
        raise ValueError("model must be CausalLM")
    if any(module.training for module in model.modules()):
        raise ValueError("Generation requires model.eval()")
    if type(use_cache) is not bool:
        raise ValueError("use_cache must be boolean")
    if type(position_offset) is not int or position_offset < 0:
        raise ValueError("position_offset must be a nonnegative integer")
    if on_token is not None and not callable(on_token):
        raise ValueError("on_token must be callable or None")
    config = GenerationConfig() if config is None else config
    if not isinstance(config, GenerationConfig):
        raise ValueError("config must be GenerationConfig")
    config.validate_vocab(model.config.vocab_size)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if (
        not isinstance(input_ids, Tensor)
        or input_ids.ndim != 2
        or 0 in input_ids.shape
        or input_ids.dtype not in (torch.int32, torch.int64)
        or input_ids.layout != torch.strided
        or input_ids.device.type not in ("cpu", "cuda")
        or input_ids.device != model.token_embedding.weight.device
    ):
        raise ValueError("input_ids must be nonempty int32/int64 [batch, prompt] on model device")
    if torch.any((input_ids < 0) | (input_ids >= model.config.vocab_size)):
        raise ValueError("input_ids outside vocabulary, including padding columns")
    batch, width = input_ids.shape
    if position_offset + width + config.max_new_tokens > model.config.context_length:
        raise ValueError(
            "RoPE offset + prompt physical width + max_new_tokens exceeds context limit"
        )
    if attention_mask is None:
        mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if (
            not isinstance(attention_mask, Tensor)
            or attention_mask.shape != input_ids.shape
            or attention_mask.device != input_ids.device
            or attention_mask.layout != torch.strided
            or attention_mask.dtype
            not in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
            or torch.any((attention_mask != 0) & (attention_mask != 1))
        ):
            raise ValueError("attention_mask must be binary bool/integer [batch, prompt]")
        mask = attention_mask.bool().clone()
    if not mask.any(dim=-1).all():
        raise ValueError("Every prompt row must have at least one valid token")
    sequences = input_ids.long().clone()
    columns = torch.arange(width, device=input_ids.device).expand(batch, -1)
    last_valid = columns.masked_fill(~mask, -1).max(-1).values
    rows = torch.arange(batch, device=input_ids.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=input_ids.device)
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    cache = (
        model.new_cache(batch, max_length=width + config.max_new_tokens)
        if use_cache and config.max_new_tokens
        else None
    )
    for step in range(config.max_new_tokens):
        chunk = step > 0 and use_cache
        model_ids = sequences[:, -1:] if chunk else sequences
        model_mask = mask[:, -1:] if chunk else mask
        kwargs = {}
        if use_cache:
            kwargs.update(use_cache=True, cache=cache)
        if position_offset:
            start = sequences.shape[1] - 1 if chunk else 0
            kwargs["position_ids"] = torch.arange(
                position_offset + start,
                position_offset + sequences.shape[1],
                device=input_ids.device,
            )
        output = model(model_ids, model_mask, **kwargs)
        logits = output.logits[:, 0] if chunk else output.logits[rows, last_valid]
        active = ~finished
        next_ids = torch.full(
            (batch,), config.pad_token_id, device=input_ids.device, dtype=torch.long
        )
        next_ids[active] = sample_next_token(logits[active], config, generator=generator)
        sequences = torch.cat((sequences, next_ids[:, None]), -1)
        mask = torch.cat((mask, active[:, None]), -1)
        last_valid = torch.where(active, sequences.shape[1] - 1, last_valid)
        if config.eos_token_id is not None:
            finished |= active & (next_ids == config.eos_token_id)
        done = bool(finished.all())
        if on_token is not None:
            on_token(step, next_ids, active)
        if done:
            break
    generated_mask = mask[:, width:]
    return GenerationOutput(
        sequences=sequences,
        attention_mask=mask,
        generated_ids=sequences[:, width:],
        generated_mask=generated_mask,
        generated_lengths=generated_mask.sum(-1),
        finish_reasons=tuple("eos" if done else "max_new_tokens" for done in finished.tolist()),
        seed=seed,
    )
