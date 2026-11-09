"""Diagnostic hooks for actual cached/uncached generation; never a profiling path."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import GenerationOutput, generate
from llm_scratch.model.transformer import CausalLM

TOLERANCES = {
    torch.float64: (1e-10, 1e-12),
    torch.float32: (1e-4, 1e-5),
    torch.float16: (0.005, 0.002),
    torch.bfloat16: (0.02, 0.01),
}


@dataclass(frozen=True)
class GenerationTrace:
    output: GenerationOutput
    logits: tuple[Tensor, ...]
    input_lengths: tuple[int, ...]


def trace_generation(
    model: CausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    config: GenerationConfig,
    seed: int,
    use_cache: bool,
    position_offset: int = 0,
) -> GenerationTrace:
    """Copy only next-token readout logits to CPU; remove hook even on failure.

    The decode readout for finished cached rows is irrelevant and can be zero;
    equivalence compares only rows active at each generation decision.
    """
    logits, lengths = [], []

    def record(module, args, kwargs, output):
        ids, mask = args[:2]
        if use_cache and logits:
            readout = output.logits[:, 0]
        else:
            columns = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
            last = columns.masked_fill(~mask.bool(), -1).max(-1).values
            readout = output.logits[torch.arange(ids.shape[0], device=ids.device), last]
        logits.append(readout.detach().cpu().clone())
        lengths.append(ids.shape[1])

    handle = model.register_forward_hook(record, with_kwargs=True)
    try:
        output = generate(
            model,
            input_ids,
            attention_mask,
            config=config,
            seed=seed,
            use_cache=use_cache,
            position_offset=position_offset,
        )
    finally:
        handle.remove()
    return GenerationTrace(output, tuple(logits), tuple(lengths))


def check_generation_equivalence(
    model: CausalLM,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    config: GenerationConfig,
    seed: int = 42,
    position_offset: int = 0,
) -> dict:
    """Require exact tokens/bookkeeping and dtype-specific close active logits.

    Identical stochastic outputs are a fixed-fixture check, not a guarantee across
    numerically unstable rankings, filter boundaries, devices or batch schedules.
    """
    reference = trace_generation(
        model,
        input_ids,
        attention_mask,
        config=config,
        seed=seed,
        use_cache=False,
        position_offset=position_offset,
    )
    cached = trace_generation(
        model,
        input_ids,
        attention_mask,
        config=config,
        seed=seed,
        use_cache=True,
        position_offset=position_offset,
    )
    for field in (
        "sequences",
        "attention_mask",
        "generated_ids",
        "generated_mask",
        "generated_lengths",
    ):
        torch.testing.assert_close(
            getattr(cached.output, field),
            getattr(reference.output, field),
            rtol=0,
            atol=0,
            msg=lambda msg, name=field: f"Cached/uncached {name} differ: {msg}",
        )
    if cached.output.finish_reasons != reference.output.finish_reasons:
        raise AssertionError("Cached/uncached finish reasons differ")
    count = reference.output.generated_ids.shape[1]
    width = input_ids.shape[1]
    if cached.input_lengths != ((width,) + (1,) * (count - 1) if count else ()):
        raise AssertionError("Cached generation did not prefill once then decode one token")
    if reference.input_lengths != tuple(range(width, width + count)):
        raise AssertionError("Uncached generation did not forward full prefixes")
    steps, max_error, minimum_margin = [], 0.0, None
    dtype = reference.logits[0].dtype if count else model.token_embedding.weight.dtype
    rtol, atol = TOLERANCES[dtype]
    for step, (actual, expected) in enumerate(zip(cached.logits, reference.logits, strict=True)):
        active = reference.output.generated_mask[:, step].cpu()
        actual, expected = actual[active], expected[active]
        torch.testing.assert_close(
            actual,
            expected,
            rtol=rtol,
            atol=atol,
            msg=lambda msg, index=step: f"Generation readout logits at step {index}: {msg}",
        )
        delta = (actual.double() - expected.double()).abs()
        error = delta.masked_fill(actual == expected, 0.0).max().item()
        max_error = max(max_error, error)
        margin = None
        if expected.shape[1] > 1:
            top = expected.double().topk(2, dim=-1).values
            gap = top[:, 0] - top[:, 1]
            # Controlled transition fixtures can have a single finite candidate.
            if torch.isfinite(gap).all():
                margin = gap.min().item()
                minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
        steps.append(
            {
                "step": step,
                "active_rows": int(active.sum()),
                "max_abs_logit_error": error,
                "min_reference_top2_gap": margin,
            }
        )
    output = cached.output
    return {
        "status": "passed",
        "settings": config.to_dict(),
        "seed": seed,
        "position_offset": position_offset,
        "logit_dtype": str(dtype),
        "tolerance": {"rtol": rtol, "atol": atol},
        "max_abs_logit_error": max_error,
        "min_reference_top2_gap": minimum_margin,
        "steps": steps,
        "cached_input_lengths": list(cached.input_lengths),
        "uncached_input_lengths": list(reference.input_lengths),
        "generated_ids": [
            output.generated_ids[i, :n].tolist()
            for i, n in enumerate(output.generated_lengths.tolist())
        ],
        "generated_mask": output.generated_mask.tolist(),
        "generated_lengths": output.generated_lengths.tolist(),
        "finish_reasons": list(output.finish_reasons),
        "checks": {
            "active_logits_close": True,
            "token_ids_exact": True,
            "masks_lengths_finish_reasons_exact": True,
            "prefill_once_token_decode": True,
        },
    }
