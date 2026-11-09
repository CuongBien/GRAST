"""Measured wall-clock streaming inference and CUDA allocator peaks, WP06.5."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from llm_scratch.evaluation.latency_vram import summarize_samples
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import generate
from llm_scratch.inference.profile_config import ProfileConfig, load_profile_config
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def copy_for_inference_context(source: CausalLM, context_length: int) -> CausalLM:
    """Explicit context-only copy; strict checkpoint validation remains unchanged.

    Only the new copy's in-memory extra_state context is overridden. This changes
    inference bounds, not the original checkpoint or trained long-context quality.
    """
    if type(context_length) is not int or context_length < source.config.context_length:
        raise ValueError("Inference context override only permits extension")
    config = replace(source.config, context_length=context_length)
    result = CausalLM(config, tokenizer_fingerprint=source.tokenizer_fingerprint).to(
        device=source.token_embedding.weight.device, dtype=source.token_embedding.weight.dtype
    )
    state = dict(source.state_dict())
    state["_extra_state"] = dict(state["_extra_state"], model_config=config.to_dict())
    result.load_state_dict(state)
    return result.eval()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory(device: torch.device) -> dict | None:
    if device.type != "cuda":
        return None
    return {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
    }


def measure_request(
    model: CausalLM, ids: Tensor, mask: Tensor, *, new_tokens: int, use_cache: bool, seed: int
) -> dict:
    """Actual generate API, synchronized token readiness, no logits-copy hooks."""
    if type(new_tokens) is not int or new_tokens < 2:
        raise ValueError("Profiling requires at least two new tokens")
    device = ids.device
    policy = GenerationConfig(max_new_tokens=new_tokens, eos_token_id=None)
    ticks = []

    def ready(step, token_ids, valid_rows):
        _synchronize(device)
        ticks.append(time.perf_counter_ns())

    _synchronize(device)
    baseline = _memory(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter_ns()
    output = generate(
        model, ids, mask, config=policy, seed=seed, use_cache=use_cache, on_token=ready
    )
    _synchronize(device)
    end = time.perf_counter_ns()
    memory = _memory(device)
    if memory is not None:
        memory = {
            "baseline_allocated": baseline["allocated"],
            "baseline_reserved": baseline["reserved"],
            "end_allocated": memory["allocated"],
            "end_reserved": memory["reserved"],
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
        }
        memory["peak_allocated_above_baseline"] = memory["peak_allocated"] - baseline["allocated"]
    if len(ticks) != new_tokens or not (output.generated_lengths == new_tokens).all():
        raise AssertionError("Fixed-token profiling did not produce the exact budget")
    if (
        not start < ticks[0]
        or any(a >= b for a, b in zip(ticks, ticks[1:], strict=False))
        or ticks[-1] > end
    ):
        raise AssertionError("Non-monotonic timing samples")
    return {
        "token_ready_offsets_ns": [n - start for n in ticks],
        "request_elapsed_ns": end - start,
        "ttft_ms": (ticks[0] - start) / 1e6,
        "next_token_latencies_ms": [(b - a) / 1e6 for a, b in zip(ticks, ticks[1:], strict=False)],
        "total_ms": (end - start) / 1e6,
        "vram_bytes": memory,
        "generated_ids": output.generated_ids.tolist(),
    }


def _telemetry(device: torch.device) -> str | None:
    executable = shutil.which("nvidia-smi")
    if device.type != "cuda" or not executable:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "-i",
                str(device.index),
                "--query-gpu=temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def profile_matrix(
    model: CausalLM, policy: ProfileConfig, *, seed: int, on_condition=None
) -> list[dict]:
    if model.config.context_length != policy.context_length:
        raise ValueError("Effective model context differs from profiling protocol")
    if model.token_embedding.weight.device.type != policy.device:
        raise ValueError("Model device differs from profiling protocol")
    if str(model.token_embedding.weight.dtype) != f"torch.{policy.precision}":
        raise ValueError("Model precision differs from profiling protocol")
    device = model.token_embedding.weight.device
    records = []
    for index, length in enumerate(policy.prompt_lengths):
        rng = torch.Generator().manual_seed(seed + length)
        host_ids = torch.randint(
            8, model.config.vocab_size, (policy.batch_size, length), generator=rng
        )
        host_ids[:, 0] = 1
        digest = hashlib.sha256(host_ids.numpy().tobytes()).hexdigest()
        ids = host_ids.to(device)
        mask = torch.ones_like(ids, dtype=torch.bool)
        modes = (False, True) if index % 2 == 0 else (True, False)
        expected = None
        for use_cache in modes:
            mode = "cached" if use_cache else "uncached"
            gc.collect()
            _synchronize(device)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            before = _telemetry(device)
            for _ in range(policy.warmup):
                measure_request(
                    model, ids, mask, new_tokens=policy.new_tokens, use_cache=use_cache, seed=seed
                )
            samples = []
            for _ in range(policy.repeats):
                sample = measure_request(
                    model, ids, mask, new_tokens=policy.new_tokens, use_cache=use_cache, seed=seed
                )
                if expected is None:
                    expected = sample["generated_ids"]
                if sample["generated_ids"] != expected:
                    raise AssertionError("Cached/uncached or repeated profiling token IDs differ")
                samples.append(sample)
            capacity, used = length + policy.new_tokens, length + policy.new_tokens - 1
            per_column = (
                2
                * model.config.num_layers
                * policy.batch_size
                * model.config.hidden_size
                * model.token_embedding.weight.element_size()
            )
            record = {
                "mode": mode,
                "prompt_length": length,
                "prompt_ids_sha256": digest,
                "batch_size": policy.batch_size,
                "new_tokens": policy.new_tokens,
                "condition_order": len(records),
                "warmup": policy.warmup,
                "repeats": policy.repeats,
                "cache_payload_formula_bytes": (
                    {
                        "capacity": capacity,
                        "used_columns": used,
                        "kv_capacity_bytes": capacity * per_column,
                        "kv_used_bytes": used * per_column,
                        "metadata_capacity_bytes": policy.batch_size * capacity * 9,
                    }
                    if use_cache
                    else None
                ),
                "telemetry_before": before,
                "telemetry_after": _telemetry(device),
                "samples": samples,
                "summary": summarize_samples(samples, policy.batch_size, policy.new_tokens),
                "token_ids_match_reference_so_far": True,
            }
            records.append(record)
            if on_condition:
                on_condition(record)
        del ids, mask
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/wp06_profiling.toml")
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    loaded = load_profile_config(args.config, device=args.device, output_root=args.output_root)
    policy = ProfileConfig(**loaded.data["profiling"])
    with Run(loaded) as run:
        seed_everything(**loaded.data["reproducibility"])
        environment = inspect_environment(policy.device)
        write_json(run.path / "artifacts/environment.json", environment)
        if not environment["validation"]["ok"]:
            raise RuntimeError(environment["validation"]["errors"])
        source = CausalLM(ModelConfig.from_dict(loaded.data["source_model"]))
        checkpoint_sha = None
        if args.checkpoint:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
            if payload.get("format_version") != 1:
                raise ValueError("Expected WP02 atomic checkpoint")
            source.load_state_dict(payload["state"]["model"])
            with args.checkpoint.open("rb") as stream:
                checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            del payload
        model = copy_for_inference_context(source, policy.context_length)
        weights_equal = all(
            torch.equal(a, b) for a, b in zip(source.parameters(), model.parameters(), strict=True)
        )
        if not weights_equal:
            raise AssertionError("Context-only copy changed model weights")
        del source
        model.to(device=policy.device, dtype=getattr(torch, policy.precision))
        setup = {
            "work_package": "WP06.5",
            "protocol_version": 1,
            "source_model": loaded.data["source_model"],
            "effective_model": model.config.to_dict(),
            "context_override": {
                "field": "context_length",
                "weights_equal_before_precision_cast": weights_equal,
                "checkpoint_file_unchanged": True,
                "trained_long_context_claim": False,
            },
            "checkpoint_sha256": checkpoint_sha,
            "tokenizer_fingerprint": model.tokenizer_fingerprint,
            "weight_source": "provided_strict_model_checkpoint"
            if args.checkpoint
            else "seeded_random_weights",
            "parameter_counts": model.parameter_counts(),
            "profiling": loaded.data["profiling"],
            "seed": loaded.data["reproducibility"]["seed"],
            "timing_scope": (
                "Warm streaming generate API wall clock, synchronize at request/token boundaries; "
                "includes validation/cache allocation/forward/selection/output bookkeeping; "
                "excludes model load/tokenization/H2D and report writes."
            ),
            "backend": "pure_pytorch_explicit_MHA_no_compile_no_attention_kernel_substitution",
            "generation": "greedy, eos disabled, fixed token budget, no logits-copy hooks",
            "prompt_recipe": (
                "CPU torch.Generator(seed+L), IDs uniform [8,V), first column BOS=1; "
                "all positions valid; copied to device before timing."
            ),
            "statistics": (
                "p50/p95 use NumPy linear quantiles; next-token latency pools N-1 intervals "
                "per repeat; throughput counts batch tokens."
            ),
            "allocator_protocol": (
                "Empty unused CUDA allocator cache once per condition before warmups; "
                "warmups discarded; reset peaks before each measured request, "
                "never empty_cache inside timing."
            ),
            "memory_scope": (
                "CUDA PyTorch allocator allocated/reserved/peaks, not whole-device memory or "
                "analytical cache payload; CPU memory is null."
            ),
        }
        write_json(run.path / "artifacts/profiling_setup.json", setup)

        def save(record):
            write_json(
                run.path / f"artifacts/L{record['prompt_length']}_{record['mode']}.json", record
            )
            run.log(
                "profiling_condition_completed",
                prompt_length=record["prompt_length"],
                mode=record["mode"],
                summary=record["summary"],
            )
            print(
                json.dumps(
                    {
                        "prompt_length": record["prompt_length"],
                        "mode": record["mode"],
                        "ttft_p50_ms": record["summary"]["ttft_ms"]["p50"],
                    }
                ),
                flush=True,
            )

        records = profile_matrix(model, policy, seed=setup["seed"], on_condition=save)
        write_json(
            run.path / "artifacts/profiling_report.json",
            setup | {"status": "passed", "records": records},
        )
        run.metadata.update(
            profiling_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            execution_device=policy.device,
        )
        write_json(run.path / "metadata.json", run.metadata)
        run.log("profiling_passed", conditions=len(records))
    print(json.dumps({"status": "passed", "run_directory": str(run.path)}))


if __name__ == "__main__":
    main()
