"""WP06.1 KV storage/transaction probe; not model inference or latency profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from llm_scratch.model.cache import KVCache
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.layers import RotaryEmbedding
from llm_scratch.utils.config import load_config
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/local_smoke.toml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32", "float64"), default="float32"
    )
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, device=args.device, output_root=args.output_root)
    config.data["experiment"]["name"] = "wp06-1-kv-cache"
    architecture = ModelConfig.from_dict(config.data["model"])
    capacity = architecture.context_length if args.max_length is None else args.max_length
    if capacity < 6:
        raise ValueError("This probe needs cache capacity >= 6")
    with Run(config) as run:
        seed_everything(**config.data["reproducibility"])
        environment = inspect_environment(
            args.device, config.data["environment"]["require_gpu_name"]
        )
        write_json(run.path / "artifacts/environment.json", environment)
        if not environment["validation"]["ok"]:
            raise RuntimeError(environment["validation"]["errors"])
        batch_size = config.data["training"]["batch_size"]
        dtype = getattr(torch, args.dtype)
        cache = KVCache(
            architecture, batch_size, max_length=capacity, device=args.device, dtype=dtype
        )
        rope = RotaryEmbedding(
            architecture.head_dim, architecture.context_length, architecture.rope_theta
        )
        pointers = [
            cache.get_layer(i)[0].untyped_storage().data_ptr()
            for i in range(architecture.num_layers)
        ]
        references = [[] for _ in range(architecture.num_layers)]
        for length in (4, 1, 1):
            mask = torch.ones(batch_size, length, device=cache.device, dtype=torch.bool)
            if cache.sequence_length == 0:
                mask[0, -1] = False
            with cache.begin_step(length, attention_mask=mask) as step:
                for layer in range(architecture.num_layers):
                    shape = (batch_size, architecture.num_heads, length, architecture.head_dim)
                    key = rope(
                        torch.randn(shape, device=cache.device, dtype=dtype), step.position_ids
                    )
                    value = torch.randn(shape, device=cache.device, dtype=dtype)
                    references[layer].append((key.clone(), value.clone()))
                    full_key, full_value = step.update(layer, key, value)
                    torch.testing.assert_close(
                        full_key,
                        torch.cat([pair[0] for pair in references[layer]], dim=2),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        full_value,
                        torch.cat([pair[1] for pair in references[layer]], dim=2),
                        rtol=0,
                        atol=0,
                    )
        if cache.sequence_length != 6:
            raise AssertionError("Layer-consistent committed length mismatch")
        before_abort = [
            tuple(t.clone() for t in cache.get_layer(layer))
            for layer in range(architecture.num_layers)
        ]
        if capacity > 6:
            try:
                with cache.begin_step(1):
                    pass
            except RuntimeError as error:
                if "incomplete" not in str(error):
                    raise
            else:
                raise AssertionError("Incomplete append committed")
        else:
            try:
                cache.begin_step(1)
            except ValueError as error:
                if "capacity/context" not in str(error):
                    raise
            else:
                raise AssertionError("Capacity overflow accepted")
        for layer in range(architecture.num_layers):
            for actual, expected in zip(cache.get_layer(layer), before_abort[layer], strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        memory = cache.memory_bytes()
        expected_capacity = (
            2
            * architecture.num_layers
            * batch_size
            * architecture.num_heads
            * capacity
            * architecture.head_dim
            * torch.empty((), dtype=dtype).element_size()
        )
        if (
            memory["kv_capacity_bytes"] != expected_capacity
            or memory["kv_used_bytes"] != expected_capacity * 6 // capacity
        ):
            raise AssertionError("KV tensor payload count differs from independent formula")
        positions = cache.position_ids.tolist()
        cache.reset()
        if cache.sequence_length or cache.memory_bytes()["kv_used_bytes"]:
            raise AssertionError("Reset did not invalidate committed contents")
        with cache.begin_step(1) as step:
            for layer in range(architecture.num_layers):
                shape = (batch_size, architecture.num_heads, 1, architecture.head_dim)
                value = torch.ones(shape, device=cache.device, dtype=dtype)
                step.update(layer, value, value)
        if pointers != [
            cache.get_layer(i)[0].untyped_storage().data_ptr()
            for i in range(architecture.num_layers)
        ]:
            raise AssertionError("Reset/reuse reallocated static KV buffers")
        report = {
            "work_package": "WP06.1",
            "status": "passed",
            "device": str(cache.device),
            "dtype": str(dtype),
            "model_config": architecture.to_dict(),
            "batch_size": batch_size,
            "capacity": capacity,
            "committed_length_before_reset": 6,
            "positions_before_reset": positions,
            "memory_bytes_before_reset": memory,
            "memory_bytes_after_reset_reuse": cache.memory_bytes(),
            "checks": {
                "all_layer_concat_round_trip_exact": True,
                "rotated_key_storage": True,
                "committed_prefix_preserved": True,
                "capacity_accounting_matches_formula": True,
                "reset_reuses_storage": True,
            },
            "limitation": (
                "Cache foundation only; no cached model logits, generation "
                "or latency/VRAM measurements."
            ),
        }
        write_json(run.path / "artifacts/kv_cache_report.json", report)
        run.metadata.update(
            execution_device=args.device,
            probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        write_json(run.path / "metadata.json", run.metadata)
        run.log("kv_cache_passed", checks=report["checks"], memory_bytes=memory)
    print(json.dumps({"status": "passed", "run_directory": str(run.path)}))


if __name__ == "__main__":
    main()
