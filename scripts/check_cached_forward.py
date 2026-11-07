"""WP06.2 cached prefill/token/chunk logits versus uncached forward; no profiling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.research.architecture_acceptance import fixed_batch
from llm_scratch.utils.config import load_config
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/local_smoke.toml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, device=args.device, output_root=args.output_root)
    config.data["experiment"]["name"] = "wp06-2-cached-forward"
    architecture = ModelConfig.from_dict(config.data["model"])
    with Run(config) as run:
        seed_everything(**config.data["reproducibility"])
        environment = inspect_environment(
            args.device, config.data["environment"]["require_gpu_name"]
        )
        write_json(run.path / "artifacts/environment.json", environment)
        if not environment["validation"]["ok"]:
            raise RuntimeError(environment["validation"]["errors"])
        model = CausalLM(architecture)
        checkpoint_sha = None
        if args.checkpoint:
            payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
            if payload.get("format_version") != 1:
                raise ValueError("Expected WP02 atomic model checkpoint format")
            model.load_state_dict(payload["state"]["model"])
            with args.checkpoint.open("rb") as stream:
                checkpoint_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            del payload
        model.to(args.device).eval()
        if args.tokenizer:
            tokenizer = ByteBPETokenizer.load(args.tokenizer)
            if len(tokenizer) != architecture.vocab_size:
                raise ValueError("Tokenizer/model vocabulary mismatch")
            if model.tokenizer_fingerprint not in (None, tokenizer.fingerprint):
                raise ValueError("Tokenizer/checkpoint fingerprint mismatch")
            model.tokenizer_fingerprint = tokenizer.fingerprint
            fixture_path = (
                Path(__file__).resolve().parents[1] / "experiments/wp05_overfit_fixture.json"
            )
            batch = fixed_batch(json.loads(fixture_path.read_text(encoding="utf-8")), tokenizer)
            ids, mask = batch["input_ids"], batch["attention_mask"]
            source = "wp04_tokenizer_authored_pretrain_sft_fixture"
        else:
            ids = torch.randint(architecture.vocab_size, (2, 9))
            mask = torch.ones_like(ids)
            mask[0, -2:] = 0
            mask[1, :2] = 0
            source = "seeded_synthetic_token_ids"
        ids, mask = ids.to(args.device), mask.to(args.device)
        total = ids.shape[1]
        if not 6 <= total <= architecture.context_length:
            raise ValueError("Probe needs at least six tokens within model context")
        chunks = [3, 1, 2] + [1] * (total - 6)
        results, max_error = [], 0.0
        with torch.inference_mode():
            cache = model.new_cache(ids.shape[0], max_length=total)
            pointers = [
                cache.get_layer(i)[0].untyped_storage().data_ptr()
                for i in range(architecture.num_layers)
            ]
            start = 0
            old_keys = None
            for length in chunks:
                end = start + length
                output = model(ids[:, start:end], mask[:, start:end], use_cache=True, cache=cache)
                expected = model(ids[:, :end], mask[:, :end]).logits[:, start:end]
                error = (output.logits - expected).abs().max().item()
                torch.testing.assert_close(output.logits, expected, rtol=1e-4, atol=1e-5)
                max_error = max(max_error, error)
                results.append({"start": start, "end": end, "max_abs_error": error})
                if cache.sequence_length != end or output.cache is not cache:
                    raise AssertionError("Returned cache/length mismatch")
                if start == 0:
                    old_keys = [
                        cache.get_layer(i)[0].clone() for i in range(architecture.num_layers)
                    ]
                else:
                    for layer, original in enumerate(old_keys):
                        torch.testing.assert_close(
                            cache.get_layer(layer)[0][:, :, :3], original, rtol=0, atol=0
                        )
                start = end
            torch.testing.assert_close(cache.attention_mask, mask.bool(), rtol=0, atol=0)
            torch.testing.assert_close(
                cache.position_ids, torch.arange(total, device=ids.device).expand(ids.shape[0], -1)
            )
            if pointers != [
                cache.get_layer(i)[0].untyped_storage().data_ptr()
                for i in range(architecture.num_layers)
            ]:
                raise AssertionError("Decode reallocated static KV buffers")
            try:
                model(ids[:, :1], use_cache=True, cache=cache)
            except ValueError as error:
                if "capacity/context" not in str(error):
                    raise
            else:
                raise AssertionError("Cache overflow accepted")
            # Request state cannot survive a model reload, even if values are unchanged.
            model.load_state_dict(model.state_dict())
            try:
                model(ids[:, :1], use_cache=True, cache=cache)
            except RuntimeError as error:
                if "weights/metadata changed" not in str(error):
                    raise
            else:
                raise AssertionError("Stale cache accepted after model reload")
        report = {
            "work_package": "WP06.2",
            "status": "passed",
            "device": args.device,
            "model_config": architecture.to_dict(),
            "parameter_counts": model.parameter_counts(),
            "batch_source": source,
            "input_shape": list(ids.shape),
            "tokenizer_fingerprint": model.tokenizer_fingerprint,
            "checkpoint_sha256": checkpoint_sha,
            "tolerance": {"rtol": 1e-4, "atol": 1e-5},
            "max_abs_logit_error": max_error,
            "chunks": results,
            "cache_memory_bytes": cache.memory_bytes(),
            "checks": {
                "prefill_token_chunk_logits_equivalent": True,
                "prefix_keys_unchanged": True,
                "cache_length_masks_positions_correct": True,
                "static_storage_reused": True,
                "capacity_limit_rejected": True,
                "reload_invalidates_cache": True,
            },
            "limitation": (
                "Teacher-forced logits only; sampling, generation equivalence and profiling follow."
            ),
        }
        write_json(run.path / "artifacts/cached_forward_report.json", report)
        run.metadata.update(
            execution_device=args.device,
            tokenizer_fingerprint=model.tokenizer_fingerprint,
            probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        write_json(run.path / "metadata.json", run.metadata)
        run.log("cached_forward_passed", max_abs_logit_error=max_error, checks=report["checks"])
    print(json.dumps({"status": "passed", "run_directory": str(run.path)}))


if __name__ == "__main__":
    main()
