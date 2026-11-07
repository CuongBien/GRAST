"""WP06.3 uncached greedy/sampled generation, EOS bookkeeping and seeded replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from dataclasses import replace
from pathlib import Path

import torch

from llm_scratch.data.collator import collate_tokenized
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import generate
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.reasoning.formatting import encode_reasoning_prompt
from llm_scratch.utils.config import load_config
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/local_smoke.toml"))
    parser.add_argument(
        "--policy", type=Path, default=Path("configs/sampling/wp06_generation.toml")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    source = args.policy.read_text(encoding="utf-8")
    policy_data = tomllib.loads(source)
    if set(policy_data) != {"generation"}:
        raise ValueError("Runtime policy file must contain only [generation]")
    policy = GenerationConfig.from_dict(policy_data["generation"])
    config = load_config(
        args.config, device=args.device, output_root=args.output_root, seed=args.seed
    )
    config.data["experiment"]["name"] = "wp06-3-generation"
    architecture = ModelConfig.from_dict(config.data["model"])
    with Run(config) as run:
        (run.path / "config_sources/runtime_generation.toml").write_text(source, encoding="utf-8")
        write_json(run.path / "artifacts/generation_config.json", policy.to_dict())
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
        tokenizer = ByteBPETokenizer.load(args.tokenizer) if args.tokenizer else None
        if tokenizer:
            if len(tokenizer) != architecture.vocab_size:
                raise ValueError("Tokenizer/model vocabulary mismatch")
            if model.tokenizer_fingerprint not in (None, tokenizer.fingerprint):
                raise ValueError("Tokenizer/checkpoint fingerprint mismatch")
            if (policy.pad_token_id, policy.eos_token_id) != (
                tokenizer.pad_token_id,
                tokenizer.eos_token_id,
            ):
                raise ValueError("Policy special token IDs differ from tokenizer")
            model.tokenizer_fingerprint = tokenizer.fingerprint
            fixture_path = (
                Path(__file__).resolve().parents[1] / "experiments/wp06_generation_fixture.json"
            )
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            examples = [
                {"input_ids": tokenizer.encode(fixture["story_prefix"], add_bos=True)},
                {"input_ids": encode_reasoning_prompt(tokenizer, fixture["question"])},
            ]
            collated = collate_tokenized(examples, tokenizer)
            ids = torch.tensor(collated["input_ids"], device=args.device)
            mask = torch.tensor(collated["attention_mask"], device=args.device)
            batch_source = "wp04_tokenizer_authored_story_reasoning_prefixes"
            fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        else:
            ids = torch.tensor([[1, 9, 10, 0], [0, 1, 11, 12]], device=args.device)
            mask = torch.tensor([[1, 1, 1, 0], [0, 1, 1, 1]], device=args.device)
            batch_source, fixture_sha = "fixed_synthetic_prompt_ids", None
        seed = config.data["reproducibility"]["seed"]
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
        outputs = {}
        for name, settings in (
            ("greedy", replace(policy, do_sample=False)),
            ("sampled", replace(policy, do_sample=True)),
        ):
            output = generate(model, ids, mask, config=settings, seed=seed)
            replay = generate(model, ids, mask, config=settings, seed=seed)
            for field in ("sequences", "attention_mask", "generated_lengths"):
                torch.testing.assert_close(
                    getattr(output, field), getattr(replay, field), rtol=0, atol=0
                )
            if output.finish_reasons != replay.finish_reasons:
                raise AssertionError("Replay finish reasons differ")
            for row, length in enumerate(output.generated_lengths.tolist()):
                tokens = output.generated_ids[row, :length]
                if output.generated_mask[row, length:].any():
                    raise AssertionError("Mask contains tokens after row finished")
                if output.finish_reasons[row] == "eos":
                    if (
                        tokens[-1] != settings.eos_token_id
                        or (tokens[:-1] == settings.eos_token_id).any()
                    ):
                        raise AssertionError("EOS bookkeeping mismatch")
                elif length != settings.max_new_tokens:
                    raise AssertionError("Budget bookkeeping mismatch")
            generated = [
                output.generated_ids[i, :n].tolist()
                for i, n in enumerate(output.generated_lengths.tolist())
            ]
            outputs[name] = {
                "settings": settings.to_dict(),
                "generated_ids": generated,
                "generated_mask": output.generated_mask.tolist(),
                "generated_lengths": output.generated_lengths.tolist(),
                "finish_reasons": list(output.finish_reasons),
                "decoded": [tokenizer.decode(tokens) for tokens in generated]
                if tokenizer
                else None,
                "seeded_replay_exact": True,
            }
        if not torch.equal(cpu_rng, torch.get_rng_state()) or any(
            not torch.equal(a, b)
            for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)
        ):
            raise AssertionError("Generation advanced global RNG")
        report = {
            "work_package": "WP06.3",
            "status": "passed",
            "device": args.device,
            "backend": "uncached_full_prefix",
            "seed": seed,
            "model_config": architecture.to_dict(),
            "parameter_counts": model.parameter_counts(),
            "batch_source": batch_source,
            "input_ids": ids.tolist(),
            "attention_mask": mask.tolist(),
            "checkpoint_sha256": checkpoint_sha,
            "tokenizer_fingerprint": model.tokenizer_fingerprint,
            "fixture_sha256": fixture_sha,
            "outputs": outputs,
            "checks": {
                "greedy_and_sampled_seeded_replay": True,
                "eos_mask_length_bookkeeping": True,
                "global_rng_unchanged": True,
            },
            "limitation": (
                "Diagnostic generation only; no benchmark quality claim, "
                "cached equivalence or latency/VRAM profiling."
            ),
        }
        write_json(run.path / "artifacts/generation_report.json", report)
        run.metadata.update(
            execution_device=args.device,
            tokenizer_fingerprint=model.tokenizer_fingerprint,
            runtime_policy_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        write_json(run.path / "metadata.json", run.metadata)
        run.log(
            "generation_passed",
            checks=report["checks"],
            generated_lengths={k: v["generated_lengths"] for k, v in outputs.items()},
        )
    print(json.dumps({"status": "passed", "run_directory": str(run.path)}))


if __name__ == "__main__":
    main()
