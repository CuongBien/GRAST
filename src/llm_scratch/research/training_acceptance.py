"""Short WP07.1 engine/resume diagnostic; stdout logs are mandatory, never a corpus run."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import tomllib
from pathlib import Path

import torch

from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.research.architecture_acceptance import fixed_batch
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import LoadedConfig
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def _same(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/wp07_training_smoke.toml")
    )
    parser.add_argument("--profile", choices=("tiny", "candidate"), default="tiny")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    path = args.config.resolve()
    source = path.read_text(encoding="utf-8")
    data = tomllib.loads(source)
    if (
        set(data)
        != {
            "schema_version",
            "fixture",
            "experiment",
            "profiles",
            "training",
            "diagnostic",
            "reproducibility",
            "artifacts",
        }
        or data["schema_version"] != 1
    ):
        raise ValueError("Invalid training diagnostic configuration")
    model_source = (path.parent / data.pop("profiles")[args.profile]).read_text(encoding="utf-8")
    architecture = ModelConfig.from_dict(
        tomllib.loads(model_source)["model"] | {"dropout": data["diagnostic"]["dropout"]}
    )
    data["fixture"] = str((path.parent / data["fixture"]).resolve())
    if args.precision:
        data["training"]["precision"] = args.precision
    cfg = TrainingConfig.from_dict(data["training"])
    if (
        cfg.gradient_accumulation_steps != 2
        or not 1 <= data["diagnostic"]["checkpoint_step"] < cfg.max_steps
    ):
        raise ValueError("Diagnostic uses two microbatches and an interior resume checkpoint")
    data.update(
        model=architecture.to_dict(),
        execution={
            "device": args.device,
            "profile": args.profile,
            "tokenizer": str(args.tokenizer.resolve()) if args.tokenizer else None,
        },
    )
    data["artifacts"]["root"] = str(
        args.output_root.resolve()
        if args.output_root
        else (path.parent / data["artifacts"]["root"]).resolve()
    )
    loaded = LoadedConfig(data, {"experiment": source, "model": model_source}, path)
    with Run(loaded) as run:
        rng = seed_everything(**data["reproducibility"])
        env = inspect_environment(args.device)
        write_json(run.path / "artifacts/environment.json", env)
        print(
            json.dumps(
                {
                    "event": "environment",
                    "selected_device": env["selected_device"],
                    "torch": env["pytorch_version"],
                    "gpus": env["gpus"],
                }
            ),
            flush=True,
        )
        if not env["validation"]["ok"]:
            raise RuntimeError(env["validation"]["errors"])
        fingerprint = None
        if args.tokenizer:
            codec = ByteBPETokenizer.load(args.tokenizer)
            if len(codec) != architecture.vocab_size:
                raise ValueError("Tokenizer vocabulary differs from model")
            fingerprint = codec.fingerprint
            fixture_path = Path(data["fixture"])
            batch = fixed_batch(json.loads(fixture_path.read_text(encoding="utf-8")), codec)
            microbatches = [{k: v[i : i + 1] for k, v in batch.items()} for i in range(2)]
            fixture_sha = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
        else:
            if args.profile == "candidate":
                raise ValueError("Candidate diagnostic requires the actual WP04 tokenizer")
            ids = torch.tensor([[1, 9, 10, 11, 2], [1, 12, 13, 2, 0]])
            labels = ids.clone()
            labels[1, :3] = -100
            labels[1, 4] = -100
            mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
            microbatches = [
                dict(
                    input_ids=ids[i : i + 1],
                    labels=labels[i : i + 1],
                    attention_mask=mask[i : i + 1],
                )
                for i in range(2)
            ]
            fixture_sha = None

        def log(record):
            run.log(
                record["event"],
                **{k: v for k, v in record.items() if k not in ("event", "timestamp")},
            )

        model = CausalLM(architecture, tokenizer_fingerprint=fingerprint).to(args.device)
        engine = TrainingEngine(model, cfg, generator=rng, log=log)
        initial = engine.evaluate(microbatches)
        history, expected = [], []
        checkpoint = run.path / "checkpoints/resume.pt"
        checkpoint_attempt = None
        while engine.step < cfg.max_steps:
            if engine.attempts >= cfg.max_steps * 16:
                raise RuntimeError("Too many AMP-skipped attempts; inspect printed scale/gradients")
            update = engine.train_update(microbatches)
            if checkpoint_attempt is not None:
                expected.append(update)
            if update["skipped"]:
                continue  # Normal FP16 scaler backoff is logged and must replay identically.
            validation = engine.evaluate(microbatches)
            history.append(
                {
                    "step": engine.step,
                    "train_loss": update["loss"],
                    "diagnostic_eval_loss": validation["loss"],
                    "lr": update["learning_rates"][0],
                }
            )
            if engine.step == data["diagnostic"]["checkpoint_step"]:
                checkpoint_attempt = engine.attempts
                engine.save(
                    checkpoint,
                    data_state={"fixture_sha256": fixture_sha, "next_window": checkpoint_attempt},
                )
        final = engine.evaluate(microbatches)
        expected_model = copy.deepcopy(engine.model.state_dict())
        expected_optimizer = copy.deepcopy(engine.optimizer.state_dict())
        expected_scheduler = copy.deepcopy(engine.scheduler.state_dict())
        expected_scaler = copy.deepcopy(engine.scaler.state_dict())
        # Release the full candidate + Adam slots before constructing the resume copy.
        counts = engine.model.parameter_counts()
        del engine, model
        model = CausalLM(architecture, tokenizer_fingerprint=fingerprint).to(args.device)
        resumed = TrainingEngine(model, cfg, generator=rng, log=log)
        cursor = resumed.load(checkpoint)
        if cursor != {
            "fixture_sha256": fixture_sha,
            "next_window": checkpoint_attempt,
        }:
            raise AssertionError("Diagnostic cursor did not round-trip")
        for reference in expected:
            actual = resumed.train_update(microbatches)
            if any(
                actual[k] != reference[k]
                for k in ("loss", "learning_rates", "skipped", "amp_scale")
            ):
                raise AssertionError("Resumed update loss/LR differs from uninterrupted run")
        if not all(
            (
                _same(expected_model, resumed.model.state_dict()),
                _same(expected_optimizer, resumed.optimizer.state_dict()),
                _same(expected_scheduler, resumed.scheduler.state_dict()),
                _same(expected_scaler, resumed.scaler.state_dict()),
            )
        ):
            raise AssertionError("Resumed model/optimizer/scheduler/scaler state differs")
        if not final["loss"] < initial["loss"]:
            raise AssertionError("Short diagnostic did not reduce evaluation loss")
        resumed.save(
            run.path / "checkpoints/final.pt",
            data_state={"fixture_sha256": fixture_sha, "next_window": resumed.attempts},
        )
        write_json(run.path / "artifacts/loss_history.json", history)
        with (run.path / "artifacts/loss_history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        report = {
            "work_package": "WP07.1",
            "status": "passed",
            "precision": cfg.precision,
            "device": args.device,
            "parameter_counts": counts,
            "training_config": cfg.to_dict(),
            "model_config": architecture.to_dict(),
            "tokenizer_fingerprint": fingerprint,
            "initial_loss": initial["loss"],
            "final_loss": final["loss"],
            "updates": cfg.max_steps,
            "attempts": resumed.attempts,
            "checks": {
                "loss_decreased": True,
                "exact_resume_model_optimizer_scheduler_scaler": True,
                "update_losses_and_lrs_exact": True,
                "stdout_logs_always": True,
            },
            "limitation": (
                "Fixed diagnostic windows, not corpus pretraining/SFT or held-out pilot; "
                "data ordering/packing follows in WP07.2."
            ),
        }
        write_json(run.path / "artifacts/training_engine_report.json", report)
        run.metadata.update(
            execution_device=args.device,
            tokenizer_fingerprint=fingerprint,
            acceptance_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        write_json(run.path / "metadata.json", run.metadata)
        run.log("training_engine_passed", checks=report["checks"])
    print(json.dumps({"status": "passed", "run_directory": str(run.path)}), flush=True)


if __name__ == "__main__":
    main()
