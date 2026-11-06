"""WP05 acceptance runner: frozen recipe/batch, diagnostic optimization and evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tomllib
from pathlib import Path

import torch

from llm_scratch.data.collator import collate_tokenized, encode_pretraining
from llm_scratch.data.config import _fields
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.architecture import audit_model
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.reasoning.formatting import encode_sft_example
from llm_scratch.training.checkpoint import load_checkpoint, save_checkpoint
from llm_scratch.training.overfit import overfit_batch
from llm_scratch.utils.config import LoadedConfig, config_digest
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def load_acceptance_config(
    path: str | Path,
    *,
    profile: str = "tiny",
    device: str = "cpu",
    tokenizer_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> LoadedConfig:
    path = Path(path).resolve()
    source = path.read_text(encoding="utf-8")
    data = tomllib.loads(source)
    _fields(
        data,
        {
            "schema_version": int,
            "fixture": str,
            "experiment": dict,
            "reproducibility": dict,
            "artifacts": dict,
            "profiles": dict,
            "tokenizer": dict,
            "optimization": dict,
            "acceptance": dict,
        },
        "acceptance config",
    )
    if data["schema_version"] != 1 or device not in ("cpu", "cuda"):
        raise ValueError("Unsupported schema/device")
    _fields(data["experiment"], {"name": str, "phase": int}, "experiment")
    if data["experiment"]["phase"] != 1 or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", data["experiment"]["name"]
    ):
        raise ValueError("Invalid experiment phase/name")
    _fields(
        data["reproducibility"],
        {"seed": int, "deterministic": bool, "num_threads": int},
        "reproducibility",
    )
    repro = data["reproducibility"]
    if not 0 <= repro["seed"] < 2**32 or repro["num_threads"] < 1 or not repro["deterministic"]:
        raise ValueError("Acceptance requires valid seed/threads and deterministic=true")
    _fields(data["artifacts"], {"root": str}, "artifacts")
    if not data["artifacts"]["root"].strip():
        raise ValueError("Empty artifacts root")
    _fields(data["profiles"], {"tiny": dict, "candidate": dict}, "profiles")
    if profile not in data["profiles"]:
        raise ValueError("Unknown profile")
    for spec in data["profiles"].values():
        _fields(spec, {"model_config": str, "learning_rate": float}, "profile")
        if not math.isfinite(spec["learning_rate"]) or spec["learning_rate"] <= 0:
            raise ValueError("Invalid learning rate")
    _fields(data["tokenizer"], {"fixture_vocab_size": int, "min_frequency": int}, "tokenizer")
    if data["tokenizer"]["fixture_vocab_size"] < 264 or data["tokenizer"]["min_frequency"] < 1:
        raise ValueError("Invalid fixture tokenizer settings")
    _fields(
        data["optimization"],
        {"steps": int, "betas": list, "weight_decay": float, "max_grad_norm": float},
        "optimization",
    )
    opt = data["optimization"]
    if (
        opt["steps"] < 1
        or not math.isfinite(opt["weight_decay"])
        or opt["weight_decay"] < 0
        or not math.isfinite(opt["max_grad_norm"])
        or opt["max_grad_norm"] <= 0
        or len(opt["betas"]) != 2
        or any(type(beta) not in (int, float) or not 0 <= beta < 1 for beta in opt["betas"])
    ):
        raise ValueError("Invalid optimization settings")
    _fields(
        data["acceptance"], {"max_loss_ratio": float, "min_token_accuracy": float}, "acceptance"
    )
    if (
        not 0 < data["acceptance"]["max_loss_ratio"] < 1
        or not 0 < data["acceptance"]["min_token_accuracy"] <= 1
    ):
        raise ValueError("Invalid acceptance thresholds")
    if profile == "candidate" and tokenizer_path is None:
        raise ValueError("Candidate profile requires a frozen WP04 tokenizer artifact")
    model_path = (path.parent / data["profiles"][profile]["model_config"]).resolve()
    model_source = model_path.read_text(encoding="utf-8")
    component = tomllib.loads(model_source)
    _fields(component, {"model": dict}, "model component")
    architecture = ModelConfig.from_dict(component["model"])
    if architecture.dropout != 0:
        raise ValueError("Acceptance requires dropout=0")
    data["model"] = architecture.to_dict()
    data["profile"], data["device"] = profile, device
    data["learning_rate"] = data["profiles"][profile]["learning_rate"]
    data["fixture"] = str((path.parent / data["fixture"]).resolve())
    data["artifacts"]["root"] = (
        str(Path(output_root).resolve())
        if output_root is not None
        else str((path.parent / data["artifacts"]["root"]).resolve())
    )
    data["tokenizer_path"] = (
        str(Path(tokenizer_path).resolve()) if tokenizer_path is not None else None
    )
    data["experiment"]["name"] += f"-{profile}"
    return LoadedConfig(data, {"acceptance": source, "model": model_source}, path)


def fixture_texts(fixture: dict) -> list[str]:
    _fields(fixture, {"schema_version": int, "purpose": str, "records": list}, "fixture")
    if fixture["schema_version"] != 1 or not fixture["records"]:
        raise ValueError("Need a nonempty diagnostic fixture")
    texts = []
    for record in fixture["records"]:
        if not isinstance(record, dict):
            raise ValueError("Fixture records must be objects")
        if record.get("stage") == "pretrain":
            _fields(record, {"stage": str, "text": str}, "pretrain record")
            texts.append(record["text"])
        elif record.get("stage") == "sft":
            _fields(
                record,
                {"stage": str, "question": str, "reasoning": str, "answer": str},
                "sft record",
            )
            texts.extend(record[key] for key in ("question", "reasoning", "answer"))
        else:
            raise ValueError("Unknown fixture stage")
    return texts


def fixed_batch(fixture: dict, tokenizer: ByteBPETokenizer) -> dict[str, torch.Tensor]:
    fixture_texts(fixture)
    examples = []
    for record in fixture["records"]:
        if record["stage"] == "pretrain":
            examples.append(encode_pretraining(tokenizer, record["text"]))
        else:
            examples.append(
                encode_sft_example(
                    tokenizer, record["question"], record["reasoning"], record["answer"]
                )
            )
    return {
        name: torch.tensor(value) for name, value in collate_tokenized(examples, tokenizer).items()
    }


def run_acceptance(config: LoadedConfig) -> Path:
    data = config.data
    fixture_bytes = Path(data["fixture"]).read_bytes()
    fixture = json.loads(fixture_bytes)
    texts = fixture_texts(fixture)
    if data["tokenizer_path"]:
        tokenizer = ByteBPETokenizer.load(data["tokenizer_path"])
    else:
        tokenizer = ByteBPETokenizer.train(
            texts,
            vocab_size=data["tokenizer"]["fixture_vocab_size"],
            min_frequency=data["tokenizer"]["min_frequency"],
        )
    architecture = ModelConfig.from_dict(data["model"])
    if architecture.vocab_size != len(tokenizer):
        raise ValueError("Model vocab size does not match actual tokenizer")
    cpu_batch = fixed_batch(fixture, tokenizer)
    if cpu_batch["input_ids"].shape[1] + 2 > architecture.context_length:
        raise ValueError("Fixture plus padding audit exceeds model context")
    batch_payload = {name: tensor.tolist() for name, tensor in cpu_batch.items()}
    data["fixture_sha256"] = hashlib.sha256(fixture_bytes).hexdigest()
    data["batch_sha256"] = config_digest(batch_payload)
    data["tokenizer_fingerprint"] = tokenizer.fingerprint
    with Run(config) as run:
        generator = seed_everything(**data["reproducibility"])
        environment = inspect_environment(data["device"])
        write_json(run.path / "artifacts/environment.json", environment)
        if not environment["validation"]["ok"]:
            raise RuntimeError(environment["validation"]["errors"])
        write_json(run.path / "config_sources/fixture.json", fixture)
        write_json(run.path / "artifacts/fixed_batch.json", batch_payload)
        tokenizer.save(run.path / "artifacts/tokenizer")
        batch = {name: tensor.to(data["device"]) for name, tensor in cpu_batch.items()}
        model = CausalLM(architecture, tokenizer_fingerprint=tokenizer.fingerprint).to(
            data["device"]
        )
        before = audit_model(model, batch)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=data["learning_rate"],
            betas=tuple(data["optimization"]["betas"]),
            weight_decay=data["optimization"]["weight_decay"],
            foreach=False,
            fused=False,
        )

        def progress(record):
            run.log("overfit_step", **record)
            if record["step"] % 25 == 0:
                print(
                    json.dumps(
                        {
                            "step": record["step"],
                            "loss": record["loss"],
                            "token_accuracy": record["token_accuracy"],
                        }
                    ),
                    flush=True,
                )

        history = overfit_batch(
            model,
            batch,
            optimizer,
            steps=data["optimization"]["steps"],
            max_grad_norm=data["optimization"]["max_grad_norm"],
            progress=progress,
        )
        after = audit_model(model, batch)
        if (
            config_digest({name: tensor.cpu().tolist() for name, tensor in batch.items()})
            != data["batch_sha256"]
        ):
            raise AssertionError("Diagnostic batch was mutated")
        ratio = history[-1]["loss"] / history[0]["loss"]
        passed = (
            ratio <= data["acceptance"]["max_loss_ratio"]
            and history[-1]["token_accuracy"] >= data["acceptance"]["min_token_accuracy"]
        )
        report = {
            "work_package": "WP05.5",
            "status": "pending_checkpoint" if passed else "failed",
            "profile": data["profile"],
            "device": data["device"],
            "model_config": architecture.to_dict(),
            "parameter_counts": model.parameter_counts(),
            "fixture_sha256": data["fixture_sha256"],
            "batch_sha256": data["batch_sha256"],
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "input_shape": list(batch["input_ids"].shape),
            "initial": history[0],
            "final": history[-1],
            "loss_ratio": ratio,
            "acceptance": data["acceptance"],
            "optimizer_recipe": {
                "name": "AdamW",
                "learning_rate": data["learning_rate"],
                **data["optimization"],
                "foreach": False,
                "fused": False,
            },
            "finite_gradients_all_steps": True,
            "before_training": before,
            "after_training": after,
            "history": history,
            "protocol": (
                "Exactly the configured steps on an authored fixed batch; final step only, "
                "no best-checkpoint selection; no validation/test data."
            ),
        }
        write_json(run.path / "artifacts/acceptance_report.json", report)
        if not passed:
            raise AssertionError("Fixed diagnostic overfit acceptance thresholds were not met")
        checkpoint = run.path / "checkpoints/overfit.pt"
        save_checkpoint(
            checkpoint,
            {
                "step": data["optimization"]["steps"],
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "batch_sha256": data["batch_sha256"],
            },
            generator=generator,
        )
        restored = (
            CausalLM(architecture, tokenizer_fingerprint=tokenizer.fingerprint)
            .to(data["device"])
            .eval()
        )
        state = load_checkpoint(checkpoint, generator=generator)
        if (
            state["step"] != data["optimization"]["steps"]
            or state["batch_sha256"] != data["batch_sha256"]
        ):
            raise AssertionError("Checkpoint step/batch fingerprint mismatch")
        restored.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                restored(**batch).logits, model(**batch).logits, rtol=0, atol=0
            )
        report["checkpoint_round_trip_exact"] = True
        report["status"] = "passed"
        write_json(run.path / "artifacts/acceptance_report.json", report)
        write_json(
            run.path / "result.json",
            {
                "status": "passed",
                "report": "artifacts/acceptance_report.json",
                "loss_ratio": ratio,
                "final_loss": history[-1]["loss"],
            },
        )
        run.metadata.update(
            execution_device=data["device"],
            tokenizer_fingerprint=tokenizer.fingerprint,
            batch_sha256=data["batch_sha256"],
        )
        write_json(run.path / "metadata.json", run.metadata)
        run.log(
            "architecture_accepted", loss_ratio=ratio, token_accuracy=history[-1]["token_accuracy"]
        )
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/wp05_acceptance.toml")
    )
    parser.add_argument("--profile", choices=("tiny", "candidate"), default="tiny")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_acceptance_config(
        args.config,
        profile=args.profile,
        device=args.device,
        tokenizer_path=args.tokenizer,
        output_root=args.output_root,
    )
    run = run_acceptance(config)
    print(json.dumps({"status": "passed", "run_directory": str(run)}))


if __name__ == "__main__":
    main()
