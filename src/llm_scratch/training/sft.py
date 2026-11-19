"""CoT SFT from pretrained weights, with a new optimizer and exact SFT continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.sft import SFTCorpus
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.curves import export_sft_curves
from llm_scratch.training.data import TrainingCursor
from llm_scratch.training.pretrain import PretrainingJob, load_recipe
from llm_scratch.training.sft_session import SFTSession
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def warm_start(model: CausalLM, checkpoint: Path) -> dict:
    """Load only FP32 model weights; never restore pretraining optimizer/RNG/data cursor."""
    digest = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state", {})
    if (
        payload.get("format_version") != 1
        or state.get("training_schema_version") != 1
        or state.get("training_config", {}).get("stage") != "pretrain"
        or type(state.get("step")) is not int
        or state["step"] < 1
        or type(state.get("attempts")) is not int
        or state["attempts"] < state["step"]
    ):
        raise ValueError("SFT initialization needs a trained pretraining engine checkpoint")
    weights = state.get("model", {})
    expected = model.state_dict()
    if weights.get("_extra_state") != expected["_extra_state"] or set(weights) != set(expected):
        raise ValueError("Pretrained architecture/tokenizer must match SFT exactly")
    # Reject all incompatible/nonfinite tensors before changing any model weights.
    for name, value in weights.items():
        if name == "_extra_state":
            continue
        reference = expected[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != reference.shape
            or value.dtype != reference.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid pretrained tensor: {name}")
    if model.config.tie_embeddings and not torch.equal(
        weights["token_embedding.weight"], weights["lm_head.weight"]
    ):
        raise ValueError("Tied pretrained weights disagree")
    model.load_state_dict(weights)
    return {
        "kind": "pretrain_weights_only_fresh_sft_optimizer_rng_cursor",
        "checkpoint_sha256": digest,
        "source_stage": "pretrain",
        "source_step": state["step"],
        "source_attempts": state["attempts"],
        "source_runtime": state.get("runtime"),
        "source_tokens_optimized": state.get("tokens_optimized"),
    }


def resume_lineage(checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state", {})
    lineage = (
        state.get("data_state", {}).get("extra_state", {}).get("contract", {}).get("initialization")
    )
    if (
        payload.get("format_version") != 1
        or state.get("training_config", {}).get("stage") != "sft"
        or not isinstance(lineage, dict)
        or lineage.get("source_stage") != "pretrain"
        or not re.fullmatch(r"[a-f0-9]{64}", str(lineage.get("checkpoint_sha256")))
    ):
        raise ValueError("Need a full SFT runner checkpoint to resume")
    return lineage


class SFTJob(PretrainingJob):
    def __init__(self, session, recipe, lineage):
        super().__init__(session, recipe)
        package = Path(__file__).parents[1]
        self.contract["stage"] = "sft"
        self.contract["initialization"] = lineage
        for relative in (
            "data/sft.py",
            "data/collator.py",
            "reasoning/formatting.py",
            "reasoning/answer_extraction.py",
        ):
            self.contract["implementation"][relative] = hashlib.sha256(
                (package / relative).read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()

    def validate(self):
        self.session.engine._emit(
            "sft_validation_started",
            step=self.session.engine.step,
            max_batches=self.recipe["data"]["validation_batches"] or "full",
        )
        super().validate()


def run_sft(
    config: Path,
    prepared_root: Path,
    tokenizer: Path,
    *,
    init_checkpoint=None,
    resume=None,
    device="cpu",
    precision=None,
    stop_after=None,
    output_root=None,
) -> Path:
    if (init_checkpoint is None) == (resume is None):
        raise ValueError("Use exactly one of init_checkpoint (new SFT) or resume (full SFT)")
    loaded = load_recipe(config, stage="sft")
    data = loaded.data
    if precision is not None:
        data["training"]["precision"] = precision
    cfg = TrainingConfig.from_dict(data["training"])
    goal = cfg.max_steps if stop_after is None else stop_after
    if type(goal) is not int or not 1 <= goal <= cfg.max_steps:
        raise ValueError(
            "stop_after must be an absolute successful step within the fixed SFT budget"
        )
    codec = ByteBPETokenizer.load(tokenizer)
    train, validation = (
        SFTCorpus(prepared_root, "train", codec),
        SFTCorpus(prepared_root, "validation", codec),
    )
    if train.meta["block_size"] != data["data"]["block_size"]:
        raise ValueError("SFT recipe max length differs from prepared data")
    architecture = ModelConfig.from_dict(data["model"])
    if architecture.vocab_size != len(codec):
        raise ValueError("SFT vocabulary must match frozen tokenizer")
    if output_root is not None:
        data["artifacts"]["root"] = str(Path(output_root).resolve())
    data["execution"] = {
        "device": device,
        "prepared_root": str(prepared_root.resolve()),
        "tokenizer": str(tokenizer.resolve()),
        "stop_after": stop_after,
        "init_checkpoint": str(Path(init_checkpoint).resolve()) if init_checkpoint else None,
        "resume": str(Path(resume).resolve()) if resume else None,
    }
    import matplotlib  # noqa: F401

    rng = seed_everything(**data["reproducibility"])
    env = inspect_environment(device)
    if not env["validation"]["ok"]:
        raise RuntimeError(env["validation"]["errors"])
    model = CausalLM(architecture, tokenizer_fingerprint=codec.fingerprint)
    lineage = (
        warm_start(model, Path(init_checkpoint))
        if init_checkpoint
        else resume_lineage(Path(resume))
    )
    data["initialization"] = lineage
    with Run(loaded) as run:
        write_json(run.path / "artifacts/environment.json", env)
        write_json(run.path / "artifacts/dataset_manifest.json", train.meta)
        codec.save(run.path / "artifacts/tokenizer")
        run.metadata.update(
            initialization=lineage,
            dataset_fingerprint=train.meta["fingerprint"],
            tokenizer_fingerprint=codec.fingerprint,
            execution_device=device,
        )
        write_json(run.path / "metadata.json", run.metadata)

        def log(record):
            run.log(
                record["event"],
                **{k: v for k, v in record.items() if k not in ("event", "timestamp")},
            )

        engine = TrainingEngine(model.to(device), cfg, generator=rng, log=log)
        session = SFTSession(
            engine,
            TrainingCursor(
                train,
                batch_size=data["data"]["batch_size"],
                accumulation_steps=cfg.gradient_accumulation_steps,
                seed=data["data"]["shuffle_seed"],
            ),
            validation,
        )
        job = SFTJob(session, data, lineage)
        if resume:
            job.load(Path(resume))
        if engine.step > goal:
            raise ValueError("stop_after precedes SFT checkpoint progress")
        engine._emit(
            "sft_started",
            run_directory=str(run.path),
            target_step=goal,
            planned_steps=cfg.max_steps,
            initialization=lineage,
            parameter_counts=model.parameter_counts(),
        )
        artifacts, checkpoints = run.path / "artifacts", run.path / "checkpoints"

        def save_progress(*, milestone=False, final=False):
            job.save(checkpoints / "latest.pt")
            if milestone or final:
                job.save(checkpoints / ("final.pt" if final else f"step-{engine.step:08d}.pt"))
            export_sft_curves(job.history, artifacts)
            engine._emit(
                "sft_progress_saved",
                step=engine.step,
                attempt=engine.attempts,
                latest=str(checkpoints / "latest.pt"),
            )

        rc = data["runner"]
        skips = 0
        for row in reversed(job.history["train"]):
            if not row["skipped"]:
                break
            skips += 1
        try:
            if not job.history["validation"]:
                job.validate()
                save_progress()
            else:
                # Commit a portable copy before starting continuation in this new run.
                save_progress(final=engine.step == cfg.max_steps)
            while engine.step < goal:
                if skips >= rc["max_attempts_per_step"]:
                    raise RuntimeError("SFT AMP attempt budget exceeded; inspect overflow logs")
                row = job.update()
                if row["skipped"]:
                    skips += 1
                    if engine.attempts % rc["checkpoint_every"] == 0:
                        save_progress()
                    continue
                skips = 0
                if engine.step % rc["validate_every"] == 0 or engine.step == goal:
                    job.validate()
                milestone = engine.step in rc["milestones"]
                if engine.step % rc["checkpoint_every"] == 0 or milestone or engine.step == goal:
                    save_progress(milestone=milestone, final=engine.step == cfg.max_steps)
        except KeyboardInterrupt:
            if not session._failed and not engine._failed and not engine._busy:
                save_progress()
            engine._emit(
                "sft_interrupted",
                step=engine.step,
                resume_checkpoint=str(checkpoints / "latest.pt"),
            )
            raise
        except BaseException as error:
            engine._emit(
                "sft_error",
                step=engine.step,
                error_type=type(error).__name__,
                error=str(error),
                resume_checkpoint=str(checkpoints / "latest.pt"),
            )
            raise
        status = "completed" if engine.step == cfg.max_steps else "stopped_at_boundary"
        write_json(
            artifacts / "sft_report.json",
            {
                "schema_version": 1,
                "work_package": "WP07.4",
                "status": status,
                "step": engine.step,
                "attempts": engine.attempts,
                "planned_steps": cfg.max_steps,
                "tokens_seen": engine.tokens_seen,
                "tokens_optimized": engine.tokens_optimized,
                "source_origin": train.meta["source_origin"],
                "dataset_fingerprint": train.meta["fingerprint"],
                "tokenizer_fingerprint": codec.fingerprint,
                "initialization": lineage,
                "parameter_counts": model.parameter_counts(),
                "runtime": engine._runtime(),
                "contract": job.contract,
                "history": job.history,
                "validation_batches": data["data"]["validation_batches"] or None,
                "latest_checkpoint": str((checkpoints / "latest.pt").resolve()),
                "scope": (
                    "Supervised CoT train/validation only; "
                    "generation/pilot capability follows WP07.5."
                ),
            },
        )
        engine._emit("sft_finished", status=status, step=engine.step, run_directory=str(run.path))
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/wp07_sft.toml"))
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--init-checkpoint", type=Path)
    initialization.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--stop-after", type=int)
    path = run_sft(**vars(parser.parse_args()))
    print(json.dumps({"event": "run_directory", "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
