"""Logged corpus pretraining with self-contained, atomic continuation checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
import tomllib
from pathlib import Path

from llm_scratch.data.pretraining import PackedCorpus
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.data import TrainingCursor
from llm_scratch.training.session import PretrainingSession
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import LoadedConfig, config_digest
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def load_recipe(path: Path, *, stage="pretrain") -> LoadedConfig:
    source = path.read_text(encoding="utf-8")
    data = tomllib.loads(source)
    keys = {
        "schema_version",
        "experiment",
        "model",
        "training",
        "data",
        "runner",
        "reproducibility",
        "artifacts",
    }
    if set(data) != keys or type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported pretraining recipe schema")
    for section, expected in {
        "experiment": {"name"},
        "data": {"block_size", "batch_size", "shuffle_seed", "validation_batches"},
        "runner": {"validate_every", "checkpoint_every", "milestones", "max_attempts_per_step"},
        "reproducibility": {"seed", "deterministic", "num_threads"},
        "artifacts": {"root"},
    }.items():
        if not isinstance(data[section], dict) or set(data[section]) != expected:
            raise ValueError(f"Invalid {section} settings")
    if (
        not isinstance(data["experiment"]["name"], str)
        or not data["experiment"]["name"]
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in data["experiment"]["name"]
        )
    ):
        raise ValueError("Invalid experiment name")
    cfg, model = TrainingConfig.from_dict(data["training"]), ModelConfig.from_dict(data["model"])
    if stage not in ("pretrain", "sft") or cfg.stage != stage:
        raise ValueError(f"This runner requires stage={stage}")
    dc, rc, repro = data["data"], data["runner"], data["reproducibility"]
    for label, value, minimum in [
        ("block_size", dc["block_size"], 3),
        ("batch_size", dc["batch_size"], 1),
        ("validation_batches", dc["validation_batches"], 0),
        ("validate_every", rc["validate_every"], 1),
        ("checkpoint_every", rc["checkpoint_every"], 1),
        ("max_attempts_per_step", rc["max_attempts_per_step"], 1),
        ("num_threads", repro["num_threads"], 1),
    ]:
        if type(value) is not int or value < minimum:
            raise ValueError(f"Invalid {label}")
    for value in (dc["shuffle_seed"], repro["seed"]):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError("Seeds must be integers in [0,2**32)")
    if type(repro["deterministic"]) is not bool or dc["block_size"] > model.context_length:
        raise ValueError("Invalid deterministic flag or block/context dimensions")
    milestones = rc["milestones"]
    if (
        not isinstance(milestones, list)
        or any(type(s) is not int or not 1 <= s <= cfg.max_steps for s in milestones)
        or milestones != sorted(set(milestones))
    ):
        raise ValueError("Milestones must be sorted unique successful steps within budget")
    if not isinstance(data["artifacts"]["root"], str) or not data["artifacts"]["root"]:
        raise ValueError("Invalid artifact root")
    data["model"], data["training"] = model.to_dict(), cfg.to_dict()
    data["artifacts"]["root"] = str((path.parent / data["artifacts"]["root"]).resolve())
    return LoadedConfig(data, {"pretraining": source}, path.resolve())


def atomic_text(path: Path, value: str):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def export_curves(history: dict, directory: Path):
    """Rebuild derived metrics from checkpoint-owned history, never from old log tails."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "history.json", history)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["series", "step", "attempt", "loss", "loss_tokens", "skipped"])
    for series, rows in history.items():
        for row in rows:
            writer.writerow(
                [
                    series,
                    row["step"],
                    row["attempt"],
                    row["loss"],
                    row["loss_tokens"],
                    row.get("skipped", False),
                ]
            )
    atomic_text(directory / "losses.csv", output.getvalue())
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    try:
        for series, rows in history.items():
            selected = [r for r in rows if not r.get("skipped", False)]
            if selected:
                ax.plot(
                    [r["step"] for r in selected],
                    [r["loss"] for r in selected],
                    label=series,
                    marker="o" if series == "validation" else None,
                )
        ax.set(
            xlabel="Successful optimizer updates",
            ylabel="Token-weighted cross entropy",
            title="Pretraining loss (validation scope in run config)",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        # The checkpoint is authoritative if interrupted while regenerating these files.
        for suffix in ("png", "svg"):
            fig.savefig(directory / f"loss_curves.{suffix}", dpi=160)
    finally:
        plt.close(fig)


class PretrainingJob:
    """Run-level history and recipe travel inside the full engine/data checkpoint."""

    def __init__(self, session: PretrainingSession, recipe: dict):
        self.session, self.recipe = session, recipe
        self.history = {"train": [], "validation": []}
        # Guard implementation drift as well as recipe/data/runtime drift on exact resume.
        package = Path(__file__).parents[1]
        files = sorted(
            list((package / "model").glob("*.py"))
            + list((package / "training").glob("*.py"))
            + [package / "data/pretraining.py", package / "utils/seed.py"]
        )
        self.contract = {
            "schema_version": 1,
            "recipe": {
                k: recipe[k]
                for k in (
                    "model",
                    "training",
                    "data",
                    "runner",
                    "reproducibility",
                )
            },
            "implementation": {
                p.relative_to(package).as_posix(): hashlib.sha256(
                    p.read_text(encoding="utf-8").encode("utf-8")
                ).hexdigest()
                for p in files
            },
        }

    def validate(self):
        result = self.session.validate(
            max_batches=self.recipe["data"]["validation_batches"] or None
        )
        self.history["validation"].append(
            {
                "step": self.session.engine.step,
                "attempt": self.session.engine.attempts,
                "loss": result["loss"],
                "loss_tokens": result["loss_tokens"],
            }
        )

    def update(self):
        row = self.session.update()
        self.history["train"].append(
            {
                k: row[k]
                for k in (
                    "step",
                    "attempt",
                    "loss",
                    "loss_tokens",
                    "skipped",
                    "learning_rates",
                    "amp_scale",
                    "batch_sha256",
                    "tokens_seen",
                    "tokens_optimized",
                )
            }
        )
        return row

    def save(self, path: Path):
        extra = {
            "contract": self.contract,
            "history": self.history,
            "history_sha256": config_digest(self.history),
        }
        self._validate_extra(extra, self.session.engine.attempts)
        if sum(not r["skipped"] for r in self.history["train"]) != self.session.engine.step:
            raise ValueError("Engine and runner history steps disagree")
        self.session.engine._emit(
            "runner_checkpoint_started", path=str(path), step=self.session.engine.step
        )
        self.session.save(path, extra_state=extra)

    def _validate_extra(self, extra, attempts):
        if (
            not isinstance(extra, dict)
            or set(extra) != {"contract", "history", "history_sha256"}
            or extra["contract"] != self.contract
        ):
            raise ValueError("Runner recipe/implementation mismatch or missing runner state")
        h = extra["history"]
        if (
            not isinstance(h, dict)
            or set(h) != {"train", "validation"}
            or not all(isinstance(v, list) for v in h.values())
            or config_digest(h) != extra["history_sha256"]
            or len(h["train"]) != attempts
        ):
            raise ValueError("Invalid checkpoint-owned loss history")
        step = 0
        for attempt, row in enumerate(h["train"], 1):
            if type(row.get("skipped")) is not bool:
                raise ValueError("Invalid skipped update history")
            step += not row["skipped"]
            if row.get("attempt") != attempt or row.get("step") != step:
                raise ValueError("History progress counters disagree")
        previous = (-1, -1)
        for row in h["validation"]:
            current = (row["step"], row["attempt"])
            if not previous < current <= (step, attempts):
                raise ValueError("Invalid validation history order")
            previous = current
        for rows in h.values():
            for row in rows:
                if (
                    type(row.get("loss")) not in (float, int)
                    or not math.isfinite(row["loss"])
                    or type(row.get("loss_tokens")) is not int
                    or row["loss_tokens"] < 1
                ):
                    raise ValueError("Invalid loss history values")

    def load(self, path: Path):
        extra = self.session.load(path, extra_validator=self._validate_extra)
        self.history = extra["history"]
        if sum(not r["skipped"] for r in self.history["train"]) != self.session.engine.step:
            raise ValueError("Engine and runner history steps disagree")


def run_pretraining(
    config: Path,
    prepared_root: Path,
    tokenizer: Path,
    *,
    device="cpu",
    output_root=None,
    resume: Path | None = None,
    stop_after: int | None = None,
    precision: str | None = None,
) -> Path:
    loaded = load_recipe(config)
    data = loaded.data
    if precision is not None:
        data["training"]["precision"] = precision
    cfg = TrainingConfig.from_dict(data["training"])
    if stop_after is not None and (
        type(stop_after) is not int or not 1 <= stop_after <= cfg.max_steps
    ):
        raise ValueError("stop_after must be an absolute successful step within the fixed budget")
    architecture = ModelConfig.from_dict(data["model"])
    train, validation = (
        PackedCorpus(prepared_root, "train"),
        PackedCorpus(prepared_root, "validation"),
    )
    codec = ByteBPETokenizer.load(
        tokenizer, expected_fingerprint=train.meta["tokenizer_fingerprint"]
    )
    if (
        len(codec) != architecture.vocab_size
        or train.meta["block_size"] != data["data"]["block_size"]
    ):
        raise ValueError(
            "Recipe vocabulary/block_size differs from frozen tokenizer/prepared corpus"
        )
    if output_root is not None:
        data["artifacts"]["root"] = str(Path(output_root).resolve())
    data["execution"] = {
        "device": device,
        "prepared_root": str(prepared_root.resolve()),
        "tokenizer": str(tokenizer.resolve()),
        "resume": str(resume.resolve()) if resume else None,
        "stop_after": stop_after,
    }
    # Fail before training if the plotting dependency is missing.
    import matplotlib  # noqa: F401

    rng = seed_everything(**data["reproducibility"])
    env = inspect_environment(device)
    if not env["validation"]["ok"]:
        raise RuntimeError(env["validation"]["errors"])
    with Run(loaded) as run:
        write_json(run.path / "artifacts/environment.json", env)
        run.metadata.update(
            dataset_fingerprint=train.meta["fingerprint"],
            tokenizer_fingerprint=codec.fingerprint,
            execution_device=device,
        )
        write_json(run.path / "metadata.json", run.metadata)

        def log(record):
            run.log(
                record["event"],
                **{k: v for k, v in record.items() if k not in ("timestamp", "event")},
            )

        engine = TrainingEngine(
            CausalLM(architecture, tokenizer_fingerprint=codec.fingerprint).to(device),
            cfg,
            generator=rng,
            log=log,
        )
        session = PretrainingSession(
            engine,
            TrainingCursor(
                train,
                batch_size=data["data"]["batch_size"],
                accumulation_steps=cfg.gradient_accumulation_steps,
                seed=data["data"]["shuffle_seed"],
            ),
            validation,
        )
        job = PretrainingJob(session, data)
        if resume is not None:
            job.load(resume)
        goal = cfg.max_steps if stop_after is None else stop_after
        if goal < engine.step:
            raise ValueError("stop_after precedes checkpoint progress")
        artifacts, checkpoints = run.path / "artifacts", run.path / "checkpoints"
        codec.save(artifacts / "tokenizer")
        write_json(artifacts / "dataset_manifest.json", train.meta)
        engine._emit(
            "pretraining_started",
            run_directory=str(run.path),
            target_step=goal,
            planned_steps=cfg.max_steps,
            parameter_counts=engine.model.parameter_counts(),
            validation_batches=data["data"]["validation_batches"] or "full",
            resumed_from=str(resume) if resume else None,
        )

        def save_progress(*, milestone=False, final=False):
            # latest.pt is the authoritative transaction: model, RNG, data cursor AND metrics.
            job.save(checkpoints / "latest.pt")
            if milestone or final:
                job.save(checkpoints / ("final.pt" if final else f"step-{engine.step:08d}.pt"))
            export_curves(job.history, artifacts)
            engine._emit(
                "pretraining_progress_saved",
                step=engine.step,
                attempt=engine.attempts,
                latest=str(checkpoints / "latest.pt"),
            )

        if not job.history["validation"]:
            job.validate()
            save_progress()
        else:
            export_curves(job.history, artifacts)
        rc = data["runner"]
        consecutive_skips = 0
        for row in reversed(job.history["train"]):
            if not row["skipped"]:
                break
            consecutive_skips += 1
        try:
            while engine.step < goal:
                if consecutive_skips >= rc["max_attempts_per_step"]:
                    raise RuntimeError("AMP attempt budget exceeded; inspect overflow logs")
                row = job.update()
                if row["skipped"]:
                    consecutive_skips += 1
                    # Keep scaler and consumed windows durable even during prolonged backoff.
                    if engine.attempts % rc["checkpoint_every"] == 0:
                        save_progress()
                    continue
                consecutive_skips = 0
                if engine.step % rc["validate_every"] == 0 or engine.step == goal:
                    job.validate()
                milestone = engine.step in rc["milestones"]
                if engine.step % rc["checkpoint_every"] == 0 or milestone or engine.step == goal:
                    save_progress(milestone=milestone, final=engine.step == cfg.max_steps)
        except KeyboardInterrupt:
            # Interrupts during an update poison the engine/session; preserve the previous latest.
            if not session._failed and not engine._failed and not engine._busy:
                save_progress()
            engine._emit(
                "pretraining_interrupted",
                step=engine.step,
                resume_checkpoint=str(checkpoints / "latest.pt"),
            )
            raise
        except BaseException as error:
            engine._emit(
                "pretraining_error",
                step=engine.step,
                error_type=type(error).__name__,
                error=str(error),
                resume_checkpoint=str(checkpoints / "latest.pt"),
            )
            raise
        # A no-op continuation also produces a complete portable checkpoint in its new run.
        if not (checkpoints / "latest.pt").exists():
            save_progress(final=engine.step == cfg.max_steps)
        status = "completed" if engine.step == cfg.max_steps else "stopped_at_boundary"
        write_json(
            artifacts / "pretraining_report.json",
            {
                "schema_version": 1,
                "work_package": "WP07.3",
                "status": status,
                "step": engine.step,
                "attempts": engine.attempts,
                "planned_steps": cfg.max_steps,
                "tokens_seen": engine.tokens_seen,
                "tokens_optimized": engine.tokens_optimized,
                "source_origin": train.meta["source_origin"],
                "dataset_fingerprint": train.meta["fingerprint"],
                "tokenizer_fingerprint": codec.fingerprint,
                "parameter_counts": engine.model.parameter_counts(),
                "runtime": engine._runtime(),
                "contract": job.contract,
                "history": job.history,
                "validation_batches": data["data"]["validation_batches"] or None,
                "latest_checkpoint": str((checkpoints / "latest.pt").resolve()),
                "scope": "Pretraining only; quality/pilot acceptance requires separate evaluation.",
            },
        )
        engine._emit(
            "pretraining_finished", status=status, step=engine.step, run_directory=str(run.path)
        )
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/wp07_pretrain.toml"))
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    path = run_pretraining(**vars(args))
    print(json.dumps({"event": "run_directory", "path": str(path)}), flush=True)


if __name__ == "__main__":
    main()
