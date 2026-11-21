"""Cumulative copy -> single -> multi curriculum with exact resume and generated selection."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import tomllib
from pathlib import Path

import torch

from llm_scratch.curriculum.data import TASKS, Corpus, audit
from llm_scratch.curriculum.evaluation import (
    export_selected,
    generate_validation,
    score,
    select_validation,
    snapshot,
    validate_weights,
    weights_digest,
)
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.checkpoints import load_for_evaluation
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.curves import export_sft_curves
from llm_scratch.training.data import TrainingCursor, window_fingerprint
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import LoadedConfig, canonical_json, config_digest
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def load_recipe(path):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    d = tomllib.loads(source)
    sections = {
        "experiment": {"name"},
        "initialization": {"sha256"},
        "data": {"batch_size", "max_length", "shuffle_seed"},
        "runner": {"validate_every", "checkpoint_every", "max_attempts_per_step"},
        "evaluation": {"per_task", "batch_size", "max_new_tokens", "precision", "selection_seed"},
        "reproducibility": {"seed", "deterministic", "num_threads"},
        "artifacts": {"root"},
    }
    if (
        set(d) != {"schema_version", "model", "training", "phases", *sections}
        or type(d["schema_version"]) is not int
        or d["schema_version"] != 1
    ):
        raise ValueError("Invalid curriculum recipe schema")
    for k, v in sections.items():
        if not isinstance(d[k], dict) or set(d[k]) != v:
            raise ValueError(f"Unknown/missing {k} recipe fields")
    cfg = TrainingConfig.from_dict(d["training"])
    architecture = ModelConfig.from_dict(d["model"])
    if cfg.stage != "sft" or not re.fullmatch(r"[a-f0-9]{64}", d["initialization"]["sha256"]):
        raise ValueError("Need SFT settings and an explicit source SHA256")
    if not isinstance(d["phases"], list) or len(d["phases"]) != 3:
        raise ValueError("Need the three ordered cumulative curriculum phases")
    previous = 0
    for i, p in enumerate(d["phases"]):
        if (
            set(p) != {"name", "end_step", "tasks"}
            or p["name"] != TASKS[i]
            or p["tasks"] != list(TASKS[: i + 1])
            or type(p["end_step"]) is not int
            or not previous < p["end_step"] <= cfg.max_steps
        ):
            raise ValueError("Invalid cumulative curriculum stage/successful-step boundary")
        previous = p["end_step"]
    if previous != cfg.max_steps:
        raise ValueError("Final phase must finish at max_steps")
    for section, keys in {
        "data": ["batch_size", "max_length"],
        "runner": list(sections["runner"]),
        "evaluation": ["per_task", "batch_size", "max_new_tokens"],
        "reproducibility": ["num_threads"],
    }.items():
        if any(type(d[section][k]) is not int or d[section][k] < 1 for k in keys):
            raise ValueError("Curriculum counts must be positive integers")
    for section, key in (
        ("data", "shuffle_seed"),
        ("evaluation", "selection_seed"),
        ("reproducibility", "seed"),
    ):
        if type(d[section][key]) is not int or not 0 <= d[section][key] < 2**32:
            raise ValueError("Invalid curriculum seed")
    if (
        type(d["reproducibility"]["deterministic"]) is not bool
        or d["data"]["max_length"] > architecture.context_length
        or d["evaluation"]["precision"] not in ("float32", "float16", "bfloat16")
        or d["evaluation"]["per_task"] < 4
        or d["evaluation"]["per_task"] % 4
    ):
        raise ValueError("Invalid reproducibility/context/validation policy")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", d["experiment"]["name"]):
        raise ValueError("Invalid experiment name")
    if not isinstance(d["artifacts"]["root"], str) or not d["artifacts"]["root"]:
        raise ValueError("Artifact root must be nonempty")
    d["model"], d["training"] = architecture.to_dict(), cfg.to_dict()
    d["artifacts"]["root"] = str((path.parent / d["artifacts"]["root"]).resolve())
    return LoadedConfig(d, {"curriculum": source}, path.resolve())


class CurriculumJob:
    def __init__(self, engine, train, validation, recipe, lineage):
        self.engine, self.train, self.validation, self.recipe = engine, train, validation, recipe
        audit(train.examples, validation.examples)
        self.cursors = [
            TrainingCursor(
                train.view(p["tasks"]),
                batch_size=recipe["data"]["batch_size"],
                accumulation_steps=engine.config.gradient_accumulation_steps,
                seed=recipe["data"]["shuffle_seed"],
            )
            for p in recipe["phases"]
        ]
        self.selection = select_validation(
            validation.examples,
            recipe["evaluation"]["per_task"],
            recipe["evaluation"]["selection_seed"],
        )
        self.history = {"train": [], "validation": []}
        self.generated = []
        self.best = None
        self.best_weights = None
        package = Path(__file__).parents[1]
        files = sorted(
            {
                p
                for folder in (
                    "model",
                    "training",
                    "curriculum",
                    "inference",
                    "reasoning",
                    "evaluation",
                    "data",
                )
                for p in (package / folder).glob("*.py")
            }
            | {package / "utils/seed.py"}
        )
        self.contract = {
            "schema_version": 1,
            "purpose": "curriculum_sft",
            "initialization": lineage,
            "recipe": {
                k: recipe[k]
                for k in (
                    "model",
                    "training",
                    "data",
                    "phases",
                    "runner",
                    "evaluation",
                    "reproducibility",
                )
            },
            "dataset_fingerprint": train.meta["fingerprint"],
            "tokenizer_fingerprint": train.codec.fingerprint,
            "selection_ids": [r["question_id"] for r in self.selection],
            "implementation": {
                p.relative_to(package).as_posix(): hashlib.sha256(
                    p.read_text(encoding="utf-8").encode()
                ).hexdigest()
                for p in files
            },
        }
        self.failed = False

    def phase(self, step=None):
        step = self.engine.step if step is None else step
        return next((i for i, p in enumerate(self.recipe["phases"]) if step < p["end_step"]), 2)

    def update(self):
        if self.failed or sum(c.windows_consumed for c in self.cursors) != self.engine.attempts:
            raise RuntimeError(
                "Curriculum failed/desynchronized; resume the latest full checkpoint"
            )
        index = self.phase()
        cursor = self.cursors[index]
        window = cursor.peek()
        digest = window_fingerprint(window)
        try:
            raw = self.engine.train_update(window.batches)
            cursor.commit(window)
        except BaseException:
            cursor.abort(window)
            self.failed = True
            raise
        row = {
            k: raw[k]
            for k in (
                "step",
                "attempt",
                "loss",
                "loss_tokens",
                "skipped",
                "learning_rates",
                "amp_scale",
                "tokens_seen",
                "tokens_optimized",
            )
        }
        row.update(phase=TASKS[index], batch_sha256=digest)
        self.history["train"].append(row)
        self.engine._emit(
            "curriculum_window_committed",
            phase=TASKS[index],
            batch_sha256=digest,
            next_epoch=cursor.epoch,
            next_offset=cursor.offset,
            step=self.engine.step,
        )
        return row

    def evaluate(self, directory=None):
        if self.generated and self.generated[-1]["step"] == self.engine.step:
            return False
        before = [c.state_dict() for c in self.cursors]
        size = self.recipe["data"]["batch_size"]
        view = self.validation.view()
        total = math.ceil(len(view) / size)
        self.engine._emit(
            "curriculum_validation_started", step=self.engine.step, records=len(view), batches=total
        )

        def batches():
            for index, batch in enumerate(view.validation_batches(size)):
                if index % 50 == 0:
                    self.engine._emit(
                        "curriculum_validation_progress",
                        step=self.engine.step,
                        completed_batches=index,
                        total_batches=total,
                    )
                yield batch

        loss = self.engine.evaluate(batches())
        self.history["validation"].append(
            {k: loss[k] for k in ("step", "loss", "loss_tokens")}
            | {"attempt": self.engine.attempts}
        )
        self.engine.model.eval()
        self.engine._emit(
            "generated_validation_started", step=self.engine.step, questions=len(self.selection)
        )
        predictions = generate_validation(
            self.engine.model,
            self.train.codec,
            self.selection,
            self.recipe["evaluation"],
            self.engine._emit,
        )
        if directory is not None:
            atomic_text(
                directory / f"step-{self.engine.step:08d}.predictions.jsonl",
                "".join(canonical_json(p) + "\n" for p in predictions),
            )
        scored, summary = score(predictions, self.selection, self.train.codec)
        record = {
            "step": self.engine.step,
            "attempt": self.engine.attempts,
            "summary": summary,
            "predictions": predictions,
        }
        self.generated.append(record)
        improved = (
            self.best is None or summary["macro_exact_match"] > self.best["macro_exact_match"]
        )
        if improved:
            self.best_weights = snapshot(self.engine.model)
            self.best = {
                "step": self.engine.step,
                "macro_exact_match": summary["macro_exact_match"],
                "tasks": summary["tasks"],
                "weights_sha256": weights_digest(self.best_weights),
            }
        if directory is not None:
            atomic_text(
                directory / f"step-{self.engine.step:08d}.scored.jsonl",
                "".join(canonical_json(p) + "\n" for p in scored),
            )
        if [c.state_dict() for c in self.cursors] != before:
            raise AssertionError("Validation advanced a train cursor")
        self.engine._emit(
            "generated_validation_completed",
            step=self.engine.step,
            improved=improved,
            best=self.best,
            **summary,
        )
        return improved

    def state(self):
        return {
            "curriculum_schema_version": 1,
            "contract": self.contract,
            "cursors": [c.state_dict() for c in self.cursors],
            "generated": self.generated,
            "best": self.best,
            "best_weights": self.best_weights,
            "extra_state": {
                "contract": self.contract,
                "history": self.history,
                "history_sha256": config_digest(self.history),
            },
        }

    def validate_state(self, state, attempts):
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "curriculum_schema_version",
                "contract",
                "cursors",
                "generated",
                "best",
                "best_weights",
                "extra_state",
            }
            or state["curriculum_schema_version"] != 1
            or state["contract"] != self.contract
            or len(state["cursors"]) != 3
        ):
            raise ValueError("Curriculum checkpoint recipe/implementation/data mismatch")
        for cursor, saved in zip(self.cursors, state["cursors"], strict=True):
            cursor.validate_state(saved)
        h = state["extra_state"]
        if (
            set(h) != {"contract", "history", "history_sha256"}
            or h["contract"] != self.contract
            or config_digest(h["history"]) != h["history_sha256"]
            or set(h["history"]) != {"train", "validation"}
            or len(h["history"]["train"]) != attempts
            or sum(s["windows_consumed"] for s in state["cursors"]) != attempts
        ):
            raise ValueError("Curriculum history/cursor counters mismatch")
        step = 0
        counts = [0, 0, 0]
        for attempt, row in enumerate(h["history"]["train"], 1):
            phase = self.phase(step)
            counts[phase] += 1
            if type(row["skipped"]) is not bool:
                raise ValueError("Invalid AMP history")
            step += not row["skipped"]
            if (
                row["attempt"] != attempt
                or row["step"] != step
                or row["phase"] != TASKS[phase]
                or not math.isfinite(row["loss"])
            ):
                raise ValueError("Invalid curriculum phase/update history")
        if counts != [s["windows_consumed"] for s in state["cursors"]]:
            raise ValueError("Stage cursor counts disagree with history")
        previous = (-1, -1)
        for row in h["history"]["validation"]:
            progress = (row["step"], row["attempt"])
            if (
                not previous < progress <= (step, attempts)
                or not math.isfinite(row["loss"])
                or row["loss_tokens"] < 1
            ):
                raise ValueError("Invalid curriculum validation history")
            previous = progress
        best = None
        previous = -1
        for record in state["generated"]:
            _, summary = score(record["predictions"], self.selection, self.train.codec)
            if not previous < record["step"] <= step or summary != record["summary"]:
                raise ValueError("Invalid generated validation history")
            previous = record["step"]
            if best is None or summary["macro_exact_match"] > best["macro_exact_match"]:
                best = {
                    "step": record["step"],
                    "macro_exact_match": summary["macro_exact_match"],
                    "tasks": summary["tasks"],
                }
        if best is None or {k: state["best"][k] for k in best} != best:
            raise ValueError("Best checkpoint must use generated macro EM, earliest tie")
        validate_weights(state["best_weights"], self.engine.model)
        if weights_digest(state["best_weights"]) != state["best"]["weights_sha256"]:
            raise ValueError("Best model checksum mismatch")

    def save(self, path):
        if self.failed or sum(c.windows_consumed for c in self.cursors) != self.engine.attempts:
            raise RuntimeError("Cannot checkpoint a failed/pending curriculum update")
        self.engine.save(path, data_state=self.state())

    def load(self, path):
        state = self.engine.load(path, data_validator=self.validate_state)
        for cursor, saved in zip(self.cursors, state["cursors"], strict=True):
            cursor.load_state_dict(saved)
        self.history = state["extra_state"]["history"]
        self.generated, self.best, self.best_weights = (
            state["generated"],
            state["best"],
            state["best_weights"],
        )
        self.failed = False
        if sum(not row["skipped"] for row in self.history["train"]) != self.engine.step:
            self.failed = True
            raise ValueError("Engine and curriculum history steps disagree")

    def export(self, directory):
        export_sft_curves(self.history, directory)
        for record in self.generated:
            stem = f"step-{record['step']:08d}"
            scored, _ = score(record["predictions"], self.selection, self.train.codec)
            for role, rows in (("predictions", record["predictions"]), ("scored", scored)):
                atomic_text(
                    directory / f"{stem}.{role}.jsonl",
                    "".join(canonical_json(p) + "\n" for p in rows),
                )
        write_json(
            directory / "generated_validation.json",
            [{k: v for k, v in r.items() if k != "predictions"} for r in self.generated],
        )
        write_json(directory / "best_selection.json", self.best)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
        try:
            for task in TASKS:
                ax.plot(
                    [r["step"] for r in self.generated],
                    [r["summary"]["tasks"][task]["exact_match"] for r in self.generated],
                    marker="o",
                    label=task,
                )
            ax.plot(
                [r["step"] for r in self.generated],
                [r["summary"]["macro_exact_match"] for r in self.generated],
                "k--",
                label="macro (selection)",
            )
            ax.set(
                xlabel="Successful optimizer updates",
                ylabel="Free-generated validation exact match",
                ylim=(-0.05, 1.05),
                title="Fixed validation only; not held-out test",
            )
            ax.grid(alpha=0.2)
            ax.legend()
            for ext in ("png", "svg"):
                fig.savefig(directory / f"generated_accuracy.{ext}", dpi=150)
        finally:
            plt.close(fig)


def run(
    config,
    prepared_root,
    tokenizer,
    *,
    init_checkpoint=None,
    resume=None,
    device="cpu",
    precision=None,
    stop_after=None,
    output_root=None,
):
    if (init_checkpoint is None) == (resume is None):
        raise ValueError("Use exactly one of source weights or full curriculum resume")
    loaded = load_recipe(config)
    d = loaded.data
    if precision is not None:
        d["training"]["precision"] = precision
    cfg = TrainingConfig.from_dict(d["training"])
    goal = cfg.max_steps if stop_after is None else stop_after
    if type(goal) is not int or not 1 <= goal <= cfg.max_steps:
        raise ValueError("stop_after is an absolute successful step within the fixed budget")
    codec = ByteBPETokenizer.load(tokenizer)
    train, validation = (
        Corpus(prepared_root, "train", codec),
        Corpus(prepared_root, "validation", codec),
    )
    if train.meta["block_size"] != d["data"]["max_length"]:
        raise ValueError("Prepared context cap differs from recipe")
    rng = seed_everything(**d["reproducibility"])
    env = inspect_environment(device)
    if not env["validation"]["ok"]:
        raise RuntimeError(env["validation"]["errors"])
    architecture = ModelConfig.from_dict(d["model"])
    if architecture.vocab_size != len(codec):
        raise ValueError("Recipe vocabulary differs from frozen tokenizer")
    if init_checkpoint is not None:
        model, meta, _ = load_for_evaluation(
            Path(init_checkpoint), codec, expected_sha256=d["initialization"]["sha256"]
        )
        if model.config != architecture or meta["stage"] != "sft":
            raise ValueError("Curriculum needs matching trained SFT source weights")
        lineage = {
            "kind": "sft_weights_only_fresh_curriculum_optimizer_rng_cursor",
            "checkpoint_sha256": meta["checkpoint_sha256"],
            "source_stage": "sft",
            "source_step": meta["step"],
        }
    else:
        payload = torch.load(resume, map_location="cpu", weights_only=True, mmap=True)
        lineage = (
            payload.get("state", {})
            .get("data_state", {})
            .get("contract", {})
            .get("initialization", {})
        )
        if lineage.get("checkpoint_sha256") != d["initialization"]["sha256"]:
            raise ValueError("Resume source lineage mismatch")
        del payload
        model = CausalLM(architecture, tokenizer_fingerprint=codec.fingerprint)
    if output_root is not None:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    d["execution"] = {
        "device": device,
        "prepared_root": str(Path(prepared_root).resolve()),
        "tokenizer": str(Path(tokenizer).resolve()),
        "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "resume": str(resume) if resume else None,
        "stop_after": stop_after,
    }
    d["initialization_lineage"] = lineage
    with Run(loaded) as active:

        def log(record):
            active.log(
                record["event"],
                **{k: v for k, v in record.items() if k not in ("event", "timestamp")},
            )

        engine = TrainingEngine(model.to(device), cfg, generator=rng, log=log)
        job = CurriculumJob(engine, train, validation, d, lineage)
        if resume:
            job.load(resume)
        if engine.step > goal:
            raise ValueError("stop_after precedes checkpoint progress")
        art, ckpt = active.path / "artifacts", active.path / "checkpoints"
        write_json(art / "environment.json", env)
        write_json(art / "dataset_manifest.json", train.meta)
        write_json(art / "training_contract.json", job.contract)
        write_json(art / "selected_validation_ids.json", job.contract["selection_ids"])
        codec.save(art / "tokenizer")
        active.metadata.update(
            initialization=lineage,
            dataset_fingerprint=train.meta["fingerprint"],
            tokenizer_fingerprint=codec.fingerprint,
        )
        write_json(active.path / "metadata.json", active.metadata)

        def save(boundary=False):
            engine._emit("curriculum_checkpoint_started", step=engine.step)
            job.save(ckpt / "latest.pt")
            if boundary and engine.step < cfg.max_steps:
                job.save(ckpt / f"phase-end-{engine.step:08d}.pt")
            if engine.step == cfg.max_steps:
                job.save(ckpt / "final.pt")
            # Best weights are carried inside latest.pt, so a resume needs no sibling files.
            export_selected(ckpt / "selected_model.pt", job.best_weights, job.best, job.contract)
            job.export(art)
            engine._emit(
                "curriculum_progress_saved",
                step=engine.step,
                best=job.best,
                latest=str(ckpt / "latest.pt"),
            )

        engine._emit(
            "curriculum_started",
            run_directory=str(active.path),
            target_step=goal,
            planned_steps=cfg.max_steps,
            phases=d["phases"],
            parameters=model.parameter_counts(),
        )
        if not job.generated:
            job.evaluate(art)
        save()
        skips = 0
        for row in reversed(job.history["train"]):
            if not row["skipped"]:
                break
            skips += 1
        previous_phase = None
        boundaries = {p["end_step"] for p in d["phases"]}
        while engine.step < goal:
            if skips >= d["runner"]["max_attempts_per_step"]:
                raise RuntimeError("Curriculum AMP attempt budget exceeded")
            phase = job.phase()
            if phase != previous_phase:
                engine._emit(
                    "curriculum_phase_started",
                    phase=TASKS[phase],
                    pool_records=len(job.cursors[phase].corpus),
                    step=engine.step,
                )
                previous_phase = phase
            row = job.update()
            skips = skips + 1 if row["skipped"] else 0
            if row["skipped"]:
                if engine.attempts % d["runner"]["checkpoint_every"] == 0:
                    save()
                continue
            boundary = engine.step in boundaries
            if engine.step % d["runner"]["validate_every"] == 0 or boundary:
                job.evaluate(art)
            if (
                engine.step % d["runner"]["checkpoint_every"] == 0
                or boundary
                or engine.step == goal
            ):
                save(boundary)
        status = "completed" if engine.step == cfg.max_steps else "controlled_stop"
        improved_over_initial = (
            job.best["macro_exact_match"] > job.generated[0]["summary"]["macro_exact_match"]
        )
        if not improved_over_initial:
            engine._emit("no_generated_validation_improvement", best_step=job.best["step"])
        write_json(
            art / "curriculum_report.json",
            {
                "status": status,
                "work_package": "WP07.6.2",
                "step": engine.step,
                "attempts": engine.attempts,
                "amp_skips": engine.attempts - engine.step,
                "planned_steps": cfg.max_steps,
                "best": job.best,
                "improved_over_initial": improved_over_initial,
                "selection_rule": (
                    "Maximum fixed generated-validation macro EM across copy/single/multi; "
                    "earliest step on ties"
                ),
                "phase_windows": {TASKS[i]: c.windows_consumed for i, c in enumerate(job.cursors)},
                "dataset_fingerprint": train.meta["fingerprint"],
                "source_initialization": lineage,
                "quality_scope": (
                    "Local smoke is execution evidence only; full curriculum and downstream "
                    "pilot require user training."
                ),
            },
        )
        engine._emit(
            "curriculum_finished", status=status, run_directory=str(active.path), best=job.best
        )
    return active.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/training/wp07_curriculum.toml"))
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--init-checkpoint", type=Path)
    g.add_argument("--resume", type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=("float32", "float16", "bfloat16"))
    p.add_argument("--stop-after", type=int)
    p.add_argument("--output-root", type=Path)
    a = p.parse_args()
    run(
        a.config,
        a.prepared_root,
        a.tokenizer,
        init_checkpoint=a.init_checkpoint,
        resume=a.resume,
        device=a.device,
        precision=a.precision,
        stop_after=a.stop_after,
        output_root=a.output_root,
    )


if __name__ == "__main__":
    main()
