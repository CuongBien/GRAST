"""Selected-export warm start, full resume, selection-only validation, final gate."""

from __future__ import annotations

import argparse
import hashlib
import re
import tomllib
from pathlib import Path

from llm_scratch.curriculum.evaluation import (
    export_selected,
    generate_validation,
    load_selected,
    snapshot,
    validate_weights,
    weights_digest,
)
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.easy_recovery.data import Corpus, audit
from llm_scratch.evaluation.pilot_metrics import extract_generated_final
from llm_scratch.model.config import ModelConfig
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.curves import export_sft_curves
from llm_scratch.training.data import TrainingCursor
from llm_scratch.training.pretrain import PretrainingJob, atomic_text
from llm_scratch.training.sft_session import SFTSession
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
        "data": {"block_size", "batch_size", "shuffle_seed", "validation_batches"},
        "runner": {"validate_every", "checkpoint_every", "milestones", "max_attempts_per_step"},
        "evaluation": {"batch_size", "max_new_tokens", "precision", "selection_seed"},
        "gate": {
            "minimum_final_accuracy",
            "minimum_verified_accuracy",
            "minimum_correct_and_wrong",
        },
        "reproducibility": {"seed", "deterministic", "num_threads"},
        "artifacts": {"root"},
    }
    if set(d) != {"schema_version", "model", "training", *sections} or d["schema_version"] != 1:
        raise ValueError("Invalid easy-recovery recipe schema")
    if any(not isinstance(d[k], dict) or set(d[k]) != keys for k, keys in sections.items()):
        raise ValueError("Unknown/missing recovery settings")
    cfg, model = TrainingConfig.from_dict(d["training"]), ModelConfig.from_dict(d["model"])
    if cfg.stage != "sft" or not re.fullmatch(r"[a-f0-9]{64}", d["initialization"]["sha256"]):
        raise ValueError("SFT and source checksum required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", d["experiment"]["name"]):
        raise ValueError("Invalid experiment name")
    for section, keys in {
        "data": ("block_size", "batch_size"),
        "runner": ("validate_every", "checkpoint_every", "max_attempts_per_step"),
        "evaluation": ("batch_size", "max_new_tokens"),
        "reproducibility": ("num_threads",),
        "gate": ("minimum_correct_and_wrong",),
    }.items():
        if any(type(d[section][k]) is not int or d[section][k] < 1 for k in keys):
            raise ValueError("Recovery counts must be positive integers")
    for section, key in (
        ("data", "shuffle_seed"),
        ("evaluation", "selection_seed"),
        ("reproducibility", "seed"),
    ):
        if type(d[section][key]) is not int or not 0 <= d[section][key] < 2**32:
            raise ValueError("Invalid recovery seed")
    milestones = d["runner"]["milestones"]
    if (
        type(d["reproducibility"]["deterministic"]) is not bool
        or d["data"]["block_size"] > model.context_length
        or d["data"]["validation_batches"] != 0
        or d["evaluation"]["precision"] not in ("float32", "float16", "bfloat16")
        or not isinstance(milestones, list)
        or milestones != sorted(set(milestones))
        or any(type(v) is not int or not 1 <= v <= cfg.max_steps for v in milestones)
        or any(
            type(d["gate"][k]) not in (int, float) or not 0 <= d["gate"][k] <= 1
            for k in ("minimum_final_accuracy", "minimum_verified_accuracy")
        )
        or not isinstance(d["artifacts"]["root"], str)
        or not d["artifacts"]["root"]
    ):
        raise ValueError("Invalid recovery protocol bounds")
    d["model"], d["training"] = model.to_dict(), cfg.to_dict()
    d["artifacts"]["root"] = str((path.parent / d["artifacts"]["root"]).resolve())
    return LoadedConfig(d, {"easy_recovery": source}, path.resolve())


def score(predictions, references, codec):
    refs = {r["question_id"]: r for r in references}
    if (
        len(refs) != len(references)
        or len(predictions) != len(refs)
        or {r["question_id"] for r in predictions} != set(refs)
    ):
        raise ValueError("Recovery scoring requires each fixed question exactly once")
    scored = []
    for pred in predictions:
        ref = refs[pred["question_id"]]
        result = extract_generated_final(pred["generated_ids"], codec, pred["finish_reason"])
        correct = result["predicted_answer"] == ref["final_answer"]
        # Conservative canonical-trace match, not a general reasoning judge.
        trace = codec.decode(pred["generated_ids"]).split("</think>", 1)[0]
        verified = correct and result["format_valid"] and trace.split() == ref["reasoning"].split()
        scored.append(
            pred
            | result
            | {
                "correct": correct,
                "verified": verified,
                "origin": ref["origin"],
                "reference": ref["final_answer"],
            }
        )
    by_origin = {}
    for origin in ("wp03", "authored"):
        rows = [r for r in scored if r["origin"] == origin]
        if not rows:
            raise ValueError("Both recovery evaluation origins are required")
        by_origin[origin] = {
            "questions": len(rows),
            **{
                key: sum(r[key] for r in rows) / len(rows)
                for key in ("correct", "verified", "format_valid")
            },
        }
    return scored, {
        "questions": len(scored),
        "correct": sum(r["correct"] for r in scored),
        "verified": sum(r["verified"] for r in scored),
        "accuracy": sum(r["correct"] for r in scored) / len(scored),
        "verified_accuracy": sum(r["verified"] for r in scored) / len(scored),
        "macro_verified": sum(m["verified"] for m in by_origin.values()) / 2,
        "macro_final": sum(m["correct"] for m in by_origin.values()) / 2,
        "by_origin": by_origin,
    }


def rank(summary):
    return summary["macro_verified"], summary["macro_final"]


class RecoveryJob(PretrainingJob):
    def __init__(self, session, recipe):
        super().__init__(session, recipe)
        package = Path(__file__).parents[1]
        # Hash every executable package source used by training, generation or data validation.
        self.contract.update(
            kind="easy-recovery-v1",
            source_sha256=recipe["initialization"]["sha256"],
            evaluation=recipe["evaluation"],
            gate=recipe["gate"],
            selection_ids=[r["question_id"] for r in session.validation.examples],
            dataset_fingerprint=session.validation.meta["fingerprint"],
            implementation={
                p.relative_to(package).as_posix(): hashlib.sha256(
                    p.read_text(encoding="utf-8").encode()
                ).hexdigest()
                for p in sorted(package.rglob("*.py"))
            },
        )
        self.generated, self.best, self.best_weights = [], None, None

    def evaluate(self):
        engine = self.session.engine
        if self.generated and self.generated[-1]["step"] == engine.step:
            return
        self.validate()
        before = self.session.train.state_dict()
        was_training = engine.model.training
        engine.model.eval()
        try:
            predictions = generate_validation(
                engine.model,
                self.session.validation.codec,
                self.session.validation.examples,
                self.recipe["evaluation"],
                engine._emit,
            )
        finally:
            engine.model.train(was_training)
        _, summary = score(
            predictions, self.session.validation.examples, self.session.validation.codec
        )
        record = {
            "step": engine.step,
            "attempt": engine.attempts,
            "summary": summary,
            "predictions": predictions,
        }
        self.generated.append(record)
        if self.best is None or rank(summary) > rank(self.best["summary"]):
            self.best_weights = snapshot(engine.model)
            self.best = {
                "step": engine.step,
                "summary": summary,
                "weights_sha256": weights_digest(self.best_weights),
            }
        if before != self.session.train.state_dict():
            raise AssertionError("Generated validation changed training cursor")
        engine._emit(
            "easy_recovery_selection",
            **{k: v for k, v in record.items() if k != "predictions"},
            best_step=self.best["step"],
        )

    def _validate_extra(self, extra, attempts):
        required = {"contract", "history", "history_sha256", "generated", "best", "best_weights"}
        if not isinstance(extra, dict) or set(extra) != required:
            raise ValueError("Missing recovery checkpoint state")
        super()._validate_extra(
            {k: extra[k] for k in ("contract", "history", "history_sha256")}, attempts
        )
        generated = extra["generated"]
        if not isinstance(generated, list) or not generated or generated[0]["step"] != 0:
            raise ValueError("Missing initial generated selection")
        previous = -1
        for r in generated:
            if (
                type(r["step"]) is not int
                or not previous
                < r["step"]
                <= sum(not h["skipped"] for h in extra["history"]["train"])
                or r["step"] not in {v["step"] for v in extra["history"]["validation"]}
            ):
                raise ValueError("Invalid generated selection schedule")
            _, summary = score(
                r["predictions"], self.session.validation.examples, self.session.validation.codec
            )
            if summary != r["summary"]:
                raise ValueError("Generated selection summary mismatch")
            previous = r["step"]
        winner = max(generated, key=lambda r: rank(r["summary"]))  # earliest tie
        best = extra["best"]
        if best["step"] != winner["step"] or best["summary"] != winner["summary"]:
            raise ValueError("Selected checkpoint does not follow frozen selection rule")
        validate_weights(extra["best_weights"], self.session.engine.model)
        if weights_digest(extra["best_weights"]) != best["weights_sha256"]:
            raise ValueError("Selected weights checksum mismatch")

    def save(self, path):
        extra = {
            "contract": self.contract,
            "history": self.history,
            "history_sha256": config_digest(self.history),
            "generated": self.generated,
            "best": self.best,
            "best_weights": self.best_weights,
        }
        self._validate_extra(extra, self.session.engine.attempts)
        self.session.save(path, extra_state=extra)

    def load(self, path):
        extra = self.session.load(path, extra_validator=self._validate_extra)
        self.history, self.generated = extra["history"], extra["generated"]
        self.best, self.best_weights = extra["best"], extra["best_weights"]


def final_gate(
    source_checkpoint, selected_checkpoint, gate, settings, thresholds, device, directory, emit
):
    """Called after selection is finished; no gate predictions can change the winner."""
    summaries = {}
    for name, checkpoint in (("baseline", source_checkpoint), ("selected", selected_checkpoint)):
        model, meta = load_selected(checkpoint, gate.codec)
        model.to(device).eval()
        predictions = generate_validation(model, gate.codec, gate.examples, settings, emit)
        scored, summaries[name] = score(predictions, gate.examples, gate.codec)
        atomic_text(
            directory / f"gate-{name}.jsonl", "".join(canonical_json(r) + "\n" for r in scored)
        )
        summaries[name]["checkpoint_sha256"] = meta["sha256"]
        del model
    chosen = summaries["selected"]
    capability = all(
        origin["correct"] >= thresholds["minimum_final_accuracy"]
        and origin["verified"] >= thresholds["minimum_verified_accuracy"]
        for origin in chosen["by_origin"].values()
    )
    n = thresholds["minimum_correct_and_wrong"]
    mixed = n <= chosen["correct"] <= chosen["questions"] - n
    report = {
        "scope": "development-validation-only",
        "thresholds": thresholds,
        "results": summaries,
        "capability_screen_passed": capability,
        "mixed_outcome_screen_passed": mixed,
        "wp09_guidance": (
            "easy_pilot"
            if capability and mixed
            else "adjust_difficulty"
            if capability and chosen["questions"] - chosen["correct"] < n
            else "diagnostic_only"
        ),
        "limit": "Engineering screen only; not calibrated confidence or test-set evidence.",
        "next": "Proceed to WP09; no automatic extra recovery round.",
    }
    write_json(directory / "gate_report.json", report)
    emit("easy_recovery_gate_finished", **report)
    return report


def run(
    config,
    prepared_root,
    tokenizer,
    source_checkpoint,
    *,
    resume=None,
    device="cpu",
    precision=None,
    stop_after=None,
    output_root=None,
):
    loaded = load_recipe(config)
    d = loaded.data
    if precision is not None:
        d["training"]["precision"] = precision
    cfg = TrainingConfig.from_dict(d["training"])
    goal = cfg.max_steps if stop_after is None else stop_after
    if type(goal) is not int or not 1 <= goal <= cfg.max_steps:
        raise ValueError("stop_after is an absolute successful step within the fixed budget")
    codec = ByteBPETokenizer.load(tokenizer)
    train, selection = (Corpus(prepared_root, k, codec) for k in ("train", "selection"))
    audit({"train": train.unique_examples, "selection": selection.examples})
    if (
        train.meta["block_size"] != d["data"]["block_size"]
        or train.meta["config"]["source_checkpoint_sha256"] != d["initialization"]["sha256"]
    ):
        raise ValueError("Recipe/data source or context mismatch")
    rng = seed_everything(**d["reproducibility"])
    env = inspect_environment(device)
    if not env["validation"]["ok"]:
        raise RuntimeError(env["validation"]["errors"])
    model, source_meta = load_selected(source_checkpoint, codec, d["initialization"]["sha256"])
    if model.config != ModelConfig.from_dict(d["model"]):
        raise ValueError("Source model architecture differs from recipe")
    if output_root is not None:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    d["execution"] = {
        "device": device,
        "prepared_root": str(Path(prepared_root).resolve()),
        "tokenizer": str(Path(tokenizer).resolve()),
        "source_checkpoint": str(Path(source_checkpoint).resolve()),
        "resume": str(resume) if resume else None,
        "stop_after": stop_after,
    }
    with Run(loaded) as active:

        def log(record):
            active.log(
                record["event"],
                **{k: v for k, v in record.items() if k not in ("event", "timestamp")},
            )

        engine = TrainingEngine(model.to(device), cfg, generator=rng, log=log)
        cursor = TrainingCursor(
            train,
            batch_size=d["data"]["batch_size"],
            accumulation_steps=cfg.gradient_accumulation_steps,
            seed=d["data"]["shuffle_seed"],
        )
        job = RecoveryJob(SFTSession(engine, cursor, selection), d)
        if resume is not None:
            job.load(resume)
        if engine.step > goal:
            raise ValueError("stop_after precedes resumed progress")
        art, ckpt = active.path / "artifacts", active.path / "checkpoints"
        write_json(art / "environment.json", env)
        write_json(art / "dataset_manifest.json", train.meta)
        write_json(art / "training_contract.json", job.contract)
        write_json(
            art / "source_lineage.json",
            {
                "sha256": source_meta["sha256"],
                "source_selection": source_meta["selection"],
                "initialization": "weights_only_fresh_optimizer_scheduler_scaler_rng_cursor",
            },
        )
        codec.save(art / "tokenizer")
        engine._emit(
            "easy_recovery_started",
            run_directory=str(active.path),
            target_step=goal,
            planned_steps=cfg.max_steps,
            parameters=model.parameter_counts(),
            train_unique=len(train.unique_examples),
            train_virtual=len(train),
            wp03_repeat=train.repeat,
            selection_questions=len(selection),
        )

        def save():
            engine._emit("easy_recovery_checkpoint_started", step=engine.step, directory=str(ckpt))
            export_selected(ckpt / "selected_model.pt", job.best_weights, job.best, job.contract)
            write_json(art / "generated_selection.json", job.generated)
            export_sft_curves(job.history, art)
            job.save(ckpt / "latest.pt")
            if engine.step in d["runner"]["milestones"]:
                job.save(ckpt / f"step-{engine.step:08d}.pt")

        if not job.generated:
            job.evaluate()
        skips = 0
        for row in reversed(job.history["train"]):
            if not row["skipped"]:
                break
            skips += 1
        while engine.step < goal:
            if skips >= d["runner"]["max_attempts_per_step"]:
                raise RuntimeError("AMP attempt budget exceeded; resume last full checkpoint")
            row = job.update()
            skips = skips + 1 if row["skipped"] else 0
            if not row["skipped"] and (
                engine.step % d["runner"]["validate_every"] == 0 or engine.step == cfg.max_steps
            ):
                job.evaluate()
            if (
                engine.attempts % d["runner"]["checkpoint_every"] == 0
                or engine.step == goal
                or (not row["skipped"] and engine.step in d["runner"]["milestones"])
            ):
                save()
        if not (ckpt / "latest.pt").exists():
            save()
        status = "completed" if engine.step == cfg.max_steps else "controlled_stop"
        report = {
            "status": status,
            "step": engine.step,
            "attempts": engine.attempts,
            "amp_skips": engine.attempts - engine.step,
            "best": job.best,
            "source_sha256": source_meta["sha256"],
            "dataset_fingerprint": train.meta["fingerprint"],
            "improved_on_selection": rank(job.best["summary"]) > rank(job.generated[0]["summary"]),
            "gate_evaluated": False,
            "planned_steps": cfg.max_steps,
        }
        if status == "completed":
            gate = Corpus(prepared_root, "gate", codec)
            audit(
                {
                    "train": train.unique_examples,
                    "selection": selection.examples,
                    "gate": gate.examples,
                }
            )
            report["gate"] = final_gate(
                source_checkpoint,
                ckpt / "selected_model.pt",
                gate,
                d["evaluation"],
                d["gate"],
                device,
                art,
                engine._emit,
            )
            report["gate_evaluated"] = True
        report["selected_sha256"] = file_sha256(ckpt / "selected_model.pt")
        write_json(art / "recovery_report.json", report)
        engine._emit("easy_recovery_finished", run_directory=str(active.path), **report)
    return active.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/training/easy_recovery.toml"))
    for name in ("prepared-root", "tokenizer", "source-checkpoint"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("resume", "output-root"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--device", default="cpu")
    p.add_argument("--precision", choices=("float32", "float16", "bfloat16"))
    a = p.parse_args()
    run(
        a.config,
        a.prepared_root,
        a.tokenizer,
        a.source_checkpoint,
        resume=a.resume,
        device=a.device,
        precision=a.precision,
        stop_after=a.stop_after,
        output_root=a.output_root,
    )


if __name__ == "__main__":
    main()
