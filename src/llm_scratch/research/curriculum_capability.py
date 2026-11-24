"""Verify a completed curriculum export and evaluate fixed validation-only scopes."""

from __future__ import annotations

import argparse
import json
import math
import re
import tomllib
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path, PurePosixPath

from llm_scratch.curriculum.data import Corpus
from llm_scratch.curriculum.evaluation import (
    generate_validation,
    load_selected,
    score,
    select_validation,
)
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.capability import generate_questions, left_pad
from llm_scratch.evaluation.pilot import load_manifest, load_questions, load_references
from llm_scratch.evaluation.pilot_metrics import score_predictions, summarize
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.reasoning.formatting import encode_reasoning_prompt
from llm_scratch.research.generation_equivalence import check_generation_equivalence
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import LoadedConfig, canonical_json, config_digest
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, rows):
    atomic_text(path, "".join(canonical_json(r) + "\n" for r in rows))


def load_protocol(path):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    d = tomllib.loads(source)
    fields = {
        "experiment": {"name"},
        "reproducibility": {"seed", "deterministic", "num_threads"},
        "artifacts": {"root"},
        "checkpoint": {"path", "sha256"},
        "summary": {"path", "sha256"},
        "baseline": {"path", "sha256"},
        "evaluation": {
            "batch_size",
            "precision",
            "max_new_tokens",
            "expected_pilot_questions",
            "holdout_per_task",
            "selection_seed",
        },
    }
    if (
        set(d) != {"schema_version", *fields}
        or type(d["schema_version"]) is not int
        or d["schema_version"] != 1
    ):
        raise ValueError("Invalid curriculum capability protocol schema")
    for k, v in fields.items():
        if not isinstance(d[k], dict) or set(d[k]) != v:
            raise ValueError(f"Unknown/missing {k} fields")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", d["experiment"]["name"]):
        raise ValueError("Invalid experiment name")
    r, e = d["reproducibility"], d["evaluation"]
    if type(r["deterministic"]) is not bool or e["precision"] != "float32":
        raise ValueError("Protocol requires explicit deterministic flag and FP32 evaluation")
    for v in (
        r["num_threads"],
        e["batch_size"],
        e["max_new_tokens"],
        e["expected_pilot_questions"],
        e["holdout_per_task"],
    ):
        if type(v) is not int or v < 1:
            raise ValueError("Evaluation counts must be positive integers")
    for v in (r["seed"], e["selection_seed"]):
        if type(v) is not int or not 0 <= v < 2**32:
            raise ValueError("Invalid seed")
    if e["holdout_per_task"] < 4 or e["holdout_per_task"] % 4:
        raise ValueError("Holdout per task must be divisible by4")
    for section in ("checkpoint", "summary", "baseline"):
        if not re.fullmatch(r"[a-f0-9]{64}", d[section]["sha256"]):
            raise ValueError("Input identity requires SHA256")
        d[section]["path"] = str((path.parent / d[section]["path"]).resolve())
    if not isinstance(d["artifacts"]["root"], str) or not d["artifacts"]["root"]:
        raise ValueError("Invalid artifact root")
    d["artifacts"]["root"] = str((path.parent / d["artifacts"]["root"]).resolve())
    return LoadedConfig(d, {"curriculum_capability": source}, path.resolve())


def selection_and_holdout(rows, contract, per_task, seed):
    if any(r["split"] != "validation" for r in rows):
        raise ValueError("Evaluation accepts validation only")
    source = contract["recipe"]["evaluation"]
    selection = select_validation(rows, source["per_task"], source["selection_seed"])
    if [r["question_id"] for r in selection] != contract["selection_ids"]:
        raise ValueError("Checkpoint selection IDs differ from frozen validation")
    selected_groups = {r["group"] for r in selection}
    holdout = select_validation(
        [r for r in rows if r["group"] not in selected_groups], per_task, seed
    )
    if {r["group"] for r in holdout} & selected_groups:
        raise ValueError("Selection-group leakage into additional validation")
    return selection, holdout


def audit_summary(path, expected_sha, meta, corpus, selection):
    if file_sha256(path) != expected_sha:
        raise ValueError("Training summary SHA256 mismatch")
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if (
            len(set(z.namelist())) != len(infos)
            or sum(i.file_size for i in infos) > 256 * 1024**2
            or any(
                PurePosixPath(i.filename).is_absolute()
                or ".." in PurePosixPath(i.filename).parts
                or "\\" in i.filename
                or ":" in i.filename
                or (i.external_attr >> 16) & 0o170000 == 0o120000
                for i in infos
            )
        ):
            raise ValueError("Unsafe/duplicate/oversized summary entries")
        if z.testzip() is not None:
            raise ValueError("Training summary CRC mismatch")

        def read(name):
            return json.loads(z.read(name))

        report, history, source = (
            read("artifacts/curriculum_report.json"),
            read("artifacts/history.json"),
            read("metadata.json"),
        )
        contract = read("artifacts/training_contract.json")
        if (
            contract != meta["contract"]
            or read("artifacts/best_selection.json") != meta["selection"]
            or read("artifacts/dataset_manifest.json") != corpus.meta
        ):
            raise ValueError("Model/summary/dataset identities disagree")
        cfg = contract["recipe"]
        if (
            report["status"] != "completed"
            or report["step"] != cfg["training"]["max_steps"]
            or report["best"] != meta["selection"]
            or report["dataset_fingerprint"] != corpus.meta["fingerprint"]
        ):
            raise ValueError("Expected a completed matching curriculum run")
        train = history["train"]
        step = 0
        seen = optimized = 0
        phases = Counter()
        for attempt, row in enumerate(train, 1):
            phase = next(p["name"] for p in cfg["phases"] if step < p["end_step"])
            if (
                type(row["skipped"]) is not bool
                or not math.isfinite(row["loss"])
                or row["loss_tokens"] < 1
            ):
                raise ValueError("Invalid training history")
            step += not row["skipped"]
            seen += row["loss_tokens"]
            optimized += row["loss_tokens"] if not row["skipped"] else 0
            phases[phase] += 1
            if (row["attempt"], row["step"], row["phase"]) != (attempt, step, phase):
                raise ValueError("Training history progress mismatch")
            if (row["tokens_seen"], row["tokens_optimized"]) != (seen, optimized):
                raise ValueError("Training token counters mismatch")
        if (
            step != report["step"]
            or len(train) != report["attempts"]
            or len(train) - step != report["amp_skips"]
            or dict(phases) != report["phase_windows"]
        ):
            raise ValueError("Training report/history counters mismatch")
        previous_progress = (-1, -1)
        if not history["validation"]:
            raise ValueError("Missing full validation loss history")
        for row in history["validation"]:
            progress = (row["step"], row["attempt"])
            if (
                not previous_progress < progress <= (step, len(train))
                or not math.isfinite(row["loss"])
                or row["loss_tokens"] != corpus.meta["splits"]["validation"]["target_tokens"]
            ):
                raise ValueError("Invalid validation loss history/scope")
            previous_progress = progress
        generated = read("artifacts/generated_validation.json")
        best = None
        previous = -1
        for r in generated:
            if not previous < r["step"] <= step:
                raise ValueError("Generated history ordering mismatch")
            previous = r["step"]
            predictions = [
                json.loads(s)
                for s in z.read(f"artifacts/step-{r['step']:08d}.predictions.jsonl")
                .decode()
                .splitlines()
            ]
            scored, summary = score(predictions, selection, corpus.codec)
            saved_scored = [
                json.loads(s)
                for s in z.read(f"artifacts/step-{r['step']:08d}.scored.jsonl")
                .decode()
                .splitlines()
            ]
            if summary != r["summary"] or scored != saved_scored:
                raise ValueError("Archived generated predictions/scores mismatch")
            if best is None or summary["macro_exact_match"] > best["macro_exact_match"]:
                best = {
                    "step": r["step"],
                    "macro_exact_match": summary["macro_exact_match"],
                    "tasks": summary["tasks"],
                }
        if best is None or any(meta["selection"][k] != v for k, v in best.items()):
            raise ValueError("Selected checkpoint must maximize generated macro EM, earliest tie")
        selected_predictions = [
            json.loads(s)
            for s in z.read(f"artifacts/step-{best['step']:08d}.predictions.jsonl")
            .decode()
            .splitlines()
        ]
        return {
            "report": report,
            "history": history,
            "generated": generated,
            "metadata": source,
            "environment": read("artifacts/environment.json"),
            "selected_predictions": selected_predictions,
        }


def run_evaluation(
    config, dataset_root, curriculum_root, tokenizer, *, device="cpu", output_root=None
):
    loaded = load_protocol(config)
    d, e = loaded.data, loaded.data["evaluation"]
    seed_everything(**d["reproducibility"])
    codec = ByteBPETokenizer.load(tokenizer)
    corpus = Corpus(curriculum_root, "validation", codec)
    manifest = load_manifest(dataset_root)
    if manifest["dataset_fingerprint"] != corpus.meta["config"]["wp03_fingerprint"]:
        raise ValueError("WP03 pilot differs from the frozen curriculum source")
    questions = load_questions(
        dataset_root, manifest, "validation", expected_records=e["expected_pilot_questions"]
    )
    model, meta = load_selected(
        Path(d["checkpoint"]["path"]), codec, expected_sha256=d["checkpoint"]["sha256"]
    )
    if meta["contract"]["dataset_fingerprint"] != corpus.meta["fingerprint"]:
        raise ValueError("Selected-model curriculum identity mismatch")
    selection, holdout = selection_and_holdout(
        corpus.examples, meta["contract"], e["holdout_per_task"], e["selection_seed"]
    )
    audit = audit_summary(
        Path(d["summary"]["path"]), d["summary"]["sha256"], meta, corpus, selection
    )
    if file_sha256(d["baseline"]["path"]) != d["baseline"]["sha256"]:
        raise ValueError("Frozen baseline predictions checksum mismatch")
    env = inspect_environment(device)
    if not env["validation"]["ok"]:
        raise RuntimeError(env["validation"]["errors"])
    if output_root:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    d["execution"] = {
        "device": device,
        "dataset_root": str(Path(dataset_root).resolve()),
        "curriculum_root": str(Path(curriculum_root).resolve()),
        "tokenizer": str(Path(tokenizer).resolve()),
    }
    files = [Path(__file__), Path(__file__).parents[1] / "curriculum/evaluation.py"]
    files.extend(
        p
        for folder in ("model", "inference", "evaluation", "reasoning")
        for p in (Path(__file__).parents[1] / folder).glob("*.py")
    )
    protocol = {
        "evaluation": e,
        "reproducibility": d["reproducibility"],
        "inputs": {s: d[s]["sha256"] for s in ("checkpoint", "summary", "baseline")},
        "curriculum_fingerprint": corpus.meta["fingerprint"],
        "wp03_fingerprint": manifest["dataset_fingerprint"],
        "selection_ids": [r["question_id"] for r in selection],
        "holdout_ids": [r["question_id"] for r in holdout],
        "pilot_ids": [r["question_id"] for r in questions],
        "split": "validation_only",
        "implementation": {
            p.relative_to(Path(__file__).parents[1]).as_posix(): file_sha256(p) for p in files
        },
    }
    d["protocol_sha256"] = config_digest(protocol)
    with Run(loaded) as active:

        def emit(event, **fields):
            print(canonical_json({"event": event, **fields}), flush=True)
            active.log(event, **fields)

        art = active.path / "artifacts"
        write_json(art / "protocol.json", protocol)
        write_json(art / "environment.json", env)
        write_json(
            art / "training_audit.json",
            {k: v for k, v in audit.items() if k != "selected_predictions"},
        )
        write_jsonl(
            art / "selected_ids.jsonl",
            [
                {"question_id": r["question_id"], "group": r["group"], "task": r["task"]}
                for r in selection
            ],
        )
        write_jsonl(
            art / "holdout_ids.jsonl",
            [
                {"question_id": r["question_id"], "group": r["group"], "task": r["task"]}
                for r in holdout
            ],
        )
        emit(
            "curriculum_evaluation_started",
            run_directory=str(active.path),
            protocol_sha256=d["protocol_sha256"],
        )
        model.to(device).eval()
        policy = GenerationConfig(
            max_new_tokens=e["max_new_tokens"],
            eos_token_id=codec.eos_token_id,
            pad_token_id=codec.pad_token_id,
        )
        ids, mask = left_pad(
            [encode_reasoning_prompt(codec, questions[0]["question"])], codec.pad_token_id, device
        )
        cache = check_generation_equivalence(
            model,
            ids,
            mask,
            config=replace(policy, max_new_tokens=8),
            seed=d["reproducibility"]["seed"],
        )
        settings = {
            "batch_size": e["batch_size"],
            "max_new_tokens": e["max_new_tokens"],
            "precision": e["precision"],
            "selection_seed": e["selection_seed"],
        }
        replay = generate_validation(model, codec, selection, settings, emit)
        write_jsonl(art / "selection_replay.predictions.jsonl", replay)
        replay_scored, replay_metrics = score(replay, selection, codec)
        write_jsonl(art / "selection_replay.scored.jsonl", replay_scored)
        expected = {r["question_id"]: r for r in audit["selected_predictions"]}
        differences = [
            r["question_id"]
            for r in replay
            if any(
                r[k] != expected[r["question_id"]][k] for k in ("generated_ids", "finish_reason")
            )
        ]
        emit("selection_replay_completed", differing_questions=len(differences), **replay_metrics)
        predictions = generate_validation(model, codec, holdout, settings, emit)
        write_jsonl(art / "additional_validation.predictions.jsonl", predictions)
        holdout_scored, holdout_metrics = score(predictions, holdout, codec)
        write_jsonl(art / "additional_validation.scored.jsonl", holdout_scored)
        pilot_pred = generate_questions(
            model,
            codec,
            questions,
            policy,
            batch_size=e["batch_size"],
            seed=d["reproducibility"]["seed"],
            precision=e["precision"],
            emit=emit,
        )
        write_jsonl(art / "pilot_validation.predictions.jsonl", pilot_pred)
        refs = load_references(
            dataset_root, manifest, "validation", {r["question_id"] for r in questions}
        )
        scored = score_predictions(pilot_pred, refs, codec)
        write_jsonl(art / "pilot_validation.scored.jsonl", scored)
        baseline = score_predictions(read_jsonl(d["baseline"]["path"]), refs, codec)
        if any(
            r["question"]
            != next(q["question"] for q in questions if q["question_id"] == r["question_id"])
            for r in baseline
        ):
            raise ValueError("Baseline question texts differ from current pilot")
        selected_prompts = {r["prompt"] for r in selection}
        outside = [r for r in scored if r["question"] not in selected_prompts]
        old_outside = [r for r in baseline if r["question"] not in selected_prompts]
        correct = [r for r in scored if r["correct"]]
        failures = [
            r
            for difficulty in ("easy", "medium", "hard")
            for r in [q for q in scored if not q["correct"] and q["difficulty"] == difficulty][:2]
        ]
        write_json(art / "pilot_correct_examples.json", correct)
        write_json(art / "pilot_failure_examples.json", failures)
        report = {
            "status": "completed",
            "work_package": "WP07.6.3",
            "checkpoint": meta,
            "training_report": audit["report"],
            "protocol_sha256": d["protocol_sha256"],
            "cache_equivalence": cache,
            "selection_replay": {
                "metrics": replay_metrics,
                "differing_ids": differences,
                "exact_generated_replay": not differences,
            },
            "additional_validation": holdout_metrics,
            "pilot_validation": summarize(scored),
            "baseline_pilot_validation": summarize(baseline),
            "pilot_selection_overlap": len(scored) - len(outside),
            "pilot_outside_selection": summarize(outside) if outside else None,
            "baseline_pilot_outside_selection": summarize(old_outside) if old_outside else None,
            "source_checkpoint_unchanged": file_sha256(d["checkpoint"]["path"])
            == d["checkpoint"]["sha256"],
            "source_summary_unchanged": file_sha256(d["summary"]["path"]) == d["summary"]["sha256"],
            "limits": (
                "Validation diagnostics, not held-out test accuracy. Additional groups excluded "
                "from checkpoint-selection48; all validation was used for teacher-forced loss. "
                "No threshold/policy tuning."
            ),
        }
        if not report["source_checkpoint_unchanged"] or not report["source_summary_unchanged"]:
            raise AssertionError("Immutable inputs changed")
        write_json(art / "curriculum_evaluation.json", report)
        emit(
            "curriculum_evaluation_completed",
            run_directory=str(active.path),
            pilot=report["pilot_validation"],
            additional=holdout_metrics,
        )
    return active.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/wp07_curriculum_selected.toml")
    )
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--curriculum-root", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-root", type=Path)
    a = p.parse_args()
    run_evaluation(
        a.config,
        a.dataset_root,
        a.curriculum_root,
        a.tokenizer,
        device=a.device,
        output_root=a.output_root,
    )


if __name__ == "__main__":
    main()
