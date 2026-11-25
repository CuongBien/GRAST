"""Evidence audit and a bounded, frozen GSM8K validation-only supplementary probe."""

from __future__ import annotations

import argparse
import gc
import json
import tomllib
from pathlib import Path

import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.sft import normalize_example
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.capability import generate_questions
from llm_scratch.evaluation.pilot import load_manifest
from llm_scratch.evaluation.pilot_metrics import score_predictions, summarize
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.reasoning.formatting import encode_reasoning_prompt
from llm_scratch.research.bank_schema import digest, identifier, integer, validate_questions
from llm_scratch.research.pilot_lock import read_jsonl, write_jsonl
from llm_scratch.research.policy_replay import load_replay
from llm_scratch.research.sample_bank import (
    SampleBank,
    _load,
    emit,
)
from llm_scratch.research.sample_bank import (
    load_config as load_bank_config,
)
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import LoadedConfig, canonical_json, config_digest
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def load_config(path):
    path = Path(path).resolve()
    source = path.read_text(encoding="utf-8")
    d = tomllib.loads(source)
    keys = {
        "experiment": {"name"},
        "reproducibility": {"seed", "deterministic", "num_threads"},
        "artifacts": {"root"},
        "inputs": {
            "project_root",
            "pilot_receipt",
            "pilot_receipt_fingerprint",
            "replay_receipt",
            "replay_receipt_fingerprint",
            "dataset",
            "wp03_fingerprint",
            "tokenizer",
            "tokenizer_fingerprint",
            "checkpoint_config",
        },
        "gsm_probe": {
            "questions",
            "selection_seed",
            "expected_validation_records",
            "batch_size",
            "max_new_tokens",
            "precision",
        },
    }
    if (
        set(d) != {"schema_version", *keys}
        or type(d["schema_version"]) is not int
        or d["schema_version"] != 1
    ):
        raise ValueError("Invalid feasibility-review config schema")
    for k, fields in keys.items():
        if not isinstance(d[k], dict) or set(d[k]) != fields:
            raise ValueError(f"Unknown/missing {k} fields")
    r, g, i = d["reproducibility"], d["gsm_probe"], d["inputs"]
    if (
        not identifier(d["experiment"]["name"])
        or type(r["deterministic"]) is not bool
        or not integer(r["seed"], 0, 2**32 - 1)
        or not integer(r["num_threads"], 1)
        or any(
            not integer(g[k], 1)
            for k in ("questions", "expected_validation_records", "batch_size", "max_new_tokens")
        )
        or not integer(g["selection_seed"], 0, 2**32 - 1)
        or g["precision"] != "float32"
        or any(
            not digest(i[k])
            for k in (
                "pilot_receipt_fingerprint",
                "replay_receipt_fingerprint",
                "wp03_fingerprint",
                "tokenizer_fingerprint",
            )
        )
    ):
        raise ValueError("Invalid review dimensions/identities")
    for table, key in [
        ("artifacts", "root"),
        *(
            ("inputs", k)
            for k in (
                "project_root",
                "pilot_receipt",
                "replay_receipt",
                "dataset",
                "tokenizer",
                "checkpoint_config",
            )
        ),
    ]:
        if not isinstance(d[table][key], str) or not d[table][key]:
            raise ValueError("Invalid review path")
        d[table][key] = str((path.parent / d[table][key]).resolve())
    return LoadedConfig(d, {"feasibility_review": source}, path)


def audit_receipt(path, expected, project_root, work_package):
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if config_digest(receipt) != expected or receipt.get("work_package") != work_package:
        raise ValueError("Source receipt identity mismatch")
    root = (Path(project_root) / receipt["run_directory"]).resolve()
    files = receipt["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Source receipt lacks artifact checksums")
    for name, spec in files.items():
        p = (root / name).resolve()
        if (
            not p.is_relative_to(root)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not p.is_file()
            or p.stat().st_size != spec["size_bytes"]
            or file_sha256(p) != spec["sha256"]
        ):
            raise ValueError("Source artifact missing/changed/escaped")
    return receipt, root


def audit_sources(d):
    i = d["inputs"]
    pilot, pilot_root = audit_receipt(
        i["pilot_receipt"], i["pilot_receipt_fingerprint"], i["project_root"], "WP08.2"
    )
    replay, replay_root = audit_receipt(
        i["replay_receipt"], i["replay_receipt_fingerprint"], i["project_root"], "WP08.3"
    )
    bank = SampleBank(pilot_root / "artifacts/bank")
    manifest, decisions = load_replay(replay_root / "artifacts/replay", bank)
    if (
        bank.fingerprint != pilot["bank_fingerprint"]
        or bank.fingerprint != replay["bank_fingerprint"]
        or manifest["fingerprint"] != replay["repeat_verification"]["replay_fingerprint"]
    ):
        raise ValueError("Pilot/replay provenance chain mismatch")
    p = json.loads((pilot_root / "artifacts/pilot_report.json").read_text(encoding="utf-8"))
    r = json.loads((replay_root / "artifacts/evaluation/report.json").read_text(encoding="utf-8"))
    if (
        p["bank_fingerprint"] != bank.fingerprint
        or r["bank_fingerprint"] != bank.fingerprint
        or r["replay_fingerprint"] != manifest["fingerprint"]
        or r["decisions"] != len(decisions)
    ):
        raise ValueError("Source report provenance mismatch")
    return bank, p, r


def gsm_rows(d):
    root = Path(d["inputs"]["dataset"])
    m = load_manifest(root)
    if m["dataset_fingerprint"] != d["inputs"]["wp03_fingerprint"]:
        raise ValueError("GSM8K WP03 fingerprint mismatch")
    relative = "gsm8k/validation.jsonl"
    spec = m["files"][relative]
    path = root / relative
    if (
        spec["path"] != relative
        or file_sha256(path) != spec["sha256"]
        or path.stat().st_size != spec["size_bytes"]
    ):
        raise ValueError("GSM8K validation checksum mismatch")
    rows = read_jsonl(path)
    if (
        len(rows) != d["gsm_probe"]["expected_validation_records"]
        or len(rows) != spec["records"]
        or not rows
    ):
        raise ValueError("GSM8K validation count mismatch")
    if len({r["question_id"] for r in rows}) != len(rows) or any(
        r["source"] != "gsm8k" or r["split"] != "validation" or r["upstream_split"] != "train"
        for r in rows
    ):
        raise ValueError("Only unique GSM8K internal validation from official train is accepted")
    return rows, spec


def question_only(rows):
    # Combined upstream JSONL contains targets; the selector/generator receive only this projection.
    result = [
        {
            "question_id": r["question_id"],
            "question": r["question"],
            "source": "gsm8k",
            "split": "validation",
            "difficulty": "ungraded",
            "operation_count": 0,
            "reasoning_steps": 0,
            "prompt_length": "natural",
            "prompt_word_count": len(r["question"].split()),
        }
        for r in rows
    ]
    validate_questions(result)
    return result


def choose_gsm(questions, codec, *, count, seed, context_length, max_new_tokens):
    validate_questions(questions)
    if (
        not integer(count, 1)
        or not integer(seed, 0, 2**32 - 1)
        or not integer(context_length, 1)
        or not integer(max_new_tokens, 1)
    ):
        raise ValueError("Invalid GSM8K probe dimensions")
    eligible = [
        q
        for q in questions
        if len(encode_reasoning_prompt(codec, q["question"])) + max_new_tokens <= context_length
    ]
    if len(eligible) < count:
        raise ValueError("Not enough context-eligible GSM8K validation questions")
    selected = sorted(
        eligible,
        key=lambda q: (
            config_digest(["wp08-gsm-probe-v1", seed, q["question_id"]]),
            q["question_id"],
        ),
    )[:count]
    return selected, len(eligible)


def probe_contract(d, checkpoints, context):
    return {
        "gsm_probe": d["gsm_probe"],
        "wp03_fingerprint": d["inputs"]["wp03_fingerprint"],
        "tokenizer_fingerprint": d["inputs"]["tokenizer_fingerprint"],
        "context_length": context,
        "checkpoints": [{k: c[k] for k in ("id", "kind", "sha256")} for c in checkpoints],
    }


def prepare(config, output):
    d = load_config(config).data
    bank, _, _ = audit_sources(d)
    cp = load_bank_config(d["inputs"]["checkpoint_config"]).data["checkpoints"]
    if [{k: c[k] for k in ("id", "kind", "sha256")} for c in cp] != [
        {k: c[k] for k in ("id", "kind", "sha256")} for c in bank.manifest["checkpoints"]
    ]:
        raise ValueError("GSM8K probe must compare the same audited checkpoints")
    context = min(c["model_config"]["context_length"] for c in bank.manifest["checkpoints"])
    codec = ByteBPETokenizer.load(
        d["inputs"]["tokenizer"], expected_fingerprint=d["inputs"]["tokenizer_fingerprint"]
    )
    if codec.fingerprint != bank.manifest["tokenizer_fingerprint"]:
        raise ValueError("Probe tokenizer differs from source bank")
    rows, spec = gsm_rows(d)
    questions, eligible = choose_gsm(
        question_only(rows),
        codec,
        count=d["gsm_probe"]["questions"],
        seed=d["gsm_probe"]["selection_seed"],
        context_length=context,
        max_new_tokens=d["gsm_probe"]["max_new_tokens"],
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "questions.jsonl", questions)
    lock = {
        "schema_version": 1,
        "work_package": "WP08.4",
        "scope": "gsm8k-official-train-derived-validation-only",
        "selection_protocol": "question-id-hash-context-eligible-v1",
        "contract": probe_contract(d, cp, context),
        "validation_source_sha256": spec["sha256"],
        "validation_records": len(rows),
        "eligible_records": eligible,
        "over_context_records": len(rows) - eligible,
        "question_ids": [q["question_id"] for q in questions],
        "questions_sha256": file_sha256(output / "questions.jsonl"),
    }
    lock["fingerprint"] = config_digest(lock)
    atomic_text(output / "manifest.json", canonical_json(lock) + "\n")
    emit(
        "gsm_probe_frozen",
        directory=str(output),
        questions=len(questions),
        eligible=eligible,
        fingerprint=lock["fingerprint"],
    )
    return lock


def load_probe(root, d, bank, checkpoints):
    root = Path(root)
    lock = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    questions = read_jsonl(root / "questions.jsonl")
    validate_questions(questions)
    context = min(c["model_config"]["context_length"] for c in bank.manifest["checkpoints"])
    raw, spec = gsm_rows(d)
    codec = ByteBPETokenizer.load(
        d["inputs"]["tokenizer"], expected_fingerprint=d["inputs"]["tokenizer_fingerprint"]
    )
    if codec.fingerprint != bank.manifest["tokenizer_fingerprint"]:
        raise ValueError("Probe tokenizer differs from source bank")
    selected, eligible = choose_gsm(
        question_only(raw),
        codec,
        count=d["gsm_probe"]["questions"],
        seed=d["gsm_probe"]["selection_seed"],
        context_length=context,
        max_new_tokens=d["gsm_probe"]["max_new_tokens"],
    )
    expected = {
        "schema_version": 1,
        "work_package": "WP08.4",
        "scope": "gsm8k-official-train-derived-validation-only",
        "selection_protocol": "question-id-hash-context-eligible-v1",
        "contract": probe_contract(d, checkpoints, context),
        "validation_source_sha256": spec["sha256"],
        "validation_records": len(raw),
        "eligible_records": eligible,
        "over_context_records": len(raw) - eligible,
        "question_ids": [q["question_id"] for q in selected],
        "questions_sha256": file_sha256(root / "questions.jsonl"),
    }
    expected["fingerprint"] = config_digest(expected)
    if lock != expected or questions != selected:
        raise ValueError("Frozen GSM8K probe/config/selection changed")
    return lock, questions, codec


def build_decision(pilot, replay, gsm):
    if not gsm or any(r["summary"]["overall"]["questions"] < 1 for r in gsm):
        raise ValueError("Need actual supplementary GSM8K probe results")
    reviewed_snapshot = (
        pilot.get("bank_fingerprint")
        == "6941830812e5fe91ba583553556cbc68a10ecf9b84dcad2bb5850c97cab1c809"
        and replay.get("replay_fingerprint")
        == "7fa3de8093ae73094a74c82b95e74cc7545858c2df05f65c965f1fee242e6a20"
    )
    return {
        "assessment": "posthoc-descriptive-feasibility-review",
        "engineering_gate": "passed",
        "confidence_efficacy_gate": (
            "no-go-with-current-model-and-benchmark" if reviewed_snapshot else "requires-new-review"
        ),
        "review_bound_to_frozen_wp08_snapshot": reviewed_snapshot,
        "primary_diagnostic_benchmark": "wp03-easy-validation"
        if reviewed_snapshot
        else "undecided",
        "gsm8k_role": (
            "retained-supplementary-stress-test-not-primary"
            if reviewed_snapshot
            else "requires-review"
        ),
        "medium_hard_role": (
            "negative-controls-and-error-analysis" if reviewed_snapshot else "requires-review"
        ),
        "limitations": [
            "Validation already examined; not an untouched test.",
            "Very few correct policy decisions; no reliable efficacy/calibration claim.",
            "Canonical trace verification is conservative, not a complete reasoning judge.",
            "GSM8K probe is small, greedy and context-eligible only; "
            "no full-benchmark accuracy claim.",
        ],
        "next_gate": (
            "Improve easy arithmetic capability and prospectively freeze a new validation "
            "pilot before confidence calibration/efficacy claims."
            if reviewed_snapshot
            else "New evidence requires a new human feasibility review."
        ),
        "automatic_training_or_threshold_fit": False,
        "evidence_counts": {
            "arithmetic_candidates": pilot["candidates"],
            "replay_decisions": replay["decisions"],
            "gsm_predictions": sum(r["summary"]["overall"]["questions"] for r in gsm),
        },
    }


def run(config, probe_root, *, device="cpu", output_root=None):
    loaded = load_config(config)
    d = loaded.data
    g = d["gsm_probe"]
    bank, pilot, replay = audit_sources(d)
    checkpoints = load_bank_config(d["inputs"]["checkpoint_config"]).data["checkpoints"]
    lock, questions, codec = load_probe(probe_root, d, bank, checkpoints)
    seed_everything(**d["reproducibility"])
    env = inspect_environment(device)
    if not env["validation"]["ok"]:
        raise RuntimeError(env["validation"]["errors"])
    if output_root:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    d["execution"] = {"device": device, "probe_fingerprint": lock["fingerprint"]}
    policy = GenerationConfig(
        max_new_tokens=g["max_new_tokens"],
        do_sample=False,
        eos_token_id=codec.eos_token_id,
        pad_token_id=codec.pad_token_id,
    )
    with Run(loaded) as active:

        def log(event, **fields):
            emit(event, **fields)
            active.log(event, **fields)

        art = active.path / "artifacts"
        atomic_text(art / "probe_manifest.json", canonical_json(lock) + "\n")
        atomic_text(art / "environment.json", canonical_json(env) + "\n")
        predictions = []
        for cp in checkpoints:
            log("gsm_checkpoint_started", checkpoint_id=cp["id"], questions=len(questions))
            model, _ = _load(cp, codec)
            model.to(device).eval()
            rows = generate_questions(
                model,
                codec,
                questions,
                policy,
                batch_size=g["batch_size"],
                seed=d["reproducibility"]["seed"],
                precision=g["precision"],
                emit=log,
            )
            write_jsonl(art / f"{cp['id']}.predictions.jsonl", rows)
            predictions.append((cp, rows))
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        log("gsm_all_predictions_committed", checkpoints=len(predictions), questions=len(questions))
        # Scoring projection is deliberately created only after both checkpoints generated.
        raw, _ = gsm_rows(d)
        by_id = {r["question_id"]: r for r in raw}
        refs = {
            q["question_id"]: normalize_example(by_id[q["question_id"]], "gsm8k", "validation")[
                "final_answer"
            ]
            for q in questions
        }
        write_jsonl(
            art / "gsm.references.jsonl",
            [
                {"question_id": k, "split": "validation", "canonical_answer": v}
                for k, v in refs.items()
            ],
        )
        results = []
        for cp, rows in predictions:
            scored = score_predictions(rows, refs, codec)
            metrics = summarize(scored)
            write_jsonl(art / f"{cp['id']}.scored.jsonl", scored)
            if file_sha256(cp["path"]) != cp["sha256"]:
                raise RuntimeError("Checkpoint changed during GSM8K probe")
            results.append({"checkpoint_id": cp["id"], "summary": metrics})
            log("gsm_checkpoint_scored", checkpoint_id=cp["id"], **metrics["overall"])
        bank.assert_unchanged()
        report = {
            "work_package": "WP08.4",
            "status": "completed",
            "source_bank_fingerprint": bank.fingerprint,
            "source_replay_fingerprint": replay["replay_fingerprint"],
            "probe_fingerprint": lock["fingerprint"],
            "arithmetic": pilot["metrics"],
            "replay": replay["metrics"],
            "gsm8k": results,
            "decision": build_decision(pilot, replay, results),
        }
        atomic_text(art / "feasibility_report.json", canonical_json(report) + "\n")
        log("feasibility_review_completed", decision=report["decision"])
    emit("feasibility_review_finished", run_directory=str(active.path))
    return active.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "run"))
    p.add_argument("--config", type=Path, default=Path("configs/evaluation/wp08_feasibility.toml"))
    p.add_argument("--probe-root", type=Path, default=Path("artifacts/wp08-4/gsm16-v1"))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output-root", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.config, a.probe_root)
    else:
        run(a.config, a.probe_root, device=a.device, output_root=a.output_root)


if __name__ == "__main__":
    main()
