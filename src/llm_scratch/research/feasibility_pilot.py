"""Run a pre-frozen validation pilot, then score outside the committed sample bank."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.sft import normalize_example
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.pilot import load_manifest, load_references
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.research.pilot_lock import (
    load_pilot,
    load_settings,
    prepare_pilot,
    read_jsonl,
    write_jsonl,
)
from llm_scratch.research.pilot_metrics import summarize_bank
from llm_scratch.research.sample_bank import build_bank, emit, load_config
from llm_scratch.research.sample_bank_scoring import score_bank
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import canonical_json
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def external_inputs(settings, questions, output):
    """References/expressions are read by evaluation only, after bank commit."""
    root = Path(settings["dataset"])
    manifest = load_manifest(root)
    if manifest["dataset_fingerprint"] != settings["wp03_fingerprint"]:
        raise ValueError("Scorer dataset fingerprint mismatch")
    all_ids = {r["question_id"] for r in read_jsonl(root / "pilot/validation.questions.jsonl")}
    refs = load_references(root, manifest, "validation", all_ids)
    ids = {q["question_id"] for q in questions}
    relative = "math/validation.jsonl"
    spec = manifest["files"][relative]
    path = root / relative
    if (
        spec["path"] != relative
        or file_sha256(path) != spec["sha256"]
        or path.stat().st_size != spec["size_bytes"]
    ):
        raise ValueError("Trace source checksum mismatch")
    rows = read_jsonl(path)
    if len(rows) != spec["records"]:
        raise ValueError("Trace source count mismatch")
    chosen = {r["question_id"]: r for r in rows if r["question_id"] in ids}
    if set(chosen) != ids:
        raise ValueError("Trace source is missing pilot questions")
    for q in questions:
        row = chosen[q["question_id"]]
        normalized = normalize_example(row, "math", "validation")
        if (
            normalized["prompt"] != q["question"]
            or normalized["final_answer"] != refs[q["question_id"]]
        ):
            raise ValueError("Trace/reference/prompt disagreement")
    write_jsonl(
        output,
        [
            {
                "question_id": q["question_id"],
                "split": "validation",
                "canonical_answer": refs[q["question_id"]],
            }
            for q in questions
        ],
    )
    return {qid: chosen[qid]["expression"] for qid in ids}


def run(config, pilot_root, *, device="cpu", output_root=None):
    settings = load_settings(config)
    loaded = load_config(settings["bank_config"])
    d = loaded.data
    lock, questions = load_pilot(pilot_root, settings, d)
    codec = ByteBPETokenizer.load(
        settings["tokenizer"], expected_fingerprint=settings["tokenizer_fingerprint"]
    )
    seed_everything(**d["reproducibility"])
    environment = inspect_environment(device)
    if not environment["validation"]["ok"]:
        raise RuntimeError(environment["validation"]["errors"])
    if output_root:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    d["pilot"] = {
        "settings": settings,
        "fingerprint": lock["fingerprint"],
        "pilot_root": str(Path(pilot_root).resolve()),
        "device": device,
    }
    loaded.sources["pilot"] = Path(config).read_text(encoding="utf-8")
    with Run(loaded) as active:

        def log(event, **fields):
            emit(event, **fields)
            active.log(event, **fields)

        art = active.path / "artifacts"
        atomic_text(art / "pilot_manifest.json", canonical_json(lock) + "\n")
        atomic_text(art / "environment.json", canonical_json(environment) + "\n")
        log(
            "pilot_started",
            run_directory=str(active.path),
            pilot_fingerprint=lock["fingerprint"],
            questions=len(questions),
            expected_candidates=len(questions)
            * len(d["checkpoints"])
            * len(d["bank"]["trial_seeds"])
            * d["bank"]["candidates_per_question"],
        )
        bank = build_bank(
            art / "bank",
            questions,
            codec,
            d["checkpoints"],
            generation=GenerationConfig(
                **d["generation"], eos_token_id=codec.eos_token_id, pad_token_id=codec.pad_token_id
            ),
            trial_seeds=d["bank"]["trial_seeds"],
            candidates_per_question=d["bank"]["candidates_per_question"],
            batch_size=d["bank"]["batch_size"],
            precision=d["bank"]["precision"],
            device=device,
            log=log,
        )
        # No reference input reaches build_bank or the policy observation interface.
        reference_path = art / "pilot.references.jsonl"
        expressions = external_inputs(settings, questions, reference_path)
        scored = score_bank(bank, reference_path, art / "external-scoring")
        labels = read_jsonl(art / "external-scoring/scored.jsonl")
        audits, per_question, metrics = summarize_bank(bank, labels, expressions, codec)
        write_jsonl(art / "trace_audit.jsonl", audits)
        write_jsonl(art / "prefix_coverage.jsonl", per_question)
        report = {
            "work_package": "WP08.2",
            "status": "completed",
            "pilot_fingerprint": lock["fingerprint"],
            "bank_fingerprint": bank.fingerprint,
            "candidates": scored["records"],
            "metrics": metrics,
            "scope": "Development validation pilot only; no test, policy fitting or GSM8K decision",
        }
        atomic_text(art / "pilot_report.json", canonical_json(report) + "\n")
        bank.assert_unchanged()
        log("pilot_completed", bank_fingerprint=bank.fingerprint, candidates=scored["records"])
    emit("pilot_run_finished", run_directory=str(active.path))
    return active.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/wp08_pilot.toml"))
    parser.add_argument("--pilot-root", type=Path, default=Path("artifacts/wp08-2/pilot-v1"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_pilot(args.config, args.pilot_root)
    else:
        run(args.config, args.pilot_root, device=args.device, output_root=args.output_root)


if __name__ == "__main__":
    main()
