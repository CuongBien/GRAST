"""WP09.2 bounded calibration-fit pilot; generation and external scoring are separate."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from llm_scratch.curriculum.evaluation import load_selected
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.capability import generate_questions
from llm_scratch.evaluation.pilot_metrics import extract_generated_final
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.reasoning.formatting import encode_reasoning_prompt
from llm_scratch.reasoning.voting import majority
from llm_scratch.research.bank_schema import TOKENIZER_FILES, validate_questions
from llm_scratch.research.confidence_design import verify as verify_benchmark
from llm_scratch.research.pilot_metrics import trace_matches
from llm_scratch.research.sample_bank import SampleBank, build_bank, emit
from llm_scratch.research.sample_bank_scoring import score_bank
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import LoadedConfig, canonical_json, config_digest
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]


def write_rows(path, rows):
    atomic_text(Path(path), "".join(canonical_json(r) + "\n" for r in rows))


def select_questions(questions, index, per_stratum, seed):
    """No references or correctness values enter pilot selection."""
    validate_questions(questions)
    lookup = {r["question_id"]: r for r in questions}
    if len({r["question_id"] for r in index}) != len(index) or set(lookup) != {
        r["question_id"] for r in index
    }:
        raise ValueError("Question/index identities disagree")
    available = [r for r in index if r["partition"] == "calibration_fit"]
    selected = []
    for name in sorted({r["difficulty"] for r in available}):
        pool = sorted(
            [r for r in available if r["difficulty"] == name],
            key=lambda r: config_digest([seed, "wp09-pilot-v1", r["question_id"]]),
        )
        if len(pool) < per_stratum:
            raise ValueError("Insufficient calibration_fit groups for balanced pilot")
        selected.extend(lookup[r["question_id"]] for r in pool[:per_stratum])
    if len(selected) != 2 * per_stratum:
        raise ValueError("Pilot requires two locked strata")
    return selected


def prepare(config, benchmark, output):
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    keys = {
        "schema_version",
        "benchmark_fingerprint",
        "selection_seed",
        "questions_per_stratum",
        "minimum_independent_correct_groups",
        "minimum_independent_wrong_groups",
        "minimum_independent_verified_groups",
        "decision_rule",
    }
    if (
        set(cfg) != keys
        or cfg["schema_version"] != 1
        or any(
            type(cfg[k]) is not int or cfg[k] < 1
            for k in keys - {"benchmark_fingerprint", "decision_rule", "schema_version"}
        )
    ):
        raise ValueError("Invalid pilot configuration")
    root, out = Path(benchmark), Path(output)
    m = verify_benchmark(root, cfg["benchmark_fingerprint"])
    qs = select_questions(
        read_rows(root / "questions.jsonl"),
        read_rows(root / "index.jsonl"),
        cfg["questions_per_stratum"],
        cfg["selection_seed"],
    )
    out.mkdir(parents=True, exist_ok=False)
    write_rows(out / "questions.jsonl", qs)
    shutil.copytree(root / "tokenizer", out / "tokenizer")
    codec = ByteBPETokenizer.load(out / "tokenizer")
    if codec.fingerprint != m["config"]["tokenizer_fingerprint"]:
        raise ValueError("Benchmark/tokenizer identity mismatch")
    plan = {
        "schema_version": 1,
        "work_package": "WP09.2",
        "scope": "calibration_fit_pilot_only",
        "config": cfg,
        "benchmark_fingerprint": m["fingerprint"],
        "checkpoint_sha256": m["checkpoint_sha256"],
        "bank": m["bank"],
        "min_samples": m["future_analysis"]["min_samples"],
        "tokenizer_fingerprint": codec.fingerprint,
        "question_ids": [q["question_id"] for q in qs],
        "strata_counts": dict(Counter(q["difficulty"] for q in qs)),
        "files": {n: file_sha256(out / n) for n in sorted(TOKENIZER_FILES | {"questions.jsonl"})},
    }
    plan["fingerprint"] = config_digest(plan)
    write_json(out / "plan.json", plan)
    emit(
        "wp09_pilot_plan_locked",
        fingerprint=plan["fingerprint"],
        questions=len(qs),
        candidates=len(qs) * len(plan["bank"]["seeds"]) * plan["bank"]["max_candidates"],
    )
    return plan


def load_inputs(root, expected_fingerprint):
    root = Path(root)
    p = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    if (
        p.get("scope") != "calibration_fit_pilot_only"
        or config_digest({k: v for k, v in p.items() if k != "fingerprint"}) != p.get("fingerprint")
        or p["fingerprint"] != expected_fingerprint
    ):
        raise ValueError("Pilot plan fingerprint mismatch")
    allowed = TOKENIZER_FILES | {"questions.jsonl"}
    if set(p["files"]) != allowed or {
        f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file()
    } != allowed | {"plan.json"}:
        raise ValueError("Pilot inputs accept question-only files, not references or labels")
    for name, digest in p["files"].items():
        f = root / name
        if (
            f.is_symlink()
            or not f.resolve().is_relative_to(root.resolve())
            or file_sha256(f) != digest
        ):
            raise ValueError("Pilot input checksum mismatch")
    qs = read_rows(root / "questions.jsonl")
    validate_questions(qs)
    if [q["question_id"] for q in qs] != p["question_ids"]:
        raise ValueError("Pilot question order mismatch")
    codec = ByteBPETokenizer.load(
        root / "tokenizer", expected_fingerprint=p["tokenizer_fingerprint"]
    )
    return p, qs, codec


def generate_pilot(inputs, expected_fingerprint, checkpoint, output_root, device="cuda"):
    """No benchmark or reference path is accepted by the generation process."""
    p, qs, codec = load_inputs(inputs, expected_fingerprint)
    b = p["bank"]
    d = {
        "experiment": {"name": "wp09-2-stochastic-pilot"},
        "reproducibility": {"seed": 42, "deterministic": True, "num_threads": 1},
        "artifacts": {"root": str(Path(output_root).resolve())},
        "pilot_plan": p,
        "execution": {"device": device, "checkpoint": str(Path(checkpoint).resolve())},
    }
    loaded = LoadedConfig(d, {}, Path(inputs).resolve() / "plan.json")
    seed_everything(**d["reproducibility"])
    environment = inspect_environment(device)
    if not environment["validation"]["ok"]:
        raise ValueError(environment["validation"]["errors"])
    with Run(loaded) as active:

        def log(event, **fields):
            emit(event, **fields)
            active.log(event, **fields)

        art = active.path / "artifacts"
        write_json(art / "environment.json", environment)
        write_json(art / "pilot_plan.json", p)
        log("wp09_pilot_started", run_directory=str(active.path), questions=len(qs))
        bank = build_bank(
            art / "bank",
            qs,
            codec,
            [
                {
                    "id": b["checkpoint"],
                    "kind": "curriculum_selected",
                    "path": str(checkpoint),
                    "sha256": p["checkpoint_sha256"],
                }
            ],
            generation=GenerationConfig(
                max_new_tokens=b["max_new_tokens"],
                do_sample=True,
                temperature=b["temperature"],
                top_p=b["top_p"],
                top_k=b["top_k"],
                eos_token_id=codec.eos_token_id,
                pad_token_id=codec.pad_token_id,
            ),
            trial_seeds=b["seeds"],
            candidates_per_question=b["max_candidates"],
            batch_size=b["batch_size"],
            device=device,
            precision=b["precision"],
            log=log,
        )
        model, _ = load_selected(checkpoint, codec, p["checkpoint_sha256"])
        model.to(device).eval()
        greedy = generate_questions(
            model,
            codec,
            qs,
            GenerationConfig(
                max_new_tokens=b["max_new_tokens"],
                eos_token_id=codec.eos_token_id,
                pad_token_id=codec.pad_token_id,
            ),
            batch_size=b["batch_size"],
            seed=42,
            precision=b["precision"],
            emit=log,
        )
        write_rows(art / "greedy.predictions.jsonl", greedy)
        if file_sha256(checkpoint) != p["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during generation")
        commit = {
            "schema_version": 1,
            "work_package": "WP09.2",
            "scope": "gold_free_generation",
            "plan_fingerprint": p["fingerprint"],
            "bank_fingerprint": bank.fingerprint,
            "greedy_sha256": file_sha256(art / "greedy.predictions.jsonl"),
        }
        commit["fingerprint"] = config_digest(commit)
        write_json(art / "generation_commit.json", commit)
        log("wp09_pilot_generation_finished", run_directory=str(active.path), **commit)
    return active.path


def prefix_diagnostics(groups, references, codec, config, min_samples):
    """Scorer only. Plurality receives prefix(n) before correctness is attached."""
    if set(references) != {g.question_id for g in groups}:
        raise ValueError("References must exactly cover pilot observations")
    prefixes, trials = [], []
    positive, wrong, verified = set(), set(), set()
    for g in groups:
        ref = references[g.question_id]
        candidate_correct = [
            c.answer_valid and c.extracted_answer == ref["final_answer"] for c in g.candidates
        ]
        candidate_verified = [
            ok and c.format_valid and trace_matches(c, ref["expression"], codec)
            for c, ok in zip(g.candidates, candidate_correct, strict=True)
        ]
        for n in range(1, len(g.candidates) + 1):
            vote = majority(g.prefix(n))
            correct = vote.answer is not None and vote.answer == ref["final_answer"]
            representative = (
                None
                if vote.representative_index is None
                else g.candidates[vote.representative_index]
            )
            trace_ok = bool(
                correct
                and representative.format_valid
                and trace_matches(representative, ref["expression"], codec)
            )
            row = {
                "checkpoint_id": g.checkpoint_id,
                "question_id": g.question_id,
                "trial_seed": g.trial_seed,
                "draws": n,
                "answer": vote.answer,
                "representative_index": vote.representative_index,
                "top_votes": vote.top_votes,
                "valid_votes": vote.valid_votes,
                "vote_share": vote.share,
                "correct": correct,
                "verified": trace_ok,
                "observed_tokens": sum(c.token_count for c in g.candidates[:n]),
            }
            prefixes.append(row)
            if n >= min_samples:
                (positive if correct else wrong).add(g.question_id)
                if trace_ok:
                    verified.add(g.question_id)
        trials.append(
            {
                "question_id": g.question_id,
                "trial_seed": g.trial_seed,
                "difficulty": g.difficulty,
                "covered": any(candidate_correct),
                "verified_covered": any(candidate_verified),
                "correct_candidates": sum(candidate_correct),
                "verified_correct_candidates": sum(candidate_verified),
                "candidates": len(g.candidates),
                "fixed_k_correct": prefixes[-1]["correct"],
                "fixed_k_verified": prefixes[-1]["verified"],
            }
        )
    passed = (
        len(positive) >= config["minimum_independent_correct_groups"]
        and len(wrong) >= config["minimum_independent_wrong_groups"]
        and len(verified) >= config["minimum_independent_verified_groups"]
    )
    return (
        prefixes,
        trials,
        {
            "unique_questions": len(references),
            "question_trials": len(trials),
            "covered_question_trials": sum(t["covered"] for t in trials),
            "verified_covered_question_trials": sum(t["verified_covered"] for t in trials),
            "correct_candidates": sum(t["correct_candidates"] for t in trials),
            "verified_correct_candidates": sum(t["verified_correct_candidates"] for t in trials),
            "candidates": sum(t["candidates"] for t in trials),
            "fixed_k_correct_trials": sum(t["fixed_k_correct"] for t in trials),
            "prefix_min_samples": min_samples,
            "unique_questions_with_any_correct_prefix": len(positive),
            "unique_questions_with_any_wrong_prefix": len(wrong),
            "unique_questions_with_any_verified_prefix": len(verified),
            "screen_passed": passed,
            "next": "expand_calibration_fit_only"
            if passed
            else "diagnostic_only_no_calibrator_fit",
            "limit": (
                "Necessary pilot diversity screen, not proof of calibration sufficiency. "
                "Outcome-group sets may overlap; repeated seeds/prefixes are not independent."
            ),
        },
    )


def analyze(inputs, expected_fingerprint, run, benchmark, output):
    p, qs, codec = load_inputs(inputs, expected_fingerprint)
    m = verify_benchmark(benchmark, p["benchmark_fingerprint"])
    art = Path(run) / "artifacts"
    commit = json.loads((art / "generation_commit.json").read_text())
    bank = SampleBank(art / "bank")
    bm, b = bank.manifest, p["bank"]
    expected_generation = GenerationConfig(
        max_new_tokens=b["max_new_tokens"],
        do_sample=True,
        temperature=b["temperature"],
        top_p=b["top_p"],
        top_k=b["top_k"],
        eos_token_id=codec.eos_token_id,
        pad_token_id=codec.pad_token_id,
    ).to_dict()
    if (
        config_digest({k: v for k, v in commit.items() if k != "fingerprint"})
        != commit["fingerprint"]
        or commit["plan_fingerprint"] != p["fingerprint"]
        or commit["bank_fingerprint"] != bank.fingerprint
        or commit["greedy_sha256"] != file_sha256(art / "greedy.predictions.jsonl")
        or read_rows(art / "bank/questions.jsonl") != qs
        or bank.manifest["trial_seeds"] != p["bank"]["seeds"]
        or bank.manifest["candidates_per_question"] != p["bank"]["max_candidates"]
        or bank.manifest["checkpoints"][0]["sha256"] != p["checkpoint_sha256"]
        or len(bm["checkpoints"]) != 1
        or bm["checkpoints"][0]["id"] != b["checkpoint"]
        or bm["batch_size"] != b["batch_size"]
        or bm["precision"] != b["precision"]
        or bm["generation"] != expected_generation
        or bm["tokenizer_fingerprint"] != p["tokenizer_fingerprint"]
    ):
        raise ValueError("Committed generation/plan/bank mismatch")
    ids = set(p["question_ids"])
    if not ids <= set(m["question_ids"]["calibration_fit"]):
        raise ValueError("Pilot may only score calibration_fit")
    refs = {
        r["question_id"]: r
        for r in read_rows(Path(benchmark) / "references.jsonl")
        if r["question_id"] in ids
    }
    if set(refs) != ids or any(r["partition"] != "calibration_fit" for r in refs.values()):
        raise ValueError("Scorer reference partition mismatch")
    output = Path(output)
    if output.resolve().is_relative_to(bank.root):
        raise ValueError("Scoring must live outside bank")
    output.mkdir(parents=True, exist_ok=False)
    write_rows(
        output / "references.jsonl",
        [
            {
                "question_id": q["question_id"],
                "split": "validation",
                "canonical_answer": refs[q["question_id"]]["final_answer"],
            }
            for q in qs
        ],
    )
    score_bank(bank, output / "references.jsonl", output / "candidate_labels")
    prefixes, trials, summary = prefix_diagnostics(
        bank.observations(), refs, codec, p["config"], p["min_samples"]
    )
    write_rows(output / "prefixes.scored.jsonl", prefixes)
    write_rows(output / "trials.scored.jsonl", trials)
    greedy = read_rows(art / "greedy.predictions.jsonl")
    if [r["question_id"] for r in greedy] != p["question_ids"]:
        raise ValueError("Greedy order mismatch")
    gs = []
    for row, q in zip(greedy, qs, strict=True):
        if row["question"] != q["question"] or row["prompt_ids_sha256"] != config_digest(
            encode_reasoning_prompt(codec, q["question"])
        ):
            raise ValueError("Greedy prompt mismatch")
        result = extract_generated_final(row["generated_ids"], codec, row["finish_reason"])
        r = refs[row["question_id"]]
        correct = result["predicted_answer"] == r["final_answer"]
        verified = (
            correct
            and result["format_valid"]
            and trace_matches(
                SimpleNamespace(generated_ids=row["generated_ids"]), r["expression"], codec
            )
        )
        gs.append(row | result | {"correct": correct, "verified": verified})
    write_rows(output / "greedy.scored.jsonl", gs)
    report = {
        "schema_version": 1,
        "work_package": "WP09.2",
        "scope": "calibration_fit_pilot_only",
        "plan_fingerprint": p["fingerprint"],
        "benchmark_fingerprint": m["fingerprint"],
        "bank_fingerprint": bank.fingerprint,
        "summary": summary,
        "greedy": {
            "questions": len(gs),
            "correct": sum(r["correct"] for r in gs),
            "verified": sum(r["verified"] for r in gs),
        },
        "generated_tokens": sum(c.token_count for g in bank.observations() for c in g.candidates),
        "evaluation_scored": False,
        "threshold_partition_scored": False,
        "calibrator_fitted": False,
        "full_bank_generated": False,
    }
    write_json(output / "report.json", report)
    emit("wp09_pilot_analysis_finished", output=str(output), **report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, default=Path("configs/evaluation/wp09_pilot.json"))
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    g = sub.add_parser("generate")
    g.add_argument("--checkpoint", type=Path, required=True)
    g.add_argument("--output-root", type=Path, required=True)
    g.add_argument("--device", default="cuda")
    a = sub.add_parser("analyze")
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--benchmark", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    for x in (g, a):
        x.add_argument("--inputs", type=Path, required=True)
        x.add_argument("--plan-fingerprint", required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.config, args.benchmark, args.output)
    elif args.action == "generate":
        generate_pilot(
            args.inputs, args.plan_fingerprint, args.checkpoint, args.output_root, args.device
        )
    else:
        analyze(args.inputs, args.plan_fingerprint, args.run, args.benchmark, args.output)


if __name__ == "__main__":
    main()
