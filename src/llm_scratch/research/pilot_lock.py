"""Freeze a balanced validation pilot without using model predictions or gold in ranking."""

from __future__ import annotations

import gc
import json
import tomllib
from collections import Counter
from pathlib import Path

from llm_scratch.curriculum.data import Corpus
from llm_scratch.curriculum.evaluation import load_selected, select_validation
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.pilot import load_manifest, load_questions
from llm_scratch.research.bank_schema import digest, integer, validate_questions
from llm_scratch.research.sample_bank import emit, load_config
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import canonical_json, config_digest

DIFFICULTIES = ("easy", "medium", "hard")
LOCK_KEYS = {
    "schema_version",
    "work_package",
    "split",
    "selection_protocol",
    "selection_seed",
    "per_difficulty",
    "sources",
    "bank_contract",
    "selection_ids",
    "excluded_groups",
    "excluded_question_ids",
    "eligible_questions",
    "question_ids",
    "question_groups",
    "difficulty_counts",
    "questions_sha256",
    "scope",
    "fingerprint",
}


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, rows):
    atomic_text(path, "".join(canonical_json(r) + "\n" for r in rows))


def load_settings(path):
    path = Path(path).resolve()
    d = tomllib.loads(path.read_text(encoding="utf-8"))
    paths = {"bank_config", "dataset", "curriculum", "tokenizer"}
    fingerprints = {"wp03_fingerprint", "curriculum_fingerprint", "tokenizer_fingerprint"}
    if (
        set(d)
        != {
            "schema_version",
            "expected_questions",
            "per_difficulty",
            "selection_seed",
            *paths,
            *fingerprints,
        }
        or type(d["schema_version"]) is not int
        or d["schema_version"] != 1
    ):
        raise ValueError("Invalid WP08 pilot settings schema")
    if (
        any(not digest(d[k]) for k in fingerprints)
        or not integer(d["expected_questions"], 3)
        or not integer(d["per_difficulty"], 1)
        or not integer(d["selection_seed"], 0, 2**32 - 1)
    ):
        raise ValueError("Invalid pilot counts/fingerprints/seed")
    for k in paths:
        if not isinstance(d[k], str) or not d[k]:
            raise ValueError("Invalid pilot input path")
        d[k] = str((path.parent / d[k]).resolve())
    return d


def bank_contract(d):
    """Portable protocol identity; paths and run names do not change the experiment."""
    return {
        "bank": d["bank"],
        "generation": d["generation"],
        "reproducibility": d["reproducibility"],
        "checkpoints": [{k: cp[k] for k in ("id", "kind", "sha256")} for cp in d["checkpoints"]],
    }


def choose_questions(questions, group_by_prompt, excluded_groups, *, per_difficulty, seed):
    """Ranking consumes question-only rows and group IDs, never answers or scores."""
    validate_questions(questions)
    if not integer(per_difficulty, 1) or not integer(seed, 0, 2**32 - 1):
        raise ValueError("Invalid pilot count/seed")
    if set(q["difficulty"] for q in questions) != set(DIFFICULTIES):
        raise ValueError("Pilot requires all three declared difficulty levels")
    if any(q["question"] not in group_by_prompt for q in questions):
        raise ValueError("Pilot question missing from curriculum group index")
    excluded = [q for q in questions if group_by_prompt[q["question"]] in excluded_groups]
    eligible = [q for q in questions if group_by_prompt[q["question"]] not in excluded_groups]
    chosen, groups = [], set()
    for difficulty in DIFFICULTIES:
        pool = sorted(
            (q for q in eligible if q["difficulty"] == difficulty),
            key=lambda q: (
                config_digest(["wp08-pilot-v1", seed, q["question_id"]]),
                q["question_id"],
            ),
        )
        rows = []
        for q in pool:
            group = group_by_prompt[q["question"]]
            if group not in groups:
                rows.append(q)
                groups.add(group)
            if len(rows) == per_difficulty:
                break
        if len(rows) != per_difficulty:
            raise ValueError(f"Insufficient independent groups for {difficulty}")
        chosen.extend(rows)
    return chosen, excluded


def prepare_pilot(config, output):
    settings = load_settings(config)
    loaded = load_config(settings["bank_config"])
    codec = ByteBPETokenizer.load(
        settings["tokenizer"], expected_fingerprint=settings["tokenizer_fingerprint"]
    )
    dataset = Path(settings["dataset"])
    manifest = load_manifest(dataset)
    if manifest["dataset_fingerprint"] != settings["wp03_fingerprint"]:
        raise ValueError("Pilot WP03 identity mismatch")
    questions = load_questions(
        dataset, manifest, "validation", expected_records=settings["expected_questions"]
    )
    corpus = Corpus(settings["curriculum"], "validation", codec)
    if (
        corpus.meta["fingerprint"] != settings["curriculum_fingerprint"]
        or corpus.meta["config"]["wp03_fingerprint"] != manifest["dataset_fingerprint"]
    ):
        raise ValueError("Pilot curriculum/WP03 identity mismatch")
    selected_specs = [c for c in loaded.data["checkpoints"] if c["kind"] == "curriculum_selected"]
    if len(selected_specs) != 1:
        raise ValueError("Pilot exclusion requires exactly one audited selected checkpoint")
    cp = selected_specs[0]
    model, meta = load_selected(Path(cp["path"]), codec, expected_sha256=cp["sha256"])
    del model
    gc.collect()
    contract = meta["contract"]
    if contract["dataset_fingerprint"] != corpus.meta["fingerprint"]:
        raise ValueError("Selected checkpoint/curriculum identity mismatch")
    evaluation = contract["recipe"]["evaluation"]
    selection = select_validation(
        corpus.examples, evaluation["per_task"], evaluation["selection_seed"]
    )
    if [r["question_id"] for r in selection] != contract["selection_ids"]:
        raise ValueError("Cannot reproduce checkpoint selection IDs")
    excluded_groups = {r["group"] for r in selection}
    index = {r["prompt"]: r["group"] for r in corpus.examples}
    chosen, excluded = choose_questions(
        questions,
        index,
        excluded_groups,
        per_difficulty=settings["per_difficulty"],
        seed=settings["selection_seed"],
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "questions.jsonl", chosen)
    lock = {
        "schema_version": 1,
        "work_package": "WP08.2",
        "split": "validation",
        "selection_protocol": "hash-stratified-one-per-group-excluding-selection-v1",
        "selection_seed": settings["selection_seed"],
        "per_difficulty": settings["per_difficulty"],
        "sources": {
            k: settings[k]
            for k in ("wp03_fingerprint", "curriculum_fingerprint", "tokenizer_fingerprint")
        },
        "bank_contract": bank_contract(loaded.data),
        "selection_ids": contract["selection_ids"],
        "excluded_groups": sorted(excluded_groups),
        "excluded_question_ids": sorted(q["question_id"] for q in excluded),
        "eligible_questions": len(questions) - len(excluded),
        "question_ids": [q["question_id"] for q in chosen],
        "question_groups": [index[q["question"]] for q in chosen],
        "difficulty_counts": dict(Counter(q["difficulty"] for q in chosen)),
        "questions_sha256": file_sha256(output / "questions.jsonl"),
        "scope": "development_validation_not_an_untouched_test_set",
    }
    lock["fingerprint"] = config_digest(lock)
    atomic_text(output / "manifest.json", canonical_json(lock) + "\n")
    emit(
        "pilot_frozen",
        directory=str(output),
        fingerprint=lock["fingerprint"],
        questions=len(chosen),
        excluded_questions=len(excluded),
    )
    return lock


def load_pilot(root, settings, data):
    root = Path(root)
    lock = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    questions = read_jsonl(root / "questions.jsonl")
    validate_questions(questions)
    if (
        not isinstance(lock, dict)
        or set(lock) != LOCK_KEYS
        or lock.get("work_package") != "WP08.2"
        or type(lock.get("schema_version")) is not int
        or lock.get("schema_version") != 1
        or lock.get("selection_protocol") != "hash-stratified-one-per-group-excluding-selection-v1"
        or lock.get("scope") != "development_validation_not_an_untouched_test_set"
        or lock.get("split") != "validation"
        or config_digest({k: v for k, v in lock.items() if k != "fingerprint"})
        != lock.get("fingerprint")
        or lock.get("questions_sha256") != file_sha256(root / "questions.jsonl")
        or lock.get("question_ids") != [q["question_id"] for q in questions]
        or lock.get("bank_contract") != bank_contract(data)
        or lock.get("selection_seed") != settings["selection_seed"]
        or lock.get("per_difficulty") != settings["per_difficulty"]
        or lock.get("sources")
        != {
            k: settings[k]
            for k in ("wp03_fingerprint", "curriculum_fingerprint", "tokenizer_fingerprint")
        }
        or lock.get("difficulty_counts") != {d: settings["per_difficulty"] for d in DIFFICULTIES}
        or dict(Counter(q["difficulty"] for q in questions)) != lock["difficulty_counts"]
        or len(lock.get("question_groups", [])) != len(questions)
        or len(set(lock["question_groups"])) != len(questions)
        or set(lock["question_groups"]) & set(lock["excluded_groups"])
        or set(lock["question_ids"]) & set(lock["excluded_question_ids"])
    ):
        raise ValueError("Frozen pilot identity/config/exclusion mismatch")
    return lock, questions
