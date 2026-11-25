"""Pilot exclusion, fixed identities, coverage oracle and actual offline integration."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from llm_scratch.data.difficulty import Expression
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.research.bank_schema import CandidateObservation, PolicyObservation
from llm_scratch.research.pilot_acceptance import check
from llm_scratch.research.pilot_lock import (
    choose_questions,
    load_pilot,
    load_settings,
    prepare_pilot,
    read_jsonl,
)
from llm_scratch.research.pilot_metrics import summarize_bank, trace_matches
from llm_scratch.research.sample_bank import load_config
from llm_scratch.utils.config import config_digest

ROOT = Path(__file__).resolve().parents[1]


def pool():
    rows = []
    for d in ("easy", "medium", "hard"):
        for n in range(8):
            text = f"Question {d} {n}"
            rows.append(
                {
                    "question_id": f"q-{d}-{n}",
                    "question": text,
                    "split": "validation",
                    "source": "fixture",
                    "difficulty": d,
                    "operation_count": 1,
                    "reasoning_steps": 1,
                    "prompt_length": "short",
                    "prompt_word_count": len(text.split()),
                }
            )
    return rows, {
        r["question"]: f"g-{r['difficulty']}-{int(r['question_id'][-1]) // 2}" for r in rows
    }


def test_question_only_ranking_stable_and_excludes_whole_groups():
    rows, index = pool()
    excluded = {"g-easy-0", "g-medium-0", "g-hard-0"}
    selected, removed = choose_questions(rows, index, excluded, per_difficulty=2, seed=42)
    reordered, _ = choose_questions(
        list(reversed(rows)), index, excluded, per_difficulty=2, seed=42
    )
    assert selected == reordered and len(selected) == 6 and len(removed) == 6
    assert not {index[q["question"]] for q in selected} & excluded
    assert len({index[q["question"]] for q in selected}) == 6
    with pytest.raises(ValueError, match="Insufficient"):
        choose_questions(rows, index, excluded, per_difficulty=4, seed=42)
    with pytest.raises(ValueError, match="missing"):
        choose_questions(rows, {}, excluded, per_difficulty=1, seed=42)
    rows[0]["canonical_answer"] = "7"
    with pytest.raises(ValueError, match="gold"):
        choose_questions(rows, index, excluded, per_difficulty=1, seed=42)


def candidate(codec, rationale, index=0):
    ids = codec.encode(rationale) + [codec.special_id("</think>"), codec.special_id("<final>")]
    ids += codec.encode("7") + [codec.special_id("</final>"), codec.eos_token_id]
    return CandidateObservation(
        index,
        42,
        0,
        index,
        2,
        tuple(ids),
        tuple([-1.0] * len(ids)),
        codec.decode(ids),
        "eos",
        "7",
        True,
        True,
        len(ids),
        -len(ids),
        -1.0,
    )


def test_coverage_oracle_prefix_not_greedy_and_trace_not_lucky_final():
    codec = ByteBPETokenizer(vocab_size=264)
    tree = Expression(operator="+", left=Expression(value=3), right=Expression(value=4)).as_dict()
    wrong_trace = candidate(codec, "R1 = 3 + 4 = 8.")
    good_trace = candidate(codec, "  R1 = 3 + 4 = 7. \n", index=1)
    assert trace_matches(good_trace, tree, codec)
    assert not trace_matches(wrong_trace, tree, codec)

    class Bank:
        manifest = {
            "candidates_per_question": 2,
            "checkpoints": [{"id": "cp"}],
            "trial_seeds": [42, 43],
        }

        def observations(self):
            return tuple(
                PolicyObservation("cp", "q", "Compute", "easy", seed, (wrong_trace, good_trace))
                for seed in (42, 43)
            )

    labels = [
        {
            "checkpoint_id": "cp",
            "question_id": "q",
            "trial_seed": seed,
            "candidate_index": k,
            "correct": (k == 1 or seed == 43),
        }
        for seed in (42, 43)
        for k in range(2)
    ]
    audits, prefixes, report = summarize_bank(Bank(), labels, {"q": tree}, codec)
    summary = report["checkpoints"][0]["seed_summary"]["all"]
    assert summary["1"]["coverage_mean"] == 0.5
    assert summary["1"]["verified_coverage_mean"] == 0
    assert summary["2"]["coverage_mean"] == summary["2"]["verified_coverage_mean"] == 1
    assert report["unique_questions"] == 1
    assert len(audits) == len(prefixes) == 4
    assert sum(r["verified_correct"] for r in audits) == 2
    assert report["checkpoints"][0]["seeds"][0]["scopes"]["all"]["2"]["candidates"] == 2
    assert not trace_matches(replace(good_trace, generated_ids=()), tree, codec)
    with pytest.raises(ValueError, match="grid"):
        summarize_bank(Bank(), labels[:-1], {"q": tree}, codec)
    with pytest.raises(ValueError, match="grid"):
        summarize_bank(Bank(), labels + [labels[0]], {"q": tree}, codec)


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    return check(tmp_path_factory.mktemp("wp08-parent") / "smoke", ROOT)


def test_actual_prepare_run_scores_outside_bank_and_no_test_access(offline):
    config, pilot, run = offline
    events = read_jsonl(run / "logs/events.jsonl")
    assert events[-2]["event"] == "pilot_completed"
    assert any(e["event"] == "sample_bank_committed" for e in events)
    assert not list((run / "artifacts/bank").rglob("*scored*"))
    report = json.loads((run / "artifacts/pilot_report.json").read_text(encoding="utf-8"))
    assert len(report["metrics"]["checkpoints"]) == 2 and report["candidates"] == 24
    lock, _ = load_pilot(
        pilot, load_settings(config), load_config(load_settings(config)["bank_config"]).data
    )
    assert len(lock["excluded_question_ids"]) >= 1


def test_preparation_does_not_read_references_or_test_files(offline, tmp_path, monkeypatch):
    config, _, _ = offline
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert "references" not in path.name and "test" not in path.name
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    prepare_pilot(config, tmp_path / "new-pilot")


@pytest.mark.parametrize("mutation", ["bank", "questions", "exclusion", "extra"])
def test_frozen_contract_or_payload_change_rejected(offline, tmp_path, mutation):
    config, pilot, _ = offline
    settings = load_settings(config)
    data = load_config(settings["bank_config"]).data
    root = tmp_path / "pilot"
    root.mkdir()
    lock = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    text = (pilot / "questions.jsonl").read_text(encoding="utf-8")
    if mutation == "bank":
        data = copy.deepcopy(data)
        data["bank"]["batch_size"] += 1
    elif mutation == "questions":
        text += "\n"
    elif mutation == "exclusion":
        lock["excluded_groups"].append(lock["question_groups"][0])
    else:
        lock["canonical_answer"] = "7"
    lock["fingerprint"] = config_digest({k: v for k, v in lock.items() if k != "fingerprint"})
    (root / "manifest.json").write_text(json.dumps(lock), encoding="utf-8")
    (root / "questions.jsonl").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_pilot(root, settings, data)
