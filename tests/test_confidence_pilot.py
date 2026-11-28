"""Pilot scope, generation/scoring boundary and prefix-label viability checks."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import test_easy_recovery

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.research import confidence_design, confidence_pilot
from llm_scratch.research.bank_schema import (
    CandidateObservation,
    PolicyObservation,
    describe_output,
)
from llm_scratch.research.sample_bank import SampleBank

ROOT = Path(__file__).resolve().parents[1]


def test_selection_is_balanced_deterministic_and_fit_only():
    questions, index = [], []
    for part in ("calibration_fit", "calibration_threshold", "evaluation"):
        for level in ("a", "b"):
            for i in range(3):
                qid = f"{part}-{level}-{i}"
                q = {
                    "question_id": qid,
                    "question": qid,
                    "source": "test",
                    "split": "validation",
                    "difficulty": level,
                    "operation_count": 1,
                    "reasoning_steps": 1,
                    "prompt_length": "short",
                    "prompt_word_count": 1,
                }
                questions.append(q)
                index.append({"question_id": qid, "partition": part, "difficulty": level})
    selected = confidence_pilot.select_questions(questions, index, 2, 42)
    assert selected == confidence_pilot.select_questions(questions[::-1], index[::-1], 2, 42)
    assert len(selected) == 4 and all(
        q["question_id"].startswith("calibration_fit") for q in selected
    )
    with pytest.raises(ValueError, match="Insufficient"):
        confidence_pilot.select_questions(questions, index, 4, 42)


def observation(codec, answers, qid="q"):
    candidates = []
    for i, answer in enumerate(answers):
        ids = codec.encode(
            "R1 = 1 + 2 = " + answer + ".</think><final>" + answer + "</final>",
            allowed_special={"</think>", "<final>", "</final>"},
            add_eos=True,
        )
        probs = [-1.0] * len(ids)
        derived = describe_output(ids, probs, "eos", codec)
        candidates.append(
            CandidateObservation(
                candidate_index=i,
                generation_seed=42,
                batch_start=0,
                batch_row=i,
                batch_size=len(answers),
                generated_ids=tuple(ids),
                generated_logprobs=tuple(probs),
                finish_reason="eos",
                **derived,
            )
        )
    return PolicyObservation("cp", qid, "question", "test", 42, tuple(candidates))


def test_oracle_is_not_winner_and_future_candidate_cannot_change_prefix():
    codec = ByteBPETokenizer(vocab_size=264)
    ref = {
        "final_answer": "3",
        "expression": {"operator": "+", "left": {"value": 1}, "right": {"value": 2}},
    }
    config = {
        "minimum_independent_correct_groups": 1,
        "minimum_independent_wrong_groups": 1,
        "minimum_independent_verified_groups": 1,
    }
    first = observation(codec, ["4", "3", "4"])
    rows, trials, result = confidence_pilot.prefix_diagnostics(
        (first,), {"q": ref}, codec, config, 2
    )
    assert trials[0]["covered"] and trials[0]["verified_covered"]
    assert result["unique_questions_with_any_correct_prefix"] == 0
    assert not result["screen_passed"]
    changed = observation(codec, ["4", "3", "3"])
    more, _, mixed = confidence_pilot.prefix_diagnostics((changed,), {"q": ref}, codec, config, 2)
    assert rows[:2] == more[:2]
    assert not rows[1]["correct"] and more[2]["correct"]
    assert mixed["screen_passed"] and mixed["next"] == "expand_calibration_fit_only"


def test_group_counts_do_not_count_repeated_seeds_as_independent():
    from dataclasses import replace

    codec = ByteBPETokenizer(vocab_size=264)
    ref = {
        "final_answer": "3",
        "expression": {"operator": "+", "left": {"value": 1}, "right": {"value": 2}},
    }
    g = observation(codec, ["3", "3", "4", "4", "4"])
    groups = (g, replace(g, trial_seed=43), replace(g, trial_seed=44))
    cfg = {
        "minimum_independent_correct_groups": 2,
        "minimum_independent_wrong_groups": 1,
        "minimum_independent_verified_groups": 1,
    }
    _, _, summary = confidence_pilot.prefix_diagnostics(groups, {"q": ref}, codec, cfg, 2)
    assert summary["unique_questions_with_any_correct_prefix"] == 1
    assert summary["unique_questions_with_any_wrong_prefix"] == 1
    assert not summary["screen_passed"]


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    base = test_easy_recovery.recovery_inputs.__wrapped__(tmp_path_factory)
    root = tmp_path_factory.mktemp("confidence-pilot")
    old = base["prepared_root"].parent / "source/prepared"
    cfg = confidence_design.load_config(ROOT / "configs/evaluation/wp09_design.json")
    cfg.update(
        checkpoint_sha256=file_sha256(base["source_checkpoint"]),
        tokenizer_fingerprint=ByteBPETokenizer.load(base["tokenizer"]).fingerprint,
        curriculum_fingerprint=json.loads((old / "manifest.json").read_text())["fingerprint"],
        recovery_fingerprint=json.loads((base["prepared_root"] / "manifest.json").read_text())[
            "fingerprint"
        ],
        strata=cfg["strata"][:2],
        main_per_stratum={p: 3 for p in confidence_design.PARTITIONS[1:]},
    )
    cfg["generation"]["max_new_tokens"] = 8
    config = root / "design.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")

    def fake_design(model, codec, questions, policy, **kwargs):
        return [
            {
                "question_id": q["question_id"],
                "generated_ids": [codec.eos_token_id],
                "finish_reason": "eos",
            }
            for q in questions
        ]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(confidence_design, "generate_questions", fake_design)
        benchmark = confidence_design.build(
            config,
            old,
            base["prepared_root"],
            base["tokenizer"],
            base["source_checkpoint"],
            root / "benchmark",
            "cpu",
        )
    m = confidence_design.verify(benchmark)
    pcfg = json.loads((ROOT / "configs/evaluation/wp09_pilot.json").read_text())
    pcfg.update(benchmark_fingerprint=m["fingerprint"], questions_per_stratum=1)
    path = root / "pilot.json"
    path.write_text(json.dumps(pcfg), encoding="utf-8")
    plan = confidence_pilot.prepare(path, benchmark, root / "inputs")
    return root, benchmark, base["source_checkpoint"], plan


def test_input_integrity_and_no_gold_files_allowed(inputs, tmp_path):
    import shutil

    root, _, _, plan = inputs
    dest = Path(shutil.copytree(root / "inputs", tmp_path / "inputs"))
    with pytest.raises(ValueError, match="fingerprint"):
        confidence_pilot.load_inputs(dest, "0" * 64)
    (dest / "references.jsonl").write_text("labels", encoding="utf-8")
    with pytest.raises(ValueError, match="question-only"):
        confidence_pilot.load_inputs(dest, plan["fingerprint"])


def test_actual_generation_repeat_external_scoring_and_bank_boundary(inputs, monkeypatch):
    root, benchmark, checkpoint, plan = inputs
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert "references" not in path.name and "scored" not in path.name
        return original(path, *args, **kwargs)

    runs = []
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        for _ in range(2):
            runs.append(
                confidence_pilot.generate_pilot(
                    root / "inputs", plan["fingerprint"], checkpoint, root / "runs", "cpu"
                )
            )
    banks = [SampleBank(r / "artifacts/bank") for r in runs]
    assert banks[0].fingerprint == banks[1].fingerprint
    assert banks[0].observations() == banks[1].observations()
    assert banks[0].manifest["records"] == 96
    assert all("correct" not in asdict(c) for g in banks[0].observations() for c in g.candidates)
    report = confidence_pilot.analyze(
        root / "inputs", plan["fingerprint"], runs[0], benchmark, root / "analysis"
    )
    assert report["summary"]["question_trials"] == 6 and report["greedy"]["questions"] == 2
    assert not report["summary"]["screen_passed"]
    assert not report["evaluation_scored"] and not report["threshold_partition_scored"]
    assert not report["calibrator_fitted"] and not report["full_bank_generated"]
    assert len(confidence_pilot.read_rows(root / "analysis/prefixes.scored.jsonl")) == 96
    assert set(
        r["question_id"] for r in confidence_pilot.read_rows(root / "analysis/references.jsonl")
    ) == set(plan["question_ids"])
    with pytest.raises(ValueError, match="outside bank"):
        confidence_pilot.analyze(
            root / "inputs",
            plan["fingerprint"],
            runs[0],
            benchmark,
            runs[0] / "artifacts/bank/labels",
        )
    greedy = runs[0] / "artifacts/greedy.predictions.jsonl"
    greedy.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="generation/plan/bank"):
        confidence_pilot.analyze(
            root / "inputs", plan["fingerprint"], runs[0], benchmark, root / "corrupt-analysis"
        )
