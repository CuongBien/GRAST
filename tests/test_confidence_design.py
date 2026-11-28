"""Design-only selection and group-separated benchmark construction."""

import copy
import json
from pathlib import Path

import pytest

from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.pilot import QUESTION_KEYS
from llm_scratch.research.confidence_design import (
    PARTITIONS,
    audit_rows,
    choose_strata,
    load_config,
    make_rows,
    partition_for,
    question_only,
    score_design,
    verify,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(ROOT / "configs/evaluation/wp09_design.json")


def test_deterministic_facts_disjoint_across_strata_and_partitions(config):
    all_rows = []
    for p in PARTITIONS:
        rows = make_rows(config, config["strata"], p, 3, set())
        assert rows == make_rows(config, config["strata"], p, 3, set())
        all_rows.extend(rows)
    audit_rows(all_rows, config, set())
    assert len({r["question_id"] for r in all_rows}) == 72
    assert all(
        partition_for(g, config["partition_seed"]) == r["partition"]
        for r in all_rows
        for g in r["fact_groups"]
    )


def test_exclusions_and_leakage_and_reference_tampering(config):
    row = make_rows(config, config["strata"][:1], "design", 1, set())[0]
    excluded = set(row["fact_groups"])
    different = make_rows(config, config["strata"][:1], "design", 1, excluded)[0]
    assert not excluded & set(different["fact_groups"])
    with pytest.raises(ValueError, match="leakage"):
        audit_rows([row, row], config, set())
    broken = copy.deepcopy(row)
    broken["partition"] = "evaluation"
    with pytest.raises(ValueError, match="leakage"):
        audit_rows([broken], config, set())
    broken = row | {"final_answer": "wrong"}
    with pytest.raises(ValueError, match="reference"):
        audit_rows([broken], config, set())


def test_observations_cannot_contain_gold(config):
    rows = make_rows(config, config["strata"], "design", 1, set())
    assert all(set(r) == QUESTION_KEYS for r in question_only(rows))
    assert question_only(rows) == question_only(
        [r | {"final_answer": "poison", "reasoning": "poison", "expression": {}} for r in rows]
    )


def test_locked_selection_rule_eligibility_and_diagnostic_fallback():
    def result(n, fmt=1):
        return {"questions": 8, "correct": n, "accuracy": n / 8, "format_valid_rate": fmt}

    summary = {
        "easy": result(8),
        "hard": result(0),
        "mixed-a": result(4),
        "mixed-b": result(5),
        "invalid": result(4, 0.5),
    }
    assert choose_strata(summary)["selected_strata"] == ["mixed-a", "mixed-b"]
    assert choose_strata(summary)["status"] == "provisional_mixed_workload"
    fallback = choose_strata({"easy": result(8), "hard": result(0)})
    assert fallback["status"] == "diagnostic_only"
    assert fallback["selected_strata"] == ["easy", "hard"]


def test_design_scorer_refuses_main_partition_and_wrong_trace(config):
    rows = make_rows(config, config["strata"][:1], "design", 2, set())
    codec = ByteBPETokenizer(vocab_size=264)
    preds = [
        {
            "question_id": r["question_id"],
            "generated_ids": codec.encode(
                r["reasoning"] + "</think><final>" + r["final_answer"] + "</final>",
                allowed_special={"</think>", "<final>", "</final>"},
                add_eos=True,
            ),
            "finish_reason": "eos",
        }
        for r in rows
    ]
    scored, summary = score_design(preds, rows, codec)
    assert all(r["verified"] for r in scored)
    assert summary["cross-short"]["correct"] == 2
    with pytest.raises(ValueError, match="design partition only"):
        score_design(preds, [r | {"partition": "evaluation"} for r in rows], codec)
    with pytest.raises(ValueError, match="complete"):
        score_design(preds[:1], rows, codec)
    wrong = [r | {"reasoning": "incorrect trace"} for r in rows]
    scored, summary = score_design(preds, wrong, codec)
    assert summary["cross-short"]["correct"] == 2 and not any(r["verified"] for r in scored)


def test_build_only_generates_design_and_checks_payloads(tmp_path, monkeypatch):
    import test_easy_recovery

    from llm_scratch.data.datasets import file_sha256
    from llm_scratch.research import confidence_design as design

    class Factory:
        def mktemp(self, name):
            p = tmp_path / name
            p.mkdir()
            return p

    inputs = test_easy_recovery.recovery_inputs.__wrapped__(Factory())
    recovery_root = inputs["prepared_root"]
    old_root = recovery_root.parent / "source/prepared"
    cfg = load_config(ROOT / "configs/evaluation/wp09_design.json")
    cfg.update(
        checkpoint_sha256=file_sha256(inputs["source_checkpoint"]),
        tokenizer_fingerprint=ByteBPETokenizer.load(inputs["tokenizer"]).fingerprint,
        curriculum_fingerprint=json.loads((old_root / "manifest.json").read_text())["fingerprint"],
        recovery_fingerprint=json.loads((recovery_root / "manifest.json").read_text())[
            "fingerprint"
        ],
        strata=cfg["strata"][:2],
        main_per_stratum={p: 2 for p in PARTITIONS[1:]},
    )
    cfg["generation"]["max_new_tokens"] = 8
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    calls = []

    def generate(model, codec, questions, policy, **kwargs):
        assert len(questions) == 16 and all(set(q) == QUESTION_KEYS for q in questions)
        calls.append(questions)
        return [
            {
                "question_id": q["question_id"],
                "generated_ids": [codec.eos_token_id],
                "finish_reason": "eos",
            }
            for q in questions
        ]

    monkeypatch.setattr(design, "generate_questions", generate)
    out = design.build(
        config,
        old_root,
        recovery_root,
        inputs["tokenizer"],
        inputs["source_checkpoint"],
        tmp_path / "benchmark",
        device="cpu",
    )
    manifest = verify(out)
    assert len(calls) == 1 and manifest["decision"]["status"] == "diagnostic_only"
    assert manifest["partition_counts"] == {p: 4 for p in PARTITIONS[1:]}
    assert not manifest["held_out_predictions_generated"] and not manifest["test_files_read"]
    with pytest.raises(ValueError, match="fingerprint"):
        verify(out, "0" * 64)
    (out / "questions.jsonl").write_text("altered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify(out, manifest["fingerprint"])
