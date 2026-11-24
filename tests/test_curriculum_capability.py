"""Completed-run verification, selection exclusion and validation-only selected-model evaluation."""

import json
import zipfile
from pathlib import Path

import pytest

from llm_scratch.curriculum.acceptance import fixture
from llm_scratch.curriculum.data import Corpus, prepare
from llm_scratch.curriculum.evaluation import load_selected
from llm_scratch.curriculum.runner import run
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.research.curriculum_capability import (
    audit_summary,
    load_protocol,
    run_evaluation,
    selection_and_holdout,
)
from llm_scratch.utils.config import canonical_json, config_digest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/evaluation/wp07_curriculum_selected.toml"


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("selected-eval")
    cfg, _, token, source = fixture(root / "inputs", ROOT)
    wp03 = cfg.parent / "wp03"
    manifest = json.loads((wp03 / "manifest.json").read_text())
    old_fp = manifest["dataset_fingerprint"]
    raw = [json.loads(s) for s in (wp03 / "math/validation.jsonl").read_text().splitlines()]
    questions, references = [], []
    for difficulty in ("easy", "medium", "hard"):
        r = next(r for r in raw if r["difficulty"] == difficulty)
        questions.append(
            {
                "question_id": r["question_id"],
                "question": r["question"],
                "source": r["source"],
                "split": "validation",
                "difficulty": difficulty,
                "operation_count": 1,
                "reasoning_steps": 1,
                "prompt_length": "short",
                "prompt_word_count": r["prompt_word_count"],
            }
        )
        references.append(
            {
                "question_id": r["question_id"],
                "canonical_answer": r["canonical_answer"],
                "split": "validation",
            }
        )
    for role, rows in (("questions", questions), ("references", references)):
        path = wp03 / f"pilot/validation.{role}.jsonl"
        path.parent.mkdir(exist_ok=True)
        path.write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8")
        manifest["files"][f"pilot/validation.{role}.jsonl"] = {
            "path": f"pilot/validation.{role}.jsonl",
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "records": len(rows),
            "role": role,
        }
    manifest["pilot"] = {
        "validation": {
            "records": 3,
            "difficulty_counts": {d: 1 for d in ("easy", "medium", "hard")},
        }
    }
    manifest["dataset_fingerprint"] = config_digest(
        {k: v for k, v in manifest.items() if k != "dataset_fingerprint"}
    )
    (wp03 / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    data_cfg = cfg.parent / "data.toml"
    data_cfg.write_text(data_cfg.read_text().replace(old_fp, manifest["dataset_fingerprint"]))
    prepared = prepare(data_cfg, wp03, token, root / "prepared-with-pilot")
    trained = run(cfg, prepared, token, init_checkpoint=source, output_root=root / "train-runs")
    summary = root / "summary.zip"
    with zipfile.ZipFile(summary, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(trained.rglob("*")):
            if path.is_file() and path.suffix != ".pt":
                z.write(path, path.relative_to(trained).as_posix())
    selected = trained / "checkpoints/selected_model.pt"
    baseline = root / "baseline.jsonl"
    baseline.write_text(
        "".join(
            canonical_json(q | {"generated_ids": [], "finish_reason": "max_new_tokens"}) + "\n"
            for q in questions
        ),
        encoding="utf-8",
    )
    text = (
        TEMPLATE.read_text()
        .replace("expected_pilot_questions = 150", "expected_pilot_questions = 3")
        .replace("holdout_per_task = 32", "holdout_per_task = 4")
        .replace("max_new_tokens = 256", "max_new_tokens = 8")
    )
    paths = {"checkpoint": selected, "summary": summary, "baseline": baseline}
    lines = []
    section = None
    for line in text.splitlines():
        if line.startswith("["):
            section = line.strip("[]")
        if section in paths and line.startswith("path ="):
            line = "path = " + json.dumps(paths[section].resolve().as_posix())
        if section in paths and line.startswith("sha256 ="):
            line = "sha256 = " + json.dumps(file_sha256(paths[section]))
        lines.append(line)
    protocol = root / "evaluation.toml"
    protocol.write_text("\n".join(lines), encoding="utf-8")
    codec = ByteBPETokenizer.load(token)
    _, meta = load_selected(selected, codec)
    corpus = Corpus(prepared, "validation", codec)
    selection, holdout = selection_and_holdout(corpus.examples, meta["contract"], 4, 42)
    return {
        "root": root,
        "config": protocol,
        "wp03": wp03,
        "prepared": prepared,
        "tokenizer": token,
        "selected": selected,
        "summary": summary,
        "meta": meta,
        "corpus": corpus,
        "selection": selection,
        "holdout": holdout,
    }


def test_summary_matches_selected_weights_metrics_and_training_history(inputs):
    result = audit_summary(
        inputs["summary"],
        file_sha256(inputs["summary"]),
        inputs["meta"],
        inputs["corpus"],
        inputs["selection"],
    )
    assert result["report"]["step"] == 6 and len(result["history"]["train"]) == 6
    assert result["report"]["best"]["step"] == 0
    assert len(result["selected_predictions"]) == 12


def test_new_validation_excludes_whole_selection_groups(inputs):
    assert len(inputs["holdout"]) == 12
    assert not {r["group"] for r in inputs["selection"]} & {r["group"] for r in inputs["holdout"]}
    with pytest.raises(ValueError, match="validation only"):
        selection_and_holdout(
            [r | {"split": "test"} for r in inputs["corpus"].examples],
            inputs["meta"]["contract"],
            4,
            42,
        )


@pytest.mark.parametrize("kind", ["metrics", "best", "tokens", "validation_loss", "unsafe"])
def test_summary_tamper_rejected_even_with_new_archive_checksum(inputs, tmp_path, kind):
    with zipfile.ZipFile(inputs["summary"]) as z:
        entries = {n: z.read(n) for n in z.namelist()}
    if kind == "unsafe":
        entries["../escape.json"] = b"{}"
    else:
        name = {
            "metrics": "artifacts/generated_validation.json",
            "best": "artifacts/best_selection.json",
            "tokens": "artifacts/history.json",
            "validation_loss": "artifacts/history.json",
        }[kind]
        data = json.loads(entries[name])
        if kind == "metrics":
            data[0]["summary"]["macro_exact_match"] = 1
        elif kind == "best":
            data["step"] = 99
        elif kind == "tokens":
            data["train"][0]["tokens_optimized"] += 1
        else:
            data["validation"][0]["loss_tokens"] += 1
        entries[name] = json.dumps(data).encode()
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, body in entries.items():
            z.writestr(name, body)
    with pytest.raises(ValueError):
        audit_summary(
            path, file_sha256(path), inputs["meta"], inputs["corpus"], inputs["selection"]
        )


@pytest.mark.parametrize(
    "old,new",
    [
        ("holdout_per_task = 32", "holdout_per_task = 3"),
        ('precision = "float32"', 'precision = "float16"'),
        ("expected_pilot_questions = 150", "expected_pilot_questions = true"),
    ],
)
def test_protocol_rejects_invalid_settings(tmp_path, old, new):
    path = tmp_path / "bad.toml"
    path.write_text(TEMPLATE.read_text().replace(old, new))
    with pytest.raises(ValueError):
        load_protocol(path)


def test_full_pipeline_never_reads_test_and_saves_predictions_before_pilot_gold(
    inputs, tmp_path, monkeypatch
):
    original = Path.open
    output = tmp_path / "runs"

    def guarded(path, *args, **kwargs):
        if path.name.startswith("test."):
            raise AssertionError("Held-out test read")
        if path.name == "validation.references.jsonl":
            assert list(output.glob("*/artifacts/pilot_validation.predictions.jsonl"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    run_path = run_evaluation(
        inputs["config"],
        inputs["wp03"],
        inputs["prepared"],
        inputs["tokenizer"],
        output_root=output,
    )
    report = json.loads((run_path / "artifacts/curriculum_evaluation.json").read_text())
    assert report["status"] == "completed"
    assert report["selection_replay"]["exact_generated_replay"]
    assert report["pilot_validation"]["overall"]["questions"] == 3
    assert report["additional_validation"]["tasks"]["copy"]["questions"] == 4
    assert report["cache_equivalence"]["status"] == "passed"
    assert report["source_checkpoint_unchanged"] and report["source_summary_unchanged"]
