"""Offline tiny WP08.2 integration fixture, including selection exclusion and scoring."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from llm_scratch.curriculum.acceptance import fixture
from llm_scratch.curriculum.data import prepare
from llm_scratch.curriculum.runner import run as train_fixture
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.difficulty import expression_from_dict
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.research.feasibility_pilot import run
from llm_scratch.research.pilot_lock import prepare_pilot, read_jsonl, write_jsonl
from llm_scratch.research.sample_bank import SampleBank, emit
from llm_scratch.utils.config import config_digest


def make_inputs(root, source_root):
    root, source_root = Path(root), Path(source_root)
    cfg, _, token, source = fixture(root / "inputs", source_root)
    dataset = cfg.parent / "wp03"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    old = manifest["dataset_fingerprint"]
    raw = read_jsonl(dataset / "math/validation.jsonl")
    questions, references = [], []
    for r in raw:
        tree = expression_from_dict(r["expression"])
        questions.append(
            {
                k: r[k]
                for k in (
                    "question_id",
                    "question",
                    "source",
                    "split",
                    "difficulty",
                    "prompt_length",
                    "prompt_word_count",
                )
            }
            | {"operation_count": tree.operations, "reasoning_steps": tree.depth}
        )
        references.append(
            {
                "question_id": r["question_id"],
                "split": "validation",
                "canonical_answer": r["canonical_answer"],
            }
        )
    for role, rows in (("questions", questions), ("references", references)):
        path = dataset / f"pilot/validation.{role}.jsonl"
        path.parent.mkdir(exist_ok=True)
        write_jsonl(path, rows)
        manifest["files"][f"pilot/validation.{role}.jsonl"] = {
            "path": f"pilot/validation.{role}.jsonl",
            "role": role,
            "records": len(rows),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    manifest["pilot"] = {
        "validation": {
            "records": 12,
            "difficulty_counts": {d: 4 for d in ("easy", "medium", "hard")},
        }
    }
    manifest["dataset_fingerprint"] = config_digest(
        {k: v for k, v in manifest.items() if k != "dataset_fingerprint"}
    )
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    data_config = cfg.parent / "data.toml"
    data_config.write_text(
        data_config.read_text(encoding="utf-8").replace(old, manifest["dataset_fingerprint"]),
        encoding="utf-8",
    )
    prepared = prepare(data_config, dataset, token, root / "curriculum")
    trained = train_fixture(
        cfg, prepared, token, init_checkpoint=source, output_root=root / "fixture-training"
    )
    selected = trained / "checkpoints/selected_model.pt"
    bank_text = (
        (source_root / "configs/experiments/wp08_pilot_bank.toml")
        .read_text(encoding="utf-8")
        .split("[[checkpoints]]")[0]
    )
    for old, new in (
        ("candidates_per_question = 16", "candidates_per_question = 2"),
        ("batch_size = 8", "batch_size = 2"),
        ("trial_seeds = [42, 43, 44]", "trial_seeds = [42, 43]"),
        ("max_new_tokens = 256", "max_new_tokens = 8"),
    ):
        bank_text = bank_text.replace(old, new)
    for label, kind, p in (
        ("sft-fixture", "engine", source),
        ("selected-fixture", "curriculum_selected", selected),
    ):
        bank_text += (
            f'\n[[checkpoints]]\nid = "{label}"\nkind = "{kind}"\n'
            f'path = {json.dumps(p.resolve().as_posix())}\nsha256 = "{file_sha256(p)}"\n'
        )
    bank_config = root / "bank.toml"
    bank_config.write_text(bank_text, encoding="utf-8")
    codec = ByteBPETokenizer.load(token)
    curriculum_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    settings = {
        "schema_version": 1,
        "bank_config": bank_config.resolve().as_posix(),
        "dataset": dataset.resolve().as_posix(),
        "curriculum": prepared.resolve().as_posix(),
        "tokenizer": token.resolve().as_posix(),
        "wp03_fingerprint": manifest["dataset_fingerprint"],
        "curriculum_fingerprint": curriculum_manifest["fingerprint"],
        "tokenizer_fingerprint": codec.fingerprint,
        "expected_questions": 12,
        "per_difficulty": 1,
        "selection_seed": 42,
    }
    config = root / "pilot.toml"
    config.write_text(
        "\n".join(f"{k} = {json.dumps(v)}" for k, v in settings.items()), encoding="utf-8"
    )
    return config


def check(root, source_root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    config = make_inputs(root, source_root)
    first = prepare_pilot(config, root / "pilot")
    second = prepare_pilot(config, root / "repeat-pilot")
    assert first == second
    path = run(config, root / "pilot", output_root=root / "runs")
    bank = SampleBank(path / "artifacts/bank")
    assert bank.manifest["records"] == 24
    report = json.loads((path / "artifacts/pilot_report.json").read_text(encoding="utf-8"))
    assert report["metrics"]["unique_questions"] == 3
    assert report["metrics"]["question_difficulty_counts"] == {
        d: 1 for d in ("easy", "medium", "hard")
    }
    emit("feasibility_pilot_smoke_passed", run_directory=str(path), candidates=24)
    return config, root / "pilot", path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/wp08-pilot-smoke"))
    parser.add_argument("--source-root", type=Path, default=Path("."))
    args = parser.parse_args()
    check(args.output_root / uuid.uuid4().hex[:12], args.source_root)


if __name__ == "__main__":
    main()
