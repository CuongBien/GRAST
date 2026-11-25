"""Tiny offline provenance/GSM8K-review fixture; no benchmark-performance claim."""

import argparse
import json
import shutil
import tomllib
import uuid
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.research.feasibility_review import prepare, run
from llm_scratch.research.pilot_lock import write_jsonl
from llm_scratch.research.replay_acceptance import check as make_replay
from llm_scratch.research.sample_bank import emit
from llm_scratch.utils.config import config_digest


def receipt(root, package, **extra):
    return {
        "work_package": package,
        "run_directory": root.resolve().as_posix(),
        "files": {
            p.relative_to(root).as_posix(): {
                "sha256": file_sha256(p),
                "size_bytes": p.stat().st_size,
            }
            for p in root.rglob("*")
            if p.is_file()
        },
        **extra,
    }


def write_config(path, data):
    text = "schema_version = 1\n"
    for section, values in data.items():
        if section != "schema_version":
            text += (
                f"\n[{section}]\n"
                + "\n".join(f"{k} = {json.dumps(v)}" for k, v in values.items())
                + "\n"
            )
    Path(path).write_text(text, encoding="utf-8")


def make_inputs(root, source_root):
    root, source_root = Path(root), Path(source_root)
    bank, _, replay_root = make_replay(root / "inputs", source_root)
    pilot_root = root / "pilot-run"
    shutil.copytree(bank.root, pilot_root / "artifacts/bank")
    pilot = {"bank_fingerprint": bank.fingerprint, "candidates": 48, "metrics": {"fixture": True}}
    (pilot_root / "artifacts/pilot_report.json").write_text(json.dumps(pilot), encoding="utf-8")
    replay_manifest = json.loads(
        (replay_root / "artifacts/replay/manifest.json").read_text(encoding="utf-8")
    )
    receipts = {
        "pilot": receipt(pilot_root, "WP08.2", bank_fingerprint=bank.fingerprint),
        "replay": receipt(
            replay_root,
            "WP08.3",
            bank_fingerprint=bank.fingerprint,
            repeat_verification={"replay_fingerprint": replay_manifest["fingerprint"]},
        ),
    }
    for key, r in receipts.items():
        (root / f"{key}.json").write_text(json.dumps(r), encoding="utf-8")
    dataset = root / "inputs/datasets"
    m = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        {
            "question_id": f"gsm-fixture-{i}",
            "source": "gsm8k",
            "split": "validation",
            "upstream_split": "train",
            "question": f"A child has {i + 2} apples and gets 3 more. How many apples now?",
            "answer": f"Add the quantities. #### {i + 5}",
            "canonical_answer": str(i + 5),
            "difficulty": "ungraded",
            "source_index": i,
        }
        for i in range(4)
    ]
    path = dataset / "gsm8k/validation.jsonl"
    path.parent.mkdir()
    write_jsonl(path, rows)
    m["files"]["gsm8k/validation.jsonl"] = {
        "path": "gsm8k/validation.jsonl",
        "records": 4,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    m["dataset_fingerprint"] = config_digest(
        {k: v for k, v in m.items() if k != "dataset_fingerprint"}
    )
    (dataset / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    template = (
        (source_root / "configs/experiments/wp08_pilot_bank.toml")
        .read_text(encoding="utf-8")
        .split("[[checkpoints]]")[0]
    )
    for i, p in enumerate((root / "inputs/pretrain.pt", root / "inputs/sft.pt")):
        template += (
            f'\n[[checkpoints]]\nid = "cp-{i}"\nkind = "engine"\n'
            f'path = {json.dumps(p.resolve().as_posix())}\nsha256 = "{file_sha256(p)}"\n'
        )
    (root / "checkpoints.toml").write_text(template, encoding="utf-8")
    d = tomllib.loads(
        (source_root / "configs/evaluation/wp08_feasibility.toml").read_text(encoding="utf-8")
    )
    d["artifacts"]["root"] = (root / "runs").resolve().as_posix()
    d["inputs"] = {
        "project_root": root.resolve().as_posix(),
        "dataset": dataset.resolve().as_posix(),
        "wp03_fingerprint": m["dataset_fingerprint"],
        "tokenizer": (root / "inputs/tokenizer").resolve().as_posix(),
        "tokenizer_fingerprint": ByteBPETokenizer.load(root / "inputs/tokenizer").fingerprint,
        "checkpoint_config": (root / "checkpoints.toml").resolve().as_posix(),
        **{f"{k}_receipt": (root / f"{k}.json").resolve().as_posix() for k in receipts},
        **{f"{k}_receipt_fingerprint": config_digest(v) for k, v in receipts.items()},
    }
    d["gsm_probe"].update(
        questions=2, expected_validation_records=4, batch_size=2, max_new_tokens=8
    )
    config = root / "review.toml"
    write_config(config, d)
    return config


def check(root, source_root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    config = make_inputs(root, source_root)
    a = prepare(config, root / "probe")
    b = prepare(config, root / "repeat-probe")
    assert a == b
    path = run(config, root / "probe", output_root=root / "reviews")
    r = json.loads((path / "artifacts/feasibility_report.json").read_text(encoding="utf-8"))
    assert r["decision"]["evidence_counts"]["gsm_predictions"] == 4
    assert r["decision"]["confidence_efficacy_gate"] == "requires-new-review"
    emit("feasibility_review_smoke_passed", run_directory=str(path), gsm_predictions=4)
    return config, root / "probe", path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=Path("."))
    p.add_argument("--output-root", type=Path, default=Path("artifacts/wp08-review-smoke"))
    a = p.parse_args()
    check(a.output_root / uuid.uuid4().hex[:12], a.source_root)


if __name__ == "__main__":
    main()
