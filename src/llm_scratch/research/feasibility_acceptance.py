"""Tiny offline execution of both diagnostics; no real model-quality claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.research.capability_acceptance import make_fixture
from llm_scratch.research.feasibility import run_feasibility


def make_inputs(root, source_config):
    capability_config = source_config.parent / "wp07_capability.toml"
    _, dataset, codec, checkpoints = make_fixture(root, capability_config)
    text = source_config.read_text(encoding="utf-8")
    text = text.replace(
        'path = "../../artifacts/checkpoints/wp07-cot-sft-math-500/final.pt"',
        f"path = {json.dumps(checkpoints[-1].resolve().as_posix())}",
    ).replace(
        "5f75909261c79e9f7f0c6e37a7e9d76ba5670be8b4be5133b6e148e0cd06ff42",
        file_sha256(checkpoints[-1]),
    )
    for old, new in (
        ("per_difficulty = 10", "per_difficulty = 1"),
        ("candidates = 16", "candidates = 2"),
        ("batch_size = 8", "batch_size = 4"),
        ("max_new_tokens = 256", "max_new_tokens = 8"),
        ("max_new_tokens = 80", "max_new_tokens = 8"),
        ("train_pairs = 32", "train_pairs = 2"),
        ("validation_pairs = 8", "validation_pairs = 2"),
        ("max_steps = 200", "max_steps = 1"),
        ("eval_steps = [0, 25, 50, 100, 200]", "eval_steps = [0, 1]"),
        ('precision = "bfloat16"', 'precision = "float32"'),
    ):
        text = text.replace(old, new)
    config = root / "feasibility.toml"
    config.write_text(text, encoding="utf-8")
    return config, dataset, codec, checkpoints[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/feasibility-smoke"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/wp07_feasibility.toml")
    )
    args = parser.parse_args()
    import uuid

    fixture_root = args.output_root.resolve() / f"fixture-{uuid.uuid4().hex[:8]}"
    config, dataset, codec, _ = make_inputs(fixture_root, args.config)
    run = run_feasibility(config, dataset, codec, output_root=args.output_root.resolve())
    report = json.loads((run / "artifacts/feasibility_report.json").read_text(encoding="utf-8"))
    assert report["coverage"]["overall"]["candidates"] == 6
    assert report["overfit"]["steps"] == 1
    assert report["source_checkpoint_unchanged"]
    print(json.dumps({"event": "feasibility_smoke_passed", "run_directory": str(run)}), flush=True)


if __name__ == "__main__":
    main()
