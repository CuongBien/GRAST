"""Bounded authored CPU fixture: all stages, stop/resume, selected model and bundle."""

from __future__ import annotations

import argparse
import uuid
from collections import Counter
from pathlib import Path

import torch

from llm_scratch.curriculum.bundle import create, extract
from llm_scratch.curriculum.data import prepare
from llm_scratch.curriculum.evaluation import load_selected
from llm_scratch.curriculum.runner import run
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.difficulty import Expression, render_question
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.research.training_acceptance import _same
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import canonical_json, config_digest
from llm_scratch.utils.io import write_json
from llm_scratch.utils.seed import seed_everything


def fixture(root, source_root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    codec = ByteBPETokenizer(vocab_size=264)
    codec.save(root / "tokenizer")
    files = {}
    dataset = root / "wp03"
    for split, offset in (("train", 0), ("validation", 10)):
        records = []
        for difficulty, operations in (("easy", 1), ("medium", 3), ("hard", 5)):
            for index in range(4):
                tree = Expression(
                    operator="+",
                    left=Expression(value=offset + index + 1),
                    right=Expression(value=2),
                )
                for step in range(operations - 1):
                    tree = Expression(
                        operator="-" if step % 2 else "+",
                        left=tree,
                        right=Expression(value=step + 3),
                    )
                semantic = config_digest(tree.canonical())
                q = render_question(tree, "short")
                records.append(
                    {
                        "question_id": "math-" + semantic,
                        "semantic_id": semantic,
                        "source": "grast-arithmetic-v1",
                        "split": split,
                        "difficulty": difficulty,
                        "prompt_length": "short",
                        "prompt_word_count": len(q.split()),
                        "question": q,
                        "answer": str(tree.evaluate()),
                        "canonical_answer": str(tree.evaluate()),
                        "expression": tree.as_dict(),
                    }
                )
        path = dataset / f"math/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(canonical_json(r) + "\n" for r in records), encoding="utf-8")
        files[f"math/{split}.jsonl"] = {
            "path": f"math/{split}.jsonl",
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "records": len(records),
        }
    manifest = {
        "schema_version": 1,
        "work_package": "WP03",
        "scope": "math-only",
        "answer_protocol": "numeric-v1",
        "files": files,
    }
    manifest["dataset_fingerprint"] = config_digest(manifest)
    write_json(dataset / "manifest.json", manifest)
    config = (source_root / "configs/data/wp07_curriculum.toml").read_text(encoding="utf-8")
    for old, new in (
        ("operand_max = 99", "operand_max = 40"),
        ("multi_train_per_level = 1500", "multi_train_per_level = 4"),
        ("multi_validation_per_level = 300", "multi_validation_per_level = 4"),
        (
            "79ff92320ebb64e283ce342076c5e570989ef5433d8791231b2d78f370baf485",
            manifest["dataset_fingerprint"],
        ),
        ("49187f5c5a806801ae6fc86cce7a9e6ca26d54d81cc989b361bbc76724abada3", codec.fingerprint),
    ):
        config = config.replace(old, new)
    (root / "data.toml").write_text(config, encoding="utf-8")
    prepared = prepare(root / "data.toml", dataset, root / "tokenizer", root / "prepared")
    rng = seed_everything(42)
    architecture = ModelConfig(
        vocab_size=264,
        context_length=512,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=32,
        dropout=0.1,
    )
    engine = TrainingEngine(
        CausalLM(architecture, tokenizer_fingerprint=codec.fingerprint),
        TrainingConfig(stage="sft", max_steps=2, warmup_steps=0, gradient_accumulation_steps=1),
        generator=rng,
    )
    ids = torch.tensor([codec.encode("A child learns.", add_bos=True, add_eos=True)])
    batch = {"input_ids": ids, "labels": ids.clone()}
    history = {
        "train": [engine.train_update([batch])],
        "validation": [engine.evaluate([batch]) | {"attempt": engine.attempts}],
    }
    checkpoint = root / "source-sft.pt"
    engine.save(
        checkpoint,
        data_state={
            "extra_state": {
                "history": history,
                "history_sha256": config_digest(history),
                "contract": {},
            }
        },
    )
    recipe = (source_root / "configs/training/wp07_curriculum.toml").read_text(encoding="utf-8")
    replacements = {
        "vocab_size = 8192": "vocab_size = 264",
        "context_length = 1024": "context_length = 512",
        "hidden_size = 512": "hidden_size = 16",
        "num_layers = 10": "num_layers = 1",
        "num_heads = 8": "num_heads = 4",
        "ffn_hidden_size = 2048": "ffn_hidden_size = 32",
        "dropout = 0.0": "dropout = 0.1",
        "max_steps = 2800": "max_steps = 6",
        "warmup_steps = 50": "warmup_steps = 0",
        'precision = "float16"': 'precision = "float32"',
        "gradient_accumulation_steps = 4": "gradient_accumulation_steps = 2",
        "end_step = 400": "end_step = 2",
        "end_step = 1600": "end_step = 4",
        "end_step = 2800": "end_step = 6",
        "validate_every = 200": "validate_every = 2",
        "checkpoint_every = 100": "checkpoint_every = 2",
        "per_task = 16": "per_task = 4",
        "max_new_tokens = 256": "max_new_tokens = 8",
        "5f75909261c79e9f7f0c6e37a7e9d76ba5670be8b4be5133b6e148e0cd06ff42": file_sha256(checkpoint),
    }
    for old, new in replacements.items():
        recipe = recipe.replace(old, new)
    (root / "training.toml").write_text(recipe, encoding="utf-8")
    return root / "training.toml", prepared, root / "tokenizer", checkpoint


def check(root, source_root):
    config, data, tokenizer, checkpoint = fixture(root / "fixture", source_root)
    bundle = create(data, tokenizer, root / "fixture.zip")
    extract(root / "fixture.zip", root / "extracted", bundle["sha256"])
    kwargs = {
        "config": config,
        "prepared_root": data,
        "tokenizer": tokenizer,
        "output_root": root / "runs",
    }
    full = run(**kwargs, init_checkpoint=checkpoint)
    partial = run(**kwargs, init_checkpoint=checkpoint, stop_after=3)
    # Resume from a single copied file, with no parent best/phase checkpoint siblings.
    import shutil

    lone = root / "lone-latest.pt"
    shutil.copyfile(partial / "checkpoints/latest.pt", lone)
    resumed = run(**kwargs, resume=lone)
    a = torch.load(full / "checkpoints/final.pt", map_location="cpu", weights_only=True)
    b = torch.load(resumed / "checkpoints/final.pt", map_location="cpu", weights_only=True)
    assert _same(a, b), "Full checkpoint differs after stop/resume"
    selected, meta = load_selected(
        resumed / "checkpoints/selected_model.pt", ByteBPETokenizer.load(tokenizer)
    )
    assert _same(selected.state_dict(), b["state"]["data_state"]["best_weights"])
    report = {
        "status": "passed",
        "scope": "Tiny authored fixture; not corpus reasoning quality",
        "all_stages": dict(
            Counter(r["phase"] for r in b["state"]["data_state"]["extra_state"]["history"]["train"])
        ),
        "full_vs_resume_exact": True,
        "selected_model_reloads": True,
        "selected_step": meta["selection"]["step"],
        "bundle_verified": True,
        "run_directory": str(resumed),
    }
    write_json(root / "acceptance.json", report)
    print(canonical_json({"event": "curriculum_smoke_passed", **report}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=Path("artifacts/curriculum-smoke"))
    p.add_argument("--source-root", type=Path, default=Path.cwd())
    a = p.parse_args()
    check(a.output_root.resolve() / uuid.uuid4().hex[:8], a.source_root.resolve())


if __name__ == "__main__":
    main()
