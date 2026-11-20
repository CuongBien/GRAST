"""Offline tiny checkpoints and fixed-pilot fixture for evaluator verification only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.difficulty import Expression, render_question
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.capability import run_capability
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.sft import warm_start
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import LoadedConfig, canonical_json, config_digest
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def make_fixture(root: Path, source_config: Path):
    codec = ByteBPETokenizer(vocab_size=264)
    codec_root = root / "tokenizer"
    codec.save(codec_root)
    dataset = root / "datasets"
    files, pilot = {}, {}
    for split, offset in (("validation", 0), ("test", 10)):
        questions, references = [], []
        for index, difficulty in enumerate(("easy", "medium", "hard")):
            tree = Expression(
                operator="+", left=Expression(value=2 + offset + index), right=Expression(value=3)
            )
            question = render_question(tree, "short")
            qid = f"fixture-{split}-{index}"
            questions.append(
                {
                    "question_id": qid,
                    "question": question,
                    "source": "grast-arithmetic-v1",
                    "split": split,
                    "difficulty": difficulty,
                    "operation_count": 1,
                    "reasoning_steps": 1,
                    "prompt_length": "short",
                    "prompt_word_count": len(question.split()),
                }
            )
            references.append(
                {"question_id": qid, "split": split, "canonical_answer": str(tree.evaluate())}
            )
        for role, rows in (("questions", questions), ("references", references)):
            relative = f"pilot/{split}.{role}.jsonl"
            p = dataset / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write("".join(canonical_json(r) + "\n" for r in rows))
            files[relative] = {
                "path": relative,
                "role": role,
                "records": len(rows),
                "size_bytes": p.stat().st_size,
                "sha256": file_sha256(p),
            }
        pilot[split] = {
            "records": 3,
            "difficulty_counts": dict(Counter(r["difficulty"] for r in questions)),
        }
    manifest = {
        "schema_version": 1,
        "work_package": "WP03",
        "scope": "math-only",
        "origin": "authored_offline_fixture",
        "answer_protocol": "numeric-v1",
        "files": files,
        "pilot": pilot,
    }
    manifest["dataset_fingerprint"] = config_digest(manifest)
    write_json(dataset / "manifest.json", manifest)
    architecture = ModelConfig(
        vocab_size=264,
        context_length=512,
        hidden_size=16,
        num_layers=1,
        num_heads=4,
        ffn_hidden_size=32,
        dropout=0.0,
    )
    checkpoints, model = [], None
    for stage in ("pretrain", "sft"):
        rng = seed_everything(42)
        model = CausalLM(architecture, tokenizer_fingerprint=codec.fingerprint)
        lineage = warm_start(model, checkpoints[0]) if checkpoints else None
        engine = TrainingEngine(
            model,
            TrainingConfig(stage=stage, max_steps=2, warmup_steps=0, gradient_accumulation_steps=1),
            generator=rng,
        )
        ids = torch.tensor([codec.encode("A small child reads.", add_bos=True, add_eos=True)])
        batch = {"input_ids": ids, "labels": ids.clone()}
        row = engine.train_update([batch])
        history = {
            "train": [row],
            "validation": [engine.evaluate([batch]) | {"attempt": engine.attempts}],
        }
        path = root / f"{stage}.pt"
        engine.save(
            path,
            data_state={
                "extra_state": {
                    "history": history,
                    "history_sha256": config_digest(history),
                    "contract": {"initialization": lineage},
                }
            },
        )
        checkpoints.append(path)
    template = source_config.read_text(encoding="utf-8").split("[[checkpoints]]", 1)[0]
    template = (
        template.replace("expected_questions = 150", "expected_questions = 3")
        .replace("max_new_tokens = 256", "max_new_tokens = 8")
        .replace("max_new_tokens = 96", "max_new_tokens = 8")
        .replace("batch_size = 8", "batch_size = 2")
    )
    for stage, path in zip(("pretrain", "sft"), checkpoints, strict=True):
        template += (
            f'\n[[checkpoints]]\nlabel = "{stage}-fixture"\n'
            f"path = {json.dumps(path.resolve().as_posix())}\n"
            f'sha256 = "{file_sha256(path)}"\nstage = "{stage}"\nstep = 1\n'
        )
    config = root / "fixture.toml"
    config.write_text(template, encoding="utf-8")
    return config, dataset, codec_root, checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/wp07_capability.toml")
    )
    args = parser.parse_args()
    outer = LoadedConfig(
        {
            "experiment": {"name": "wp07-5-capability-smoke"},
            "reproducibility": {"seed": 42},
            "artifacts": {"root": str(args.output_root.resolve())},
        },
        {"capability": args.config.read_text(encoding="utf-8")},
        args.config.resolve(),
    )
    with Run(outer) as run:
        config, dataset, codec, _ = make_fixture(run.path / "artifacts/fixture", args.config)
        child = run_capability(
            config, dataset, codec, split="both", output_root=run.path / "artifacts/evaluation"
        )
        report = json.loads(
            (child / "artifacts/capability_report.json").read_text(encoding="utf-8")
        )
        if report["status"] != "completed" or len(report["checkpoints"]) != 2:
            raise AssertionError("Tiny capability evaluator failed")
        for record in report["checkpoints"].values():
            if any(m["overall"]["questions"] != 3 for m in record["pilot"].values()):
                raise AssertionError("Fixture questions were dropped")
        write_json(
            run.path / "artifacts/capability_acceptance.json",
            {
                "work_package": "WP07.5",
                "status": "passed",
                "evaluation_run": str(child),
                "scope": (
                    "Tiny authored offline fixture only; not actual checkpoint/pilot accuracy."
                ),
            },
        )
    print(
        json.dumps({"event": "capability_smoke_passed", "run_directory": str(run.path)}), flush=True
    )


if __name__ == "__main__":
    main()
