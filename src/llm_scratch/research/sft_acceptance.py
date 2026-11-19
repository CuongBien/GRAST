"""Bounded CoT SFT warm-start and exact continuation acceptance, not long training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.difficulty import generate_math
from llm_scratch.data.sft import prepare_sft
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.research.training_acceptance import _same
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.pretrain import load_recipe
from llm_scratch.training.sft import run_sft
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import canonical_json, config_digest
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def offline_math(root: Path):
    level = {
        "name": "fixture",
        "operations": 1,
        "reasoning_steps": 1,
        "prompt_length": "short",
        "operand_min": 1,
        "operand_max": 25,
        "operators": ["+", "-", "*", "/"],
    }
    rows = generate_math([level], 14, 73)
    files = {}
    for split, selected in (("train", rows[:10]), ("validation", rows[10:])):
        relative = f"math/{split}.jsonl"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(canonical_json(row | {"split": split}) + "\n" for row in selected),
            encoding="utf-8",
        )
        files[relative] = {
            "path": relative,
            "sha256": file_sha256(path),
            "records": len(selected),
            "size_bytes": path.stat().st_size,
            "role": "training_or_evaluation",
        }
    manifest = {
        "schema_version": 1,
        "work_package": "WP03",
        "scope": "math-only",
        "origin": "authored_offline_fixture",
        "files": files,
    }
    manifest["dataset_fingerprint"] = config_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return root


def tiny_pretrained(root: Path, architecture: ModelConfig, codec: ByteBPETokenizer):
    rng = seed_everything(42)
    model = CausalLM(architecture, tokenizer_fingerprint=codec.fingerprint)
    engine = TrainingEngine(
        model,
        TrainingConfig(max_steps=2, warmup_steps=0, gradient_accumulation_steps=1),
        generator=rng,
    )
    ids = torch.tensor([codec.encode("A child reads a little story.", add_bos=True, add_eos=True)])
    engine.train_update([{"input_ids": ids, "labels": ids.clone()}])
    path = root / "pretrain_fixture.pt"
    engine.save(path, data_state={"scope": "fixture_only_one_pretraining_update"})
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training/wp07_sft_smoke.toml"))
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--stop-after", type=int, default=4)
    parser.add_argument("--split-step", type=int, default=2)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    loaded = load_recipe(args.config, stage="sft")
    if not 1 <= args.split_step < args.stop_after <= min(8, loaded.data["training"]["max_steps"]):
        raise ValueError(
            "Acceptance must be 2..8 successful SFT updates with a split strictly inside"
        )
    if any((args.prepared_root, args.tokenizer, args.init_checkpoint)) and not all(
        (args.prepared_root, args.tokenizer, args.init_checkpoint)
    ):
        raise ValueError(
            "Pass prepared-root, tokenizer and init-checkpoint together, or offline defaults"
        )
    if args.output_root:
        loaded.data["artifacts"]["root"] = str(args.output_root.resolve())
    with Run(loaded) as run:
        root, codec_root, parent = args.prepared_root, args.tokenizer, args.init_checkpoint
        if root is None:
            codec = ByteBPETokenizer(vocab_size=264)
            codec_root = run.path / "artifacts/tokenizer"
            codec.save(codec_root)
            root = prepare_sft(
                offline_math(run.path / "artifacts/source"),
                codec_root,
                run.path / "artifacts/sft_data",
                max_length=loaded.data["data"]["block_size"],
            )
            parent = tiny_pretrained(
                run.path / "artifacts", ModelConfig.from_dict(loaded.data["model"]), codec
            )
        original_sha = file_sha256(parent)
        common = dict(
            device=args.device,
            precision=args.precision,
            output_root=run.path / "artifacts/sft_runs",
        )
        full = run_sft(
            args.config,
            root,
            codec_root,
            init_checkpoint=parent,
            stop_after=args.stop_after,
            **common,
        )
        first = run_sft(
            args.config,
            root,
            codec_root,
            init_checkpoint=parent,
            stop_after=args.split_step,
            **common,
        )
        resumed = run_sft(
            args.config,
            root,
            codec_root,
            resume=first / "checkpoints/latest.pt",
            stop_after=args.stop_after,
            **common,
        )
        a, b = [
            torch.load(p / "checkpoints/latest.pt", map_location="cpu", weights_only=True)
            for p in (full, resumed)
        ]
        histories = [p["state"]["data_state"].pop("extra_state")["history"] for p in (a, b)]
        if not _same(a, b) or histories[0]["train"] != histories[1]["train"]:
            raise AssertionError(
                "SFT engine/data/RNG state or per-attempt trace differs after resume"
            )
        actual_val = {r["step"]: r for r in histories[1]["validation"]}
        if any(actual_val.get(r["step"]) != r for r in histories[0]["validation"]):
            raise AssertionError("SFT common validation losses differ")
        if file_sha256(parent) != original_sha:
            raise AssertionError("SFT changed the source pretraining checkpoint")
        report = {
            "work_package": "WP07.4",
            "status": "passed",
            "device": args.device,
            "precision": args.precision or loaded.data["training"]["precision"],
            "verified_steps": args.stop_after,
            "planned_steps": loaded.data["training"]["max_steps"],
            "parent_checkpoint_sha256": original_sha,
            "full_run": str(full),
            "first_run": str(first),
            "resumed_run": str(resumed),
            "train_history": histories[0]["train"],
            "validation_history": histories[0]["validation"],
            "checks": {
                "model_optimizer_scheduler_scaler_cursor_rng_exact": True,
                "per_attempt_batch_loss_lr_amp_trace_exact": True,
                "common_validation_losses_exact": True,
                "source_checkpoint_unchanged": True,
            },
            "limitation": "Short SFT runner verification; no long SFT/pilot capability claim.",
        }
        write_json(run.path / "artifacts/sft_acceptance.json", report)
        run.log("sft_acceptance_passed", checks=report["checks"])
    print(
        json.dumps({"event": "sft_acceptance_passed", "run_directory": str(run.path)}), flush=True
    )


if __name__ == "__main__":
    main()
