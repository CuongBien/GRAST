"""Short real-runner save/restart verification; never launch the long training budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llm_scratch.data.pretraining import prepare_pretraining
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.research.data_acceptance import offline_sources
from llm_scratch.research.training_acceptance import _same
from llm_scratch.training.pretrain import load_recipe, run_pretraining
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training/wp07_pretrain_smoke.toml")
    )
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--stop-after", type=int, default=4)
    parser.add_argument("--split-step", type=int, default=2)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    loaded = load_recipe(args.config)
    if not 1 <= args.split_step < args.stop_after <= min(8, loaded.data["training"]["max_steps"]):
        raise ValueError("Acceptance must be 2..8 successful steps, split strictly inside")
    if bool(args.prepared_root) != bool(args.tokenizer):
        raise ValueError("Pass both prepared-root and tokenizer, or neither for authored fixture")
    if args.output_root:
        loaded.data["artifacts"]["root"] = str(args.output_root.resolve())
    with Run(loaded) as run:
        prepared, codec = args.prepared_root, args.tokenizer
        if prepared is None:
            codec = run.path / "artifacts/tokenizer"
            ByteBPETokenizer(vocab_size=264).save(codec)
            prepared = prepare_pretraining(
                offline_sources(run.path / "artifacts/source"),
                codec,
                run.path / "artifacts/prepared",
                block_size=loaded.data["data"]["block_size"],
            )
        common = dict(
            device=args.device,
            precision=args.precision,
            output_root=run.path / "artifacts/runner_runs",
        )
        full = run_pretraining(args.config, prepared, codec, stop_after=args.stop_after, **common)
        first = run_pretraining(args.config, prepared, codec, stop_after=args.split_step, **common)
        resumed = run_pretraining(
            args.config,
            prepared,
            codec,
            stop_after=args.stop_after,
            resume=first / "checkpoints/latest.pt",
            **common,
        )
        a, b = [
            torch.load(p / "checkpoints/latest.pt", map_location="cpu", weights_only=True)
            for p in (full, resumed)
        ]
        histories = [v["state"]["data_state"].pop("extra_state")["history"] for v in (a, b)]
        if not _same(a, b) or histories[0]["train"] != histories[1]["train"]:
            raise AssertionError("Runner full engine/data/RNG state or per-attempt trace differs")
        expected_val = {r["step"]: r for r in histories[0]["validation"]}
        actual_val = {r["step"]: r for r in histories[1]["validation"]}
        if any(actual_val.get(step) != row for step, row in expected_val.items()):
            raise AssertionError("Common validation measurements differ")
        report = {
            "work_package": "WP07.3",
            "status": "passed",
            "device": args.device,
            "precision": args.precision or loaded.data["training"]["precision"],
            "verified_steps": args.stop_after,
            "planned_steps": loaded.data["training"]["max_steps"],
            "full_run": str(full),
            "first_run": str(first),
            "resumed_run": str(resumed),
            "train_history": histories[0]["train"],
            "validation_history": histories[0]["validation"],
            "checks": {
                "model_optimizer_scheduler_scaler_cursor_rng_exact": True,
                "per_attempt_batch_loss_lr_amp_trace_exact": True,
                "common_validation_losses_exact": True,
                "checkpoint_owned_history_recovered": True,
            },
            "limitation": "Short runner verification; no corpus baseline/Colab/quality claim.",
        }
        write_json(run.path / "artifacts/pretraining_acceptance.json", report)
        run.log("pretraining_runner_passed", checks=report["checks"])
    print(
        json.dumps({"event": "pretraining_runner_passed", "run_directory": str(run.path)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
