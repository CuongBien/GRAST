"""Short WP07.2 data-order/full-resume diagnostic; never a corpus training claim."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tomllib
from dataclasses import replace
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.pretraining import PackedCorpus, prepare_pretraining
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.research.training_acceptance import _same
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.data import TrainingCursor
from llm_scratch.training.session import PretrainingSession
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import LoadedConfig, config_digest
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def offline_sources(root: Path) -> Path:
    """Authored offline fixtures matching WP03 schema, not real TinyStories records."""
    texts = {
        "train": [
            "A small fox finds a red ball beside the tree. "
            "The fox brings it home and shares it with a friend.",
            "Rain falls on the garden. A child opens the door, "
            "puts on a blue coat, and watches the flowers grow.",
            "Minh reads a story about a boat. "
            "Then Minh draws a picture and says: xin chào, thế giới!",
        ],
        "validation": [
            "The moon shines above a quiet hill. A bird sleeps in a warm nest.",
            "A girl counts three shells on the sand and walks back to her family.",
        ],
    }
    files = {}
    for split, rows in texts.items():
        path = root / f"tinystories/{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for i, text in enumerate(rows):
                stream.write(
                    json.dumps(
                        {
                            "source": "tinystories",
                            "split": split,
                            "question_id": f"authored-{split}-{i}",
                            "text": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        relative = path.relative_to(root).as_posix()
        files[relative] = {
            "path": relative,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "records": len(rows),
            "role": "pretraining",
        }
    manifest = {
        "schema_version": 1,
        "work_package": "WP03",
        "scope": "full",
        "origin": "authored_offline_fixture",
        "files": files,
    }
    manifest["dataset_fingerprint"] = config_digest(manifest)
    write_json(root / "manifest.json", manifest)
    return root


def make_session(root, architecture, cfg, data_cfg, device, rng, log=None):
    train, validation = PackedCorpus(root, "train"), PackedCorpus(root, "validation")
    model = CausalLM(architecture, tokenizer_fingerprint=train.meta["tokenizer_fingerprint"]).to(
        device
    )
    engine = TrainingEngine(model, cfg, generator=rng, log=log)
    cursor = TrainingCursor(
        train,
        batch_size=data_cfg["batch_size"],
        accumulation_steps=cfg.gradient_accumulation_steps,
        seed=data_cfg["shuffle_seed"],
    )
    return PretrainingSession(engine, cursor, validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/wp07_data_smoke.toml")
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--profile", choices=("tiny", "candidate"), default="tiny")
    parser.add_argument("--block-size", type=int)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if (
        bool(args.dataset_root) != bool(args.tokenizer)
        or args.prepared_root
        and (args.dataset_root or args.tokenizer)
    ):
        raise ValueError(
            "Use --dataset-root + --tokenizer, or --prepared-root, or offline defaults"
        )
    path = args.config.resolve()
    source = path.read_text(encoding="utf-8")
    data = tomllib.loads(source)
    if (
        set(data)
        != {
            "schema_version",
            "experiment",
            "profiles",
            "training",
            "data",
            "diagnostic",
            "reproducibility",
            "artifacts",
        }
        or data["schema_version"] != 1
    ):
        raise ValueError("Invalid data diagnostic config")
    if args.precision:
        data["training"]["precision"] = args.precision
    if args.block_size is not None:
        data["data"]["block_size"] = args.block_size
    cfg = TrainingConfig.from_dict(data["training"])
    if cfg.stage != "pretrain":
        raise ValueError("WP07.2 pipeline is for pretraining; CoT formatting follows WP07.4")
    dc = data["data"]
    if (
        any(
            type(dc[k]) is not int or dc[k] < 1
            for k in ("block_size", "batch_size", "validation_batches")
        )
        or dc["block_size"] < 3
    ):
        raise ValueError("Invalid block/batch/validation dimensions")
    if not 1 <= data["diagnostic"]["checkpoint_step"] < cfg.max_steps:
        raise ValueError("Resume point must be inside the short diagnostic")
    codec = None
    if args.prepared_root:
        preview = PackedCorpus(args.prepared_root, "train")
        vocab_size = preview.meta["vocab_size"]
        if dc["block_size"] != preview.meta["block_size"]:
            raise ValueError("Configured block_size differs from prepared corpus")
        source_mode = preview.meta["source_origin"]
    else:
        codec = (
            ByteBPETokenizer.load(args.tokenizer)
            if args.tokenizer
            else ByteBPETokenizer(vocab_size=264)
        )
        vocab_size = len(codec)
        source_mode = (
            json.loads((args.dataset_root / "manifest.json").read_text(encoding="utf-8")).get(
                "origin", "wp03_prepared_sources"
            )
            if args.dataset_root
            else "authored_offline_fixture"
        )
    if args.profile == "candidate" and source_mode == "authored_offline_fixture":
        raise ValueError("Candidate smoke requires actual WP03/WP04 inputs")
    model_source = (path.parent / data.pop("profiles")[args.profile]).read_text(encoding="utf-8")
    base = ModelConfig.from_dict(tomllib.loads(model_source)["model"])
    if args.profile == "candidate" and vocab_size != base.vocab_size:
        raise ValueError("Candidate requires the WP04 vocabulary size, no implicit resize")
    architecture = replace(base, vocab_size=vocab_size, dropout=data["diagnostic"]["dropout"])
    if dc["block_size"] > architecture.context_length:
        raise ValueError("Packing block_size exceeds source model context")
    data.update(
        model=architecture.to_dict(),
        source_mode=source_mode,
        execution={
            "device": args.device,
            "profile": args.profile,
            "dataset_root": str(args.dataset_root.resolve()) if args.dataset_root else None,
            "tokenizer": str(args.tokenizer.resolve()) if args.tokenizer else None,
            "prepared_root": str(args.prepared_root.resolve()) if args.prepared_root else None,
        },
    )
    data["artifacts"]["root"] = str(
        args.output_root.resolve()
        if args.output_root
        else (path.parent / data["artifacts"]["root"]).resolve()
    )
    loaded = LoadedConfig(data, {"experiment": source, "model": model_source}, path)
    with Run(loaded) as run:
        rng = seed_everything(**data["reproducibility"])
        env = inspect_environment(args.device)
        write_json(run.path / "artifacts/environment.json", env)
        print(
            json.dumps(
                {
                    "event": "data_smoke_started",
                    "device": env["selected_device"],
                    "source_mode": source_mode,
                }
            ),
            flush=True,
        )
        if not env["validation"]["ok"]:
            raise RuntimeError(env["validation"]["errors"])
        root = args.prepared_root
        if root is None:
            source_root, codec_root = args.dataset_root, args.tokenizer
            if source_root is None:
                source_root = offline_sources(run.path / "artifacts/offline_sources")
                codec_root = run.path / "artifacts/offline_tokenizer"
                codec.save(codec_root)
            root = prepare_pretraining(
                source_root,
                codec_root,
                run.path / "artifacts/pretraining_data",
                block_size=dc["block_size"],
            )

        def log(record):
            run.log(
                record["event"],
                **{k: v for k, v in record.items() if k not in ("event", "timestamp")},
            )

        session = make_session(root, architecture, cfg, dc, args.device, rng, log)
        initial = session.validate(max_batches=dc["validation_batches"])
        history, expected = [], []
        checkpoint = run.path / "checkpoints/resume.pt"
        checkpoint_written = False
        while session.engine.step < cfg.max_steps:
            if session.engine.attempts >= cfg.max_steps * 16:
                raise RuntimeError("Too many AMP overflow attempts; inspect logs")
            update = session.update()
            if checkpoint_written:
                expected.append(update)
            if update["skipped"]:
                continue
            validation = session.validate(max_batches=dc["validation_batches"])
            history.append(
                {
                    "step": session.engine.step,
                    "train_loss": update["loss"],
                    "validation_loss": validation["loss"],
                    "batch_sha256": update["batch_sha256"],
                }
            )
            if (
                session.engine.step == data["diagnostic"]["checkpoint_step"]
                and not checkpoint_written
            ):
                session.save(checkpoint)
                checkpoint_written = True
        final = session.validate(max_batches=dc["validation_batches"])
        states = [
            copy.deepcopy(obj.state_dict())
            for obj in (
                session.engine.model,
                session.engine.optimizer,
                session.engine.scheduler,
                session.engine.scaler,
                session.train,
            )
        ]
        corpus_meta = session.train.corpus.meta
        del session
        resumed = make_session(root, architecture, cfg, dc, args.device, rng, log)
        resumed.load(checkpoint)
        trace = []
        for reference in expected:
            actual = resumed.update()
            fields = (
                "step",
                "attempt",
                "loss",
                "learning_rates",
                "skipped",
                "amp_scale",
                "block_indices",
                "batch_sha256",
            )
            if any(actual[k] != reference[k] for k in fields):
                raise AssertionError("Resumed batch/token IDs/loss/LR/AMP state differ")
            trace.append({k: actual[k] for k in fields})
        for expected_state, obj in zip(
            states,
            (
                resumed.engine.model,
                resumed.engine.optimizer,
                resumed.engine.scheduler,
                resumed.engine.scaler,
                resumed.train,
            ),
            strict=True,
        ):
            if not _same(expected_state, obj.state_dict()):
                raise AssertionError("Full engine/data state differs after resume")
        if resumed.validate(max_batches=dc["validation_batches"])["loss"] != final["loss"]:
            raise AssertionError("Resumed validation loss differs")
        resumed.save(run.path / "checkpoints/final.pt")
        report = {
            "work_package": "WP07.2",
            "status": "passed",
            "source_mode": source_mode,
            "device": args.device,
            "precision": cfg.precision,
            "model_config": architecture.to_dict(),
            "parameter_counts": resumed.engine.model.parameter_counts(),
            "corpus_manifest": corpus_meta,
            "prepared_root": str(Path(root).resolve()),
            "initial_validation_loss": initial["loss"],
            "final_validation_loss": final["loss"],
            "validation_max_batches": dc["validation_batches"],
            "history": history,
            "resume_trace": trace,
            "final_train_cursor": resumed.train.state_dict(),
            "checks": {
                "next_batch_token_ids_and_losses_exact": True,
                "model_optimizer_scheduler_scaler_cursor_exact": True,
                "heldout_validation_order_and_loss_exact": True,
                "no_test_set_read": True,
                "stdout_logs_always": True,
            },
            "limitation": (
                "Short data/resume verification only; no corpus baseline, "
                "CoT SFT, pilot or Colab execution claim."
            ),
        }
        write_json(run.path / "artifacts/data_resume_report.json", report)
        run.metadata.update(
            dataset_fingerprint=corpus_meta["fingerprint"],
            tokenizer_fingerprint=corpus_meta["tokenizer_fingerprint"],
            execution_device=args.device,
            acceptance_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        write_json(run.path / "metadata.json", run.metadata)
        run.log("data_resume_passed", checks=report["checks"])
    print(json.dumps({"status": "passed", "run_directory": str(run.path)}), flush=True)


if __name__ == "__main__":
    main()
