"""Bounded copy/arithmetic overfit and frozen validation coverage diagnostics (WP07.6.1)."""

from __future__ import annotations

import argparse
import gc
import math
import re
import time
import tomllib
from pathlib import Path

import torch

from llm_scratch.data.collator import collate_tokenized
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.capability import generate_questions
from llm_scratch.evaluation.checkpoints import load_for_evaluation
from llm_scratch.evaluation.coverage import balanced_validation, sample_candidates, score_coverage
from llm_scratch.evaluation.pilot import load_manifest, load_questions, load_references
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.reasoning.formatting import encode_sft_example
from llm_scratch.research.feasibility_data import make_tasks, question_only, score_tasks
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import LoadedConfig, canonical_json, config_digest
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def load_protocol(path):
    source = path.read_text(encoding="utf-8")
    data = tomllib.loads(source)
    fields = {
        "experiment": {"name"},
        "reproducibility": {"seed", "deterministic", "num_threads"},
        "artifacts": {"root"},
        "checkpoint": {"path", "sha256"},
        "coverage": {
            "per_difficulty",
            "candidates",
            "batch_size",
            "precision",
            "max_new_tokens",
            "temperature",
            "top_k",
            "top_p",
        },
        "overfit": {
            "train_pairs",
            "validation_pairs",
            "batch_size",
            "max_steps",
            "eval_steps",
            "learning_rate",
            "precision",
            "generation_precision",
            "max_new_tokens",
        },
    }
    if (
        set(data) != {"schema_version", *fields}
        or type(data["schema_version"]) is not int
        or (data["schema_version"] != 1)
    ):
        raise ValueError("Invalid feasibility protocol schema")
    for name, keys in fields.items():
        if not isinstance(data[name], dict) or set(data[name]) != keys:
            raise ValueError(f"Unknown/missing {name} protocol fields")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", data["experiment"]["name"]):
        raise ValueError("Invalid experiment name")
    r, c, o = (data[k] for k in ("reproducibility", "coverage", "overfit"))
    if (
        type(r["seed"]) is not int
        or not 0 <= r["seed"] < 2**32
        or (type(r["deterministic"]) is not bool)
    ):
        raise ValueError("Invalid reproducibility seed/flags")
    for value in (
        r["num_threads"],
        c["per_difficulty"],
        c["candidates"],
        c["batch_size"],
        c["max_new_tokens"],
        o["batch_size"],
        o["max_steps"],
        o["max_new_tokens"],
    ):
        if type(value) is not int or value < 1:
            raise ValueError("Protocol counts must be positive integers")
    if not isinstance(o["eval_steps"], list) or any(type(n) is not int for n in o["eval_steps"]):
        raise ValueError("Evaluation steps must be integers")
    if o["eval_steps"] != sorted(set(o["eval_steps"])) or (
        not o["eval_steps"] or o["eval_steps"][0] != 0 or o["eval_steps"][-1] != o["max_steps"]
    ):
        raise ValueError("Evaluation schedule must span step zero to the fixed budget")
    for precision in (c["precision"], o["precision"], o["generation_precision"]):
        if precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unknown precision")
    make_tasks(o["train_pairs"], o["validation_pairs"], r["seed"])
    TrainingConfig(max_steps=o["max_steps"], warmup_steps=0, learning_rate=o["learning_rate"])
    GenerationConfig(**{k: c[k] for k in ("max_new_tokens", "temperature", "top_k", "top_p")})
    if not re.fullmatch(r"[a-f0-9]{64}", data["checkpoint"]["sha256"]):
        raise ValueError("Checkpoint digest must be SHA256")
    for group, key in (("checkpoint", "path"), ("artifacts", "root")):
        value = data[group][key]
        if not isinstance(value, str) or not value:
            raise ValueError("Paths must be nonempty strings")
        data[group][key] = str((path.parent / value).resolve())
    return LoadedConfig(data, {"feasibility": source}, path.resolve())


def jsonl(path, rows):
    atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def diagnostic_batches(rows, codec, size, order=None):
    order = list(range(len(rows))) if order is None else list(order)
    if sorted(order) != list(range(len(rows))):
        raise ValueError("Every diagnostic example must appear exactly once per update")
    batches = []
    for start in range(0, len(order), size):
        examples = [
            encode_sft_example(
                codec, rows[i]["question"], rows[i]["reasoning"], rows[i]["final_answer"]
            )
            for i in order[start : start + size]
        ]
        batches.append(
            {
                k: torch.tensor(v, dtype=torch.long)
                for k, v in collate_tokenized(examples, codec).items()
            }
        )
    return batches


def run_overfit(
    model,
    codec,
    settings,
    *,
    seed,
    directory,
    emit,
    log=None,
    deterministic=True,
    initialization=None,
):
    """Fresh optimizer; all tiny authored train examples consumed once per update.

    No resume, pilot examples, threshold search, or source optimizer/RNG state.
    A failed memorization gate is an observed result, not an excuse to extend the budget.
    """
    directory.mkdir(parents=True)
    tasks = make_tasks(settings["train_pairs"], settings["validation_pairs"], seed)
    for split, rows in tasks.items():
        jsonl(directory / f"{split}.jsonl", rows)
    generator = seed_everything(
        seed, deterministic=deterministic, num_threads=torch.get_num_threads()
    )
    config = TrainingConfig(
        stage="sft",
        max_steps=settings["max_steps"],
        warmup_steps=0,
        gradient_accumulation_steps=math.ceil(len(tasks["train"]) / settings["batch_size"]),
        learning_rate=settings["learning_rate"],
        weight_decay=0.0,
        min_lr_ratio=0.1,
        precision=settings["precision"],
    )
    engine = TrainingEngine(model, config, generator=generator, log=log)
    if engine.optimizer.state or engine.step or engine.attempts:
        raise AssertionError("Diagnostic optimizer must start empty at step zero")
    policy = GenerationConfig(
        max_new_tokens=settings["max_new_tokens"],
        eos_token_id=codec.eos_token_id,
        pad_token_id=codec.pad_token_id,
    )
    history = {"train": [], "validation": []}
    evaluations = []

    def evaluate():
        model.eval()
        metrics = {}
        for split, rows in tasks.items():
            loss = engine.evaluate(diagnostic_batches(rows, codec, settings["batch_size"]))
            if split == "validation":
                history["validation"].append(loss | {"attempt": engine.attempts})
            predictions = generate_questions(
                model,
                codec,
                question_only(rows),
                policy,
                batch_size=settings["batch_size"],
                seed=seed,
                precision=settings["generation_precision"],
                emit=emit,
            )
            stem = directory / f"step-{engine.step:04d}-{split}"
            jsonl(stem.with_suffix(".predictions.jsonl"), predictions)
            scored, summary = score_tasks(predictions, rows, codec)
            jsonl(stem.with_suffix(".scored.jsonl"), scored)
            metrics[split] = {"loss": loss["loss"], "tasks": summary}
        result = {"step": engine.step, "attempt": engine.attempts, **metrics}
        evaluations.append(result)
        write_json(directory / "evaluations.json", evaluations)
        emit("overfit_generated_evaluation", **result)

    evaluate()
    attempt_limit = settings["max_steps"] + max(10, settings["max_steps"] // 5)
    while engine.step < settings["max_steps"] and engine.attempts < attempt_limit:
        order = generator.permutation(len(tasks["train"])).tolist()
        row = engine.train_update(
            diagnostic_batches(tasks["train"], codec, settings["batch_size"], order)
        )
        history["train"].append(row)
        if not row["skipped"] and engine.step in settings["eval_steps"]:
            evaluate()
    if evaluations[-1]["step"] != engine.step:
        evaluate()
    engine.save(
        directory / "diagnostic-final.pt",
        data_state={
            "extra_state": {
                "history": history,
                "history_sha256": config_digest(history),
                "contract": {
                    "purpose": "authored-overfit-diagnostic-only",
                    "seed": seed,
                    "initialization": initialization,
                },
            }
        },
    )
    final = evaluations[-1]
    report = {
        "purpose": "Bounded memorization diagnostic; not a replacement SFT baseline",
        "data_fingerprint": config_digest(tasks),
        "training_config": config.to_dict(),
        "optimizer_initial_state_entries": 0,
        "initial_step": 0,
        "steps": engine.step,
        "attempts": engine.attempts,
        "amp_skips": engine.attempts - engine.step,
        "budget_completed": engine.step == config.max_steps,
        "train_memorized": all(m["exact_match"] == 1.0 for m in final["train"]["tasks"].values()),
        "evaluations": evaluations,
        "checkpoint_sha256": file_sha256(directory / "diagnostic-final.pt"),
    }
    write_json(directory / "report.json", report)
    return report


def run_feasibility(
    config_path, dataset_root, tokenizer_root, *, device="cpu", output_root=None, part="all"
):
    if part not in ("all", "coverage", "overfit"):
        raise ValueError("Unknown diagnostic part")
    loaded = load_protocol(Path(config_path))
    data = loaded.data
    if output_root is not None:
        data["artifacts"]["root"] = str(Path(output_root).resolve())
    data["execution"] = {"part": part, "device": str(device)}
    codec = ByteBPETokenizer.load(tokenizer_root)
    root = Path(dataset_root)
    manifest = load_manifest(root)
    questions = load_questions(
        root, manifest, "validation", expected_records=manifest["pilot"]["validation"]["records"]
    )
    selection = balanced_validation(
        questions, data["coverage"]["per_difficulty"], data["reproducibility"]["seed"]
    )
    data["inputs"] = {
        "dataset_root": str(root.resolve()),
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "tokenizer_root": str(Path(tokenizer_root).resolve()),
        "tokenizer_fingerprint": codec.fingerprint,
        "validation_question_ids": [r["question_id"] for r in selection],
    }
    seed_everything(**data["reproducibility"])
    with Run(loaded) as run:
        began = time.perf_counter()

        def emit(event, **fields):
            print(canonical_json({"event": event, **fields}), flush=True)
            run.log(event, **fields)

        def engine_log(record):
            run.log(
                record["event"],
                **{k: v for k, v in record.items() if k not in ("event", "timestamp")},
            )

        directory = run.path / "artifacts"
        sources = {
            f"{p.parent.name}/{p.name}": file_sha256(p)
            for folder in ("evaluation", "research", "training", "model", "inference", "reasoning")
            for p in (Path(__file__).parents[1] / folder).glob("*.py")
        }
        protocol = {"config": data, "source_sha256": sources}
        write_json(directory / "protocol.json", protocol | {"sha256": config_digest(protocol)})
        jsonl(directory / "selected_validation.questions.jsonl", selection)
        emit(
            "feasibility_started",
            run_directory=str(run.path),
            protocol_sha256=config_digest(protocol),
        )
        model, checkpoint, _ = load_for_evaluation(
            Path(data["checkpoint"]["path"]), codec, expected_sha256=data["checkpoint"]["sha256"]
        )
        model.to(device).eval()
        report = {
            "work_package": "WP07.6.1",
            "checkpoint": checkpoint,
            "protocol_sha256": config_digest(protocol),
        }
        if part in ("all", "coverage"):
            settings = data["coverage"]
            policy = GenerationConfig(
                **{k: settings[k] for k in ("max_new_tokens", "temperature", "top_k", "top_p")},
                do_sample=True,
                eos_token_id=codec.eos_token_id,
                pad_token_id=codec.pad_token_id,
            )
            predictions = sample_candidates(
                model,
                codec,
                selection,
                policy,
                candidates=settings["candidates"],
                batch_size=settings["batch_size"],
                seed=data["reproducibility"]["seed"],
                precision=settings["precision"],
                emit=emit,
            )
            jsonl(directory / "coverage.predictions.jsonl", predictions)
            # Persist every model-only prediction before any validation reference read.
            all_refs = load_references(
                root, manifest, "validation", {q["question_id"] for q in questions}
            )
            references = {q["question_id"]: all_refs[q["question_id"]] for q in selection}
            scored, per_question, summary = score_coverage(
                predictions, references, codec, candidates=settings["candidates"]
            )
            jsonl(directory / "coverage.scored.jsonl", scored)
            write_json(directory / "coverage.questions.json", per_question)
            report["coverage"] = summary
            emit("coverage_completed", **summary)
        if part in ("all", "overfit"):
            # Model weights only; the source optimizer/scaler/RNG/cursor are never restored.
            report["overfit"] = run_overfit(
                model,
                codec,
                data["overfit"],
                seed=data["reproducibility"]["seed"],
                directory=directory / "overfit",
                emit=emit,
                log=engine_log,
                deterministic=data["reproducibility"]["deterministic"],
                initialization={
                    "mode": "weights-only-fresh-optimizer",
                    "source_checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "source_stage": checkpoint["stage"],
                    "source_step": checkpoint["step"],
                },
            )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        source_sha = file_sha256(Path(data["checkpoint"]["path"]))
        if source_sha != data["checkpoint"]["sha256"]:
            raise AssertionError("Immutable source checkpoint changed")
        report.update(
            status="completed",
            source_checkpoint_unchanged=True,
            elapsed_seconds=time.perf_counter() - began,
        )
        export_plot(report, directory)
        write_json(directory / "feasibility_report.json", report)
        emit(
            "feasibility_completed",
            run_directory=str(run.path),
            elapsed_seconds=report["elapsed_seconds"],
        )
    return run.path


def export_plot(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    panels = int("coverage" in report) + 2 * int("overfit" in report)
    fig, axes = plt.subplots(
        1, panels, figsize=(6 * panels, 4.5), squeeze=False, layout="constrained"
    )
    index = 0
    if "coverage" in report:
        ax = axes[0, index]
        strata = report["coverage"]["by_difficulty"]
        bars = ax.bar(list(strata), [m["observed_coverage_at_k"] for m in strata.values()])
        ax.bar_label(
            bars, labels=[f"{m['covered_questions']}/{m['questions']}" for m in strata.values()]
        )
        ax.set(
            title="Original SFT: observed coverage@K (validation)",
            ylim=(0, 1.1),
            ylabel="Questions with >=1 correct candidate",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        index += 1
    if "overfit" in report:
        rows = report["overfit"]["evaluations"]
        for split, style in (("train", "-"), ("validation", "--")):
            for task in ("copy", "arithmetic"):
                axes[0, index].plot(
                    [r["step"] for r in rows],
                    [r[split]["tasks"][task]["exact_match"] for r in rows],
                    style,
                    marker="o",
                    label=f"{split}: {task}",
                )
            axes[0, index + 1].plot(
                [r["step"] for r in rows],
                [r[split]["loss"] for r in rows],
                style,
                marker="o",
                label=split,
            )
        axes[0, index].set(
            title="Authored tasks: free-generated answer EM",
            ylim=(-0.05, 1.05),
            xlabel="Diagnostic optimizer updates",
            ylabel="Exact match",
        )
        axes[0, index].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0, index + 1].set(
            title="Authored tasks: teacher-forced loss",
            xlabel="Diagnostic optimizer updates",
            ylabel="Target-token cross-entropy",
        )
        axes[0, index].legend()
        axes[0, index + 1].legend()
    try:
        for ax in axes[0]:
            ax.grid(axis="y", alpha=0.2)
        for ext in ("png", "svg"):
            fig.savefig(directory / f"feasibility.{ext}", dpi=150)
    finally:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/wp07_feasibility.toml")
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--part", choices=("all", "coverage", "overfit"), default="all")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    run_feasibility(
        args.config,
        args.dataset_root,
        args.tokenizer,
        device=args.device,
        output_root=args.output_root,
        part=args.part,
    )


if __name__ == "__main__":
    main()
