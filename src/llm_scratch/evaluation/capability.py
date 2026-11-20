"""Frozen greedy pilot, checkpoint comparison and authored text generation for WP07.5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
import tomllib
from dataclasses import replace
from pathlib import Path

import torch

from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.checkpoints import load_for_evaluation
from llm_scratch.evaluation.pilot import (
    QUESTION_KEYS,
    audit_question_splits,
    load_manifest,
    load_questions,
    load_references,
)
from llm_scratch.evaluation.pilot_metrics import score_predictions, summarize
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import generate
from llm_scratch.reasoning.formatting import encode_reasoning_prompt
from llm_scratch.research.generation_equivalence import check_generation_equivalence
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import LoadedConfig, canonical_json, config_digest
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def load_protocol(path: Path):
    source = path.read_text(encoding="utf-8")
    data = tomllib.loads(source)
    if (
        set(data)
        != {
            "schema_version",
            "experiment",
            "reproducibility",
            "artifacts",
            "evaluation",
            "generation",
            "text",
            "checkpoints",
        }
        or type(data["schema_version"]) is not int
        or data["schema_version"] != 1
    ):
        raise ValueError("Invalid capability protocol schema")
    for name, keys in {
        "experiment": {"name"},
        "reproducibility": {"seed", "deterministic", "num_threads"},
        "artifacts": {"root"},
        "evaluation": {"batch_size", "precision", "expected_questions", "cache_check_tokens"},
        "generation": {"max_new_tokens", "do_sample", "temperature", "top_k", "top_p"},
        "text": {"prefixes", "max_new_tokens", "seed", "temperature", "top_k", "top_p"},
    }.items():
        if not isinstance(data[name], dict) or set(data[name]) != keys:
            raise ValueError(f"Unknown/missing {name} protocol fields")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", data["experiment"]["name"]):
        raise ValueError("Invalid experiment name")
    ev, repro, text = data["evaluation"], data["reproducibility"], data["text"]
    for value in (
        ev["batch_size"],
        ev["expected_questions"],
        ev["cache_check_tokens"],
        text["max_new_tokens"],
        repro["num_threads"],
    ):
        if type(value) is not int or value < 1:
            raise ValueError("Protocol dimensions must be positive integers")
    if (
        ev["precision"] not in ("float32", "float16", "bfloat16")
        or type(repro["deterministic"]) is not bool
    ):
        raise ValueError("Invalid evaluation precision/determinism")
    if any(type(v) is not int or not 0 <= v < 2**32 for v in (repro["seed"], text["seed"])):
        raise ValueError("Invalid protocol seed")
    if (
        not isinstance(text["prefixes"], list)
        or not text["prefixes"]
        or any(not isinstance(s, str) or not s.strip() for s in text["prefixes"])
    ):
        raise ValueError("Authored text prefixes must be nonempty strings")
    GenerationConfig(**data["generation"])
    if data["generation"]["do_sample"] or data["generation"]["max_new_tokens"] < 1:
        raise ValueError("WP07.5 primary pilot is fixed greedy generation with a positive budget")
    GenerationConfig(
        **{k: v for k, v in text.items() if k not in ("seed", "prefixes")}, do_sample=True
    )
    if not isinstance(data["checkpoints"], list) or len(data["checkpoints"]) < 2:
        raise ValueError("Compare at least two real checkpoint stages")
    labels, digests = set(), set()
    for cp in data["checkpoints"]:
        if set(cp) != {"label", "path", "sha256", "stage", "step"} or (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", cp["label"])
            or not re.fullmatch(r"[a-f0-9]{64}", cp["sha256"])
            or cp["label"] in labels
            or cp["sha256"] in digests
            or cp["stage"] not in ("pretrain", "sft")
            or type(cp["step"]) is not int
            or cp["step"] < 1
        ):
            raise ValueError("Invalid/duplicate checkpoint identities")
        labels.add(cp["label"])
        digests.add(cp["sha256"])
        cp["path"] = str((path.parent / cp["path"]).resolve())
    if not isinstance(data["artifacts"]["root"], str) or not data["artifacts"]["root"]:
        raise ValueError("Artifact root must be nonempty")
    data["artifacts"]["root"] = str((path.parent / data["artifacts"]["root"]).resolve())
    return LoadedConfig(data, {"capability": source}, path.resolve())


def left_pad(rows, pad_id, device):
    if not rows or any(not row for row in rows):
        raise ValueError("Generation prefixes must be nonempty")
    width = max(map(len, rows))
    ids = [[pad_id] * (width - len(r)) + r for r in rows]
    masks = [[0] * (width - len(r)) + [1] * len(r) for r in rows]
    return torch.tensor(ids, dtype=torch.long, device=device), torch.tensor(
        masks, dtype=torch.bool, device=device
    )


def generate_questions(model, codec, questions, policy, *, batch_size, seed, precision, emit):
    """This function receives question-only rows. It cannot read reference files."""
    if not questions or any(set(row) != QUESTION_KEYS for row in questions):
        raise ValueError("Generation requires the question-only pilot schema, without gold fields")
    predictions = []
    device = model.token_embedding.weight.device
    for start in range(0, len(questions), batch_size):
        rows = questions[start : start + batch_size]
        prefixes = [encode_reasoning_prompt(codec, r["question"]) for r in rows]
        ids, mask = left_pad(prefixes, codec.pad_token_id, device)
        emit("pilot_batch_started", completed=start, total=len(questions), batch_size=len(rows))
        began = time.perf_counter()

        def progress(step, token_ids, active, completed=start):
            if (step + 1) % 32 == 0:
                emit(
                    "pilot_decode_progress",
                    completed=completed,
                    decode_step=step + 1,
                    active_rows=int(active.sum()),
                )

        with torch.autocast(
            device.type, dtype=getattr(torch, precision), enabled=precision != "float32"
        ):
            output = generate(
                model, ids, mask, config=policy, seed=seed, use_cache=True, on_token=progress
            )
        for index, row in enumerate(rows):
            generated = output.generated_ids[index][output.generated_mask[index]].tolist()
            predictions.append(
                {
                    "question_id": row["question_id"],
                    "difficulty": row["difficulty"],
                    "split": row["split"],
                    "question": row["question"],
                    "prompt_ids_sha256": config_digest(prefixes[index]),
                    "generated_ids": generated,
                    "finish_reason": output.finish_reasons[index],
                    "generated_text": codec.decode(generated, skip_special_tokens=False),
                }
            )
        emit(
            "pilot_batch_finished",
            completed=start + len(rows),
            total=len(questions),
            elapsed_seconds=time.perf_counter() - began,
        )
    return predictions


def export_comparison(records, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        ["checkpoint", "stage", "series", "step", "attempt", "loss", "loss_tokens", "skipped"]
    )
    fig, axes = plt.subplots(
        1, len(records), figsize=(7 * len(records), 4.5), squeeze=False, layout="constrained"
    )
    try:
        for ax, (label, record) in zip(axes[0], records.items(), strict=True):
            for series, rows in record["history"].items():
                for row in rows:
                    writer.writerow(
                        [
                            label,
                            record["checkpoint"]["stage"],
                            series,
                            row["step"],
                            row["attempt"],
                            row["loss"],
                            row["loss_tokens"],
                            row.get("skipped", False),
                        ]
                    )
                kept = [r for r in rows if not r.get("skipped", False)]
                ax.plot([r["step"] for r in kept], [r["loss"] for r in kept], label=series)
            ax.set(
                title=f"{label} ({record['checkpoint']['stage']})",
                xlabel="Stage optimizer updates",
                ylabel="CE on that stage's dataset/target scope",
            )
            ax.grid(alpha=0.25)
            ax.legend()
        for ext in ("png", "svg"):
            fig.savefig(directory / f"stage_loss_curves.{ext}", dpi=160)
    finally:
        plt.close(fig)
    atomic_text(directory / "stage_losses.csv", output.getvalue())
    metrics = io.StringIO(newline="")
    writer = csv.writer(metrics)
    writer.writerow(
        [
            "checkpoint",
            "split",
            "difficulty",
            "questions",
            "correct",
            "exact_match",
            "answer_valid_rate",
            "format_valid_rate",
            "mean_generated_tokens",
        ]
    )
    for label, record in records.items():
        for split, results in record["pilot"].items():
            for difficulty, m in {"all": results["overall"], **results["by_difficulty"]}.items():
                writer.writerow(
                    [
                        label,
                        split,
                        difficulty,
                        *[
                            m[k]
                            for k in (
                                "questions",
                                "correct",
                                "exact_match",
                                "answer_valid_rate",
                                "format_valid_rate",
                                "mean_generated_tokens",
                            )
                        ],
                    ]
                )
    atomic_text(directory / "pilot_metrics.csv", metrics.getvalue())
    from matplotlib.ticker import PercentFormatter

    splits = list(next(iter(records.values()))["pilot"])
    fig, axes = plt.subplots(
        1, len(splits), figsize=(6 * len(splits), 4.5), squeeze=False, layout="constrained"
    )
    try:
        for ax, split in zip(axes[0], splits, strict=True):
            present = set(next(iter(records.values()))["pilot"][split]["by_difficulty"])
            levels = [v for v in ("easy", "medium", "hard") if v in present]
            levels += sorted(present - set(levels))
            width = 0.8 / len(records)
            for index, (label, record) in enumerate(records.items()):
                values = [record["pilot"][split]["by_difficulty"][v] for v in levels]
                positions = [i - 0.4 + width / 2 + index * width for i in range(len(levels))]
                bars = ax.bar(
                    positions, [v["exact_match"] for v in values], width=width, label=label
                )
                ax.bar_label(
                    bars,
                    labels=[f"{v['correct']}/{v['questions']}" for v in values],
                    padding=3,
                    fontsize=8,
                )
            ax.set(
                title=f"Fixed pilot {split}",
                xlabel="WP03 difficulty tier",
                ylabel="Final-answer exact match",
                ylim=(0, 1.08),
            )
            ax.set_xticks(range(len(levels)), levels)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.legend()
        for ext in ("png", "svg"):
            fig.savefig(directory / f"pilot_exact_match.{ext}", dpi=160)
    finally:
        plt.close(fig)


def run_capability(
    config: Path,
    dataset_root: Path,
    tokenizer: Path,
    *,
    device="cpu",
    split="validation",
    diagnostic_limit=None,
    output_root=None,
):
    loaded = load_protocol(config)
    data, ev = loaded.data, loaded.data["evaluation"]
    splits = ("validation", "test") if split == "both" else (split,)
    if any(s not in ("validation", "test") for s in splits):
        raise ValueError("Choose validation, test or both")
    if diagnostic_limit is not None and (
        split != "validation"
        or type(diagnostic_limit) is not int
        or not 1 <= diagnostic_limit <= ev["expected_questions"]
    ):
        raise ValueError(
            "Diagnostic subsets are validation-only; test must use the full frozen pilot"
        )
    if device == "cpu" and ev["precision"] == "float16":
        raise ValueError("FP16 evaluation requires CUDA")
    codec = ByteBPETokenizer.load(tokenizer)
    manifest = load_manifest(dataset_root)
    questions = {
        s: load_questions(dataset_root, manifest, s, expected_records=ev["expected_questions"])
        for s in splits
    }
    audit_question_splits(questions.values())
    if diagnostic_limit:
        questions["validation"] = questions["validation"][:diagnostic_limit]
    policy = GenerationConfig(
        **data["generation"], eos_token_id=codec.eos_token_id, pad_token_id=codec.pad_token_id
    )
    policy.validate_vocab(len(codec))
    # Paths and device are execution metadata; protocol identity covers policies + immutable inputs.
    protocol = {
        "schema_version": 1,
        "prompt_protocol": "wp04-bos-question-lf-think-v1",
        "answer_protocol": "numeric-v1-marked-final-generated-only-v1",
        "batching": "file-order-static-left-padding-physical-rope-v1",
        "metric": "exact-canonical-final-answer-all-questions-invalid-as-wrong",
        "evaluation": ev,
        "generation": policy.to_dict(),
        "text": data["text"],
        "reproducibility": data["reproducibility"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "tokenizer_fingerprint": codec.fingerprint,
        "checkpoints": [
            {k: cp[k] for k in ("label", "sha256", "stage", "step")} for cp in data["checkpoints"]
        ],
    }
    package = Path(__file__).parents[1]
    files = sorted(
        [
            p
            for folder in ("model", "inference", "evaluation")
            for p in (package / folder).glob("*.py")
        ]
        + [
            package / relative
            for relative in (
                "reasoning/formatting.py",
                "reasoning/answer_extraction.py",
                "data/tokenizer.py",
                "utils/seed.py",
            )
        ]
    )
    protocol["implementation"] = {
        p.relative_to(package).as_posix(): hashlib.sha256(
            p.read_text(encoding="utf-8").encode("utf-8")
        ).hexdigest()
        for p in files
    }
    protocol_sha = config_digest(protocol)
    if output_root:
        data["artifacts"]["root"] = str(Path(output_root).resolve())
    data["execution"] = {
        "device": device,
        "split": split,
        "diagnostic_limit": diagnostic_limit,
        "dataset_root": str(dataset_root.resolve()),
        "tokenizer": str(tokenizer.resolve()),
    }
    data["protocol_sha256"] = protocol_sha
    seed_everything(**data["reproducibility"])
    env = inspect_environment(device)
    if not env["validation"]["ok"]:
        raise RuntimeError(env["validation"]["errors"])
    with Run(loaded) as run:

        def emit(event, **fields):
            print(json.dumps({"event": event, **fields}, ensure_ascii=False), flush=True)
            run.log(event, **fields)

        emit(
            "capability_started",
            protocol_sha256=protocol_sha,
            splits=splits,
            diagnostic_limit=diagnostic_limit,
        )
        artifacts = run.path / "artifacts"
        write_json(artifacts / "protocol.json", protocol)
        write_json(artifacts / "environment.json", env)
        records, identities = {}, []
        for cp in data["checkpoints"]:
            label = cp["label"]
            emit("checkpoint_loading", label=label)
            model, meta, history = load_for_evaluation(
                Path(cp["path"]), codec, expected_sha256=cp["sha256"]
            )
            if (meta["stage"], meta["step"]) != (cp["stage"], cp["step"]):
                raise ValueError("Checkpoint stage/step differs from frozen inventory")
            if identities and meta["model_config"] != identities[0]["model_config"]:
                raise ValueError(
                    "Compare identical architectures/tokenizers, without implicit resizing"
                )
            identities.append(meta)
            if meta["stage"] == "sft" and (
                not isinstance(meta["initialization"], dict)
                or meta["initialization"].get("checkpoint_sha256")
                not in {c["sha256"] for c in data["checkpoints"] if c["stage"] == "pretrain"}
            ):
                raise ValueError("SFT parent must be one of the compared pretrained checkpoints")
            model.to(device).eval()
            directory = artifacts / label
            directory.mkdir()
            emit("checkpoint_loaded", label=label, checkpoint=meta)
            record = {"checkpoint": meta, "history": history, "pilot": {}, "text_samples": []}
            prefix = [codec.bos_token_id, *codec.encode(data["text"]["prefixes"][0])]
            ids, mask = left_pad([prefix], codec.pad_token_id, model.token_embedding.weight.device)
            with torch.autocast(
                device, dtype=getattr(torch, ev["precision"]), enabled=ev["precision"] != "float32"
            ):
                record["cache_equivalence"] = check_generation_equivalence(
                    model,
                    ids,
                    mask,
                    config=replace(policy, max_new_tokens=ev["cache_check_tokens"]),
                    seed=data["reproducibility"]["seed"],
                )
            for i, text in enumerate(data["text"]["prefixes"]):
                ids, mask = left_pad(
                    [[codec.bos_token_id, *codec.encode(text)]],
                    codec.pad_token_id,
                    model.token_embedding.weight.device,
                )
                settings = GenerationConfig(
                    **{k: v for k, v in data["text"].items() if k not in ("seed", "prefixes")},
                    do_sample=True,
                    eos_token_id=codec.eos_token_id,
                    pad_token_id=codec.pad_token_id,
                )
                with torch.autocast(
                    device,
                    dtype=getattr(torch, ev["precision"]),
                    enabled=ev["precision"] != "float32",
                ):
                    output = generate(
                        model,
                        ids,
                        mask,
                        config=settings,
                        seed=(data["text"]["seed"] + i) % 2**32,
                        use_cache=True,
                    )
                generated = output.generated_ids[0][output.generated_mask[0]].tolist()
                record["text_samples"].append(
                    {
                        "prefix": text,
                        "generated_ids": generated,
                        "continuation": codec.decode(generated, skip_special_tokens=False),
                        "finish_reason": output.finish_reasons[0],
                        "seed": (data["text"]["seed"] + i) % 2**32,
                    }
                )
            write_json(directory / "text_samples.json", record["text_samples"])
            write_json(directory / "checkpoint.json", meta)
            write_json(directory / "history.json", history)
            for current_split, rows in questions.items():
                emit(
                    "pilot_generation_started",
                    label=label,
                    split=current_split,
                    questions=len(rows),
                )
                predictions = generate_questions(
                    model,
                    codec,
                    rows,
                    policy,
                    batch_size=ev["batch_size"],
                    seed=data["reproducibility"]["seed"],
                    precision=ev["precision"],
                    emit=emit,
                )
                # Persist model-only predictions BEFORE the scorer reads any gold reference file.
                pred_path = directory / f"{current_split}.predictions.jsonl"
                atomic_text(pred_path, "".join(canonical_json(r) + "\n" for r in predictions))
                full_ids = {
                    r["question_id"]
                    for r in load_questions(
                        dataset_root,
                        manifest,
                        current_split,
                        expected_records=ev["expected_questions"],
                    )
                }
                refs = load_references(dataset_root, manifest, current_split, full_ids)
                if diagnostic_limit:
                    refs = {r["question_id"]: refs[r["question_id"]] for r in rows}
                scored = score_predictions(predictions, refs, codec)
                result = summarize(scored)
                record["pilot"][current_split] = result
                atomic_text(
                    directory / f"{current_split}.scored.jsonl",
                    "".join(canonical_json(r) + "\n" for r in scored),
                )
                write_json(
                    directory / f"{current_split}.failures.json",
                    [r for r in scored if not r["correct"]][:5],
                )
                emit("pilot_scored", label=label, split=current_split, metrics=result)
            records[label] = record
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        source_hashes = {m["checkpoint_sha256"] for m in identities if m["stage"] == "pretrain"}
        for meta in identities:
            if (
                meta["stage"] == "sft"
                and meta["initialization"]["checkpoint_sha256"] not in source_hashes
            ):
                raise ValueError("SFT parent must be one of the compared pretrained checkpoints")
        export_comparison(records, artifacts)
        report = {
            "schema_version": 1,
            "work_package": "WP07.5",
            "status": "diagnostic_subset" if diagnostic_limit else "completed",
            "protocol_sha256": protocol_sha,
            "dataset_fingerprint": manifest["dataset_fingerprint"],
            "runtime": env,
            "checkpoints": records,
            "scope": (
                "Fixed-K=1 greedy final-answer EM; no confidence/calibration/threshold tuning "
                "or reasoning-trace correctness claim."
            ),
        }
        write_json(artifacts / "capability_report.json", report)
        run.metadata.update(
            protocol_sha256=protocol_sha,
            dataset_fingerprint=manifest["dataset_fingerprint"],
            tokenizer_fingerprint=codec.fingerprint,
        )
        write_json(run.path / "metadata.json", run.metadata)
        emit("capability_completed", run_directory=str(run.path), status=report["status"])
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/wp07_capability.toml")
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--split", choices=("validation", "test", "both"), default="validation")
    parser.add_argument("--diagnostic-limit", type=int)
    parser.add_argument("--output-root", type=Path)
    run_capability(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
