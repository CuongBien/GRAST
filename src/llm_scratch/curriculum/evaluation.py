"""Balanced fixed generated validation and explicit selected-model inference export."""

from __future__ import annotations

import copy
import hashlib
import os
import tempfile
from collections import Counter
from pathlib import Path

import torch

from llm_scratch.curriculum.data import TASKS
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.difficulty import expression_from_dict
from llm_scratch.evaluation.capability import generate_questions
from llm_scratch.evaluation.pilot_metrics import extract_generated_final
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.research.feasibility_data import _copy_answer
from llm_scratch.utils.config import canonical_json, config_digest


def select_validation(rows, per_task, seed):
    if type(per_task) is not int or per_task < 4 or per_task % 4:
        raise ValueError("Generated validation needs per_task divisible by4 and >=4")
    if any(r["split"] != "validation" for r in rows):
        raise ValueError("Selection accepts validation only")
    groups = [
        ("copy", "authored", None, per_task),
        ("single", "authored", None, per_task // 2),
        ("single", "wp03", None, per_task // 2),
        ("multi", "authored", "ops-2", per_task // 4),
        ("multi", "authored", "ops-3", per_task // 4),
        ("multi", "wp03", "medium", per_task // 4),
        ("multi", "wp03", "hard", per_task // 4),
    ]
    selected = []
    for task, origin, difficulty, n in groups:
        pool = [
            r
            for r in rows
            if r["task"] == task
            and r["origin"] == origin
            and (difficulty is None or r["difficulty"] == difficulty)
        ]
        if len(pool) < n:
            raise ValueError(
                f"Insufficient generated validation stratum: {task}/{origin}/{difficulty}"
            )
        selected.extend(sorted(pool, key=lambda r: config_digest([seed, r["question_id"]]))[:n])
    if len({r["question_id"] for r in selected}) != 3 * per_task:
        raise ValueError("Duplicate validation selection")
    return selected


def questions_only(rows):
    result = []
    for r in rows:
        tree = None if r["task"] == "copy" else expression_from_dict(r["expression"])
        result.append(
            {
                "question_id": r["question_id"],
                "question": r["prompt"],
                "source": "curriculum-v1",
                "split": r["split"],
                "difficulty": r["difficulty"],
                "operation_count": 0 if tree is None else tree.operations,
                "reasoning_steps": 1 if tree is None else tree.depth,
                "prompt_length": r["prompt_length"],
                "prompt_word_count": len(r["prompt"].split()),
            }
        )
    return result


def score(predictions, references, codec):
    refs = {r["question_id"]: r for r in references}
    if (
        len(predictions) != len(refs)
        or len(refs) != len(references)
        or {r["question_id"] for r in predictions} != set(refs)
    ):
        raise ValueError("Generated validation must score every selected question exactly once")
    scored = []
    for pred in predictions:
        ref = refs[pred["question_id"]]
        result = extract_generated_final(pred["generated_ids"], codec, pred["finish_reason"])
        if ref["task"] == "copy":
            pair = _copy_answer(pred["generated_ids"], codec)
            correct = pair == ref["operands"]
            result.update(
                predicted_answer=None if pair is None else f"{pair[0]},{pair[1]}",
                answer_valid=pair is not None,
            )
        else:
            correct = result["predicted_answer"] == ref["final_answer"]
        scored.append(
            pred
            | result
            | {
                "task": ref["task"],
                "origin": ref["origin"],
                "reference": ref["final_answer"],
                "correct": correct,
            }
        )
    summary = {}
    for task in TASKS:
        group = [r for r in scored if r["task"] == task]
        if not group:
            raise ValueError("Generated validation requires all three tasks")
        summary[task] = {
            "questions": len(group),
            "correct": sum(r["correct"] for r in group),
            "exact_match": sum(r["correct"] for r in group) / len(group),
            "format_valid_rate": sum(r["format_valid"] for r in group) / len(group),
            "answer_valid_rate": sum(r["answer_valid"] for r in group) / len(group),
        }
    return scored, {
        "tasks": summary,
        "macro_exact_match": sum(m["exact_match"] for m in summary.values()) / 3,
        "finish_reasons": dict(Counter(p["finish_reason"] for p in scored)),
    }


def generate_validation(model, codec, selection, settings, emit):
    policy = GenerationConfig(
        max_new_tokens=settings["max_new_tokens"],
        eos_token_id=codec.eos_token_id,
        pad_token_id=codec.pad_token_id,
    )
    return generate_questions(
        model,
        codec,
        questions_only(selection),
        policy,
        batch_size=settings["batch_size"],
        seed=settings["selection_seed"],
        precision=settings["precision"],
        emit=emit,
    )


def snapshot(model):
    return {
        k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        for k, v in model.state_dict().items()
    }


def weights_digest(weights):
    digest = hashlib.sha256()
    for k, v in sorted(weights.items()):
        digest.update(k.encode())
        digest.update(
            v.detach().cpu().contiguous().numpy().tobytes()
            if isinstance(v, torch.Tensor)
            else canonical_json(v).encode()
        )
    return digest.hexdigest()


def validate_weights(weights, model):
    expected = model.state_dict()
    if set(weights) != set(expected) or weights["_extra_state"] != expected["_extra_state"]:
        raise ValueError("Selected model architecture/tokenizer mismatch")
    for k, v in weights.items():
        if k != "_extra_state" and (
            not isinstance(v, torch.Tensor)
            or v.shape != expected[k].shape
            or v.dtype != expected[k].dtype
            or not torch.isfinite(v).all()
        ):
            raise ValueError("Invalid selected-model tensor")
    if model.config.tie_embeddings and not torch.equal(
        weights["token_embedding.weight"], weights["lm_head.weight"]
    ):
        raise ValueError("Selected tied weights disagree")


def export_selected(path, weights, record, contract):
    """Inference-only export: explicitly cannot resume an optimizer from this file."""
    path = Path(path)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".selected-")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(
                {
                    "format": "grast-curriculum-selected-v1",
                    "model": weights,
                    "selection": record,
                    "contract": contract,
                    "purpose": "inference_only_not_training_resume",
                },
                f,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def load_selected(path, codec, expected_sha256=None):
    digest = file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Selected-model checksum mismatch")
    p = torch.load(path, map_location="cpu", weights_only=True)
    if (
        p.get("format") != "grast-curriculum-selected-v1"
        or p.get("purpose") != "inference_only_not_training_resume"
    ):
        raise ValueError("Need an explicitly exported curriculum selected model")
    architecture = ModelConfig.from_dict(p["model"]["_extra_state"]["model_config"])
    if architecture.vocab_size != len(codec):
        raise ValueError("Selected-model vocabulary mismatch")
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        model = CausalLM(architecture, tokenizer_fingerprint=codec.fingerprint)
    validate_weights(p["model"], model)
    if weights_digest(p["model"]) != p["selection"].get("weights_sha256"):
        raise ValueError("Selected-model weights checksum mismatch")
    model.load_state_dict(p["model"])
    return model.eval(), {"sha256": digest, "selection": p["selection"], "contract": p["contract"]}
