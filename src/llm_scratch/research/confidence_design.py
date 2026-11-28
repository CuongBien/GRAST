"""WP09.1: bounded design-only probe and immutable grouped development benchmark."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from llm_scratch.curriculum.data import Corpus as OldCorpus
from llm_scratch.curriculum.evaluation import load_selected
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.difficulty import Expression, expression_from_dict, render_question
from llm_scratch.data.sft import math_reasoning
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.easy_recovery.data import Corpus as RecoveryCorpus
from llm_scratch.easy_recovery.data import facts
from llm_scratch.evaluation.capability import generate_questions
from llm_scratch.evaluation.pilot import QUESTION_KEYS
from llm_scratch.evaluation.pilot_metrics import extract_generated_final
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.reasoning.formatting import encode_reasoning_prompt
from llm_scratch.utils.config import canonical_json, config_digest
from llm_scratch.utils.environment import collect_metadata, inspect_environment
from llm_scratch.utils.io import write_json
from llm_scratch.utils.seed import seed_everything

PARTITIONS = ("design", "calibration_fit", "calibration_threshold", "evaluation")


def emit(event, **fields):
    print(canonical_json({"event": event, **fields}), flush=True)


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "seed",
        "partition_seed",
        "checkpoint_sha256",
        "tokenizer_fingerprint",
        "curriculum_fingerprint",
        "recovery_fingerprint",
        "design_per_stratum",
        "selected_strata",
        "main_per_stratum",
        "generation",
        "selection_rule",
        "strata",
    }
    if set(cfg) != expected or cfg["schema_version"] != 1:
        raise ValueError("Unknown design protocol fields")
    for k in ("seed", "partition_seed", "design_per_stratum", "selected_strata"):
        if type(cfg[k]) is not int or cfg[k] < 1:
            raise ValueError("Invalid design count/seed")
    if (
        cfg["design_per_stratum"] != 8
        or cfg["selected_strata"] != 2
        or set(cfg["main_per_stratum"]) != set(PARTITIONS[1:])
        or any(type(v) is not int or v < 1 for v in cfg["main_per_stratum"].values())
        or len({s["name"] for s in cfg["strata"]}) != len(cfg["strata"])
    ):
        raise ValueError("Invalid frozen pilot/partition sizes")
    for s in cfg["strata"]:
        if (
            set(s) != {"name", "operations", "a", "b", "c", "prompt_length"}
            or s["operations"] not in (1, 2)
            or s["prompt_length"] not in ("short", "medium")
            or any(
                len(s[k]) != 2
                or any(type(v) is not int for v in s[k])
                or not 1 <= s[k][0] <= s[k][1]
                for k in ("a", "b", "c")
            )
        ):
            raise ValueError("Invalid design stratum")
    g = cfg["generation"]
    if (
        set(g) != {"batch_size", "max_new_tokens", "seed", "precision"}
        or any(type(g[k]) is not int or g[k] < 1 for k in ("batch_size", "max_new_tokens"))
        or g["precision"] != "float32"
    ):
        raise ValueError("Design must use fixed FP32 greedy generation")
    return cfg


def partition_for(fact, seed):
    slot = int(config_digest([seed, fact])[:16], 16) % 100
    return PARTITIONS[0 if slot < 20 else 1 if slot < 50 else 2 if slot < 70 else 3]


def tree_facts(tree):
    return facts([{"task": "multi", "expression": tree.as_dict()}])


def question_only(rows):
    return [{k: r[k] for k in QUESTION_KEYS} for r in rows]


def make_rows(cfg, strata, partition, counts, excluded):
    """All evaluated integer operand pairs belong to ONE immutable hash partition."""
    used, rows = set(), []
    for spec in strata:
        rng = random.Random(int(config_digest([cfg["seed"], partition, spec["name"]]), 16))
        accepted = 0
        for _ in range(200000):
            if accepted == counts:
                break
            a, b, c = (rng.randint(*spec[k]) for k in ("a", "b", "c"))
            tree = Expression(
                operator=rng.choice(("+", "-")), left=Expression(value=a), right=Expression(value=b)
            )
            if spec["operations"] == 2:
                tree = Expression(
                    operator=rng.choice(("+", "-")), left=tree, right=Expression(value=c)
                )
            groups = tree_facts(tree)
            if groups & (excluded | used) or any(
                partition_for(g, cfg["partition_seed"]) != partition for g in groups
            ):
                continue
            semantic = config_digest(tree.canonical())
            prompt = render_question(tree, spec["prompt_length"])
            rows.append(
                {
                    "question_id": "wp09-" + semantic,
                    "question": prompt,
                    "source": "grast-wp09-development-v1",
                    "split": "validation",
                    "difficulty": spec["name"],
                    "operation_count": tree.operations,
                    "reasoning_steps": tree.depth,
                    "prompt_length": spec["prompt_length"],
                    "prompt_word_count": len(prompt.split()),
                    "partition": partition,
                    "semantic_id": semantic,
                    "fact_groups": sorted(groups),
                    "expression": tree.as_dict(),
                    "reasoning": math_reasoning(tree.as_dict()),
                    "final_answer": str(tree.evaluate()),
                }
            )
            used |= groups
            accepted += 1
        if accepted != counts:
            raise ValueError(f"Insufficient disjoint facts: {partition}/{spec['name']}")
    return rows


def audit_rows(rows, cfg, excluded):
    seen_ids, seen_prompts, seen_facts = set(), set(), set()
    for r in rows:
        tree = expression_from_dict(r["expression"])
        groups = tree_facts(tree)
        if (
            r["question_id"] != "wp09-" + config_digest(tree.canonical())
            or r["semantic_id"] != config_digest(tree.canonical())
            or r["fact_groups"] != sorted(groups)
            or groups & (excluded | seen_facts)
            or r["question_id"] in seen_ids
            or r["question"] in seen_prompts
            or any(partition_for(g, cfg["partition_seed"]) != r["partition"] for g in groups)
            or r["question"] != render_question(tree, r["prompt_length"])
            or r["final_answer"] != str(tree.evaluate())
            or r["reasoning"] != math_reasoning(r["expression"])
        ):
            raise ValueError("Question identity, reference or operand-pair partition leakage")
        seen_ids.add(r["question_id"])
        seen_prompts.add(r["question"])
        seen_facts |= groups


def score_design(predictions, rows, codec):
    refs = {r["question_id"]: r for r in rows}
    if (
        any(r["partition"] != "design" for r in rows)
        or len(refs) != len(rows)
        or len(predictions) != len(refs)
        or {p["question_id"] for p in predictions} != set(refs)
    ):
        raise ValueError("Probe scoring accepts the complete design partition only")
    scored = []
    for p in predictions:
        r = refs[p["question_id"]]
        result = extract_generated_final(p["generated_ids"], codec, p["finish_reason"])
        correct = result["predicted_answer"] == r["final_answer"]
        trace = codec.decode(p["generated_ids"]).split("</think>", 1)[0]
        scored.append(
            p
            | result
            | {
                "correct": correct,
                "stratum": r["difficulty"],
                "verified": correct
                and result["format_valid"]
                and trace.split() == r["reasoning"].split(),
            }
        )
    summary = {}
    for name in sorted({r["difficulty"] for r in rows}):
        group = [r for r in scored if r["stratum"] == name]
        n = len(group)
        summary[name] = {
            "questions": n,
            "correct": sum(r["correct"] for r in group),
            "accuracy": sum(r["correct"] for r in group) / n,
            "verified": sum(r["verified"] for r in group),
            "format_valid_rate": sum(r["format_valid"] for r in group) / n,
        }
    return scored, summary


def choose_strata(summary):
    if len(summary) < 2 or any(v["questions"] != 8 for v in summary.values()):
        raise ValueError("Need at least two complete eight-question design strata")
    ranked = sorted(summary, key=lambda s: (abs(summary[s]["accuracy"] - 0.5), s))
    eligible = [
        s
        for s in ranked
        if 1 <= summary[s]["correct"] <= 7 and summary[s]["format_valid_rate"] >= 0.75
    ]
    chosen = (eligible + [s for s in ranked if s not in eligible])[:2]
    return {
        "selected_strata": chosen,
        "status": "provisional_mixed_workload" if len(eligible) >= 2 else "diagnostic_only",
        "eligible_strata": eligible,
        "limit": "Eight design questions per stratum do not guarantee held-out outcome balance.",
    }


def write_rows(path, rows):
    path.write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8", newline="\n")


def verify(root, expected_fingerprint=None):
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        m.get("work_package") != "WP09.1"
        or config_digest({k: v for k, v in m.items() if k != "fingerprint"}) != m.get("fingerprint")
        or (expected_fingerprint is not None and m["fingerprint"] != expected_fingerprint)
    ):
        raise ValueError("Benchmark fingerprint mismatch")
    allowed = {
        "design.questions.jsonl",
        "design.references.jsonl",
        "design.predictions.jsonl",
        "design.scored.jsonl",
        "questions.jsonl",
        "references.jsonl",
        "index.jsonl",
    }
    if set(m["files"]) != allowed:
        raise ValueError("Unknown benchmark payload files")
    for name, spec in m["files"].items():
        p = root / name
        if p.is_symlink() or file_sha256(p) != spec["sha256"] or p.stat().st_size != spec["bytes"]:
            raise ValueError("Benchmark payload checksum mismatch")

    def read(name):
        return [json.loads(s) for s in (root / name).read_text(encoding="utf-8").splitlines()]

    rows = read("references.jsonl")
    design = read("design.references.jsonl")
    audit_rows(design + rows, m["config"], set())
    if (
        read("questions.jsonl") != question_only(rows)
        or read("design.questions.jsonl") != question_only(design)
        or dict(Counter(r["partition"] for r in rows)) != m["partition_counts"]
        or any(
            m["question_ids"][p] != [r["question_id"] for r in rows if r["partition"] == p]
            for p in PARTITIONS[1:]
        )
        or read("index.jsonl")
        != [
            {
                k: r[k]
                for k in ("question_id", "partition", "difficulty", "semantic_id", "fact_groups")
            }
            for r in rows
        ]
    ):
        raise ValueError("Benchmark question/index/partition mismatch")
    return m


def build(config, curriculum_root, recovery_root, tokenizer, checkpoint, output, device="cuda"):
    cfg = load_config(config)
    codec = ByteBPETokenizer.load(tokenizer, expected_fingerprint=cfg["tokenizer_fingerprint"])
    old = [OldCorpus(curriculum_root, p, codec) for p in ("train", "validation")]
    recovery = [RecoveryCorpus(recovery_root, p, codec) for p in ("train", "selection", "gate")]
    if (
        old[0].meta["fingerprint"] != cfg["curriculum_fingerprint"]
        or recovery[0].meta["fingerprint"] != cfg["recovery_fingerprint"]
    ):
        raise ValueError("Frozen source dataset mismatch")
    excluded = facts([r for corpus in old for r in corpus.examples]) | facts(
        [r for corpus in recovery for r in corpus.unique_examples]
    )

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "prospective_config.json", cfg)
    rows = make_rows(cfg, cfg["strata"], "design", cfg["design_per_stratum"], excluded)
    audit_rows(rows, cfg, excluded)
    write_rows(output / "design.questions.jsonl", question_only(rows))
    write_rows(output / "design.references.jsonl", rows)
    emit("wp09_design_locked_before_probe", questions=len(rows), config_sha256=config_digest(cfg))
    seed_everything(42)
    write_json(
        output / "metadata.json", collect_metadata(Path(config).resolve(), config_digest(cfg), 42)
    )
    environment = inspect_environment(device)
    if not environment["validation"]["ok"]:
        raise ValueError(environment["validation"]["errors"])
    write_json(output / "environment.json", environment)
    model, meta = load_selected(checkpoint, codec, cfg["checkpoint_sha256"])
    model.to(device).eval()
    g = cfg["generation"]
    if any(
        len(encode_reasoning_prompt(codec, r["question"])) + g["max_new_tokens"]
        > model.config.context_length
        for r in rows
    ):
        raise ValueError("Design generation exceeds context")
    predictions = generate_questions(
        model,
        codec,
        question_only(rows),
        GenerationConfig(
            max_new_tokens=g["max_new_tokens"],
            eos_token_id=codec.eos_token_id,
            pad_token_id=codec.pad_token_id,
        ),
        batch_size=g["batch_size"],
        seed=g["seed"],
        precision=g["precision"],
        emit=emit,
    )
    write_rows(output / "design.predictions.jsonl", predictions)
    scored, summary = score_design(predictions, rows, codec)
    decision = choose_strata(summary)
    write_rows(output / "design.scored.jsonl", scored)
    selected = [s for s in cfg["strata"] if s["name"] in decision["selected_strata"]]
    main = []
    for partition, count in cfg["main_per_stratum"].items():
        main.extend(make_rows(cfg, selected, partition, count, excluded))
    audit_rows(rows + main, cfg, excluded)
    if any(
        len(encode_reasoning_prompt(codec, r["question"])) + g["max_new_tokens"]
        > model.config.context_length
        for r in main
    ):
        raise ValueError("Held-out benchmark exceeds context")
    write_rows(output / "questions.jsonl", question_only(main))
    write_rows(output / "references.jsonl", main)
    write_rows(
        output / "index.jsonl",
        [
            {
                k: r[k]
                for k in ("question_id", "partition", "difficulty", "semantic_id", "fact_groups")
            }
            for r in main
        ],
    )
    codec.save(output / "tokenizer")
    protocol = {
        "schema_version": 1,
        "work_package": "WP09.1",
        "scope": "development_validation_only",
        "config": cfg,
        "design_summary": summary,
        "decision": decision,
        "checkpoint_sha256": meta["sha256"],
        "selected_step": meta["selection"]["step"],
        "excluded_facts": len(excluded),
        "excluded_facts_sha256": config_digest(sorted(excluded)),
        "partition_counts": dict(Counter(r["partition"] for r in main)),
        "question_ids": {
            p: [r["question_id"] for r in main if r["partition"] == p] for p in PARTITIONS[1:]
        },
        "bank": {
            "checkpoint": f"easy-recovery-selected-{meta['selection']['step']}",
            "seeds": [42, 43, 44],
            "max_candidates": 16,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 0,
            "max_new_tokens": g["max_new_tokens"],
            "precision": "float32",
            "batch_size": 8,
            "greedy_separate": True,
            "share_candidate_order_across_policies": True,
        },
        "future_analysis": {
            "fixed_k": [1, 2, 4, 8, 16],
            "consensus_counts": [2, 3, 4],
            "min_samples": 2,
            "calibration_target": "current prefix winner correct",
            "invalid_votes": "keep in denominator; invalid cannot trigger stop",
            "tie_break": "earliest occurrence",
            "max_accuracy_drop_vs_fixed16": 0.02,
            "threshold_selection_partition": "calibration_threshold",
            "calibrator_fit_partition": "calibration_fit",
            "report_partition": "evaluation",
            "uncertainty_unit": "question/fact group, cluster all seeds and prefixes",
            "confidence_features": [
                "agreement",
                "answer_entropy",
                "length_normalized_log_probability",
            ],
        },
        "test_files_read": False,
        "held_out_predictions_generated": False,
        "training_performed": False,
        "main_sampling_rule": "No filtering main questions by model outputs",
        "files": {
            p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
            for p in sorted(output.glob("*.jsonl"))
        },
    }
    protocol["fingerprint"] = config_digest(protocol)
    write_json(output / "manifest.json", protocol)
    verify(output, protocol["fingerprint"])
    emit(
        "wp09_design_ready",
        output=str(output),
        fingerprint=protocol["fingerprint"],
        decision=decision,
        counts=protocol["partition_counts"],
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/evaluation/wp09_design.json"))
    for name in ("curriculum-root", "recovery-root", "tokenizer", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    build(
        a.config, a.curriculum_root, a.recovery_root, a.tokenizer, a.checkpoint, a.output, a.device
    )


if __name__ == "__main__":
    main()
