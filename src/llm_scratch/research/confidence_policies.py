"""WP09.5 threshold engineering: synthetic selection/freeze, real diagnostic replay."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from llm_scratch.data.datasets import file_sha256
from llm_scratch.reasoning.answer_extraction import normalize_answer
from llm_scratch.reasoning.calibrators import CalibrationContext, CalibrationSpec, fit_calibrator
from llm_scratch.reasoning.confidence_stopping import PROTOCOL, ConfidencePolicy, replay_confidence
from llm_scratch.reasoning.stopping import Policy
from llm_scratch.reasoning.voting import majority
from llm_scratch.research import confidence_baselines as baselines
from llm_scratch.research.bank_schema import (
    CandidateObservation,
    PolicyObservation,
    digest,
    integer,
)
from llm_scratch.research.confidence_calibration import synthetic_examples
from llm_scratch.research.confidence_diagnostics import read_references
from llm_scratch.research.confidence_pilot import load_inputs, read_rows, write_rows
from llm_scratch.research.policy_replay import parse_policies, replay_one
from llm_scratch.research.sample_bank import SampleBank, emit
from llm_scratch.utils.config import LoadedConfig, config_digest
from llm_scratch.utils.environment import package_fingerprint
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run

PAPER = "https://arxiv.org/html/2511.02603v1"
FIXTURE = "ordered-four-family-late-correction-minority-confidence-v1"


def policy_grid(d, calibrator=None):
    policies = []
    for kind in ("raw_threshold", "calibrated_threshold", "cges_lns"):
        if kind == "calibrated_threshold" and calibrator is None:
            continue
        for minimum in d["min_draws"]:
            for threshold in d["thresholds"]:
                policies.append(
                    ConfidencePolicy(
                        f"{kind}-min{minimum}-t{int(threshold * 100):02d}",
                        kind,
                        minimum,
                        16,
                        threshold,
                        d["capacity"],
                        d["epsilon"],
                        calibrator.fingerprint if kind == "calibrated_threshold" else None,
                    )
                )
    return tuple(policies)


def synthetic_groups(partition, count, seed):
    """Gold-free ordered observations and a separate scorer-only reference mapping."""
    if (
        partition not in {"synthetic_threshold", "synthetic_evaluation"}
        or not integer(count, 4)
        or count % 4
    ):
        raise ValueError("Invalid synthetic policy split/groups")
    groups, refs = [], {}
    for i in range(count):
        q = f"wp09-5-{partition}-{i:04d}"
        correct, wrong = str(1000 + i), str(2000 + i)
        refs[q] = correct
        family = i % 4
        for trial in (42, 43, 44):
            candidates = []
            for j in range(16):
                if family == 0:
                    a, c = correct, 0.9
                elif family == 1:
                    a, c = (wrong, 0.6) if j < 2 else (correct, 0.9)
                elif family == 2:
                    a, c = wrong, 0.98
                else:
                    a, c = (correct, 0.95) if j in (4, 8, 12, 15) else (wrong, 0.04)
                h = int(config_digest([FIXTURE, seed, q, trial, j])[:8], 16)
                c += (h % 11 - 5) * 0.0005
                tokens = 20 + h % 3
                probs = tuple([math.log(c)] * tokens)
                candidates.append(
                    CandidateObservation(
                        j,
                        trial,
                        j // 4 * 4,
                        j % 4,
                        4,
                        tuple([10] * tokens),
                        probs,
                        a,
                        "eos",
                        a,
                        True,
                        True,
                        tokens,
                        math.fsum(probs),
                        math.fsum(probs) / tokens,
                    )
                )
            groups.append(
                PolicyObservation(
                    "synthetic-cp",
                    q,
                    f"Authored fixture {i}",
                    f"family-{family}",
                    trial,
                    tuple(candidates),
                )
            )
    return tuple(groups), refs


def replay_method(group, policy, calibrator, context):
    if type(policy) is Policy:
        return replay_one(group, policy)
    return replay_confidence(
        group,
        policy,
        calibrator=calibrator if policy.kind == "calibrated_threshold" else None,
        context=context,
    )


def commit_replay(groups, policies, source_fingerprint, output, *, calibrator=None, context=None):
    """No references, labels, threshold-selection callbacks or future-answer universe."""
    if (
        not groups
        or any(type(g) is not PolicyObservation for g in groups)
        or not policies
        or any(type(p) not in (Policy, ConfidencePolicy) for p in policies)
        or len({p.id for p in policies}) != len(policies)
        or not digest(source_fingerprint)
        or type(context) is not CalibrationContext
    ):
        raise ValueError("Require distinct frozen policy IDs")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, g in enumerate(groups, 1):
        rows.extend(replay_method(g, p, calibrator, context) for p in policies)
        if i == 1 or i % 60 == 0 or i == len(groups):
            emit("confidence_policy_replay_progress", groups_completed=i, groups_total=len(groups))
    write_rows(output / "decisions.jsonl", rows)
    manifest = {
        "schema_version": 1,
        "work_package": "WP09.5",
        "scope": "gold_free_prefix_decisions",
        "protocol": PROTOCOL,
        "source_fingerprint": source_fingerprint,
        "policies": [
            {"type": "baseline" if type(p) is Policy else "confidence", **asdict(p)}
            for p in policies
        ],
        "context": asdict(context),
        "records": len(rows),
        "decisions_sha256": file_sha256(output / "decisions.jsonl"),
        "implementation": package_fingerprint(),
    }
    manifest["fingerprint"] = config_digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


def full_reference(groups, prompt_sizes):
    return {
        q: {
            k: mean(
                baselines.costs(g, len(g.candidates), prompt_sizes[q])[k]
                for g in groups
                if g.question_id == q
            )
            for k in baselines.COST_KEYS
        }
        for q in prompt_sizes
    }


def score_committed(
    groups,
    policies,
    source_fingerprint,
    replay,
    refs,
    output,
    prompt_sizes,
    *,
    calibrator=None,
    context=None,
):
    replay, output = Path(replay).resolve(), Path(output).resolve()
    if output.is_relative_to(replay):
        raise ValueError("Scoring must stay outside the gold-free replay")
    if {p.name for p in replay.iterdir()} != {"manifest.json", "decisions.jsonl"}:
        raise ValueError("Unexpected gold-free replay payload")
    m = json.loads((replay / "manifest.json").read_text())
    manifest_sha = file_sha256(replay / "manifest.json")
    expected_keys = {
        "schema_version",
        "work_package",
        "scope",
        "protocol",
        "source_fingerprint",
        "policies",
        "context",
        "records",
        "decisions_sha256",
        "implementation",
        "fingerprint",
    }
    if (
        set(m) != expected_keys
        or m["schema_version"] != 1
        or m["work_package"] != "WP09.5"
        or m["scope"] != "gold_free_prefix_decisions"
        or m["protocol"] != PROTOCOL
        or m["source_fingerprint"] != source_fingerprint
        or m["context"] != asdict(context)
        or m["implementation"] != package_fingerprint()
        or m["policies"]
        != [
            {"type": "baseline" if type(p) is Policy else "confidence", **asdict(p)}
            for p in policies
        ]
        or m["fingerprint"] != config_digest({k: v for k, v in m.items() if k != "fingerprint"})
        or m["decisions_sha256"] != file_sha256(replay / "decisions.jsonl")
    ):
        raise ValueError("Frozen confidence replay binding mismatch")
    rows = read_rows(replay / "decisions.jsonl")
    if (
        rows != [replay_method(g, p, calibrator, context) for g in groups for p in policies]
        or len(rows) != m["records"]
    ):
        raise ValueError("Decisions disagree with observed-prefix replay")
    if (
        set(refs) != {g.question_id for g in groups}
        or set(prompt_sizes) != set(refs)
        or any(not isinstance(a, str) or normalize_answer(a) != a for a in refs.values())
    ):
        raise ValueError("References must exactly cover source questions")
    index = {(g.checkpoint_id, g.question_id, g.trial_seed): g for g in groups}
    scored = []
    specs = {p.id: p for p in policies}
    for row in rows:
        g = index[row["checkpoint_id"], row["question_id"], row["trial_seed"]]
        correct = row["answer"] is not None and row["answer"] == refs[g.question_id]
        full_correct = majority(g).answer == refs[g.question_id]
        p = specs[row["policy_id"]]
        adaptive = type(p) is ConfidencePolicy or p.kind == "agreement"
        early_wrong = adaptive and row["draws"] < len(g.candidates) and not correct
        step = row["history"][-1]
        scored.append(
            {k: v for k, v in row.items() if k != "history"}
            | {
                "reference": refs[g.question_id],
                "correct": correct,
                **baselines.costs(g, row["draws"], prompt_sizes[g.question_id]),
                "wrong_early_stop": early_wrong,
                "avoidable_wrong_early_stop": early_wrong and full_correct,
                "wrong_with_later_correct": not correct
                and any(
                    c.answer_valid and c.extracted_answer == refs[g.question_id]
                    for c in g.candidates[row["draws"] :]
                ),
                "full_budget_correct": full_correct,
                "stopping_score": step.get("confidence", step.get("vote_share")),
            }
        )
    reference = full_reference(groups, prompt_sizes)
    metrics = [
        {
            "policy_id": p.id,
            **baselines.grouped_summary([r for r in scored if r["policy_id"] == p.id], reference)[
                1
            ],
        }
        for p in policies
    ]
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "scored.jsonl", scored)
    write_json(output / "metrics.json", metrics)
    if (
        file_sha256(replay / "decisions.jsonl") != m["decisions_sha256"]
        or file_sha256(replay / "manifest.json") != manifest_sha
    ):
        raise ValueError("Committed replay changed during scoring")
    return metrics


def select_thresholds(metrics, policies, *, partition, max_regret):
    if partition not in {"synthetic_threshold", "calibration_threshold"}:
        raise ValueError("Threshold selection accepts threshold partition only")
    if (
        type(max_regret) not in (int, float)
        or not math.isfinite(max_regret)
        or not 0 <= max_regret <= 1
    ):
        raise ValueError("Invalid predeclared accuracy regret")
    reference = next(m for m in metrics if m["policy_id"] == "fixed-k16")
    lookup = {m["policy_id"]: m for m in metrics}
    result = {}
    for kind in ("raw_threshold", "calibrated_threshold", "cges_lns"):
        eligible = [
            p
            for p in policies
            if type(p) is ConfidencePolicy
            and p.kind == kind
            and lookup[p.id]["accuracy"] >= reference["accuracy"] - max_regret - 1e-12
        ]
        best = (
            min(
                eligible,
                key=lambda p: (
                    lookup[p.id]["mean_costs"]["prefix_tokens"],
                    -lookup[p.id]["accuracy"],
                    -p.threshold,
                    p.id,
                ),
            )
            if eligible
            else None
        )
        result[kind] = {
            "selected_policy": asdict(best) if best else None,
            "status": "selected_on_threshold_only" if best else "no_feasible_policy_use_fixed_k16",
        }
    return {
        "partition": partition,
        "criterion": "min_mean_prefix_tokens_subject_to_accuracy_regret",
        "max_accuracy_regret": max_regret,
        "reference_policy": "fixed-k16",
        "reference_accuracy": reference["accuracy"],
        "choices": result,
    }


def load_selection(path, *, expected_fingerprint, expected_scope):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    keys = {
        "partition",
        "criterion",
        "max_accuracy_regret",
        "reference_policy",
        "reference_accuracy",
        "choices",
        "work_package",
        "scope",
        "threshold_replay_fingerprint",
        "threshold_metrics_sha256",
        "calibrator_fingerprint",
        "fingerprint",
    }
    if (
        set(d) != keys
        or d["work_package"] != "WP09.5"
        or d["scope"] != expected_scope
        or d["scope"] not in {"synthetic_only", "model_validation"}
        or d["partition"]
        != {"synthetic_only": "synthetic_threshold", "model_validation": "calibration_threshold"}[
            d["scope"]
        ]
        or d["criterion"] != "min_mean_prefix_tokens_subject_to_accuracy_regret"
        or not 0 <= d["max_accuracy_regret"] <= 1
        or d["reference_policy"] != "fixed-k16"
        or not 0 <= d["reference_accuracy"] <= 1
        or any(
            not digest(d[k])
            for k in (
                "threshold_replay_fingerprint",
                "threshold_metrics_sha256",
                "calibrator_fingerprint",
                "fingerprint",
            )
        )
        or d["fingerprint"] != expected_fingerprint
        or config_digest({k: v for k, v in d.items() if k != "fingerprint"}) != d["fingerprint"]
        or set(d["choices"]) != {"raw_threshold", "calibrated_threshold", "cges_lns"}
    ):
        raise ValueError("Frozen threshold selection scope/schema/fingerprint mismatch")
    result = []
    for kind, r in d["choices"].items():
        if set(r) != {"selected_policy", "status"}:
            raise ValueError("Invalid frozen selection choice")
        if r["selected_policy"] is None:
            if r["status"] != "no_feasible_policy_use_fixed_k16":
                raise ValueError("Invalid fixed-budget fallback")
        else:
            p = ConfidencePolicy(**r["selected_policy"])
            if (
                p.kind != kind
                or r["status"] != "selected_on_threshold_only"
                or (
                    kind == "calibrated_threshold"
                    and p.calibrator_fingerprint != d["calibrator_fingerprint"]
                )
            ):
                raise ValueError("Frozen policy kind/calibrator mismatch")
            result.append(p)
    return tuple(result)


def synthetic_acceptance(d, base_policies, calibration_cfg, output):
    output.mkdir(parents=True, exist_ok=False)
    fit_rows = synthetic_examples(
        "calibration_fit",
        calibration_cfg["synthetic"]["fit_groups"],
        calibration_cfg["synthetic"]["seed"],
    )
    calibrator = fit_calibrator(
        fit_rows,
        CalibrationSpec(**calibration_cfg["calibration_spec"]),
        CalibrationContext("synthetic"),
    )
    calibrator.save(output / "synthetic-calibrator.json")
    grid = policy_grid(d, calibrator)
    threshold, refs = synthetic_groups("synthetic_threshold", d["synthetic_groups"], d["seed"])
    if set(calibrator.payload["training"]["group_ids"]) & set(refs):
        raise ValueError("Synthetic threshold groups overlap calibration fit")
    source = config_digest([asdict(g) for g in threshold])
    all_policies = (*base_policies, *grid)
    commit = commit_replay(
        threshold,
        all_policies,
        source,
        output / "threshold-replay",
        calibrator=calibrator,
        context=CalibrationContext("synthetic"),
    )
    metrics = score_committed(
        threshold,
        all_policies,
        source,
        output / "threshold-replay",
        refs,
        output / "threshold-analysis",
        {q: 12 for q in refs},
        calibrator=calibrator,
        context=CalibrationContext("synthetic"),
    )
    frozen = select_thresholds(
        metrics, all_policies, partition="synthetic_threshold", max_regret=d["max_accuracy_regret"]
    )
    frozen.update(
        {
            "work_package": "WP09.5",
            "scope": "synthetic_only",
            "threshold_replay_fingerprint": commit["fingerprint"],
            "threshold_metrics_sha256": file_sha256(output / "threshold-analysis/metrics.json"),
            "calibrator_fingerprint": calibrator.fingerprint,
        }
    )
    frozen["fingerprint"] = config_digest(frozen)
    write_json(output / "frozen_selection.json", frozen)
    frozen_sha = file_sha256(output / "frozen_selection.json")
    # Evaluation trajectories and labels are not generated until selection is frozen.
    evaluation, eval_refs = synthetic_groups(
        "synthetic_evaluation", d["synthetic_groups"], d["seed"]
    )
    if set(refs) & set(eval_refs) or set(calibrator.payload["training"]["group_ids"]) & set(
        eval_refs
    ):
        raise ValueError("Synthetic fit/threshold/evaluation overlap")
    chosen = load_selection(
        output / "frozen_selection.json",
        expected_fingerprint=frozen["fingerprint"],
        expected_scope="synthetic_only",
    )
    eval_policies = (*base_policies, *chosen)
    eval_source = config_digest([asdict(g) for g in evaluation])
    eval_commit = commit_replay(
        evaluation,
        eval_policies,
        eval_source,
        output / "evaluation-replay",
        calibrator=calibrator,
        context=CalibrationContext("synthetic"),
    )
    eval_metrics = score_committed(
        evaluation,
        eval_policies,
        eval_source,
        output / "evaluation-replay",
        eval_refs,
        output / "evaluation-analysis",
        {q: 12 for q in eval_refs},
        calibrator=calibrator,
        context=CalibrationContext("synthetic"),
    )
    if file_sha256(output / "frozen_selection.json") != frozen_sha:
        raise ValueError("Frozen threshold selection changed during evaluation")
    report = {
        "scope": "authored_synthetic_engineering_only",
        "fixture_protocol": FIXTURE,
        "groups_per_partition": d["synthetic_groups"],
        "seeds": [42, 43, 44],
        "threshold_decisions": commit["records"],
        "evaluation_decisions": eval_commit["records"],
        "selection": frozen,
        "evaluation_metrics": eval_metrics,
        "calibrator_fingerprint": calibrator.fingerprint,
        "evaluation_replay_fingerprint": eval_commit["fingerprint"],
        "evaluation_used_for_selection": False,
        "model_efficacy_claim": False,
    }
    write_json(output / "report.json", report)
    baselines.export_table(
        eval_metrics,
        output / "evaluation-analysis",
        title="WP09.5 SYNTHETIC verification: cost proxies, not online latency",
    )
    emit(
        "synthetic_confidence_policies_verified",
        threshold_decisions=commit["records"],
        evaluation_decisions=eval_commit["records"],
    )
    return report


def real_replay(base_cfg, d, output):
    inp = base_cfg["inputs"]
    bank = SampleBank(inp["bank"])
    plan, qs, codec = load_inputs(inp["pilot_inputs"], inp["plan_fingerprint"])
    g = json.loads(Path(inp["generation_commit"]).read_text())
    if (
        bank.fingerprint != inp["bank_fingerprint"]
        or len(bank.manifest["checkpoints"]) != 1
        or bank.manifest["question_ids"] != [q["question_id"] for q in qs]
        or bank.manifest["checkpoints"][0]["sha256"] != plan["checkpoint_sha256"]
        or config_digest({k: v for k, v in g.items() if k != "fingerprint"})
        != inp["generation_fingerprint"]
        or g["fingerprint"] != inp["generation_fingerprint"]
        or g["bank_fingerprint"] != bank.fingerprint
        or g["plan_fingerprint"] != plan["fingerprint"]
        or g["greedy_sha256"] != inp["greedy_sha256"]
        or file_sha256(inp["greedy"]) != inp["greedy_sha256"]
    ):
        raise ValueError("Frozen real bank/generation/greedy binding mismatch")
    if output.resolve().is_relative_to(bank.root):
        raise ValueError("Real policy outputs must stay outside bank")
    output.mkdir(parents=True, exist_ok=False)
    baseline_policies = parse_policies(base_cfg["policies"], 16)
    raw_greedy = read_rows(inp["greedy"])
    baselines.commit_decisions(
        bank, baseline_policies, raw_greedy, codec, output / "baseline-replay"
    )
    grid = policy_grid(d)
    context = CalibrationContext(
        "model_validation", plan["checkpoint_sha256"], bank.manifest["tokenizer_fingerprint"]
    )
    commit = commit_replay(
        bank.observations(), grid, bank.fingerprint, output / "confidence-replay", context=context
    )
    # All policy decisions, including baselines and dedicated greedy, are committed before gold.
    old = baselines.score_baselines(
        bank,
        codec,
        output / "baseline-replay",
        inp["references"],
        inp["references_sha256"],
        output / "baseline-analysis",
        baseline_policies,
        "fixed-k16",
        raw_greedy,
    )
    refs = read_references(
        bank, output / "confidence-replay", inp["references"], inp["references_sha256"]
    )
    from llm_scratch.reasoning.formatting import encode_reasoning_prompt

    prompts = {
        g.question_id: len(encode_reasoning_prompt(codec, g.question)) for g in bank.observations()
    }
    metrics = score_committed(
        bank.observations(),
        grid,
        bank.fingerprint,
        output / "confidence-replay",
        refs,
        output / "confidence-analysis",
        prompts,
        context=context,
    )
    merged = [*old["metrics"], *metrics]
    output.joinpath("comparison").mkdir()
    baselines.export_table(
        merged,
        output / "comparison",
        title="WP09.5 real pilot DIAGNOSTIC: cost proxies, not online latency",
    )
    report = {
        "scope": "real_calibration_fit_diagnostic_only",
        "bank_fingerprint": bank.fingerprint,
        "replay_fingerprint": commit["fingerprint"],
        "confidence_decisions": commit["records"],
        "baseline_sampled_decisions": old["sampled_decisions"],
        "greedy_decisions": old["greedy_decisions"],
        "unique_questions": old["unique_questions"],
        "question_seed_trials": old["question_seed_trials"],
        "metrics": merged,
        "skipped_real_calibrated_policy_count": len(d["min_draws"]) * len(d["thresholds"]),
        "skip_reason": "No real-model calibrator has been fitted; synthetic artifact forbidden",
        "real_threshold_selected": False,
        "real_calibrator_fitted": False,
        "real_threshold_partition_scored": False,
        "real_evaluation_scored": False,
        "new_model_inference_performed": False,
        "scientific_efficacy_accepted": False,
    }
    write_json(output / "report.json", report)
    bank.assert_unchanged()
    if (
        file_sha256(inp["references"]) != inp["references_sha256"]
        or file_sha256(inp["greedy"]) != inp["greedy_sha256"]
    ):
        raise ValueError("Frozen real sources changed during replay")
    emit(
        "real_confidence_policies_replayed",
        confidence_decisions=commit["records"],
        methods=len(merged),
    )
    return report


def load_config(path):
    path = Path(path).resolve()
    d = json.loads(path.read_text(encoding="utf-8"))
    keys = {
        "schema_version",
        "experiment",
        "reproducibility",
        "artifacts",
        "protocol",
        "baseline_config",
        "baseline_config_fingerprint",
        "calibration_config",
        "calibration_config_fingerprint",
        "thresholds",
        "min_draws",
        "capacity",
        "epsilon",
        "max_accuracy_regret",
        "synthetic_groups",
        "seed",
    }
    if (
        set(d) != keys
        or d["schema_version"] != 1
        or d["protocol"] != PROTOCOL
        or set(d["experiment"]) != {"name"}
        or set(d["reproducibility"]) != {"seed"}
        or set(d["artifacts"]) != {"root"}
        or not integer(d["reproducibility"]["seed"], 0, 2**32 - 1)
        or d["thresholds"] != [0.5, 0.75, 0.9, 0.95, 0.99]
        or d["min_draws"] != [2, 4]
        or d["capacity"] != 17
        or d["epsilon"] != 1e-6
        or d["max_accuracy_regret"] != 0.05
        or d["synthetic_groups"] != 64
        or not integer(d["seed"], 0, 2**32 - 1)
    ):
        raise ValueError("Invalid frozen confidence policy config")
    for key in ("baseline_config", "calibration_config"):
        d[key] = str((path.parent / d[key]).resolve())
        if (
            config_digest(json.loads(Path(d[key]).read_text(encoding="utf-8")))
            != d[key + "_fingerprint"]
        ):
            raise ValueError("Source config fingerprint mismatch")
    d["artifacts"]["root"] = str((path.parent / d["artifacts"]["root"]).resolve())
    return LoadedConfig(d, {}, path)


def run(config, *, synthetic_only=False, output_root=None):
    loaded = load_config(config)
    d = loaded.data
    base_cfg = baselines.load_config(d["baseline_config"]).data
    calibration_cfg = json.loads(Path(d["calibration_config"]).read_text(encoding="utf-8"))
    if output_root is not None:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    if Path(d["artifacts"]["root"]).resolve().is_relative_to(Path(base_cfg["inputs"]["bank"])):
        raise ValueError("Policy runs must stay outside bank")
    base_policies = parse_policies(base_cfg["policies"], 16)
    with Run(loaded) as active:
        emit(
            "confidence_policies_started",
            run_directory=str(active.path),
            synthetic_only=synthetic_only,
        )
        art = active.path / "artifacts"
        synthetic = synthetic_acceptance(d, base_policies, calibration_cfg, art / "synthetic")
        real = None if synthetic_only else real_replay(base_cfg, d, art / "real")
        report = {
            "work_package": "WP09.5",
            "status": "engineering_verified_only",
            "protocol": PROTOCOL,
            "paper": PAPER,
            "synthetic": synthetic,
            "real": real,
            "cges_adaptation": (
                "fixed capacity17, anonymous unseen slots, clipped response LNS; "
                "not a paper reproduction or calibrated correctness posterior"
            ),
            "real_calibrator_fitted": False,
            "real_threshold_selected": False,
            "real_threshold_partition_scored": False,
            "real_evaluation_scored": False,
            "new_model_inference_performed": False,
            "scientific_efficacy_accepted": False,
        }
        write_json(art / "report.json", report)
        active.log(
            "confidence_policies_completed",
            synthetic_selection_fingerprint=synthetic["selection"]["fingerprint"],
        )
        emit(
            "confidence_policies_finished",
            run_directory=str(active.path),
            real_threshold_selected=False,
        )
    return active.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/evaluation/wp09_policies.json"))
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--output-root", type=Path)
    a = p.parse_args()
    run(a.config, synthetic_only=a.synthetic_only, output_root=a.output_root)


if __name__ == "__main__":
    main()
