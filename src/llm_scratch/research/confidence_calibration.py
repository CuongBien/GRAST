"""WP09.4B synthetic engineering acceptance and bounded real-pilot fit eligibility."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import asdict
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.reasoning.calibrators import (
    PROTOCOL,
    CalibrationContext,
    CalibrationExample,
    CalibrationSpec,
    FeatureVector,
    example_weights,
    features_from_prefix,
    fit_calibrator,
    fit_support,
    load_calibrator,
    require_disjoint_assessment,
)
from llm_scratch.reasoning.confidence import prefix_signals
from llm_scratch.research.bank_schema import digest, identifier, integer
from llm_scratch.research.confidence_diagnostics import (
    create_trace,
    read_references,
)
from llm_scratch.research.confidence_diagnostics import (
    load_config as load_diagnostic_config,
)
from llm_scratch.research.confidence_pilot import load_inputs, read_rows, write_rows
from llm_scratch.research.sample_bank import SampleBank, emit
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import LoadedConfig, config_digest
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run

FIXTURE_PROTOCOL = "authored-two-outcome-overconfidence-disjoint-groups-v1"
INPUT_KEYS = {"group_id", "question_id", "trial_seed", "draws", "features"}


def synthetic_examples(partition, groups, seed):
    """Authored feature fixtures, not generations or measurements of a real model."""
    if partition not in ("calibration_fit", "synthetic_assessment") or not groups or groups % 32:
        raise ValueError("Synthetic groups must be a positive multiple of 32")
    rows = []
    shift = seed % 8 + (3 if partition == "synthetic_assessment" else 0)
    for i in range(groups):
        family = i % 4
        correct = ((i // 4 + shift) % 8) < [5, 3, 2, 1][family]
        identity = f"synthetic-{partition}-{i:04d}"
        for trial_seed in (42, 43, 44):
            for n in (2, 4, 8, 16):
                h = int(config_digest([FIXTURE_PROTOCOL, seed, identity, trial_seed, n])[:8], 16)
                votes = round(n * (1 - family / 8))
                votes = min(n, max(n // 2, votes + (h % 3 - 1 if n >= 8 else 0)))
                a = votes / n
                entropy = -sum(p * math.log(p) for p in (a, 1 - a) if p)
                logprob = -0.015 - 0.08 * (1 - a) - (h % 11) * 0.0002
                rows.append(
                    CalibrationExample(
                        partition,
                        identity,
                        identity,
                        trial_seed,
                        n,
                        FeatureVector(
                            (a, entropy / math.log(n), logprob, math.log(n) / math.log(16))
                        ),
                        correct,
                    )
                )
    return rows


def observable_inputs(examples):
    return [
        {
            "group_id": e.group_id,
            "question_id": e.question_id,
            "trial_seed": e.trial_seed,
            "draws": e.draws,
            "features": list(e.features.values),
        }
        for e in examples
    ]


def predict_inputs(model, rows, context):
    """Prediction boundary accepts no correctness, references or label callbacks."""
    if not rows or any(set(r) != INPUT_KEYS for r in rows):
        raise ValueError("Prediction inputs must be observable-only")
    features = [FeatureVector(tuple(r["features"])) for r in rows]
    if len({(r["question_id"], r["trial_seed"], r["draws"]) for r in rows}) != len(rows) or any(
        not identifier(r["question_id"])
        or not identifier(r["group_id"])
        or not integer(r["trial_seed"], 0, 2**32 - 1)
        or not integer(r["draws"], 2, 16)
        or not math.isclose(f.values[3], math.log(r["draws"]) / math.log(16), abs_tol=1e-12)
        for r, f in zip(rows, features, strict=True)
    ):
        raise ValueError("Invalid observable prediction identity/prefix")
    probs = model.predict_features(features, context=context)
    return [
        {k: v for k, v in r.items() if k != "features"}
        | {
            "raw_agreement": r["features"][0],
            "calibrated_confidence": p,
        }
        for r, p in zip(rows, probs, strict=True)
    ]


def probability_metrics(examples, scores):
    weights = example_weights(examples)
    if len(scores) != len(examples) or any(
        type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in scores
    ):
        raise ValueError("Invalid probability score vector")
    bins = []
    for j in range(10):
        indices = [i for i, p in enumerate(scores) if min(int(p * 10), 9) == j]
        mass = math.fsum(float(weights[i]) for i in indices)
        conf = math.fsum(float(weights[i]) * scores[i] for i in indices) / mass if mass else None
        acc = (
            math.fsum(float(weights[i]) * examples[i].correct for i in indices) / mass
            if mass
            else None
        )
        bins.append(
            {
                "lower": j / 10,
                "upper": (j + 1) / 10,
                "observations": len(indices),
                "group_weight_mass": mass,
                "mean_confidence": conf,
                "accuracy": acc,
            }
        )
    return {
        "accuracy": math.fsum(float(w) * e.correct for e, w in zip(examples, weights, strict=True)),
        "brier": math.fsum(
            float(w) * (p - e.correct) ** 2
            for e, w, p in zip(examples, weights, scores, strict=True)
        ),
        "log_loss": -math.fsum(
            float(w) * (math.log(max(1e-15, p)) if e.correct else math.log(max(1e-15, 1 - p)))
            for e, w, p in zip(examples, weights, scores, strict=True)
        ),
        "ece": math.fsum(
            b["group_weight_mass"] * abs(b["mean_confidence"] - b["accuracy"])
            for b in bins
            if b["observations"]
        ),
        "groups": len({e.group_id for e in examples}),
        "rows": len(examples),
        "bins": bins,
    }


def synthetic_acceptance(spec, cfg, output):
    output.mkdir(parents=True, exist_ok=False)
    fit = synthetic_examples("calibration_fit", cfg["fit_groups"], cfg["seed"])
    heldout = synthetic_examples("synthetic_assessment", cfg["assessment_groups"], cfg["seed"])
    write_rows(output / "fit.examples.jsonl", [asdict(e) for e in fit])
    inputs = observable_inputs(heldout)
    write_rows(output / "assessment.inputs.jsonl", inputs)
    model = fit_calibrator(fit, spec, CalibrationContext("synthetic"))
    require_disjoint_assessment(model, heldout)
    model.save(output / "calibrator.json")
    loaded = load_calibrator(output / "calibrator.json", expected_fingerprint=model.fingerprint)
    predictions = predict_inputs(
        loaded, read_rows(output / "assessment.inputs.jsonl"), CalibrationContext("synthetic")
    )
    if predictions != predict_inputs(model, inputs, CalibrationContext("synthetic")):
        raise AssertionError("Save/load changed calibrated predictions")
    write_rows(output / "assessment.predictions.jsonl", predictions)
    commit = {
        "calibrator_fingerprint": model.fingerprint,
        "inputs_sha256": file_sha256(output / "assessment.inputs.jsonl"),
        "predictions_sha256": file_sha256(output / "assessment.predictions.jsonl"),
    }
    commit["fingerprint"] = config_digest(commit)
    write_json(output / "prediction_commit.json", commit)
    # Correctness is attached only after observable-only predictions are committed.
    scored = [r | {"correct": e.correct} for r, e in zip(predictions, heldout, strict=True)]
    write_rows(output / "assessment.scored.jsonl", scored)
    raw = probability_metrics(heldout, [r["raw_agreement"] for r in predictions])
    calibrated = probability_metrics(heldout, [r["calibrated_confidence"] for r in predictions])
    if not calibrated["brier"] < raw["brier"] or not calibrated["log_loss"] < raw["log_loss"]:
        raise AssertionError("Calibration smoke failed on the authored overconfidence fixture")
    report = {
        "scope": "synthetic_verification_only",
        "fixture_protocol": FIXTURE_PROTOCOL,
        "calibrator_fingerprint": model.fingerprint,
        "fit_support": fit_support(fit, spec),
        "assessment_support": fit_support(heldout, spec),
        "raw": raw,
        "calibrated": calibrated,
        "save_load_predictions_exact": True,
        "prediction_commit": commit,
        "model_efficacy_claim": False,
        "optimizer": model.payload["optimizer"],
    }
    write_json(output / "report.json", report)
    export_plot(report, output)
    emit(
        "synthetic_calibration_verified",
        fit_groups=cfg["fit_groups"],
        assessment_groups=cfg["assessment_groups"],
        raw_brier=raw["brier"],
        calibrated_brier=calibrated["brier"],
    )
    return report


def real_eligibility(diagnostic_config, spec, output):
    d = load_diagnostic_config(diagnostic_config).data
    inp = d["inputs"]
    bank = SampleBank(inp["bank"])
    if Path(output).resolve().is_relative_to(bank.root):
        raise ValueError("Calibration eligibility outputs must stay outside bank")
    plan, qs, _ = load_inputs(inp["pilot_inputs"], inp["plan_fingerprint"])
    if (
        bank.fingerprint != inp["bank_fingerprint"]
        or len(bank.manifest["checkpoints"]) != 1
        or bank.manifest["checkpoints"][0]["sha256"] != plan["checkpoint_sha256"]
        or bank.manifest["question_ids"] != [q["question_id"] for q in qs]
    ):
        raise ValueError("Frozen calibration-fit pilot binding mismatch")
    output.mkdir(parents=True, exist_ok=False)
    trace = create_trace(bank, output / "trace")
    refs = read_references(bank, output / "trace", inp["references"], inp["references_sha256"])
    examples = []
    for g in bank.observations():
        for n in range(2, len(g.candidates) + 1):
            prefix = g.prefix(n)
            f = features_from_prefix(prefix)
            if f is not None:
                answer = prefix_signals(prefix)["answer"]
                examples.append(
                    CalibrationExample(
                        "calibration_fit",
                        g.question_id,
                        g.question_id,
                        g.trial_seed,
                        n,
                        f,
                        answer == refs[g.question_id],
                    )
                )
    support = (
        fit_support(examples, spec)
        if examples
        else {"rows": 0, "groups": 0, "positive_groups": 0, "negative_groups": 0, "eligible": False}
    )
    write_rows(output / "fit.examples.jsonl", [asdict(e) for e in examples])
    report = {
        "scope": "real_pilot_eligibility_only",
        "bank_fingerprint": bank.fingerprint,
        "trace_fingerprint": trace["fingerprint"],
        "references_sha256": inp["references_sha256"],
        "context": asdict(
            CalibrationContext(
                "model_validation",
                plan["checkpoint_sha256"],
                bank.manifest["tokenizer_fingerprint"],
            )
        ),
        "support": support,
        "real_calibrator_fitted": False,
        "next": "separate_scientific_fit_review_required"
        if support["eligible"]
        else "blocked_insufficient_positive_negative_groups",
    }
    # This acceptance command never fits a real calibrator, including on a future changed bank.
    write_json(output / "report.json", report)
    bank.assert_unchanged()
    if file_sha256(inp["references"]) != inp["references_sha256"]:
        raise ValueError("References changed during eligibility scan")
    emit("real_calibration_eligibility_checked", **support)
    return report


def export_plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    try:
        axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Identity reference")
        for name, marker in (("raw", "o"), ("calibrated", "s")):
            bins = [b for b in report[name]["bins"] if b["observations"]]
            axes[0].plot(
                [b["mean_confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                marker=marker,
                label=name,
            )
        axes[0].set(
            xlabel="Mean score in bin",
            ylabel="Synthetic accuracy",
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.03),
        )
        axes[0].legend()
        names = ["brier", "log_loss", "ece"]
        for j, name in enumerate(("raw", "calibrated")):
            axes[1].bar(
                [i + (j - 0.5) * 0.35 for i in range(3)],
                [report[name][k] for k in names],
                width=0.35,
                label=name,
            )
        axes[1].set_xticks(range(3), ["Brier", "Log loss", "ECE"])
        axes[1].set_ylabel("Synthetic assessment metric")
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
        fig.suptitle("WP09.4B SYNTHETIC verification only - no real-model calibration claim")
        for ext in ("png", "svg"):
            fig.savefig(output / f"synthetic_calibration.{ext}", dpi=160)
    finally:
        plt.close(fig)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["scope", "score", "groups", "rows", "accuracy", "brier", "log_loss", "ece"])
    for key in ("raw", "calibrated"):
        m = report[key]
        writer.writerow(
            [
                "synthetic_assessment_only",
                key,
                *[m[k] for k in ("groups", "rows", "accuracy", "brier", "log_loss", "ece")],
            ]
        )
    atomic_text(output / "synthetic_metrics.csv", stream.getvalue())


def load_config(path):
    path = Path(path).resolve()
    d = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(d)
        != {
            "schema_version",
            "experiment",
            "reproducibility",
            "artifacts",
            "protocol",
            "diagnostic_config",
            "diagnostic_config_fingerprint",
            "calibration_spec",
            "synthetic",
        }
        or d["schema_version"] != 1
        or d["protocol"] != PROTOCOL
        or set(d["experiment"]) != {"name"}
        or set(d["reproducibility"]) != {"seed"}
        or set(d["artifacts"]) != {"root"}
        or set(d["synthetic"]) != {"fit_groups", "assessment_groups", "seed", "fixture_protocol"}
        or d["synthetic"]["fit_groups"] != 128
        or d["synthetic"]["assessment_groups"] != 128
        or d["synthetic"]["fixture_protocol"] != FIXTURE_PROTOCOL
        or not integer(d["synthetic"]["seed"], 0, 2**32 - 1)
        or not integer(d["reproducibility"]["seed"], 0, 2**32 - 1)
        or not isinstance(d["experiment"]["name"], str)
        or not d["experiment"]["name"]
        or not isinstance(d["artifacts"]["root"], str)
        or not d["artifacts"]["root"]
        or set(d["calibration_spec"]) != set(asdict(CalibrationSpec()))
        or not digest(d["diagnostic_config_fingerprint"])
    ):
        raise ValueError("Invalid frozen WP09.4B acceptance config")
    CalibrationSpec(**d["calibration_spec"])
    for k in ("diagnostic_config",):
        d[k] = str((path.parent / d[k]).resolve())
    source_config = json.loads(Path(d["diagnostic_config"]).read_text(encoding="utf-8"))
    if config_digest(source_config) != d["diagnostic_config_fingerprint"]:
        raise ValueError("Diagnostic source config checksum mismatch")
    d["artifacts"]["root"] = str((path.parent / d["artifacts"]["root"]).resolve())
    return LoadedConfig(d, {}, path)


def run(config, *, synthetic_only=False, output_root=None):
    loaded = load_config(config)
    d = loaded.data
    if output_root is not None:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    bank_root = Path(load_diagnostic_config(d["diagnostic_config"]).data["inputs"]["bank"])
    if Path(d["artifacts"]["root"]).resolve().is_relative_to(bank_root):
        raise ValueError("Calibration run outputs must stay outside bank")
    spec = CalibrationSpec(**d["calibration_spec"])
    with Run(loaded) as active:
        emit(
            "calibration_acceptance_started",
            run_directory=str(active.path),
            synthetic_only=synthetic_only,
        )
        art = active.path / "artifacts"
        synthetic = synthetic_acceptance(spec, d["synthetic"], art / "synthetic")
        real = (
            None
            if synthetic_only
            else real_eligibility(d["diagnostic_config"], spec, art / "real-eligibility")
        )
        report = {
            "work_package": "WP09.4B",
            "status": "engineering_verified_only",
            "synthetic": synthetic,
            "real_pilot": real,
            "real_calibrator_fitted": False,
            "threshold_search_performed": False,
            "threshold_partition_scored": False,
            "evaluation_scored": False,
            "new_model_inference_performed": False,
            "scientific_efficacy_accepted": False,
        }
        write_json(art / "report.json", report)
        active.log(
            "calibration_acceptance_completed",
            real_calibrator_fitted=False,
            synthetic_calibrator_fingerprint=synthetic["calibrator_fingerprint"],
        )
        emit(
            "calibration_acceptance_finished",
            run_directory=str(active.path),
            real_calibrator_fitted=False,
        )
    return active.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/evaluation/wp09_calibration.json"))
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--output-root", type=Path)
    a = p.parse_args()
    run(a.config, synthetic_only=a.synthetic_only, output_root=a.output_root)


if __name__ == "__main__":
    main()
