"""WP09.4A frozen prefix signals and post-commit, single-class-aware diagnostics."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from llm_scratch.data.datasets import file_sha256
from llm_scratch.reasoning.answer_extraction import normalize_answer
from llm_scratch.reasoning.calibration import agreement_reliability
from llm_scratch.reasoning.confidence import PROTOCOL, prefix_signals
from llm_scratch.research.bank_schema import digest, integer
from llm_scratch.research.confidence_pilot import load_inputs, read_rows, write_rows
from llm_scratch.research.sample_bank import SampleBank, emit
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import LoadedConfig, config_digest
from llm_scratch.utils.environment import package_fingerprint
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run

MANIFEST_KEYS = {
    "schema_version",
    "work_package",
    "scope",
    "protocol",
    "bank_fingerprint",
    "records",
    "signals_sha256",
    "implementation",
    "fingerprint",
}
FEATURE_MEANS = (
    "answer_entropy_nats",
    "answer_entropy_normalized",
    "valid_fraction",
    "mean_sequence_logprob_sum",
    "mean_candidate_logprob_mean",
    "pooled_token_logprob_mean",
    "winner_member_logprob_mean",
    "representative_logprob_mean",
)


def signal_rows(bank):
    for group in bank.observations():
        for n in range(1, len(group.candidates) + 1):
            yield {
                "checkpoint_id": group.checkpoint_id,
                "question_id": group.question_id,
                "difficulty": group.difficulty,
                "trial_seed": group.trial_seed,
                **prefix_signals(group.prefix(n)),
            }


def create_trace(bank, output):
    """No references, label paths or scoring callbacks accepted here."""
    output = Path(output).resolve()
    if output.is_relative_to(bank.root):
        raise ValueError("Signal trace must stay outside bank")
    output.mkdir(parents=True, exist_ok=False)
    rows = list(signal_rows(bank))
    write_rows(output / "signals.jsonl", rows)
    bank.assert_unchanged()
    m = {
        "schema_version": 1,
        "work_package": "WP09.4A",
        "scope": "gold_free_prefix_signals",
        "protocol": PROTOCOL,
        "bank_fingerprint": bank.fingerprint,
        "records": len(rows),
        "signals_sha256": file_sha256(output / "signals.jsonl"),
        "implementation": package_fingerprint(),
    }
    m["fingerprint"] = config_digest(m)
    write_json(output / "manifest.json", m)
    emit("confidence_trace_committed", records=len(rows), fingerprint=m["fingerprint"])
    return m


def load_trace(bank, root):
    root = Path(root).resolve()
    if {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} != {
        "manifest.json",
        "signals.jsonl",
    }:
        raise ValueError("Trace requires exactly the gold-free payload")
    m = json.loads((root / "manifest.json").read_text())
    if (
        set(m) != MANIFEST_KEYS
        or m["schema_version"] != 1
        or m["work_package"] != "WP09.4A"
        or m["scope"] != "gold_free_prefix_signals"
        or m["protocol"] != PROTOCOL
        or m["bank_fingerprint"] != bank.fingerprint
        or config_digest({k: v for k, v in m.items() if k != "fingerprint"}) != m["fingerprint"]
        or m["implementation"] != package_fingerprint()
        or m["signals_sha256"] != file_sha256(root / "signals.jsonl")
    ):
        raise ValueError("Signal trace manifest/checksum/implementation mismatch")
    rows = read_rows(root / "signals.jsonl")
    if rows != list(signal_rows(bank)) or len(rows) != m["records"]:
        raise ValueError("Signals disagree with the frozen bank prefixes")
    return m, rows


def read_references(bank, trace, path, expected_sha):
    path = Path(path).resolve()
    if path.is_relative_to(bank.root) or path.is_relative_to(Path(trace).resolve()):
        raise ValueError("Gold must stay outside bank/trace")
    if file_sha256(path) != expected_sha:
        raise ValueError("Reference checksum mismatch")
    ids = set(bank.manifest["question_ids"])
    refs = {}
    for r in read_rows(path):
        if (
            set(r) != {"question_id", "split", "canonical_answer"}
            or r["question_id"] not in ids
            or r["question_id"] in refs
            or r["split"] != "validation"
            or not isinstance(r["canonical_answer"], str)
            or normalize_answer(r["canonical_answer"]) != r["canonical_answer"]
        ):
            raise ValueError("Invalid/duplicate/noncanonical external reference")
        refs[r["question_id"]] = r["canonical_answer"]
    if set(refs) != ids or file_sha256(path) != expected_sha:
        raise ValueError("References must exactly cover the stable pilot")
    return refs


def summarize(rows, edges):
    result = agreement_reliability(rows, edges)
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)
    features = {}
    for k in FEATURE_MEANS:
        # Do not silently exclude abstaining seeds from winner-related means.
        features[k] = (
            mean(mean(r[k] for r in rs) for rs in by_q.values())
            if all(r[k] is not None for r in rows)
            else None
        )
    return result | {
        "mean_signals": features,
        "abstained_observations": sum(r["answer"] is None for r in rows),
        "unanimous_observations": sum(r["agreement"] == 1 for r in rows),
        "wrong_unanimous_observations": sum(r["agreement"] == 1 and not r["correct"] for r in rows),
        "high_agreement_observations": sum(r["agreement"] >= 0.9 for r in rows),
        "wrong_high_agreement_observations": sum(
            r["agreement"] >= 0.9 and not r["correct"] for r in rows
        ),
    }


def score_trace(bank, trace, references, expected_sha, output, edges, plotted_prefixes):
    trace, output = Path(trace).resolve(), Path(output).resolve()
    if output.is_relative_to(bank.root) or output.is_relative_to(trace):
        raise ValueError("Scored outputs must stay outside bank/trace")
    manifest, rows = load_trace(bank, trace)
    manifest_sha = file_sha256(trace / "manifest.json")
    refs = read_references(bank, trace, references, expected_sha)
    scored = [
        r
        | {
            "reference": refs[r["question_id"]],
            "correct": r["answer"] is not None and r["answer"] == refs[r["question_id"]],
        }
        for r in rows
    ]
    metrics = []
    for n in range(1, bank.manifest["candidates_per_question"] + 1):
        selected = [r for r in scored if r["draws"] == n]
        metrics.append(
            {
                "draws": n,
                **summarize(selected, edges),
                "strata": {
                    s: summarize([r for r in selected if r["difficulty"] == s], edges)
                    for s in sorted({r["difficulty"] for r in selected})
                },
            }
        )
    # Fixed diagnostic cutpoint; no policy or threshold is selected by these examples.
    examples, seen = [], set()
    for r in scored:
        if (
            r["draws"] >= 2
            and r["agreement"] >= 0.9
            and not r["correct"]
            and r["question_id"] not in seen
        ):
            examples.append(r)
            seen.add(r["question_id"])
        if len(examples) == 5:
            break
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / "prefixes.scored.jsonl", scored)
    write_rows(output / "error_cases.jsonl", examples)
    report = {
        "schema_version": 1,
        "work_package": "WP09.4A",
        "status": "diagnostic_only",
        "bank_fingerprint": bank.fingerprint,
        "trace_fingerprint": manifest["fingerprint"],
        "references_sha256": expected_sha,
        "prefix_observations": len(scored),
        "unique_questions": len(refs),
        "question_seed_trials": len(bank.observations()),
        "correct_prefix_observations": sum(r["correct"] for r in scored),
        "correct_prefix_observations_n_ge_2": sum(r["correct"] for r in scored if r["draws"] >= 2),
        "metrics": metrics,
        "reliability_bin_edges": edges,
        "plotted_prefixes": plotted_prefixes,
        "new_inference_performed": False,
        "calibrator_fitted": False,
        "threshold_search_performed": False,
        "threshold_partition_scored": False,
        "evaluation_scored": False,
        "protocol": {
            "signal_protocol": PROTOCOL,
            "entropy": (
                "Natural log; canonical answers plus one invalid bucket; "
                "normalized by log(draws), n=1 -> 0"
            ),
            "logprob": (
                "Raw model FP32 natural-log probabilities, all generated tokens incl EOS, "
                "no padding; not sampling-distribution or correctness probabilities"
            ),
            "winner_logprob": (
                "Equal mean of candidate length-normalized logprobs within the observed "
                "winner class; earliest member also reported"
            ),
            "normalization": (
                "One prefix size at a time; equal question weight, seeds averaged within "
                "question; no pooling prefixes as independent trials"
            ),
            "reliability": (
                "Raw agreement only; fixed bins [lower, upper), final bin includes 1.0; "
                "empty bins null; ECE/Brier descriptive, no fitting"
            ),
            "high_agreement": "Fixed descriptive cutpoint >=0.9, not a fitted stopping threshold",
            "n1": (
                "Agreement=1 for any valid single answer; entropy=0, "
                "neither demonstrates correctness"
            ),
            "invalid": (
                "Abstain from voting, remain in denominator and one entropy bucket; "
                "zero entropy can mean all-invalid"
            ),
        },
        "interpretation": (
            "Development pilot only. Single-outcome data cannot assess discrimination "
            "or support calibration/threshold efficacy; no AUROC, fit or population-accuracy claim."
        ),
    }
    write_json(output / "report.json", report)
    export_artifacts(report, output)
    bank.assert_unchanged()
    if (
        file_sha256(trace / "manifest.json") != manifest_sha
        or file_sha256(trace / "signals.jsonl") != manifest["signals_sha256"]
        or file_sha256(references) != expected_sha
    ):
        raise ValueError("Trace/references changed during scoring")
    return report


def export_artifacts(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stream = io.StringIO(newline="")
    fields = [
        "draws",
        "accuracy",
        "mean_agreement",
        "raw_agreement_ece",
        "raw_agreement_brier",
        "unanimous_observations",
        "wrong_unanimous_observations",
        *FEATURE_MEANS,
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for m in report["metrics"]:
        writer.writerow({k: m[k] for k in fields if k not in FEATURE_MEANS} | m["mean_signals"])
    atomic_text(output / "prefix_summary.csv", stream.getvalue())
    stream = io.StringIO(newline="")
    fields = ["draws", *report["metrics"][0]["bins"][0].keys()]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for m in report["metrics"]:
        writer.writerows({"draws": m["draws"], **b} for b in m["bins"])
    atomic_text(output / "reliability_bins.csv", stream.getvalue())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    try:
        axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Identity reference")
        chosen = [m for m in report["metrics"] if m["draws"] in report["plotted_prefixes"]]
        markers = ["o", "s", "^", "D"]
        for j, m in enumerate(chosen):
            bins = [b for b in m["bins"] if b["observations"]]
            axes[0].plot(
                [b["mean_agreement"] for b in bins],
                [b["accuracy"] for b in bins],
                marker=markers[j % 4],
                linestyle=":",
                label=f"K={m['draws']}",
            )
            axes[1].bar(
                [i + (j - (len(chosen) - 1) / 2) * 0.18 for i in range(len(m["bins"]))],
                [b["question_weight_mass"] for b in m["bins"]],
                width=0.17,
                label=f"K={m['draws']}",
            )
        axes[0].set(
            xlabel="Mean raw agreement in bin",
            ylabel="Winner accuracy in bin",
            xlim=(-0.03, 1.03),
            ylim=(-0.04, 1.04),
        )
        axes[0].legend(loc="upper left")
        bins = report["metrics"][0]["bins"]
        axes[1].set_xticks(range(len(bins)), [f"{b['lower']:.1f}" for b in bins])
        axes[1].set(
            xlabel="Agreement bin lower edge (last includes 1.0)",
            ylabel="Question-weighted mass",
            ylim=(0, 1.05),
        )
        axes[1].legend(loc="upper left")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.set_axisbelow(True)
        fig.suptitle(
            f"WP09.4A raw agreement: {report['unique_questions']} questions, no calibration fit"
        )
        for ext in ("png", "svg"):
            fig.savefig(output / f"reliability_diagnostic.{ext}", dpi=160)
    finally:
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    try:
        ms = report["metrics"]
        ns = [m["draws"] for m in ms]
        axes[0].plot(ns, [m["mean_agreement"] for m in ms], label="Raw agreement")
        axes[0].plot(ns, [m["accuracy"] for m in ms], label="Winner accuracy")
        axes[0].set(ylabel="Question-weighted mean", ylim=(-0.04, 1.04))
        axes[1].plot(
            ns, [m["mean_signals"]["answer_entropy_normalized"] for m in ms], label="H / log(n)"
        )
        axes[1].set(ylabel="Normalized answer entropy", ylim=(-0.04, 1.04))
        for key, label in [
            ("mean_candidate_logprob_mean", "All observed candidates"),
            ("winner_member_logprob_mean", "Observed winner class"),
        ]:
            axes[2].plot(ns, [m["mean_signals"][key] for m in ms], label=label)
        axes[2].set(ylabel="Mean raw model logprob / generated token")
        for ax in axes:
            ax.set(xlabel="Observed prefix draws", xticks=[1, 2, 4, 8, 16])
            ax.grid(alpha=0.2)
            ax.legend(loc="best")
        fig.suptitle("WP09.4A prefix signals: descriptive scores, not calibrated correctness")
        for ext in ("png", "svg"):
            fig.savefig(output / f"prefix_signals.{ext}", dpi=160)
    finally:
        plt.close(fig)


def load_config(path):
    path = Path(path).resolve()
    d = json.loads(path.read_text(encoding="utf-8"))
    keys = {
        "pilot_inputs",
        "plan_fingerprint",
        "bank",
        "bank_fingerprint",
        "references",
        "references_sha256",
    }
    if (
        set(d)
        != {
            "schema_version",
            "experiment",
            "reproducibility",
            "artifacts",
            "inputs",
            "signal_protocol",
            "reliability_bin_edges",
            "plotted_prefixes",
        }
        or d["schema_version"] != 1
        or set(d["inputs"]) != keys
        or set(d["experiment"]) != {"name"}
        or not d["experiment"]["name"]
        or set(d["reproducibility"]) != {"seed"}
        or not integer(d["reproducibility"]["seed"], 0, 2**32 - 1)
        or set(d["artifacts"]) != {"root"}
        or not isinstance(d["artifacts"]["root"], str)
        or not d["artifacts"]["root"]
        or d["signal_protocol"] != PROTOCOL
        or d["reliability_bin_edges"] != [i / 10 for i in range(11)]
        or d["plotted_prefixes"] != [2, 4, 8, 16]
    ):
        raise ValueError("Invalid frozen confidence diagnostic config")
    for k, v in d["inputs"].items():
        if k.endswith(("fingerprint", "sha256")):
            if not digest(v):
                raise ValueError("Invalid diagnostic input fingerprint")
        elif not isinstance(v, str) or not v:
            raise ValueError("Invalid diagnostic input path")
        else:
            d["inputs"][k] = str((path.parent / v).resolve())
    d["artifacts"]["root"] = str((path.parent / d["artifacts"]["root"]).resolve())
    return LoadedConfig(d, {}, path)


def run(config, output_root=None):
    loaded = load_config(config)
    d, inp = loaded.data, loaded.data["inputs"]
    bank = SampleBank(inp["bank"])
    p, qs, _ = load_inputs(inp["pilot_inputs"], inp["plan_fingerprint"])
    if (
        bank.fingerprint != inp["bank_fingerprint"]
        or len(bank.manifest["checkpoints"]) != 1
        or bank.manifest["checkpoints"][0]["sha256"] != p["checkpoint_sha256"]
        or bank.manifest["question_ids"] != [q["question_id"] for q in qs]
        or bank.manifest["candidates_per_question"] != 16
    ):
        raise ValueError("Frozen calibration-fit pilot binding mismatch")
    if output_root is not None:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    if Path(d["artifacts"]["root"]).is_relative_to(bank.root):
        raise ValueError("Run outputs must stay outside bank")
    with Run(loaded) as active:
        emit("confidence_diagnostics_started", run_directory=str(active.path))
        art = active.path / "artifacts"
        create_trace(bank, art / "trace")
        report = score_trace(
            bank,
            art / "trace",
            inp["references"],
            inp["references_sha256"],
            art / "analysis",
            d["reliability_bin_edges"],
            d["plotted_prefixes"],
        )
        active.log(
            "confidence_diagnostics_completed",
            trace_fingerprint=report["trace_fingerprint"],
            prefix_observations=report["prefix_observations"],
        )
        emit(
            "confidence_diagnostics_finished",
            run_directory=str(active.path),
            prefix_observations=report["prefix_observations"],
            correct_prefix_observations=report["correct_prefix_observations"],
        )
    return active.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluation/wp09_confidence_diagnostics.json")
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    run(args.config, args.output_root)


if __name__ == "__main__":
    main()
