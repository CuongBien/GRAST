"""WP09.3 offline diagnostic baselines; no new inference or calibration fitting."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from llm_scratch.data.datasets import file_sha256
from llm_scratch.evaluation.pilot_metrics import extract_generated_final
from llm_scratch.reasoning.formatting import encode_reasoning_prompt
from llm_scratch.reasoning.voting import majority
from llm_scratch.research.bank_schema import digest, integer
from llm_scratch.research.confidence_pilot import load_inputs, read_rows, write_rows
from llm_scratch.research.policy_replay import create_replay, load_replay, parse_policies
from llm_scratch.research.replay_scoring import score_replay
from llm_scratch.research.sample_bank import SampleBank, emit
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import LoadedConfig, config_digest
from llm_scratch.utils.io import write_json
from llm_scratch.utils.logging import Run

COST_KEYS = (
    "prefix_tokens",
    "batch_tokens",
    "padded_generation_slots",
    "prefix_prompt_tokens",
    "batch_prompt_tokens",
)


def costs(group, draws, prompt_tokens):
    """Post-decision costs may see complete launched batches, never a policy input."""
    cs = group.candidates
    if not integer(draws, 1, len(cs)) or not integer(prompt_tokens, 1):
        raise ValueError("Invalid cost prefix/prompt size")
    for i, c in enumerate(cs):
        if c.candidate_index != i or not c.batch_start <= i < c.batch_start + c.batch_size:
            raise ValueError("Invalid ordered batch layout")
    starts = {c.batch_start for c in cs[:draws]}
    charged = [c for c in cs if c.batch_start in starts]
    slots = 0
    for start in starts:
        batch = [c for c in charged if c.batch_start == start]
        if len(batch) != batch[0].batch_size or [c.candidate_index for c in batch] != list(
            range(start, start + len(batch))
        ):
            raise ValueError("Incomplete launched batch")
        slots += len(batch) * max(c.token_count for c in batch)
    return {
        "prefix_tokens": sum(c.token_count for c in cs[:draws]),
        "batch_tokens": sum(c.token_count for c in charged),
        "padded_generation_slots": slots,
        "prefix_prompt_tokens": draws * prompt_tokens,
        "batch_prompt_tokens": len(charged) * prompt_tokens,
        "charged_candidates": len(charged),
        "launched_batches": len(starts),
    }


def greedy_decisions(raw, bank, codec):
    """Dedicated greedy completions, one per question; never sampled K1."""
    questions = read_rows(bank.root / "questions.jsonl")
    if [r["question_id"] for r in raw] != [q["question_id"] for q in questions]:
        raise ValueError("Dedicated greedy question order mismatch")
    result = []
    cp = bank.manifest["checkpoints"][0]["id"]
    size, budget = bank.manifest["batch_size"], bank.manifest["generation"]["max_new_tokens"]
    for row, q in zip(raw, questions, strict=True):
        if set(row) != {
            "question_id",
            "difficulty",
            "split",
            "question",
            "prompt_ids_sha256",
            "generated_ids",
            "finish_reason",
            "generated_text",
        }:
            raise ValueError("Greedy decisions require raw model-only completions")
        ids = row["generated_ids"]
        prompt = encode_reasoning_prompt(codec, q["question"])
        if (
            row["question"] != q["question"]
            or row["difficulty"] != q["difficulty"]
            or row["split"] != q["split"]
            or row["prompt_ids_sha256"] != config_digest(prompt)
            or not ids
            or len(ids) > budget
            or any(not integer(t, 0, len(codec) - 1) for t in ids)
            or row["finish_reason"] not in ("eos", "max_new_tokens")
            or (row["finish_reason"] == "eos") != (ids[-1] == codec.eos_token_id)
            or codec.eos_token_id in ids[:-1]
            or (row["finish_reason"] == "max_new_tokens" and len(ids) != budget)
            or row["generated_text"] != codec.decode(ids, skip_special_tokens=False)
        ):
            raise ValueError("Invalid dedicated greedy prompt/tokens/finish")
        extracted = extract_generated_final(ids, codec, row["finish_reason"])
        result.append(
            {
                "checkpoint_id": cp,
                "question_id": q["question_id"],
                "difficulty": q["difficulty"],
                "trial_seed": None,
                "policy_id": "greedy",
                "draws": 1,
                "answer": extracted["predicted_answer"],
                "stop_reason": "dedicated_greedy",
                "prefix_tokens": len(ids),
                "batch_tokens": len(ids),
                "prefix_prompt_tokens": len(prompt),
                "batch_prompt_tokens": len(prompt),
                "charged_candidates": 1,
            }
        )
    for start in range(0, len(result), size):
        batch = result[start : start + size]
        longest = max(r["prefix_tokens"] for r in batch)
        for r in batch:
            r["padded_generation_slots"] = longest
    return result


def commit_decisions(bank, policies, greedy, codec, output):
    """Reference paths and labels are not accepted by the decision stage."""
    output = Path(output)
    if output.resolve().is_relative_to(bank.root):
        raise ValueError("Decision outputs must stay outside bank")
    output.mkdir(parents=True, exist_ok=False)
    native = create_replay(bank, policies, output / "sampled")
    gs = greedy_decisions(greedy, bank, codec)
    write_rows(output / "greedy.decisions.jsonl", gs)
    commit = {
        "schema_version": 1,
        "work_package": "WP09.3",
        "scope": "gold_free_decisions",
        "bank_fingerprint": bank.fingerprint,
        "sampled_replay_fingerprint": native["fingerprint"],
        "greedy_decisions_sha256": file_sha256(output / "greedy.decisions.jsonl"),
        "sampled_decisions": native["records"],
        "greedy_decisions": len(gs),
    }
    commit["fingerprint"] = config_digest(commit)
    write_json(output / "commit.json", commit)
    return commit


def load_config(path):
    path = Path(path).resolve()
    d = json.loads(path.read_text(encoding="utf-8"))
    input_keys = {
        "pilot_inputs",
        "plan_fingerprint",
        "bank",
        "bank_fingerprint",
        "generation_commit",
        "generation_fingerprint",
        "greedy",
        "greedy_sha256",
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
            "reference_policy",
            "policies",
        }
        or d["schema_version"] != 1
        or set(d["inputs"]) != input_keys
        or set(d["experiment"]) != {"name"}
        or set(d["reproducibility"]) != {"seed"}
        or set(d["artifacts"]) != {"root"}
        or not integer(d["reproducibility"]["seed"], 0, 2**32 - 1)
    ):
        raise ValueError("Invalid baseline config schema")
    parse_policies(d["policies"])
    if "greedy" in {r["id"] for r in d["policies"]}:
        raise ValueError("Greedy is a separate committed baseline")
    for k, v in d["inputs"].items():
        if k.endswith(("fingerprint", "sha256")):
            if not digest(v):
                raise ValueError("Invalid source fingerprint")
        elif not isinstance(v, str) or not v:
            raise ValueError("Invalid input path")
        else:
            d["inputs"][k] = str((path.parent / v).resolve())
    d["artifacts"]["root"] = str((path.parent / d["artifacts"]["root"]).resolve())
    return LoadedConfig(d, {}, path)


def grouped_summary(rows, reference):
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["question_id"]].append(r)
    questions = [
        {
            "question_id": q,
            "difficulty": rs[0]["difficulty"],
            "seeds_or_runs": len(rs),
            "accuracy": mean(r["correct"] for r in rs),
            "draws": mean(r["draws"] for r in rs),
            **{k: mean(r[k] for r in rs) for k in COST_KEYS},
        }
        for q, rs in by_q.items()
    ]
    avg = {k: mean(r[k] for r in questions) for k in COST_KEYS}
    base = {k: mean(reference[q][k] for q in by_q) for k in COST_KEYS}
    return questions, {
        "unique_questions": len(questions),
        "observations": len(rows),
        "correct_observations": sum(r["correct"] for r in rows),
        "accuracy": mean(r["accuracy"] for r in questions),
        "mean_draws": mean(r["draws"] for r in questions),
        "mean_costs": avg,
        "reference_mean_costs": base,
        "prefix_savings": 1 - avg["prefix_tokens"] / base["prefix_tokens"],
        "batch_savings": 1 - avg["batch_tokens"] / base["batch_tokens"],
        "padded_slot_savings": 1 - avg["padded_generation_slots"] / base["padded_generation_slots"],
        "wrong_early_stops": sum(r.get("wrong_early_stop", False) for r in rows),
        "avoidable_wrong_early_stops": sum(
            r.get("avoidable_wrong_early_stop", False) for r in rows
        ),
        "wrong_with_later_correct": sum(r.get("wrong_with_later_correct", False) for r in rows),
    }


def score_baselines(
    bank,
    codec,
    replay,
    references,
    expected_reference_sha,
    output,
    policies,
    reference_id,
    source_greedy,
):
    """All future-sample diagnostics and batch charging occur after decisions commit."""
    output, replay = Path(output), Path(replay)
    if output.resolve().is_relative_to(bank.root) or output.resolve().is_relative_to(
        replay.resolve()
    ):
        raise ValueError("Scored outputs must stay outside bank and replay")
    commit = json.loads((replay / "commit.json").read_text())
    native, decisions = load_replay(replay / "sampled", bank)
    if (
        config_digest({k: v for k, v in commit.items() if k != "fingerprint"})
        != commit["fingerprint"]
        or commit["bank_fingerprint"] != bank.fingerprint
        or commit["sampled_replay_fingerprint"] != native["fingerprint"]
        or commit["greedy_decisions_sha256"] != file_sha256(replay / "greedy.decisions.jsonl")
        or native["policies"] != [asdict(p) for p in policies]
    ):
        raise ValueError("Decision commit mismatch")
    output.mkdir(parents=True, exist_ok=False)
    score_replay(
        replay / "sampled",
        bank,
        references,
        output / "sampled",
        expected_sha256=expected_reference_sha,
    )
    sampled = read_rows(output / "sampled/scored.jsonl")
    refs = {r["question_id"]: r["reference"] for r in sampled}
    groups = {(g.checkpoint_id, g.question_id, g.trial_seed): g for g in bank.observations()}
    spec = {p.id: p for p in policies}
    reference_policy = spec.get(reference_id)
    if (
        reference_policy is None
        or reference_policy.kind != "fixed"
        or reference_policy.max_draws != bank.manifest["candidates_per_question"]
    ):
        raise ValueError("Reference policy must consume the full bank budget")
    reference_q = defaultdict(list)
    for group in groups.values():
        size = len(encode_reasoning_prompt(codec, group.question))
        reference_q[group.question_id].append(costs(group, len(group.candidates), size))
    reference = {q: {k: mean(r[k] for r in rs) for k in COST_KEYS} for q, rs in reference_q.items()}
    all_rows = []
    for row in sampled:
        group = groups[(row["checkpoint_id"], row["question_id"], row["trial_seed"])]
        charge = costs(group, row["draws"], len(encode_reasoning_prompt(codec, group.question)))
        if charge["prefix_tokens"] != row["observed_generated_tokens"]:
            raise ValueError("Native prefix cost mismatch")
        early = (
            spec[row["policy_id"]].kind == "agreement"
            and row["draws"] < len(group.candidates)
            and not row["correct"]
        )
        full_correct = majority(group).answer == refs[row["question_id"]]
        all_rows.append(
            row
            | charge
            | {
                "wrong_early_stop": early,
                "avoidable_wrong_early_stop": early and full_correct,
                "full_budget_correct": full_correct,
            }
        )
    greedy = read_rows(replay / "greedy.decisions.jsonl")
    if greedy != greedy_decisions(source_greedy, bank, codec):
        raise ValueError("Greedy decisions disagree with frozen raw completions")
    if [r["question_id"] for r in greedy] != bank.manifest["question_ids"]:
        raise ValueError("Greedy decisions must cover questions exactly once")
    for r in greedy:
        all_rows.append(
            r
            | {
                "reference": refs[r["question_id"]],
                "correct": r["answer"] is not None and r["answer"] == refs[r["question_id"]],
            }
        )
    write_rows(output / "scored.jsonl", all_rows)
    table, qrows = [], []
    for name in ["greedy", *[p.id for p in policies]]:
        selected = [r for r in all_rows if r["policy_id"] == name]
        per_question, summary = grouped_summary(selected, reference)
        qrows.extend(r | {"policy_id": name} for r in per_question)
        strata = {
            s: grouped_summary([r for r in selected if r["difficulty"] == s], reference)[1]
            for s in sorted({r["difficulty"] for r in selected})
        }
        table.append({"policy_id": name, **summary, "strata": strata})
    write_rows(output / "per_question.jsonl", qrows)
    examples, seen = [], set()
    history = {(d["policy_id"], d["question_id"], d["trial_seed"]): d for d in decisions}
    for r in all_rows:
        if r.get("wrong_early_stop") and r["question_id"] not in seen and len(examples) < 5:
            raw = history[(r["policy_id"], r["question_id"], r["trial_seed"])]
            examples.append(
                {
                    k: r[k]
                    for k in (
                        "policy_id",
                        "question_id",
                        "trial_seed",
                        "draws",
                        "answer",
                        "reference",
                        "full_budget_correct",
                        "wrong_with_later_correct",
                    )
                }
                | {"observed_vote_share": raw["history"][-1]["vote_share"]}
            )
            seen.add(r["question_id"])
    write_json(output / "error_cases.json", examples)
    oracle_count = sum(
        any(c.answer_valid and c.extracted_answer == refs[g.question_id] for c in g.candidates)
        for g in groups.values()
    )
    report = {
        "schema_version": 1,
        "work_package": "WP09.3",
        "status": "diagnostic_only",
        "bank_fingerprint": bank.fingerprint,
        "replay_fingerprint": native["fingerprint"],
        "decision_commit_fingerprint": commit["fingerprint"],
        "reference_policy": reference_id,
        "sampled_decisions": len(sampled),
        "greedy_decisions": len(greedy),
        "metrics": table,
        "unique_questions": len(refs),
        "question_seed_trials": len(groups),
        "oracle_covered_trials": oracle_count,
        "sampled_acquisition_generated_tokens": sum(
            c.token_count for g in groups.values() for c in g.candidates
        ),
        "greedy_acquisition_generated_tokens": sum(r["prefix_tokens"] for r in greedy),
        "new_inference_performed": False,
        "calibrator_fitted": False,
        "threshold_partition_scored": False,
        "evaluation_scored": False,
        "cost_protocol": {
            "prefix": "Valid generated tokens incl EOS in consumed prefix; ideal sequential proxy",
            "batch": "All valid tokens from every candidate in launched native batches",
            "padded_slots": (
                "Batch size times longest completion per launched batch; not FLOPs or latency"
            ),
            "prefill": "Prompt tokens reported separately, not included in generated-token savings",
            "greedy": ("Dedicated completions; padded slots use actual cross-question batches"),
            "normalization": (
                "Equal question weight, seeds averaged within question; "
                "ratios of means vs full fixed-K"
            ),
            "sunk_cost": "Full bank acquisition already paid; replay does not save this past cost",
        },
        "interpretation": (
            "Pilot diagnostic only; no calibration/efficacy or population-accuracy claim. "
            + (
                "Oracle=0 bounds selectors to zero accuracy on this bank."
                if oracle_count == 0
                else "Oracle coverage is a ceiling, not achieved policy accuracy."
            )
        ),
    }
    write_json(output / "report.json", report)
    export_table(table, output)
    return report


def export_table(table, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stream = io.StringIO(newline="")
    columns = [
        "policy",
        "questions",
        "observations",
        "accuracy",
        "mean_draws",
        "prefix_tokens",
        "batch_tokens",
        "padded_slots",
        "prefix_savings",
        "batch_savings",
        "wrong_early_stops",
        "avoidable_wrong_early_stops",
    ]
    writer = csv.writer(stream)
    writer.writerow(columns)
    for m in table:
        c = m["mean_costs"]
        writer.writerow(
            [
                m["policy_id"],
                m["unique_questions"],
                m["observations"],
                m["accuracy"],
                m["mean_draws"],
                c["prefix_tokens"],
                c["batch_tokens"],
                c["padded_generation_slots"],
                m["prefix_savings"],
                m["batch_savings"],
                m["wrong_early_stops"],
                m["avoidable_wrong_early_stops"],
            ]
        )
    atomic_text(output / "accuracy_token_table.csv", stream.getvalue())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    try:
        y = list(range(len(table)))
        axes[0].barh(
            [i - 0.18 for i in y],
            [r["mean_costs"]["prefix_tokens"] for r in table],
            height=0.34,
            label="Consumed prefix",
        )
        axes[0].barh(
            [i + 0.18 for i in y],
            [r["mean_costs"]["batch_tokens"] for r in table],
            height=0.34,
            label="Whole launched batches",
        )
        axes[0].legend(loc="upper right")
        axes[1].barh(
            y, [r["mean_costs"]["padded_generation_slots"] for r in table], color="#476582"
        )
        for ax in axes:
            ax.set_yticks(y, [r["policy_id"] for r in table])
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.2)
            ax.set_axisbelow(True)
        axes[0].set_xlabel("Mean generated tokens / question (seeds averaged)")
        axes[1].set_xlabel("Mean allocated generation slots / question")
        fig.suptitle("WP09.3 diagnostic pilot: cost proxies, not online latency")
        for ext in ("png", "svg"):
            fig.savefig(output / f"cost_proxies.{ext}", dpi=160)
    finally:
        plt.close(fig)


def run(config, output_root=None):
    loaded = load_config(config)
    d = loaded.data
    inp = d["inputs"]
    bank = SampleBank(inp["bank"])
    p, qs, codec = load_inputs(inp["pilot_inputs"], inp["plan_fingerprint"])
    g = json.loads(Path(inp["generation_commit"]).read_text())
    if (
        bank.fingerprint != inp["bank_fingerprint"]
        or len(bank.manifest["checkpoints"]) != 1
        or config_digest({k: v for k, v in g.items() if k != "fingerprint"}) != g["fingerprint"]
        or g["fingerprint"] != inp["generation_fingerprint"]
        or g["bank_fingerprint"] != bank.fingerprint
        or g["plan_fingerprint"] != p["fingerprint"]
        or g["greedy_sha256"] != inp["greedy_sha256"]
        or file_sha256(inp["greedy"]) != inp["greedy_sha256"]
        or [q["question_id"] for q in qs] != bank.manifest["question_ids"]
        or bank.manifest["checkpoints"][0]["sha256"] != p["checkpoint_sha256"]
    ):
        raise ValueError("Frozen pilot bank/greedy/generation binding mismatch")
    policies = parse_policies(d["policies"], bank.manifest["candidates_per_question"])
    reference = next((p for p in policies if p.id == d["reference_policy"]), None)
    if (
        reference is None
        or reference.kind != "fixed"
        or reference.max_draws != bank.manifest["candidates_per_question"]
    ):
        raise ValueError("Reference policy must consume the full bank budget")
    if output_root is not None:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    if Path(d["artifacts"]["root"]).is_relative_to(bank.root):
        raise ValueError("Run outputs must stay outside bank")
    with Run(loaded) as active:
        art = active.path / "artifacts"
        emit("wp09_baselines_started", run_directory=str(active.path))
        raw_greedy = read_rows(inp["greedy"])
        commit_decisions(bank, policies, raw_greedy, codec, art / "replay")
        report = score_baselines(
            bank,
            codec,
            art / "replay",
            inp["references"],
            inp["references_sha256"],
            art / "analysis",
            policies,
            d["reference_policy"],
            raw_greedy,
        )
        bank.assert_unchanged()
        if file_sha256(inp["greedy"]) != inp["greedy_sha256"]:
            raise ValueError("Greedy source changed during replay")
        active.log(
            "wp09_baselines_completed",
            bank_fingerprint=bank.fingerprint,
            sampled_decisions=report["sampled_decisions"],
            greedy_decisions=report["greedy_decisions"],
        )
        emit(
            "wp09_baselines_finished",
            run_directory=str(active.path),
            sampled_decisions=report["sampled_decisions"],
            greedy_decisions=report["greedy_decisions"],
        )
    return active.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/evaluation/wp09_baselines.json"))
    p.add_argument("--output-root", type=Path)
    a = p.parse_args()
    run(a.config, a.output_root)


if __name__ == "__main__":
    main()
