"""Post-decision scoring; no labels or future-candidate diagnostics reach policies."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from llm_scratch.data.datasets import file_sha256
from llm_scratch.reasoning.answer_extraction import normalize_answer
from llm_scratch.research.policy_replay import load_replay
from llm_scratch.research.sample_bank import emit
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import canonical_json


def score_replay(replay_root, bank, references, output, *, expected_sha256=None):
    manifest, decisions = load_replay(replay_root, bank)
    replay_manifest_sha = file_sha256(Path(replay_root) / "manifest.json")
    references, output = Path(references).resolve(), Path(output).resolve()
    if (
        output.is_relative_to(bank.root)
        or output.is_relative_to(Path(replay_root).resolve())
        or references.is_relative_to(bank.root)
        or references.is_relative_to(Path(replay_root).resolve())
    ):
        raise ValueError("Gold and scored outputs must stay outside bank/replay roots")
    digest = file_sha256(references)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Replay reference checksum mismatch")
    ids = {g.question_id for g in bank.observations()}
    refs = {}
    for line in references.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if (
            not isinstance(r, dict)
            or set(r) != {"question_id", "split", "canonical_answer"}
            or not isinstance(r["question_id"], str)
            or r["question_id"] not in ids
            or r["question_id"] in refs
            or r["split"] != "validation"
            or not isinstance(r["canonical_answer"], str)
            or normalize_answer(r["canonical_answer"]) != r["canonical_answer"]
        ):
            raise ValueError("Invalid/duplicate/noncanonical replay reference")
        refs[r["question_id"]] = r["canonical_answer"]
    if set(refs) != ids or file_sha256(references) != digest:
        raise ValueError("References must exactly cover the stable pilot")
    groups = {(g.checkpoint_id, g.question_id, g.trial_seed): g for g in bank.observations()}
    scored = []
    for d in decisions:
        group = groups[(d["checkpoint_id"], d["question_id"], d["trial_seed"])]
        matches = [
            c.answer_valid and c.extracted_answer == refs[d["question_id"]]
            for c in group.candidates
        ]
        correct = d["answer"] is not None and d["answer"] == refs[d["question_id"]]
        full_tokens = sum(c.token_count for c in group.candidates)
        row = {k: v for k, v in d.items() if k != "history"}
        scored.append(
            row
            | {
                "reference": refs[d["question_id"]],
                "correct": correct,
                "abstained": d["answer"] is None,
                "full_bank_generated_tokens": full_tokens,
                "token_savings_fraction": 1 - d["observed_generated_tokens"] / full_tokens,
                "wrong_despite_observed_correct": not correct and any(matches[: d["draws"]]),
                "wrong_with_later_correct": not correct and any(matches[d["draws"] :]),
                "oracle_covered_at_bank_budget": any(matches),
            }
        )

    def summary(rows):
        tokens = sum(r["observed_generated_tokens"] for r in rows)
        full = sum(r["full_bank_generated_tokens"] for r in rows)
        return {
            "question_trials": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "accuracy": mean(r["correct"] for r in rows),
            "abstained": sum(r["abstained"] for r in rows),
            "mean_draws": mean(r["draws"] for r in rows),
            "observed_generated_tokens": tokens,
            "full_bank_generated_tokens": full,
            "token_savings_fraction": 1 - tokens / full,
            "wrong_with_later_correct": sum(r["wrong_with_later_correct"] for r in rows),
            "wrong_despite_observed_correct": sum(
                r["wrong_despite_observed_correct"] for r in rows
            ),
            "oracle_covered_at_bank_budget": sum(r["oracle_covered_at_bank_budget"] for r in rows),
        }

    metrics = []
    for cp in bank.manifest["checkpoints"]:
        for policy in manifest["policies"]:
            selected = [
                r
                for r in scored
                if r["checkpoint_id"] == cp["id"] and r["policy_id"] == policy["id"]
            ]
            scopes = ["all", *sorted({r["difficulty"] for r in selected})]
            per_seed = [
                {
                    "seed": seed,
                    "scopes": {
                        scope: summary(
                            [
                                r
                                for r in selected
                                if r["trial_seed"] == seed
                                and (scope == "all" or r["difficulty"] == scope)
                            ]
                        )
                        for scope in scopes
                    },
                }
                for seed in bank.manifest["trial_seeds"]
            ]
            metrics.append(
                {
                    "checkpoint_id": cp["id"],
                    "policy_id": policy["id"],
                    "scopes": {
                        scope: summary(
                            [r for r in selected if scope == "all" or r["difficulty"] == scope]
                        )
                        for scope in scopes
                    },
                    "per_seed": per_seed,
                    "seed_accuracy_range": {
                        scope: {
                            "mean": mean(s["scopes"][scope]["accuracy"] for s in per_seed),
                            "min": min(s["scopes"][scope]["accuracy"] for s in per_seed),
                            "max": max(s["scopes"][scope]["accuracy"] for s in per_seed),
                        }
                        for scope in scopes
                    },
                }
            )
    bank.assert_unchanged()
    if (
        file_sha256(Path(replay_root) / "manifest.json") != replay_manifest_sha
        or file_sha256(Path(replay_root) / "decisions.jsonl") != manifest["decisions_sha256"]
    ):
        raise RuntimeError("Committed replay changed during scoring")
    output.mkdir(parents=True, exist_ok=False)
    atomic_text(output / "scored.jsonl", "".join(canonical_json(r) + "\n" for r in scored))
    report = {
        "work_package": "WP08.3",
        "status": "completed",
        "decisions": len(scored),
        "bank_fingerprint": bank.fingerprint,
        "replay_fingerprint": manifest["fingerprint"],
        "references_sha256": digest,
        "metrics": metrics,
        "unique_questions": len(ids),
        "scope": "Development-validation answer exact match; no fitting or benchmark decision",
        "cost_protocol": (
            "Consumed-prefix generated tokens including EOS, excluding masked tails; "
            "no latency/VRAM claim. Full bank is a sunk offline acquisition cost."
        ),
        "diagnostic_protocol": (
            "Wrong-with-later-correct and oracle coverage use labels/future samples only "
            "after decisions; not policy inputs. Seeds reuse the same questions."
        ),
    }
    atomic_text(output / "report.json", canonical_json(report) + "\n")
    emit("replay_scored", decisions=len(scored), report_directory=str(output))
    return report
