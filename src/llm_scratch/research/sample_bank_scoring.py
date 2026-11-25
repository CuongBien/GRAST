"""External evaluation labels. Never imported by bank construction or policy observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.reasoning.answer_extraction import normalize_answer
from llm_scratch.research.sample_bank import SampleBank, emit
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import canonical_json


def score_bank(bank, references, output):
    """Score all stored candidates after generation; no decisions or thresholds fitted."""
    if not isinstance(bank, SampleBank):
        bank = SampleBank(bank)
    output, references = Path(output).resolve(), Path(references).resolve()
    if output.is_relative_to(bank.root) or references.is_relative_to(bank.root):
        raise ValueError("Gold and scored outputs must remain outside the model-only bank")
    groups = bank.observations()
    ids = {g.question_id for g in groups}
    refs = {}
    for line in references.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or set(row) != {"question_id", "split", "canonical_answer"}
            or not isinstance(row["question_id"], str)
            or row["question_id"] not in ids
            or row["question_id"] in refs
            or row["split"] != "validation"
            or not isinstance(row["canonical_answer"], str)
            or normalize_answer(row["canonical_answer"]) != row["canonical_answer"]
        ):
            raise ValueError("Invalid/duplicate/noncanonical external reference")
        refs[row["question_id"]] = row["canonical_answer"]
    if set(refs) != ids:
        raise ValueError("References must exactly cover the bank question IDs")
    rows = []
    for group in groups:
        for candidate in group.candidates:
            rows.append(
                {
                    "checkpoint_id": group.checkpoint_id,
                    "question_id": group.question_id,
                    "trial_seed": group.trial_seed,
                    "candidate_index": candidate.candidate_index,
                    "extracted_answer": candidate.extracted_answer,
                    "canonical_answer": refs[group.question_id],
                    "correct": candidate.answer_valid
                    and candidate.extracted_answer == refs[group.question_id],
                }
            )
    bank.assert_unchanged()
    output.mkdir(parents=True, exist_ok=False)
    atomic_text(output / "scored.jsonl", "".join(canonical_json(r) + "\n" for r in rows))
    report = {
        "work_package": "WP08.1",
        "scope": "external_candidate_labels_only_no_feasibility_decision",
        "bank_fingerprint": bank.fingerprint,
        "references_sha256": file_sha256(references),
        "records": len(rows),
        "correct_candidates": sum(r["correct"] for r in rows),
        "scored_sha256": file_sha256(output / "scored.jsonl"),
    }
    atomic_text(output / "report.json", canonical_json(report) + "\n")
    emit("bank_scoring_finished", output_directory=str(output), **report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    score_bank(args.bank, args.references, args.output)


if __name__ == "__main__":
    main()
