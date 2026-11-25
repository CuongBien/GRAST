"""Bounded repeated-bank engineering checks; this is not the WP08 feasibility pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.research.capability_acceptance import make_fixture
from llm_scratch.research.sample_bank import build_bank, emit, load_config
from llm_scratch.research.sample_bank_scoring import score_bank
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import canonical_json
from llm_scratch.utils.environment import inspect_environment
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything


def questions():
    return [
        {
            "question_id": f"wp08-engineering-{i}",
            "question": text,
            "source": "authored-engineering-fixture",
            "split": "validation",
            "difficulty": "easy",
            "operation_count": 1,
            "reasoning_steps": 1,
            "prompt_length": "short",
            "prompt_word_count": len(text.split()),
        }
        for i, text in enumerate(("Evaluate (3 + 4).", "Evaluate (9 - 2)."))
    ]


def check(
    config,
    *,
    device="cpu",
    output_root=None,
    real_tokenizer=None,
    fixture_config=Path("configs/evaluation/wp07_capability.toml"),
):
    loaded = load_config(config)
    d = loaded.data
    if output_root:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    real = real_tokenizer is not None
    d["execution"] = {"device": device, "mode": "real-checkpoint" if real else "tiny-offline"}
    if real:
        # A bounded engineering check: 2 questions x 2 draws x 1 trial seed.
        d["bank"]["trial_seeds"] = [42]
    else:
        d["generation"]["max_new_tokens"] = 8
    active = Run(loaded)
    if not real:
        root = active.path / "artifacts/fixture"
        root.mkdir()
        _, _, tokenizer, paths = make_fixture(root, fixture_config)
        d["checkpoints"] = [
            {
                "id": f"{stage}-fixture",
                "kind": "engine",
                "path": str(p.resolve()),
                "sha256": file_sha256(p),
            }
            for stage, p in zip(("pretrain", "sft"), paths, strict=True)
        ]
    else:
        tokenizer = real_tokenizer
    # Config snapshot includes the actual checkpoint paths/digests and bounded grid.
    with active:

        def log(event, **fields):
            emit(event, **fields)
            active.log(event, **fields)

        seed_everything(**d["reproducibility"])
        environment = inspect_environment(device)
        if not environment["validation"]["ok"]:
            raise RuntimeError(environment["validation"]["errors"])
        atomic_text(active.path / "artifacts/environment.json", canonical_json(environment) + "\n")
        codec = ByteBPETokenizer.load(tokenizer)
        policy = GenerationConfig(
            **d["generation"], eos_token_id=codec.eos_token_id, pad_token_id=codec.pad_token_id
        )
        banks = []
        for index in range(2):
            seed_everything(**d["reproducibility"])
            banks.append(
                build_bank(
                    active.path / f"artifacts/bank-{index}",
                    questions(),
                    codec,
                    d["checkpoints"],
                    generation=policy,
                    trial_seeds=d["bank"]["trial_seeds"],
                    candidates_per_question=d["bank"]["candidates_per_question"],
                    batch_size=d["bank"]["batch_size"],
                    device=device,
                    precision=d["bank"]["precision"],
                    log=log,
                )
            )
        assert banks[0].fingerprint == banks[1].fingerprint
        assert banks[0].observations() == banks[1].observations()
        # References become an evaluation input only after both banks have committed.
        refs = active.path / "artifacts/engineering.references.jsonl"
        atomic_text(
            refs,
            "".join(
                canonical_json(
                    {
                        "question_id": q["question_id"],
                        "split": "validation",
                        "canonical_answer": "7",
                    }
                )
                + "\n"
                for q in questions()
            ),
        )
        score_bank(banks[0], refs, active.path / "artifacts/external-scoring")
        manifest = banks[0].manifest
        report = {
            "work_package": "WP08.1",
            "status": "passed",
            "device": device,
            "mode": d["execution"]["mode"],
            "records_per_bank": manifest["records"],
            "bank_fingerprint": banks[0].fingerprint,
            "repeated_banks_exact": True,
            "ordered_groups": len(banks[0].observations()),
            "generated_tokens_per_bank": sum(
                c.token_count for g in banks[0].observations() for c in g.candidates
            ),
            "gold_outside_bank": True,
            "scope": "Engineering reproducibility only; no feasibility or benchmark decision",
        }
        atomic_text(active.path / "artifacts/report.json", canonical_json(report) + "\n")
        log("sample_bank_smoke_passed", **report)
    emit("sample_bank_acceptance_finished", run_directory=str(active.path))
    return active.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/wp08_sample_bank_smoke.toml")
    )
    parser.add_argument(
        "--fixture-config", type=Path, default=Path("configs/evaluation/wp07_capability.toml")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--real-tokenizer",
        type=Path,
        help="Use the declared real checkpoint instead of tiny authored fixtures",
    )
    args = parser.parse_args()
    check(
        args.config,
        device=args.device,
        output_root=args.output_root,
        real_tokenizer=args.real_tokenizer,
        fixture_config=args.fixture_config,
    )


if __name__ == "__main__":
    main()
