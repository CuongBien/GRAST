"""Small offline replay fixture; model-quality results are not research evidence."""

import argparse
import json
import uuid
from dataclasses import asdict
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.reasoning.stopping import Policy
from llm_scratch.research.capability_acceptance import make_fixture
from llm_scratch.research.pilot_lock import read_jsonl
from llm_scratch.research.policy_replay import create_replay, load_replay, run
from llm_scratch.research.sample_bank import build_bank, emit
from llm_scratch.utils.seed import seed_everything


def make_bank(root, source_root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    _, dataset, token, paths = make_fixture(
        root, Path(source_root) / "configs/evaluation/wp07_capability.toml"
    )
    codec = ByteBPETokenizer.load(token)
    specs = [
        {"id": f"cp-{i}", "kind": "engine", "path": str(p.resolve()), "sha256": file_sha256(p)}
        for i, p in enumerate(paths)
    ]
    seed_everything(42, deterministic=True, num_threads=1)
    bank = build_bank(
        root / "bank",
        read_jsonl(dataset / "pilot/validation.questions.jsonl"),
        codec,
        specs,
        generation=GenerationConfig(max_new_tokens=8, do_sample=True, top_k=20, eos_token_id=2),
        candidates_per_question=4,
        batch_size=2,
        trial_seeds=[42, 43],
    )
    return bank, dataset / "pilot/validation.references.jsonl"


def write_config(path, bank, references, policies):
    text = (
        f'schema_version = 1\n[experiment]\nname = "wp08-3-replay-smoke"\n'
        f'[reproducibility]\nseed = 42\n[artifacts]\nroot = "runs"\n[inputs]\n'
        f"bank = {json.dumps(bank.root.as_posix())}\n"
        f'bank_fingerprint = "{bank.fingerprint}"\n'
        f"references = {json.dumps(Path(references).resolve().as_posix())}\n"
        f'references_sha256 = "{file_sha256(references)}"\n'
    )
    for policy in policies:
        text += (
            "\n[[policies]]\n"
            + "\n".join(f"{k} = {json.dumps(v)}" for k, v in asdict(policy).items())
            + "\n"
        )
    Path(path).write_text(text, encoding="utf-8")


def check(root, source_root):
    root = Path(root)
    bank, references = make_bank(root, source_root)
    policies = (
        Policy("k1", "fixed", 1, 1),
        Policy("k4", "fixed", 4, 4),
        Policy("unanimous2", "agreement", 2, 4),
        Policy("vote80", "agreement", 2, 4, 4, 5),
    )
    config = root / "replay.toml"
    write_config(config, bank, references, policies)
    path = run(config, output_root=root / "runs")
    first, decisions = load_replay(path / "artifacts/replay", bank)
    second = create_replay(bank, policies, root / "repeat")
    assert first == second and len(decisions) == 48
    emit(
        "policy_replay_smoke_passed",
        decisions=48,
        repeated_fingerprint_exact=True,
        run_directory=str(path),
    )
    return bank, references, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("."))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/wp08-replay-smoke"))
    a = parser.parse_args()
    check(a.output_root / uuid.uuid4().hex[:12], a.source_root)


if __name__ == "__main__":
    main()
