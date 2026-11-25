"""Commit gold-free decisions on stored prefixes; evaluation is a separate later read."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import asdict, fields
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.reasoning.stopping import Policy, evaluate_prefix
from llm_scratch.research.bank_schema import digest, identifier, integer
from llm_scratch.research.sample_bank import SampleBank, emit
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import LoadedConfig, canonical_json, config_digest
from llm_scratch.utils.environment import package_fingerprint
from llm_scratch.utils.logging import Run

PROTOCOL = "ordered-prefix-plurality-earliest-tie-all-draws-denominator-v1"
MANIFEST_KEYS = {
    "schema_version",
    "work_package",
    "scope",
    "protocol",
    "bank_fingerprint",
    "policies",
    "records",
    "decisions_sha256",
    "implementation",
    "fingerprint",
}


def parse_policies(rows, available=16):
    keys = {f.name for f in fields(Policy)}
    if not isinstance(rows, list) or not rows:
        raise ValueError("Declare at least one policy")
    result = []
    for r in rows:
        if not isinstance(r, dict) or set(r) != keys:
            raise ValueError("Unknown/missing replay policy fields")
        p = Policy(**r)
        if p.max_draws > available or p.id in {v.id for v in result}:
            raise ValueError("Duplicate policy or budget exceeds available bank")
        result.append(p)
    return tuple(result)


def replay_one(group, policy):
    """The driver retains the bank; evaluate_prefix receives only prefix(n)."""
    if len(group.candidates) < policy.max_draws:
        raise ValueError("Bank is too short for the declared policy")
    history = []
    for n in range(1, policy.max_draws + 1):
        observation = group.prefix(n)
        step = evaluate_prefix(observation, policy)
        history.append(step)
        if step["stop"]:
            return {
                "checkpoint_id": group.checkpoint_id,
                "question_id": group.question_id,
                "difficulty": group.difficulty,
                "trial_seed": group.trial_seed,
                "policy_id": policy.id,
                "draws": n,
                "answer": step["answer"],
                "representative_index": step["representative_index"],
                "stop_reason": step["reason"],
                "observed_generated_tokens": sum(c.token_count for c in observation.candidates),
                "history": history,
            }
    raise RuntimeError("A bounded declarative policy did not terminate")


def create_replay(bank, policies, output, *, log=emit):
    """No references, label paths or correctness callbacks are accepted."""
    if not isinstance(bank, SampleBank):
        raise ValueError("Replay requires a verified SampleBank")
    policies = parse_policies(
        [asdict(p) for p in policies], bank.manifest["candidates_per_question"]
    )
    output = Path(output).resolve()
    if output.is_relative_to(bank.root):
        raise ValueError("Replay must be written outside the model-only bank")
    output.mkdir(parents=True, exist_ok=False)
    count = 0
    groups = bank.observations()
    with (output / "decisions.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for index, group in enumerate(groups, 1):
            for policy in policies:
                stream.write(canonical_json(replay_one(group, policy)) + "\n")
                count += 1
            stream.flush()
            if index == 1 or index % 30 == 0 or index == len(groups):
                log(
                    "replay_progress",
                    groups_completed=index,
                    groups_total=len(groups),
                    decisions=count,
                )
    bank.assert_unchanged()
    manifest = {
        "schema_version": 1,
        "work_package": "WP08.3",
        "scope": "gold_free_decisions",
        "protocol": PROTOCOL,
        "bank_fingerprint": bank.fingerprint,
        "policies": [asdict(p) for p in policies],
        "records": count,
        "decisions_sha256": file_sha256(output / "decisions.jsonl"),
        "implementation": package_fingerprint(),
    }
    manifest["fingerprint"] = config_digest(manifest)
    atomic_text(output / "manifest.json", canonical_json(manifest) + "\n")
    load_replay(output, bank)
    log(
        "replay_committed",
        replay_directory=str(output),
        decisions=count,
        fingerprint=manifest["fingerprint"],
    )
    return manifest


def load_replay(root, bank):
    root = Path(root).resolve()
    if root.is_relative_to(bank.root):
        raise ValueError("Replay must live outside the bank")
    if {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} != {
        "manifest.json",
        "decisions.jsonl",
    }:
        raise ValueError("Only gold-free committed replay files are accepted")
    if any(p.is_symlink() for p in root.rglob("*")):
        raise ValueError("Replay symlinks forbidden")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or set(manifest) != MANIFEST_KEYS
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["work_package"] != "WP08.3"
        or manifest["scope"] != "gold_free_decisions"
        or manifest["protocol"] != PROTOCOL
        or manifest["bank_fingerprint"] != bank.fingerprint
        or config_digest({k: v for k, v in manifest.items() if k != "fingerprint"})
        != manifest["fingerprint"]
        or file_sha256(root / "decisions.jsonl") != manifest["decisions_sha256"]
    ):
        raise ValueError("Replay commit/protocol/bank/checksum mismatch")
    implementation = manifest["implementation"]
    if (
        not isinstance(implementation, dict)
        or set(implementation) != {"sha256", "files"}
        or not isinstance(implementation["files"], dict)
        or not implementation["files"]
        or any(
            not re.fullmatch(r"[a-z0-9_/]+\.py", k) or not digest(v)
            for k, v in implementation["files"].items()
        )
        or config_digest(implementation["files"]) != implementation["sha256"]
    ):
        raise ValueError("Invalid replay source fingerprint")
    policies = parse_policies(manifest["policies"], bank.manifest["candidates_per_question"])
    rows = [
        json.loads(s) for s in (root / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected = [replay_one(g, p) for g in bank.observations() for p in policies]
    if (
        not integer(manifest["records"], 1)
        or len(rows) != manifest["records"]
        or canonical_json(rows) != canonical_json(expected)
    ):
        raise ValueError("Replay order/grid/prefix/answer/cost/history mismatch")
    return manifest, rows


def load_config(path):
    path = Path(path).resolve()
    source = path.read_text(encoding="utf-8")
    d = tomllib.loads(source)
    sections = {
        "experiment": {"name"},
        "reproducibility": {"seed"},
        "artifacts": {"root"},
        "inputs": {"bank", "bank_fingerprint", "references", "references_sha256"},
    }
    if (
        set(d) != {"schema_version", "policies", *sections}
        or type(d["schema_version"]) is not int
        or d["schema_version"] != 1
    ):
        raise ValueError("Invalid replay config")
    for k, keys in sections.items():
        if not isinstance(d[k], dict) or set(d[k]) != keys:
            raise ValueError(f"Invalid {k} config fields")
    if (
        not identifier(d["experiment"]["name"])
        or not integer(d["reproducibility"]["seed"], 0, 2**32 - 1)
        or not digest(d["inputs"]["bank_fingerprint"])
        or not digest(d["inputs"]["references_sha256"])
    ):
        raise ValueError("Invalid replay identities/seed")
    parse_policies(d["policies"])
    for section, key in (("inputs", "bank"), ("inputs", "references"), ("artifacts", "root")):
        if not isinstance(d[section][key], str) or not d[section][key]:
            raise ValueError("Invalid replay path")
        d[section][key] = str((path.parent / d[section][key]).resolve())
    return LoadedConfig(d, {"replay": source}, path)


def run(config, *, output_root=None):
    loaded = load_config(config)
    d = loaded.data
    bank = SampleBank(d["inputs"]["bank"])
    if bank.fingerprint != d["inputs"]["bank_fingerprint"]:
        raise ValueError("Expected the exact frozen bank")
    if output_root:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    if Path(d["artifacts"]["root"]).resolve().is_relative_to(bank.root) or Path(
        d["inputs"]["references"]
    ).resolve().is_relative_to(bank.root):
        raise ValueError("Run outputs and references must stay outside the bank")
    policies = parse_policies(d["policies"], bank.manifest["candidates_per_question"])
    with Run(loaded) as active:

        def log(event, **fields):
            emit(event, **fields)
            active.log(event, **fields)

        create_replay(bank, policies, active.path / "artifacts/replay", log=log)
        from llm_scratch.research.replay_scoring import score_replay

        report = score_replay(
            active.path / "artifacts/replay",
            bank,
            d["inputs"]["references"],
            active.path / "artifacts/evaluation",
            expected_sha256=d["inputs"]["references_sha256"],
        )
        log(
            "replay_finished",
            replay_fingerprint=report["replay_fingerprint"],
            decisions=report["decisions"],
        )
    emit("replay_run_finished", run_directory=str(active.path))
    return active.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/wp08_replay.toml"))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    run(args.config, output_root=args.output_root)


if __name__ == "__main__":
    main()
