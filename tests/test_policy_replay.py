"""Independent policy oracles, noninterference, immutable replay and exact scoring."""

import copy
import json
import random
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.reasoning.stopping import Policy, evaluate_prefix
from llm_scratch.reasoning.voting import majority
from llm_scratch.research.bank_schema import (
    CandidateObservation,
    PolicyObservation,
    describe_output,
)
from llm_scratch.research.policy_replay import (
    create_replay,
    load_config,
    load_replay,
    parse_policies,
    replay_one,
    run,
)
from llm_scratch.research.replay_acceptance import check, write_config
from llm_scratch.research.replay_scoring import score_replay
from llm_scratch.research.sample_bank import SampleBank
from llm_scratch.utils.config import canonical_json, config_digest

ROOT = Path(__file__).resolve().parents[1]


def observation(answers, counts=None):
    counts = counts or [i + 2 for i in range(len(answers))]
    candidates = tuple(
        CandidateObservation(
            i,
            42,
            0,
            i,
            len(answers),
            (10,) * n,
            (-1.0,) * n,
            str(a),
            "eos",
            a,
            a is not None,
            a is not None,
            n,
            -float(n),
            -1.0,
        )
        for i, (a, n) in enumerate(zip(answers, counts, strict=True))
    )
    return PolicyObservation("cp", "question", "Compute", "easy", 42, candidates)


def test_tie_earliest_class_invalid_denominator_and_empty_abstention():
    vote = majority(observation(["8", "7", "7", "8", None]))
    assert (
        vote.answer,
        vote.representative_index,
        vote.top_votes,
        vote.valid_votes,
        vote.share,
    ) == ("8", 0, 2, 4, 0.4)
    assert majority(observation([None, "7", "8"])).representative_index == 1
    assert majority(observation([None, None])).answer is None
    assert majority(observation([])).share == 0
    with pytest.raises(ValueError, match="canonical"):
        majority(observation(["07"]))


def test_fixed_budget_unanimity_vote_threshold_and_fallback_oracles():
    group = observation([None, "7", "7", "7", "7", "8"])
    fixed = replay_one(group, Policy("fixed", "fixed", 4, 4))
    assert fixed["draws"] == 4 and fixed["answer"] == "7"
    assert fixed["observed_generated_tokens"] == sum([2, 3, 4, 5])
    assert fixed["stop_reason"] == "fixed_budget"
    assert [r["stop"] for r in fixed["history"]] == [False, False, False, True]
    unanimous = replay_one(group, Policy("unanimous", "agreement", 2, 6))
    assert unanimous["draws"] == 6 and unanimous["stop_reason"] == "budget_exhausted"
    threshold = replay_one(group, Policy("threshold", "agreement", 4, 6, 4, 5))
    assert threshold["draws"] == 5 and threshold["history"][-1]["vote_share"] == 0.8
    abstain = replay_one(observation([None] * 4), Policy("all-invalid", "agreement", 2, 4))
    assert abstain["answer"] is None and abstain["draws"] == 4
    assert abstain["observed_generated_tokens"] == 14
    with pytest.raises(ValueError, match="short"):
        replay_one(group.prefix(1), Policy("too-long", "fixed", 2, 2))


def test_future_and_gold_noninterference_and_only_prefix_exposed(monkeypatch):
    import llm_scratch.research.policy_replay as module

    group = observation(["8", "8", "7", "7"])
    policy = Policy("stop2", "agreement", 2, 4)
    seen = []
    original = module.evaluate_prefix

    def spy(prefix, spec):
        seen.append(len(prefix.candidates))
        assert set(asdict(prefix)) == {
            "checkpoint_id",
            "question_id",
            "question",
            "difficulty",
            "trial_seed",
            "candidates",
        }
        assert [c.candidate_index for c in prefix.candidates] == list(range(len(prefix.candidates)))
        assert all(
            not ({"correct", "reference", "canonical_answer"} & set(asdict(c)))
            for c in prefix.candidates
        )
        return original(prefix, spec)

    monkeypatch.setattr(module, "evaluate_prefix", spy)
    before = random.getstate(), torch.get_rng_state().clone()
    first = replay_one(group, policy)
    assert seen == [1, 2] and first["answer"] == "8"
    changed_future = replace(
        group, candidates=group.candidates[:2] + observation(["1", None, "999", "2"]).candidates[2:]
    )
    assert replay_one(changed_future, policy) == first
    assert random.getstate() == before[0] and torch.equal(torch.get_rng_state(), before[1])
    with pytest.raises(ValueError, match="nonempty"):
        evaluate_prefix(group.prefix(0), policy)


@pytest.mark.parametrize(
    "values",
    [
        ("p", "unknown", 2, 4),
        ("p", "fixed", 1, 2),
        ("p", "fixed", True, 1),
        ("p", "agreement", 0, 4),
        ("p", "agreement", 2, 17),
        ("p", "agreement", 2, 4, 0, 5),
        ("p", "agreement", 2, 4, 6, 5),
    ],
)
def test_invalid_specs(values):
    with pytest.raises(ValueError):
        Policy(*values)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    return check(tmp_path_factory.mktemp("wp08-3-parent") / "fixture", ROOT)


def test_actual_smoke_determinism_and_gold_is_not_opened(fixture, tmp_path, monkeypatch):
    bank, _, path = fixture
    m, rows = load_replay(path / "artifacts/replay", bank)
    assert len(rows) == 48
    policies = tuple(Policy(**r) for r in m["policies"])
    original = Path.open

    def guarded(p, *args, **kwargs):
        assert "references" not in p.name and "scored" not in p.name
        return original(p, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        repeated = create_replay(bank, policies, tmp_path / "repeat")
    assert repeated == m
    assert (tmp_path / "repeat/decisions.jsonl").read_bytes() == (
        path / "artifacts/replay/decisions.jsonl"
    ).read_bytes()
    assert json.loads((path / "run.json").read_text(encoding="utf-8"))["status"] == "completed"
    with pytest.raises(ValueError, match="outside"):
        create_replay(bank, policies, bank.root / "bad")
    with pytest.raises(FileExistsError):
        create_replay(bank, policies, tmp_path / "repeat")
    with pytest.raises(ValueError):
        parse_policies([m["policies"][0], m["policies"][0]])


@pytest.mark.parametrize("mutation", ["cost", "history", "gold", "order", "missing", "bool_type"])
def test_rehashed_decision_tampering_rejected(fixture, tmp_path, mutation):
    bank, _, path = fixture
    root = Path(shutil.copytree(path / "artifacts/replay", tmp_path / "replay"))
    rows = [
        json.loads(s) for s in (root / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if mutation == "cost":
        rows[0]["observed_generated_tokens"] += 1
    elif mutation == "history":
        rows[0]["history"][0]["top_votes"] = 999
    elif mutation == "gold":
        rows[0]["correct"] = True
    elif mutation == "order":
        rows.reverse()
    elif mutation == "missing":
        rows.pop()
    else:
        rows[0]["history"][0]["stop"] = int(rows[0]["history"][0]["stop"])
    (root / "decisions.jsonl").write_text(
        "".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8"
    )
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    m["decisions_sha256"] = file_sha256(root / "decisions.jsonl")
    m["fingerprint"] = config_digest({k: v for k, v in m.items() if k != "fingerprint"})
    (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        load_replay(root, bank)


def test_partial_replay_extra_gold_file_and_policy_config_guards(fixture, tmp_path):
    bank, _, path = fixture
    root = Path(shutil.copytree(path / "artifacts/replay", tmp_path / "replay"))
    (root / "references.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="gold-free"):
        load_replay(root, bank)
    (root / "references.jsonl").unlink()
    (root / "manifest.json").unlink()
    with pytest.raises(ValueError):
        load_replay(root, bank)
    config = load_config(ROOT / "configs/experiments/wp08_replay.toml")
    assert len(config.data["policies"]) == 8
    policies = copy.deepcopy(config.data["policies"])
    policies[0]["reference"] = "7"
    with pytest.raises(ValueError, match="fields"):
        parse_policies(policies)
    with pytest.raises(ValueError, match="exceeds"):
        parse_policies(config.data["policies"], available=4)


def authored_bank(fixture, tmp_path):
    source, _, _ = fixture
    root = Path(shutil.copytree(source.root, tmp_path / "authored-bank"))
    codec = ByteBPETokenizer.load(root / "tokenizer")
    rows = [
        json.loads(s) for s in (root / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for row in rows:
        answer = "8" if row["candidate_index"] < 2 else "7"
        ids = codec.encode("x") + [5, 6] + codec.encode(answer) + [7, 2]
        probs = [-1.0] * len(ids)
        row.update(generated_ids=ids, generated_logprobs=probs, finish_reason="eos")
        row.update(describe_output(ids, probs, "eos", codec))
    (root / "candidates.jsonl").write_text(
        "".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8"
    )
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    m["files"]["candidates.jsonl"] = {
        "sha256": file_sha256(root / "candidates.jsonl"),
        "size_bytes": (root / "candidates.jsonl").stat().st_size,
    }
    m["fingerprint"] = config_digest({k: v for k, v in m.items() if k != "fingerprint"})
    (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return SampleBank(root)


def test_external_gold_changes_scores_only_and_oracle_diagnostics_are_posthoc(fixture, tmp_path):
    # Authored counterfactual labels/completions for exact diagnostic arithmetic only.
    bank = authored_bank(fixture, tmp_path)
    policies = (Policy("stop2", "agreement", 2, 4), Policy("k4", "fixed", 4, 4))
    manifest = create_replay(bank, policies, tmp_path / "replay")
    decisions_sha = file_sha256(tmp_path / "replay/decisions.jsonl")
    refs = tmp_path / "references.jsonl"
    for answer in ("7", "8"):
        refs.write_text(
            "".join(
                canonical_json(
                    {"question_id": qid, "split": "validation", "canonical_answer": answer}
                )
                + "\n"
                for qid in sorted({g.question_id for g in bank.observations()})
            ),
            encoding="utf-8",
        )
        report = score_replay(tmp_path / "replay", bank, refs, tmp_path / f"score-{answer}")
        for item in report["metrics"]:
            s = item["scopes"]["all"]
            assert s["accuracy"] == (0 if answer == "7" else 1)
            assert s["question_trials"] == 6
            assert s["wrong_with_later_correct"] == (
                6 if answer == "7" and item["policy_id"] == "stop2" else 0
            )
            assert s["wrong_despite_observed_correct"] == (
                6 if answer == "7" and item["policy_id"] == "k4" else 0
            )
            assert s["token_savings_fraction"] == (0.5 if item["policy_id"] == "stop2" else 0)
        assert load_replay(tmp_path / "replay", bank)[0] == manifest
        assert file_sha256(tmp_path / "replay/decisions.jsonl") == decisions_sha
    with pytest.raises(ValueError, match="checksum"):
        score_replay(
            tmp_path / "replay", bank, refs, tmp_path / "bad-sha", expected_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="outside"):
        score_replay(tmp_path / "replay", bank, refs, tmp_path / "replay/gold")


def test_runner_reads_gold_only_after_committing_all_decisions(fixture, tmp_path, monkeypatch):
    bank, refs, _ = fixture
    config = tmp_path / "replay.toml"
    write_config(config, bank, refs, (Policy("k1", "fixed", 1, 1),))
    original = Path.open
    reads = []

    def guarded(p, *args, **kwargs):
        if p.resolve() == refs.resolve():
            markers = list((tmp_path / "runs").glob("*/artifacts/replay/manifest.json"))
            assert markers, "Gold was accessed before decisions committed"
            reads.append(str(p))
        return original(p, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    output = run(config, output_root=tmp_path / "runs")
    assert reads and (output / "artifacts/evaluation/report.json").is_file()
