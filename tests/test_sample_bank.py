"""Ordered repeatable storage, corruption rejection and separate gold boundary."""

import copy
import json
import shutil
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

import pytest

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.research.bank_schema import describe_output, validate_grid, validate_questions
from llm_scratch.research.capability_acceptance import make_fixture
from llm_scratch.research.sample_bank import SampleBank, build_bank, load_config, run_builder
from llm_scratch.research.sample_bank_acceptance import questions
from llm_scratch.research.sample_bank_scoring import score_bank
from llm_scratch.utils.config import canonical_json, config_digest
from llm_scratch.utils.seed import seed_everything

CONFIG = Path("configs/experiments/wp08_sample_bank_smoke.toml")


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("bank-fixture")
    _, _, codec_root, paths = make_fixture(root, Path("configs/evaluation/wp07_capability.toml"))
    codec = ByteBPETokenizer.load(codec_root)
    specs = [
        {"id": str(i), "kind": "engine", "path": str(p), "sha256": file_sha256(p)}
        for i, p in enumerate(paths)
    ]
    kwargs = dict(
        generation=GenerationConfig(
            max_new_tokens=8, do_sample=True, top_k=20, temperature=0.8, eos_token_id=2
        ),
        candidates_per_question=3,
        batch_size=2,
        trial_seeds=[42, 7],
        log=lambda *a, **k: None,
    )
    seed_everything(42, deterministic=True, num_threads=1)
    bank = build_bank(root / "bank", questions(), codec, specs, **kwargs)
    return root, codec, specs, kwargs, bank


def clone(fixture, tmp_path):
    return Path(shutil.copytree(fixture[-1].root, tmp_path / "bank"))


def edit(root, mutate_rows=None, mutate_manifest=None):
    m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if mutate_rows:
        rows = [
            json.loads(s)
            for s in (root / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        mutate_rows(rows)
        (root / "candidates.jsonl").write_text(
            "".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8"
        )
        m["files"]["candidates.jsonl"] = {
            "sha256": file_sha256(root / "candidates.jsonl"),
            "size_bytes": (root / "candidates.jsonl").stat().st_size,
        }
    if mutate_manifest:
        mutate_manifest(m)
    m.pop("fingerprint")
    m["fingerprint"] = config_digest(m)
    (root / "manifest.json").write_text(canonical_json(m), encoding="utf-8")


def test_repeat_grid_order_immutable_prefix_and_gold_free_fields(fixture, tmp_path, monkeypatch):
    _, codec, specs, kwargs, first = fixture
    original = Path.open

    def guarded(path, *args, **kw):
        assert "references" not in path.name and "scored" not in path.name
        return original(path, *args, **kw)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", guarded)
        second = build_bank(tmp_path / "repeat", questions(), codec, specs, **kwargs)
    assert second.fingerprint == first.fingerprint
    assert second.observations() == first.observations()
    groups = first.observations()
    assert [(g.checkpoint_id, g.question_id, g.trial_seed) for g in groups] == [
        (cp["id"], q["question_id"], seed) for cp in specs for q in questions() for seed in [42, 7]
    ]
    assert first.manifest["records"] == 24
    candidate = groups[0].candidates[2]
    assert (
        candidate.candidate_index,
        candidate.batch_start,
        candidate.batch_size,
        candidate.batch_row,
    ) == (2, 2, 1, 0)
    observed = groups[0].prefix(1)
    assert len(observed.candidates) == 1 and observed.candidates[0].candidate_index == 0
    assert groups[0].prefix(0).candidates == ()
    assert set(asdict(observed)) == {
        "checkpoint_id",
        "question_id",
        "question",
        "difficulty",
        "trial_seed",
        "candidates",
    }
    assert not (
        {"correct", "gold", "reference", "canonical_answer", "selection"}
        & set(asdict(observed.candidates[0]))
    )
    with pytest.raises(FrozenInstanceError):
        observed.candidates = ()
    with pytest.raises(ValueError):
        observed.prefix(2)
    with pytest.raises(FileExistsError):
        build_bank(first.root, questions(), codec, specs, **kwargs)


def test_max_16_candidates_and_checkpoint_hash_guard(fixture, tmp_path):
    _, codec, specs, kwargs, _ = fixture
    bank = build_bank(
        tmp_path / "max16",
        questions()[:1],
        codec,
        specs[:1],
        **(kwargs | {"candidates_per_question": 16, "trial_seeds": [42], "batch_size": 5}),
    )
    assert [c.candidate_index for c in bank.observations()[0].candidates] == list(range(16))
    bad = copy.deepcopy(specs[:1])
    bad[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="(?i)sha|checksum"):
        build_bank(tmp_path / "bad-checkpoint", questions(), codec, bad, **kwargs)
    assert not (tmp_path / "bad-checkpoint/manifest.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("canonical_answer", "7"),
        ("correct", True),
        ("split", "test"),
        ("question", []),
        ("prompt_word_count", True),
    ],
)
def test_questions_reject_gold_test_and_malformed(field, value):
    rows = questions()
    rows[0][field] = value
    with pytest.raises(ValueError):
        validate_questions(rows)


@pytest.mark.parametrize(
    "k,b,seeds",
    [(17, 2, [42]), (0, 2, [42]), (2, 0, [42]), (2, 2, [42, 42]), (2, 2, [True]), (2, 2, [[]])],
)
def test_invalid_grid(k, b, seeds):
    with pytest.raises(ValueError):
        validate_grid(k, b, seeds)


@pytest.mark.parametrize(
    "key,value",
    [
        ("candidate_index", 1),
        ("generation_seed", 4),
        ("logprob_sum", 0.0),
        ("token_count", 999),
        ("answer_valid", True),
        ("correct", True),
        ("generated_text", "forged"),
        ("batch_row", 2),
    ],
)
def test_rehashed_semantic_corruption_rejected(fixture, tmp_path, key, value):
    root = clone(fixture, tmp_path)
    edit(root, lambda rows: rows[0].update({key: value}))
    with pytest.raises(ValueError):
        SampleBank(root)


def test_missing_extra_swapped_rows_and_post_read_mutation(fixture, tmp_path):
    root = clone(fixture, tmp_path)
    edit(root, lambda rows: rows.reverse())
    with pytest.raises(ValueError, match="order"):
        SampleBank(root)
    root = clone(fixture, tmp_path / "second")
    edit(root, lambda rows: rows.pop())
    with pytest.raises(ValueError, match="grid"):
        SampleBank(root)
    root = clone(fixture, tmp_path / "third")
    bank = SampleBank(root)
    (root / "gold.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unlisted"):
        bank.observations()
    with pytest.raises(ValueError, match="Unlisted"):
        SampleBank(root)


def test_checksum_and_uncommitted_directory_rejected(fixture, tmp_path):
    root = clone(fixture, tmp_path)
    with (root / "candidates.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        SampleBank(root)
    (root / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        SampleBank(root)


def test_gold_changes_external_labels_only(fixture, tmp_path):
    root = clone(fixture, tmp_path)
    codec = fixture[1]
    # Authored valid completion to test both true and false scorer outcomes.
    ids = (
        codec.encode("7")
        + [codec.special_id("</think>"), codec.special_id("<final>")]
        + codec.encode("7")
        + [codec.special_id("</final>"), codec.eos_token_id]
    )
    probs = [-1.0] * len(ids)

    def replace(rows):
        rows[0].update(generated_ids=ids, generated_logprobs=probs, finish_reason="eos")
        rows[0].update(describe_output(ids, probs, "eos", codec))

    edit(root, replace)
    bank = SampleBank(root)
    before = bank.fingerprint, bank.observations()
    refs = tmp_path / "references.jsonl"
    for answer, expected in (("7", 1), ("8", 0)):
        refs.write_text(
            "".join(
                canonical_json(
                    {
                        "question_id": q["question_id"],
                        "split": "validation",
                        "canonical_answer": answer,
                    }
                )
                + "\n"
                for q in questions()
            ),
            encoding="utf-8",
        )
        report = score_bank(bank, refs, tmp_path / f"score-{answer}")
        assert report["correct_candidates"] == expected
        assert (bank.fingerprint, bank.observations()) == before
    with pytest.raises(ValueError, match="outside"):
        score_bank(bank, refs, root / "scores")
    refs.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        score_bank(bank, refs, tmp_path / "missing")


def test_runner_writes_snapshot_logs_and_no_gold(fixture, tmp_path):
    _, codec, specs, _, _ = fixture
    config = tmp_path / "bank.toml"
    text = CONFIG.read_text(encoding="utf-8").split("[[checkpoints]]")[0]
    text = text.replace("max_new_tokens = 64", "max_new_tokens = 2")
    for cp in specs[:1]:
        text += "\n[[checkpoints]]\n" + "\n".join(f"{k} = {json.dumps(v)}" for k, v in cp.items())
    config.write_text(text, encoding="utf-8")
    codec.save(tmp_path / "codec")
    qfile = tmp_path / "questions.jsonl"
    qfile.write_text("".join(canonical_json(q) + "\n" for q in questions()), encoding="utf-8")
    assert load_config(config).data["bank"]["candidates_per_question"] == 2
    run = run_builder(config, qfile, tmp_path / "codec", output_root=tmp_path / "runs")
    assert (run / "config.json").is_file() and (run / "metadata.json").is_file()
    assert "bank_batch_finished" in (run / "logs/events.jsonl").read_text(encoding="utf-8")
    assert SampleBank(run / "artifacts/bank").manifest["records"] == 8
