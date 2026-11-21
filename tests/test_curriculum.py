"""Grouped data, fixed generated selection, cumulative stages and exact portable resume."""

import copy
import json
import zipfile
from collections import Counter
from pathlib import Path

import pytest
import torch

from llm_scratch.curriculum import evaluation, runner
from llm_scratch.curriculum.acceptance import check, fixture
from llm_scratch.curriculum.bundle import create, extract
from llm_scratch.curriculum.data import (
    Corpus,
    audit,
    authored,
    group_split,
    load_data_config,
    prepare,
)
from llm_scratch.curriculum.evaluation import (
    load_selected,
    questions_only,
    score,
    select_validation,
)
from llm_scratch.curriculum.runner import CurriculumJob, load_recipe
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.checkpoints import load_for_evaluation
from llm_scratch.evaluation.pilot import QUESTION_KEYS
from llm_scratch.research.training_acceptance import _same
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import config_digest
from llm_scratch.utils.seed import seed_everything

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(tmp_path):
    return fixture(tmp_path / "inputs", ROOT)


def job_from(inputs):
    config, data, tokenizer, checkpoint = inputs
    codec = ByteBPETokenizer.load(tokenizer)
    recipe = load_recipe(config).data
    rng = seed_everything(42)
    model, meta, _ = load_for_evaluation(checkpoint, codec)
    engine = TrainingEngine(model, TrainingConfig.from_dict(recipe["training"]), generator=rng)
    lineage = {
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "source_stage": "sft",
        "source_step": meta["step"],
    }
    return CurriculumJob(
        engine, Corpus(data, "train", codec), Corpus(data, "validation", codec), recipe, lineage
    )


def response(codec, final, reasoning="The copied operands are correct."):
    return [
        *codec.encode(reasoning),
        codec.special_id("</think>"),
        codec.special_id("<final>"),
        *codec.encode(final),
        codec.special_id("</final>"),
        codec.eos_token_id,
    ]


def test_grouped_authored_data_reproducible_and_swaps_ops_share_split(inputs):
    cfg = load_data_config(inputs[0].parent / "data.toml")
    first, second = authored(cfg), authored(cfg)
    assert first == second
    groups = {s: {r["group"] for r in rows} for s, rows in first.items()}
    assert not groups["train"] & groups["validation"]
    by_pair = {}
    for split, rows in first.items():
        for r in rows:
            assert group_split(r["group"], cfg) == split
            assert min(r["operands"]) >= 31
            if r["task"] != "multi":
                key = tuple(sorted(r["operands"]))
                assert key not in by_pair or by_pair[key] == split
                by_pair[key] = split
    assert len(by_pair) == 55


def test_preparation_never_reads_test_or_pilot_and_does_not_overwrite(
    inputs, tmp_path, monkeypatch
):
    root = inputs[0].parent
    original = Path.open

    def guarded(p, *args, **kwargs):
        if "test" in p.name or "pilot" in p.parts:
            raise AssertionError("Curriculum must not read test/pilot files")
        return original(p, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    target = tmp_path / "prepared2"
    prepare(root / "data.toml", root / "wp03", root / "tokenizer", target)
    assert (target / "train.jsonl").read_bytes() == (inputs[1] / "train.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(root / "data.toml", root / "wp03", root / "tokenizer", target)


def test_audit_and_source_checksums_reject_leakage(inputs, tmp_path):
    codec = ByteBPETokenizer.load(inputs[2])
    train = Corpus(inputs[1], "train", codec).examples
    val = Corpus(inputs[1], "validation", codec).examples
    audit(train, val)
    with pytest.raises(ValueError):
        audit(train, val + [train[0]])
    source = inputs[0].parent / "wp03/math/train.jsonl"
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        prepare(inputs[0].parent / "data.toml", source.parents[1], inputs[2], tmp_path / "bad")


def test_loader_checks_semantics_and_prompt_mask_even_when_file_rehashed(inputs):
    config, data, tokenizer, _ = inputs
    codec = ByteBPETokenizer.load(tokenizer)
    corpus = Corpus(data, "train", codec)
    batch = corpus.view(["copy"]).batch([0, 1])
    for ids, labels, mask in zip(
        batch["input_ids"], batch["labels"], batch["attention_mask"], strict=True
    ):
        first = int((labels != -100).nonzero()[0])
        assert (
            ids[first - 1] == codec.special_id("<think>") and (labels[~mask.bool()] == -100).all()
        )
    rows = corpus.examples
    rows[0]["final_answer"] = "99,99"
    payload = data / "train.jsonl"
    payload.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["splits"]["train"].update(
        sha256=file_sha256(payload), size_bytes=payload.stat().st_size
    )
    manifest["fingerprint"] = config_digest(
        {k: v for k, v in manifest.items() if k != "fingerprint"}
    )
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Copy"):
        Corpus(data, "train", codec)
    with pytest.raises(RuntimeError):
        corpus.view().batch([0])


def test_no_silent_truncation(inputs, tmp_path):
    root = inputs[0].parent
    text = (root / "data.toml").read_text().replace("max_length = 512", "max_length = 8")
    short = root / "short.toml"
    short.write_text(text)
    with pytest.raises(ValueError, match="no truncation"):
        prepare(short, root / "wp03", inputs[2], tmp_path / "bad")


def test_fixed_selection_balanced_by_task_source_and_never_train(inputs):
    codec = ByteBPETokenizer.load(inputs[2])
    rows = Corpus(inputs[1], "validation", codec).examples
    selected = select_validation(rows, 4, 42)
    assert Counter(r["task"] for r in selected) == {"copy": 4, "single": 4, "multi": 4}
    assert Counter((r["task"], r["origin"]) for r in selected) == {
        ("copy", "authored"): 4,
        ("single", "authored"): 2,
        ("single", "wp03"): 2,
        ("multi", "authored"): 2,
        ("multi", "wp03"): 2,
    }
    altered = [r | {"final_answer": "999"} for r in rows[::-1]]
    assert [r["question_id"] for r in select_validation(altered, 4, 42)] == [
        r["question_id"] for r in selected
    ]
    assert all(set(q) == QUESTION_KEYS for q in questions_only(selected))
    with pytest.raises(ValueError):
        select_validation([r | {"split": "train"} for r in rows], 4, 42)


def test_copy_pair_and_numeric_scoring_different_protocols_invalid_counts_wrong(inputs):
    job = job_from(inputs)
    codec = job.train.codec
    predictions = [
        {
            "question_id": r["question_id"],
            "generated_ids": response(codec, r["final_answer"]),
            "finish_reason": "eos",
        }
        for r in job.selection
    ]
    _, summary = score(predictions, job.selection, codec)
    assert summary["macro_exact_match"] == 1
    predictions[0]["generated_ids"] = codec.encode(job.selection[0]["final_answer"])
    _, summary = score(predictions, job.selection, codec)
    assert summary["tasks"]["copy"]["correct"] == 3 and summary["tasks"]["copy"]["questions"] == 4
    with pytest.raises(ValueError):
        score(predictions[:-1], job.selection, codec)


def test_best_uses_generated_macro_not_loss_and_ties_keep_earliest(inputs, monkeypatch):
    job = job_from(inputs)
    correct_tasks = set()

    def generated(model, codec, selection, settings, emit):
        return [
            {
                "question_id": r["question_id"],
                "generated_ids": response(codec, r["final_answer"])
                if r["task"] in correct_tasks
                else [],
                "finish_reason": "eos",
            }
            for r in selection
        ]

    monkeypatch.setattr(runner, "generate_validation", generated)
    monkeypatch.setattr(
        job.engine,
        "evaluate",
        lambda _: {"step": job.engine.step, "loss": 10 - job.engine.step, "loss_tokens": 10},
    )
    job.evaluate()
    assert job.best["step"] == 0
    job.update()
    correct_tasks.add("copy")
    job.evaluate()
    assert job.best["step"] == 1 and job.best["macro_exact_match"] == pytest.approx(1 / 3)
    correct_tasks.update(("single", "multi"))
    job.update()
    job.evaluate()
    job.update()
    job.evaluate()
    assert job.best["step"] == 2 and job.best["macro_exact_match"] == 1
    job.validate_state(job.state(), job.engine.attempts)


def test_generator_receives_question_only_rows(inputs, monkeypatch):
    job = job_from(inputs)
    seen = []

    def generate(model, codec, questions, *args, **kwargs):
        seen.extend(questions)
        assert all(set(q) == QUESTION_KEYS for q in questions)
        return [
            {
                "question_id": q["question_id"],
                "generated_ids": [],
                "finish_reason": "max_new_tokens",
            }
            for q in questions
        ]

    monkeypatch.setattr(evaluation, "generate_questions", generate)
    job.evaluate()
    assert len(seen) == 12


@pytest.mark.parametrize("mutation", ["recipe", "cursor", "history", "best", "best_weights"])
def test_resume_rejects_changed_contract_or_selection_before_weights(inputs, mutation):
    job = job_from(inputs)
    job.evaluate()
    state = copy.deepcopy(job.state())
    if mutation == "recipe":
        state["contract"]["recipe"]["phases"][0]["end_step"] = 99
    elif mutation == "cursor":
        state["cursors"][0]["windows_consumed"] = 1
    elif mutation == "history":
        state["extra_state"]["history"]["validation"][0]["loss"] = 999
    elif mutation == "best":
        state["best"]["step"] = 9
    else:
        state["best_weights"]["final_norm.weight"][0] += 1
    before = copy.deepcopy(job.engine.model.state_dict())
    with pytest.raises(ValueError):
        job.validate_state(state, 0)
    assert _same(before, job.engine.model.state_dict())


def test_boundary_resume_next_stage_keeps_global_optimizer_and_cursor(inputs, tmp_path):
    job = job_from(inputs)
    job.evaluate()
    job.update()
    job.update()
    assert job.phase() == 1 and [c.windows_consumed for c in job.cursors] == [2, 0, 0]
    path = tmp_path / "boundary.pt"
    job.save(path)
    restored = job_from(inputs)
    restored.load(path)
    assert restored.phase() == 1 and restored.engine.step == 2
    assert _same(job.engine.optimizer.state_dict(), restored.engine.optimizer.state_dict())
    assert restored.cursors[1].state_dict() == job.cursors[1].state_dict()


def test_bundle_allowlist_checksum_and_no_overwrite(inputs, tmp_path):
    output = tmp_path / "bundle.zip"
    info = create(inputs[1], inputs[2], output)
    extract(output, tmp_path / "unpack", info["sha256"])
    with pytest.raises(FileExistsError):
        extract(output, tmp_path / "unpack", info["sha256"])
    with pytest.raises(ValueError):
        extract(output, tmp_path / "wrong", "0" * 64)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(output) as source, zipfile.ZipFile(bad, "w") as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        target.writestr("../escape", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract(bad, tmp_path / "bad-output", file_sha256(bad))
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "old,new",
    [
        ("end_step = 400", "end_step = 0"),
        ('tasks = ["copy", "single"]', 'tasks = ["single"]'),
        ("per_task = 16", "per_task = 3"),
        ("validate_every = 200", "validate_every = true"),
    ],
)
def test_recipe_rejects_invalid_phases_or_validation(tmp_path, old, new):
    p = tmp_path / "bad.toml"
    p.write_text((ROOT / "configs/training/wp07_curriculum.toml").read_text().replace(old, new))
    with pytest.raises(ValueError):
        load_recipe(p)


def test_full_pipeline_dropout_resume_all_stages_selected_and_bundle(tmp_path):
    report = check(tmp_path / "smoke", ROOT)
    assert report["full_vs_resume_exact"] and report["selected_model_reloads"]
    assert report["all_stages"] == {"copy": 2, "single": 2, "multi": 2}
    path = Path(report["run_directory"]) / "checkpoints/selected_model.pt"
    codec = ByteBPETokenizer.load(tmp_path / "smoke/fixture/tokenizer")
    rng = torch.get_rng_state().clone()
    model, meta = load_selected(path, codec)
    assert torch.equal(rng, torch.get_rng_state())
    assert meta["selection"]["step"] == 0  # Equal zero quality: retain earliest, not final.
    assert not model.training
    with pytest.raises(ValueError, match="checksum"):
        load_selected(path, codec, expected_sha256="0" * 64)
