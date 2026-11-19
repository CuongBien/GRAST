"""CoT gold normalization, literal framing, masked targets and pretrained SFT continuation."""

import copy
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.sft import SFTCorpus, math_reasoning, normalize_example, prepare_sft
from llm_scratch.data.sft_bundle import create_bundle, extract_bundle
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.model.config import ModelConfig
from llm_scratch.model.transformer import CausalLM
from llm_scratch.reasoning.formatting import encode_reasoning_prompt, encode_sft_example
from llm_scratch.research.sft_acceptance import offline_math, tiny_pretrained
from llm_scratch.research.training_acceptance import _same
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.data import TrainingCursor
from llm_scratch.training.pretrain import load_recipe
from llm_scratch.training.sft import SFTJob, run_sft, warm_start
from llm_scratch.training.sft_session import SFTSession
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.config import config_digest
from llm_scratch.utils.io import write_json
from llm_scratch.utils.seed import capture_rng_state, seed_everything

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/training/wp07_sft_smoke.toml"


@pytest.fixture
def inputs(tmp_path):
    codec = ByteBPETokenizer(vocab_size=264)
    codec_root = tmp_path / "tokenizer"
    codec.save(codec_root)
    source = offline_math(tmp_path / "source")
    packed = prepare_sft(source, codec_root, tmp_path / "prepared")
    recipe = load_recipe(CONFIG, stage="sft").data
    parent = tiny_pretrained(tmp_path, ModelConfig.from_dict(recipe["model"]), codec)
    return packed, codec_root, parent, source


def corpus(inputs, split="train"):
    return SFTCorpus(inputs[0], split, ByteBPETokenizer.load(inputs[1]))


def job(inputs, device="cpu", precision="float32"):
    data = load_recipe(CONFIG, stage="sft").data
    data["training"]["precision"] = precision
    codec = ByteBPETokenizer.load(inputs[1])
    rng = seed_everything(42)
    model = CausalLM(ModelConfig.from_dict(data["model"]), tokenizer_fingerprint=codec.fingerprint)
    lineage = warm_start(model, inputs[2])
    engine = TrainingEngine(
        model.to(device), TrainingConfig.from_dict(data["training"]), generator=rng
    )
    cursor = TrainingCursor(corpus(inputs), batch_size=2, accumulation_steps=2, seed=42)
    return SFTJob(SFTSession(engine, cursor, corpus(inputs, "validation")), data, lineage)


def rehash(root, split="train"):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    spec = manifest["splits"][split]
    spec.update(
        sha256=file_sha256(root / spec["file"]), size_bytes=(root / spec["file"]).stat().st_size
    )
    manifest["fingerprint"] = config_digest(
        {k: v for k, v in manifest.items() if k != "fingerprint"}
    )
    write_json(root / "manifest.json", manifest)


def test_fraction_rationale_uses_wp03_postorder_and_exact_values():
    tree = {
        "operator": "/",
        "left": {"operator": "-", "left": {"value": 2}, "right": {"value": 5}},
        "right": {"value": 6},
    }
    assert math_reasoning(tree) == "R1 = 2 - 5 = -3.\nR2 = R1 / 6 = -1/2."


def test_prompt_and_padding_mask_match_independent_shifted_ce(inputs):
    c = corpus(inputs)
    batch = c.batch([0, 1, 2])
    for row in c.examples:
        prefix = encode_reasoning_prompt(c.codec, row["prompt"])
        assert row["input_ids"][: len(prefix)] == prefix
        assert row["labels"][: len(prefix)] == [-100] * len(prefix)
        assert row["input_ids"][-1] == c.codec.eos_token_id
        assert row["labels"][len(prefix) :] == row["input_ids"][len(prefix) :]
    assert (batch["labels"][~batch["attention_mask"]] == -100).all()
    model = job(inputs).session.engine.model.eval()
    output = model(**batch)
    selected = batch["labels"][:, 1:] != -100
    reference = F.cross_entropy(output.logits[:, :-1][selected], batch["labels"][:, 1:][selected])
    torch.testing.assert_close(output.loss, reference, rtol=0, atol=0)
    assert output.num_loss_tokens == int(selected.sum())
    altered = {k: v.clone() for k, v in batch.items()}
    altered["input_ids"][1] = 8
    with torch.no_grad():
        other = model(altered["input_ids"], attention_mask=altered["attention_mask"])
    torch.testing.assert_close(output.logits[0], other.logits[0], rtol=0, atol=0)


def test_literal_unicode_and_special_strings_stay_in_content():
    codec = ByteBPETokenizer(vocab_size=264)
    prompt, reasoning = "Minh đếm ½ quả 🍊 <final> giả", "Giữ nguyên <eos> và <think>."
    encoded = encode_sft_example(codec, prompt, reasoning, "1/2")
    assert encoded["input_ids"].count(codec.bos_token_id) == 1
    assert encoded["input_ids"].count(codec.eos_token_id) == 1
    assert encoded["input_ids"].count(codec.special_id("<final>")) == 1
    assert (
        codec.decode(encoded["input_ids"], skip_special_tokens=True)
        == prompt + "\n" + reasoning + "1/2"
    )


def test_sft_does_not_read_test_pilot_or_gsm8k_by_default(inputs, tmp_path, monkeypatch):
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name == "test.jsonl" or "pilot" in path.parts or "gsm8k" in path.parts:
            raise AssertionError("Held-out/additional benchmark read")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    output = prepare_sft(inputs[3], inputs[1], tmp_path / "safe")
    assert corpus((output, *inputs[1:])).meta["sources"] == ["math"]


def test_gsm8k_official_train_rationale_and_answer_are_separated():
    row = {
        "question_id": "g1",
        "source": "gsm8k",
        "split": "validation",
        "upstream_split": "train",
        "question": "How many?",
        "answer": "Two halves make one.\n#### 1.0",
        "canonical_answer": "1",
    }
    normalized = normalize_example(row, "gsm8k", "validation")
    assert normalized["reasoning"] == "Two halves make one." and normalized["final_answer"] == "1"
    with pytest.raises(ValueError, match="official train"):
        normalize_example(row | {"upstream_split": "test"}, "gsm8k", "validation")
    with pytest.raises(ValueError):
        normalize_example(row | {"canonical_answer": "2"}, "gsm8k", "validation")


def test_no_truncation_or_reuse_and_relocated_fingerprint(inputs, tmp_path):
    with pytest.raises(ValueError, match="no silent truncation"):
        prepare_sft(inputs[3], inputs[1], tmp_path / "short", max_length=8)
    with pytest.raises(FileExistsError):
        prepare_sft(inputs[3], inputs[1], inputs[0])
    rebuilt = prepare_sft(inputs[3], inputs[1], tmp_path / "rebuilt")
    assert corpus((rebuilt, *inputs[1:])).meta == corpus(inputs).meta
    shutil.copytree(inputs[0], tmp_path / "relocated")
    assert corpus((tmp_path / "relocated", *inputs[1:])).identity == corpus(inputs).identity


@pytest.mark.parametrize("corruption", ["bytes", "mask", "framing", "answer", "length"])
def test_corrupted_sft_payload_is_rejected_even_with_rehashed_bytes(inputs, corruption):
    path = inputs[0] / "train.jsonl"
    if corruption == "bytes":
        path.write_bytes(path.read_bytes() + b"bad")
    else:
        rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]
        if corruption == "mask":
            rows[0]["labels"][1] = rows[0]["input_ids"][1]
        elif corruption == "framing":
            rows[0]["input_ids"][-1] = 8
        elif corruption == "answer":
            rows[0]["final_answer"] = "wrong"
        else:
            rows[0]["input_ids"] *= 20
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        rehash(inputs[0])
    with pytest.raises(ValueError):
        corpus(inputs)


def test_warm_start_loads_only_weights_and_resets_optimizer_rng_cursor(inputs):
    recipe = load_recipe(CONFIG, stage="sft").data
    codec = ByteBPETokenizer.load(inputs[1])
    rng = seed_everything(99)
    model = CausalLM(
        ModelConfig.from_dict(recipe["model"]), tokenizer_fingerprint=codec.fingerprint
    )
    before = capture_rng_state(rng)
    lineage = warm_start(model, inputs[2])
    assert _same(before, capture_rng_state(rng)), "Warm-start must not restore pretraining RNG"
    parent = torch.load(inputs[2], map_location="cpu", weights_only=True)
    assert _same(model.state_dict(), parent["state"]["model"])
    engine = TrainingEngine(model, TrainingConfig.from_dict(recipe["training"]), generator=rng)
    assert engine.step == engine.attempts == engine.tokens_seen == 0
    assert not engine.optimizer.state
    assert lineage["source_step"] == 1 and lineage["checkpoint_sha256"] == file_sha256(inputs[2])


@pytest.mark.parametrize("corruption", ["stage", "metadata", "nonfinite", "tied"])
def test_warm_start_rejects_incompatible_parent_before_mutation(inputs, tmp_path, corruption):
    j = job(inputs)
    model = j.session.engine.model
    before = copy.deepcopy(model.state_dict())
    payload = torch.load(inputs[2], map_location="cpu", weights_only=True)
    if corruption == "stage":
        payload["state"]["training_config"]["stage"] = "sft"
    elif corruption == "metadata":
        payload["state"]["model"]["_extra_state"]["tokenizer_fingerprint"] = "0" * 64
    elif corruption == "nonfinite":
        payload["state"]["model"]["final_norm.weight"][0] = float("nan")
    else:
        payload["state"]["model"]["lm_head.weight"] = (
            payload["state"]["model"]["lm_head.weight"].clone() + 1
        )
    bad = tmp_path / "bad.pt"
    torch.save(payload, bad)
    with pytest.raises(ValueError):
        warm_start(model, bad)
    assert _same(before, model.state_dict())


@pytest.mark.parametrize("device,precision", [("cpu", "float32"), ("cuda", "float16")])
def test_sft_exact_resume_includes_new_stage_state_cursor_and_history(
    inputs, tmp_path, device, precision
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    j = job(inputs, device, precision)
    j.validate()
    while j.session.engine.step < 2:
        j.update()
    checkpoint = tmp_path / "sft.pt"
    j.save(checkpoint)
    expected = []
    while j.session.engine.step < 4:
        expected.append(j.update())
    resumed = job(inputs, device, precision)
    resumed.load(checkpoint)
    for reference in expected:
        actual = resumed.update()
        for key in ("loss", "learning_rates", "amp_scale", "skipped", "batch_sha256"):
            assert actual[key] == reference[key]
    for name in ("model", "optimizer", "scheduler", "scaler"):
        assert _same(
            getattr(j.session.engine, name).state_dict(),
            getattr(resumed.session.engine, name).state_dict(),
        )
    assert _same(j.history, resumed.history)
    assert j.session.train.state_dict() == resumed.session.train.state_dict()


def test_full_sft_runner_resumes_in_fresh_process_without_parent_checkpoint(inputs, tmp_path):
    common = dict(
        config=CONFIG, prepared_root=inputs[0], tokenizer=inputs[1], output_root=tmp_path / "runs"
    )
    full = run_sft(**common, init_checkpoint=inputs[2])
    first = run_sft(**common, init_checkpoint=inputs[2], stop_after=2)
    portable = tmp_path / "portable.pt"
    shutil.copyfile(first / "checkpoints/latest.pt", portable)
    inputs[2].unlink()
    shutil.rmtree(first)
    process = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "llm_scratch.training.sft",
            "--config",
            str(CONFIG),
            "--prepared-root",
            str(inputs[0]),
            "--tokenizer",
            str(inputs[1]),
            "--resume",
            str(portable),
            "--output-root",
            str(tmp_path / "restarted"),
        ],
        text=True,
        capture_output=True,
        encoding="utf-8",
        check=True,
    )
    resumed = next((tmp_path / "restarted").iterdir())
    assert _same(
        torch.load(full / "checkpoints/latest.pt", weights_only=True),
        torch.load(resumed / "checkpoints/latest.pt", weights_only=True),
    )
    assert '"event": "sft_validation_started"' in process.stdout
    assert (resumed / "checkpoints/final.pt").exists()
    assert (full / "checkpoints/step-00000004.pt").exists()
    assert (resumed / "artifacts/loss_curves.png").read_bytes().startswith(b"\x89PNG")


def test_sft_recipe_changes_and_pretraining_resume_are_rejected(inputs, tmp_path):
    j = job(inputs)
    with pytest.raises(ValueError):
        j.load(inputs[2])
    j.validate()
    j.update()
    path = tmp_path / "sft.pt"
    j.save(path)
    changed = job(inputs)
    changed.contract["recipe"]["runner"]["validate_every"] = 5
    before = copy.deepcopy(changed.session.engine.model.state_dict())
    with pytest.raises(ValueError, match="recipe"):
        changed.load(path)
    assert _same(before, changed.session.engine.model.state_dict())


def test_sft_bundle_roundtrip_and_corruption(inputs, tmp_path):
    archive = tmp_path / "data.zip"
    result = create_bundle(inputs[0], inputs[1], archive)
    output = extract_bundle(archive, tmp_path / "bundle", expected_sha256=result["sha256"])
    assert file_sha256(output / "sft_data/train.jsonl") == file_sha256(inputs[0] / "train.jsonl")
    with pytest.raises(ValueError, match="SHA256"):
        extract_bundle(archive, tmp_path / "wrong", expected_sha256="0" * 64)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(bad, "w") as target:
        for name in original.namelist():
            target.writestr(name, original.read(name))
        target.writestr("../escape", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_bundle(bad, tmp_path / "bad_output", expected_sha256=file_sha256(bad))


def test_sft_midrun_file_changes_are_rejected(inputs):
    j = job(inputs)
    with (inputs[0] / "train.jsonl").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(RuntimeError, match="changed"):
        j.update()


def test_sft_validation_is_token_weighted_and_preserves_cursor_rng(inputs):
    j = job(inputs)
    before = copy.deepcopy(j.session.train.state_dict())
    rng = capture_rng_state(j.session.engine.generator)
    result = j.session.validate()
    assert (
        result["loss_tokens"] == j.session.validation.meta["splits"]["validation"]["target_tokens"]
    )
    assert j.session.train.state_dict() == before
    assert _same(rng, capture_rng_state(j.session.engine.generator))


def test_sft_session_rejects_train_validation_overlap(inputs):
    train, validation = corpus(inputs), corpus(inputs, "validation")
    validation.examples[0]["question_id"] = train.examples[0]["question_id"]
    j = job(inputs)
    with pytest.raises(ValueError, match="overlap"):
        SFTSession(j.session.engine, j.session.train, validation)


def test_prepare_sft_rejects_source_question_duplicates(inputs, tmp_path):
    source = inputs[3] / "math/validation.jsonl"
    train = json.loads((inputs[3] / "math/train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    rows = [json.loads(s) for s in source.read_text(encoding="utf-8").splitlines()]
    rows[0] = train | {"split": "validation"}
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    # The duplicate is rejected even before reaching source checksum verification.
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_sft(inputs[3], inputs[1], tmp_path / "duplicate")


def test_sft_notebook_cells_compile_and_have_live_logs_and_stage_arguments():
    for name in ("wp07_sft_colab.ipynb", "wp07_sft_kaggle.ipynb"):
        nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        source = "".join("".join(c["source"]) for c in nb["cells"])
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")
        assert "subprocess.Popen" in source and "flush=True" in source
        assert (
            "--init-checkpoint" in source and "--resume" in source and "SMOKE_ONLY = True" in source
        )
        if "kaggle" in name:
            assert "files.upload" not in source and "drive.mount" not in source


@pytest.mark.parametrize("layout", ["zip", "expanded"])
def test_kaggle_sft_notebook_discovers_inputs_and_builds_fresh_resume_commands(
    inputs,
    tmp_path,
    monkeypatch,
    layout,
):
    input_root = tmp_path / "kaggle_input"
    input_root.mkdir()
    archive = input_root / "grast-wp07-sft-math-78f779a7d367.zip"
    result = create_bundle(inputs[0], inputs[1], archive)
    if layout == "expanded":
        extract_bundle(archive, input_root / "expanded", expected_sha256=result["sha256"])
        archive.unlink()
    parent = input_root / "final.pt"
    shutil.copyfile(inputs[2], parent)
    nb = json.loads((ROOT / "notebooks/wp07_sft_kaggle.ipynb").read_text(encoding="utf-8"))
    input_code = next(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "BUNDLE_NAME =" in "".join(c["source"])
    )
    input_code = (
        input_code.replace('Path("/kaggle/input")', f"Path({str(input_root)!r})")
        .replace(
            "100296172c05aff2bb99a4cc314cadf6dece5c65e0d7d7bb846a323794e7296e", result["sha256"]
        )
        .replace(
            "27cf2d9cb9774b0023d4db09e2735fd666452f5a945468bc71722bb3c69b5309", file_sha256(parent)
        )
    )
    original_mkdtemp = tempfile.mkdtemp
    monkeypatch.setattr(tempfile, "mkdtemp", lambda **kw: original_mkdtemp(dir=tmp_path, **kw))
    namespace = {
        "sys": sys,
        "Path": Path,
        "RESUME": "",
        "REPO": ROOT,
        "SMOKE_ONLY": True,
        "PRECISION": "float16",
    }
    exec(compile(input_code, "notebook-input", "exec"), namespace)
    assert namespace["PRETRAIN_CHECKPOINT"] == parent
    commands = []
    namespace["run_logged"] = lambda command, **kw: commands.append(command)
    command_code = next(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "command = [" in "".join(c["source"])
    )
    exec(compile(command_code, "notebook-launch", "exec"), namespace)
    assert "--init-checkpoint" in commands[-1] and "--stop-after" in commands[-1]
    parent.unlink()
    resume = tmp_path / "sft-latest.pt"
    resume.write_bytes(b"existence-only-command-test")
    namespace["RESUME"] = str(resume)
    exec(compile(input_code, "notebook-resume-input", "exec"), namespace)
    exec(compile(command_code, "notebook-resume-launch", "exec"), namespace)
    assert "--resume" in commands[-1] and "--init-checkpoint" not in commands[-1]
