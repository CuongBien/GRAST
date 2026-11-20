"""Kaggle weights discovery distinguishes absent input, renamed files and wrong hashes."""

import ast
import json
from pathlib import Path

import pytest

from llm_scratch.data.datasets import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def resolver():
    notebook = json.loads((ROOT / "notebooks/wp07_sft_kaggle.ipynb").read_text(encoding="utf-8"))
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "def resolve_pretrained_checkpoint" in "".join(c["source"])
    )
    function = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "resolve_pretrained_checkpoint"
    )
    namespace = {"Path": Path, "file_sha256": file_sha256}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "notebook-resolver", "exec"),
        namespace,
    )
    return namespace["resolve_pretrained_checkpoint"]


@pytest.mark.parametrize("summary_only", [False, True])
def test_missing_weights_explains_add_input_and_summary_is_not_weights(tmp_path, summary_only):
    if summary_only:
        (tmp_path / "pretraining-summary.zip").write_bytes(b"not-a-model-checkpoint")
    with pytest.raises(FileNotFoundError) as error:
        resolver()(tmp_path, "0" * 64)
    assert "No .pt checkpoint files" in str(error.value)
    assert "Add Input" in str(error.value)
    assert "do not contain model weights" in str(error.value)


def test_renamed_weights_match_only_the_verified_hash(tmp_path, capsys):
    (tmp_path / "final.pt").write_bytes(b"wrong-checkpoint")
    parent = tmp_path / "pretrained-2000-upload.pt"
    parent.write_bytes(b"verified-checkpoint-bytes")
    assert resolver()(tmp_path, file_sha256(parent)) == parent
    output = capsys.readouterr().out
    assert (
        "Checking pretraining checkpoint:" in output
        and "Verified pretraining checkpoint:" in output
    )
    assert file_sha256(tmp_path / "final.pt") in output


def test_wrong_checkpoint_is_a_hash_mismatch_not_a_missing_file(tmp_path, capsys):
    parent = tmp_path / "final.pt"
    parent.write_bytes(b"wrong-checkpoint")
    with pytest.raises(ValueError, match="Expected SHA256: " + "0" * 64):
        resolver()(tmp_path, "0" * 64)
    assert file_sha256(parent) in capsys.readouterr().out


def test_explicit_checkpoint_does_not_silently_fall_back(tmp_path):
    parent = tmp_path / "final.pt"
    parent.write_bytes(b"verified-checkpoint")
    wrong = tmp_path / "other.pt"
    wrong.write_bytes(b"wrong-checkpoint")
    with pytest.raises(ValueError):
        resolver()(tmp_path, file_sha256(parent), wrong)
    with pytest.raises(FileNotFoundError, match="path does not exist"):
        resolver()(tmp_path, file_sha256(parent), tmp_path / "missing.pt")
    assert resolver()(tmp_path, file_sha256(parent), parent) == parent
