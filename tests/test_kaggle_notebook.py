"""Exercise the shipped notebook's real input/launch helpers without a Kaggle account."""

import ast
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest
import torch

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.pretraining import prepare_pretraining
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.data.training_bundle import FILES, create_bundle, extract_bundle, verify_inputs
from llm_scratch.research.data_acceptance import offline_sources

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/wp07_pretraining_kaggle.ipynb"


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def helpers():
    namespace = dict(
        Path=Path,
        sys=sys,
        hashlib=hashlib,
        json=json,
        tempfile=tempfile,
        file_sha256=file_sha256,
        FILES=FILES,
        extract_bundle=extract_bundle,
        verify_inputs=verify_inputs,
    )
    for cell in notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        parsed = ast.parse("".join(cell["source"]))
        selected = [
            node
            for node in parsed.body
            if isinstance(node, ast.FunctionDef)
            or (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Constant)
                and any(
                    isinstance(t, ast.Name)
                    and t.id
                    in {
                        "BUNDLE_NAME",
                        "BUNDLE_SHA256",
                        "MANIFEST_SHA256",
                        "DATASET_FINGERPRINT",
                    }
                    for t in node.targets
                )
            )
        ]
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace


@pytest.fixture
def bundle(tmp_path):
    source = offline_sources(tmp_path / "source")
    codec = tmp_path / "codec"
    ByteBPETokenizer(vocab_size=264).save(codec)
    packed = prepare_pretraining(source, codec, tmp_path / "packed", block_size=256)
    ns = helpers()
    input_base = tmp_path / "input"
    input_base.mkdir()
    archive = input_base / ns["BUNDLE_NAME"]
    report = create_bundle(packed, codec, archive)
    with zipfile.ZipFile(archive) as z:
        ns["MANIFEST_SHA256"] = hashlib.sha256(z.read("bundle_manifest.json")).hexdigest()
    ns["BUNDLE_SHA256"] = report["sha256"]
    ns["DATASET_FINGERPRINT"] = report["manifest"]["dataset_fingerprint"]
    return ns, archive, input_base


def test_all_cells_compile_and_background_execution_needs_no_colab_dialogs():
    source = ""
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            compile(code, cell["id"], "exec")
            source += code
    assert "google.colab" not in source and "files.upload" not in source
    assert "files.download" not in source and "drive.mount" not in source
    assert 'Path("/kaggle/input")' in source and 'Path("/kaggle/working/grast-wp07")' in source
    assert "Popen" in source and "stderr=subprocess.STDOUT" in source and "flush=True" in source
    assert "aa4be36b33ca025701421ef48715a22b43c4e8db" in source
    assert 'sys.path.insert(0, str(REPO / "src"))' in source


@pytest.mark.parametrize(
    "major,bf16,expected",
    [
        (6, False, "float16"),
        (7, True, "float16"),
        (8, True, "bfloat16"),
        (8, False, "float16"),
    ],
)
def test_native_precision_selection(major, bf16, expected):
    assert helpers()["select_precision"](major, bf16) == expected


def test_zip_input_discovery_and_verified_extraction(bundle):
    ns, archive, input_base = bundle
    root = ns["find_inputs"](input_base)
    assert root != archive.parent
    assert (
        verify_inputs(root / "pretraining_data", root / "tokenizer")["fingerprint"]
        == (ns["DATASET_FINGERPRINT"])
    )


def test_expanded_input_discovery_nested_kaggle_layout(bundle):
    ns, archive, input_base = bundle
    root = input_base / "datasets/owner/frozen-grast/version1"
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)  # Locally-created trusted fixture archive.
    archive.unlink()
    assert ns["find_inputs"](input_base) == root


@pytest.mark.parametrize("target", ["pretraining_data/train.u32", "tokenizer/vocab.json"])
def test_expanded_payload_corruption_is_rejected(bundle, target):
    ns, archive, input_base = bundle
    root = input_base / "dataset"
    with zipfile.ZipFile(archive) as z:
        z.extractall(root)
    (root / target).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ns["find_inputs"](input_base)


def test_wrong_archive_and_missing_input_fail_before_training(bundle, tmp_path):
    ns, archive, input_base = bundle
    archive.write_bytes(b"wrong file")
    with pytest.raises(ValueError, match="SHA256"):
        ns["find_inputs"](input_base)
    with pytest.raises(FileNotFoundError, match="Add Input"):
        ns["find_inputs"](tmp_path / "empty")


@pytest.mark.parametrize("smoke", [True, False])
def test_training_command_keeps_recipe_budget_and_supports_attached_checkpoint(tmp_path, smoke):
    ns = helpers()
    checkpoint = tmp_path / "attached/latest.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"fixture-path-only")
    command = ns["training_command"](
        ROOT, tmp_path, tmp_path / "output", "float16", smoke, str(checkpoint)
    )
    assert command[command.index("--precision") + 1] == "float16"
    assert command[command.index("--resume") + 1] == checkpoint
    assert command[command.index("--config") + 1] == ROOT / "configs/training/wp07_pretrain.toml"
    assert ("--stop-after" in command) == smoke
    if smoke:
        assert command[command.index("--stop-after") + 1] == "4"
    with pytest.raises(FileNotFoundError, match="Attach"):
        ns["training_command"](ROOT, tmp_path, tmp_path, "float16", smoke, "missing.pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA probe requires a GPU")
def test_notebook_gpu_probe_executes_real_forward_backward(tmp_path):
    code = next(
        "".join(c["source"])
        for c in notebook()["cells"]
        if c["cell_type"] == "code" and "probe_loss.backward()" in "".join(c["source"])
    )
    namespace = {"WORK_ROOT": tmp_path, "sys": sys, "REVISION": "fixture"}
    exec(compile(code, str(NOTEBOOK), "exec"), namespace)
    report = json.loads((tmp_path / "kaggle_environment.json").read_text(encoding="utf-8"))
    assert report["device"] == "cuda:0" and report["precision"] in ("bfloat16", "float16")
