"""Portable Colab kit: pinned wheel, recipe, data and tokenizer; weights stay separate."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.easy_recovery.data import Corpus, audit
from llm_scratch.easy_recovery.runner import load_recipe

FILES = {
    "data/manifest.json",
    "data/train.jsonl",
    "data/selection.jsonl",
    "data/gate.jsonl",
    "tokenizer/vocab.json",
    "tokenizer/merges.json",
    "tokenizer/tokenizer_manifest.json",
    "tokenizer/tokenizer_config.json",
    "easy_recovery.toml",
    "grast-0.1.0-py3-none-any.whl",
}


def verify(root):
    root = Path(root)
    m = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    if (
        m.get("schema_version") != 1
        or m.get("scope") != "easy-recovery-colab-v1"
        or set(m.get("files", {})) != FILES
    ):
        raise ValueError("Invalid recovery kit manifest")
    for name, digest in m["files"].items():
        p = root / name
        if p.is_symlink() or file_sha256(p) != digest:
            raise ValueError("Recovery kit payload mismatch")
    codec = ByteBPETokenizer.load(root / "tokenizer")
    parts = {k: Corpus(root / "data", k, codec) for k in ("train", "selection", "gate")}
    audit({k: c.unique_examples for k, c in parts.items()})
    recipe = load_recipe(root / "easy_recovery.toml").data
    if (
        parts["train"].meta["fingerprint"] != m["dataset_fingerprint"]
        or codec.fingerprint != m["tokenizer_fingerprint"]
        or recipe["initialization"]["sha256"] != m["source_checkpoint_sha256"]
        or recipe["data"]["block_size"] != parts["train"].meta["block_size"]
    ):
        raise ValueError("Recovery kit data/recipe identity mismatch")
    return root


def create(data, tokenizer, recipe, wheel, output, source_commit):
    if not isinstance(source_commit, str) or not re.fullmatch(r"[a-f0-9]{40}", source_commit):
        raise ValueError("Need the implementation Git commit for the portable wheel")
    data, tokenizer, recipe, wheel, output = map(Path, (data, tokenizer, recipe, wheel, output))
    paths = {
        name: data / Path(name).name
        if name.startswith("data/")
        else tokenizer / Path(name).name
        if name.startswith("tokenizer/")
        else recipe
        if name.endswith(".toml")
        else wheel
        for name in sorted(FILES)
    }
    codec = ByteBPETokenizer.load(tokenizer)
    corpus = Corpus(data, "train", codec)
    m = {
        "schema_version": 1,
        "scope": "easy-recovery-colab-v1",
        "source_commit": source_commit,
        "dataset_fingerprint": corpus.meta["fingerprint"],
        "tokenizer_fingerprint": codec.fingerprint,
        "source_checkpoint_sha256": load_recipe(recipe).data["initialization"]["sha256"],
        "files": {n: file_sha256(p) for n, p in paths.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, p in paths.items():
            z.write(p, name)
        z.writestr("bundle_manifest.json", json.dumps(m, sort_keys=True, indent=2))
    return {
        "path": str(output),
        "sha256": file_sha256(output),
        "size_bytes": output.stat().st_size,
        "dataset_fingerprint": corpus.meta["fingerprint"],
    }


def extract(archive, output, expected_sha256):
    archive, output = Path(archive), Path(output)
    if file_sha256(archive) != expected_sha256:
        raise ValueError("Recovery kit SHA256 mismatch")
    with zipfile.ZipFile(archive) as z:
        infos, names = z.infolist(), z.namelist()
        if (
            len(set(names)) != len(names)
            or set(names) != FILES | {"bundle_manifest.json"}
            or sum(i.file_size for i in infos) > 256 * 1024**2
            or z.getinfo("bundle_manifest.json").file_size > 65536
            or any((i.external_attr >> 16) & 0o170000 == 0o120000 for i in infos)
        ):
            raise ValueError("Unsafe/duplicate/oversized recovery kit")
        output.mkdir(parents=True, exist_ok=False)
        for name in sorted(names):
            p = output / name
            p.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, p.open("xb") as target:
                shutil.copyfileobj(source, target)
    return verify(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "tokenizer", "recipe", "wheel", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-commit", required=True)
    a = p.parse_args()
    print(
        json.dumps(create(a.data, a.tokenizer, a.recipe, a.wheel, a.output, a.source_commit)),
        flush=True,
    )


if __name__ == "__main__":
    main()
