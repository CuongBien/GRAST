"""Verified portable curriculum/tokenizer bundle; model weights are a separate input."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from llm_scratch.curriculum.data import Corpus, audit
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer

FILES = {
    "curriculum_data/manifest.json",
    "curriculum_data/train.jsonl",
    "curriculum_data/validation.jsonl",
    "tokenizer/vocab.json",
    "tokenizer/merges.json",
    "tokenizer/tokenizer_manifest.json",
    "tokenizer/tokenizer_config.json",
}


def verify(root):
    m = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    if (
        m.get("schema_version") != 1
        or m.get("scope") != "curriculum_train_validation_only"
        or set(m.get("files", {})) != FILES
    ):
        raise ValueError("Invalid curriculum bundle manifest")
    for name, digest in m["files"].items():
        path = root / name
        if path.is_symlink() or file_sha256(path) != digest:
            raise ValueError("Curriculum bundle payload mismatch")
    codec = ByteBPETokenizer.load(root / "tokenizer")
    train, validation = (
        Corpus(root / "curriculum_data", "train", codec),
        Corpus(root / "curriculum_data", "validation", codec),
    )
    audit(train.examples, validation.examples)
    if (
        train.meta["fingerprint"] != m["dataset_fingerprint"]
        or codec.fingerprint != m["tokenizer_fingerprint"]
    ):
        raise ValueError("Curriculum bundle identity mismatch")
    return root


def create(root, tokenizer, output):
    codec = ByteBPETokenizer.load(tokenizer)
    train, validation = Corpus(root, "train", codec), Corpus(root, "validation", codec)
    audit(train.examples, validation.examples)
    paths = {
        name: (root if name.startswith("curriculum_data/") else tokenizer) / Path(name).name
        for name in sorted(FILES)
    }
    manifest = {
        "schema_version": 1,
        "scope": "curriculum_train_validation_only",
        "dataset_fingerprint": train.meta["fingerprint"],
        "tokenizer_fingerprint": codec.fingerprint,
        "files": {name: file_sha256(p) for name, p in paths.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for name, path in paths.items():
            z.write(path, name)
        z.writestr("bundle_manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
    return {
        "path": str(output),
        "sha256": file_sha256(output),
        "dataset_fingerprint": train.meta["fingerprint"],
        "size_bytes": output.stat().st_size,
    }


def extract(archive, output, expected_sha256):
    if file_sha256(archive) != expected_sha256:
        raise ValueError("Curriculum archive SHA256 mismatch")
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        names = z.namelist()
        if (
            len(set(names)) != len(names)
            or set(names) != FILES | {"bundle_manifest.json"}
            or sum(i.file_size for i in infos) > 1024**3
            or z.getinfo("bundle_manifest.json").file_size > 65536
            or any((i.external_attr >> 16) & 0o170000 == 0o120000 for i in infos)
        ):
            raise ValueError("Unsafe/duplicate/oversized curriculum archive")
        output.mkdir(parents=True, exist_ok=False)
        for name in sorted(names):
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
    return verify(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    c = sub.add_parser("create")
    c.add_argument("--prepared-root", type=Path, required=True)
    c.add_argument("--tokenizer", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    e = sub.add_parser("extract")
    e.add_argument("--archive", type=Path, required=True)
    e.add_argument("--output", type=Path, required=True)
    e.add_argument("--sha256", required=True)
    a = p.parse_args()
    result = (
        create(a.prepared_root, a.tokenizer, a.output)
        if a.action == "create"
        else {"extracted_root": str(extract(a.archive, a.output, a.sha256))}
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
