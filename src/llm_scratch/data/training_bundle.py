"""Portable, hash-verified prepared train/validation + frozen tokenizer bundles."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.pretraining import PackedCorpus
from llm_scratch.data.tokenizer import ByteBPETokenizer

FILES = {
    "pretraining_data/manifest.json",
    "pretraining_data/train.u32",
    "pretraining_data/validation.u32",
    "tokenizer/tokenizer_manifest.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.json",
}


def verify_inputs(root: Path, tokenizer: Path):
    train = PackedCorpus(root, "train")
    PackedCorpus(root, "validation")
    ByteBPETokenizer.load(tokenizer, expected_fingerprint=train.meta["tokenizer_fingerprint"])
    return train.meta


def create_bundle(root: Path, tokenizer: Path, output: Path) -> dict:
    meta = verify_inputs(root, tokenizer)
    files = {
        name: (root if name.startswith("pretraining_data/") else tokenizer) / Path(name).name
        for name in sorted(FILES)
    }
    manifest = {
        "schema_version": 1,
        "scope": "pretraining_train_validation_only",
        "dataset_fingerprint": meta["fingerprint"],
        "tokenizer_fingerprint": meta["tokenizer_fingerprint"],
        "block_size": meta["block_size"],
        "files": {name: file_sha256(path) for name, path in files.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    return {"path": str(output), "sha256": file_sha256(output), "manifest": manifest}


def extract_bundle(archive: Path, output: Path, *, expected_sha256: str | None = None) -> Path:
    if expected_sha256 is not None and file_sha256(archive) != expected_sha256:
        raise ValueError("Bundle SHA256 mismatch")
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        names = [entry.filename for entry in infos]
        if (
            len(set(names)) != len(names)
            or set(names) != FILES | {"bundle_manifest.json"}
            or any((i.external_attr >> 16) & 0o170000 == 0o120000 for i in infos)
            or sum(i.file_size for i in infos) > 2 * 1024**3
            or bundle.getinfo("bundle_manifest.json").file_size > 65536
        ):
            raise ValueError("Unexpected/duplicate/unsafe/oversized bundle entries")
        manifest = json.loads(bundle.read("bundle_manifest.json"))
        if (
            manifest.get("schema_version") != 1
            or manifest.get("scope") != "pretraining_train_validation_only"
            or set(manifest.get("files", {})) != FILES
        ):
            raise ValueError("Invalid training bundle manifest")
        output.mkdir(parents=True, exist_ok=False)
        # Only the fixed allowlist is extracted; no caller-controlled path from the ZIP.
        for name in sorted(FILES):
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination)
            if file_sha256(target) != manifest["files"][name]:
                raise ValueError(f"Bundle entry checksum mismatch: {name}")
        meta = verify_inputs(output / "pretraining_data", output / "tokenizer")
        if any(manifest[k] != meta[k] for k in ("tokenizer_fingerprint", "block_size")) or (
            manifest["dataset_fingerprint"] != meta["fingerprint"]
        ):
            raise ValueError("Bundle dataset metadata mismatch")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--prepared-root", type=Path, required=True)
    create.add_argument("--tokenizer", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    if args.command == "create":
        result = create_bundle(args.prepared_root, args.tokenizer, args.output)
    else:
        result = {
            "path": str(
                extract_bundle(
                    args.archive,
                    args.output,
                    expected_sha256=args.expected_sha256,
                )
            )
        }
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
