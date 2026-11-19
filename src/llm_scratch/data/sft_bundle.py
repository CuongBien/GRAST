"""Portable CoT data/tokenizer bundle; pretrained weights remain a separate input."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.sft import SFTCorpus
from llm_scratch.data.tokenizer import ByteBPETokenizer

FILES = {
    "sft_data/manifest.json",
    "sft_data/train.jsonl",
    "sft_data/validation.jsonl",
    "tokenizer/tokenizer_manifest.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.json",
}


def verify(root, tokenizer):
    codec = ByteBPETokenizer.load(tokenizer)
    train = SFTCorpus(root, "train", codec)
    SFTCorpus(root, "validation", codec)
    return train.meta


def create_bundle(root: Path, tokenizer: Path, output: Path):
    meta = verify(root, tokenizer)
    files = {
        name: (root if name.startswith("sft_data/") else tokenizer) / Path(name).name
        for name in sorted(FILES)
    }
    manifest = {
        "schema_version": 1,
        "scope": "cot_sft_train_validation_only",
        "dataset_fingerprint": meta["fingerprint"],
        "tokenizer_fingerprint": meta["tokenizer_fingerprint"],
        "files": {name: file_sha256(path) for name, path in files.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("bundle_manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
    return {"path": str(output), "sha256": file_sha256(output), "manifest": manifest}


def verify_directory(root: Path):
    manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("scope") != "cot_sft_train_validation_only"
        or (set(manifest.get("files", {})) != FILES)
    ):
        raise ValueError("Invalid CoT bundle manifest")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if (
            not path.is_relative_to(root.resolve())
            or path.is_symlink()
            or file_sha256(path) != digest
        ):
            raise ValueError(f"SFT bundle payload mismatch: {name}")
    meta = verify(root / "sft_data", root / "tokenizer")
    if manifest["dataset_fingerprint"] != meta["fingerprint"] or (
        manifest["tokenizer_fingerprint"] != meta["tokenizer_fingerprint"]
    ):
        raise ValueError("SFT bundle identity mismatch")
    return root


def extract_bundle(archive: Path, output: Path, *, expected_sha256: str):
    if file_sha256(archive) != expected_sha256:
        raise ValueError("SFT bundle SHA256 mismatch")
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        infos = bundle.infolist()
        if (
            len(set(names)) != len(names)
            or set(names) != FILES | {"bundle_manifest.json"}
            or sum(i.file_size for i in infos) > 1024**3
            or bundle.getinfo("bundle_manifest.json").file_size > 65536
            or any((i.external_attr >> 16) & 0o170000 == 0o120000 for i in infos)
        ):
            raise ValueError("Unsafe/duplicate/oversized SFT bundle entries")
        output.mkdir(parents=True, exist_ok=False)
        for name in sorted(FILES | {"bundle_manifest.json"}):
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(name) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
    return verify_directory(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_bundle(args.prepared_root, args.tokenizer, args.output)), flush=True)


if __name__ == "__main__":
    main()
