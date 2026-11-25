"""Committed, checksum-verified model-only sample banks and gold-free policy observations."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import re
import tomllib
from pathlib import Path

import torch

from llm_scratch.curriculum.evaluation import load_selected
from llm_scratch.data.datasets import file_sha256
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.evaluation.capability import left_pad
from llm_scratch.evaluation.checkpoints import load_for_evaluation
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import generate
from llm_scratch.model.config import ModelConfig
from llm_scratch.reasoning.formatting import encode_reasoning_prompt
from llm_scratch.research.bank_schema import (
    BATCHING_PROTOCOL,
    CANDIDATE_KEYS,
    LOGPROB_PROTOCOL,
    PAYLOAD_FILES,
    CandidateObservation,
    PolicyObservation,
    describe_output,
    digest,
    generation_seed,
    identifier,
    integer,
    validate_grid,
    validate_questions,
)
from llm_scratch.training.pretrain import atomic_text
from llm_scratch.utils.config import LoadedConfig, canonical_json, config_digest
from llm_scratch.utils.environment import inspect_environment, package_fingerprint
from llm_scratch.utils.logging import Run
from llm_scratch.utils.seed import seed_everything

MANIFEST_KEYS = {
    "schema_version",
    "work_package",
    "scope",
    "split",
    "tokenizer_fingerprint",
    "question_ids",
    "checkpoints",
    "trial_seeds",
    "candidates_per_question",
    "batch_size",
    "generation",
    "precision",
    "runtime",
    "implementation",
    "logprob_protocol",
    "batching_protocol",
    "records",
    "files",
    "fingerprint",
}
CHECKPOINT_KEYS = {"id", "kind", "sha256", "stage", "step", "model_config", "parameter_count"}
RUNTIME_KEYS = {
    "device",
    "torch",
    "cuda",
    "gpu_name",
    "deterministic",
    "num_threads",
    "matmul_tf32",
    "cudnn_tf32",
}


def emit(event, **fields):
    print(canonical_json({"event": event, **fields}), flush=True)


def _runtime(device):
    device = torch.device(device)
    return {
        "device": str(device),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "num_threads": torch.get_num_threads(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def _check_files(root):
    actual = set()
    for p in root.rglob("*"):
        if p.is_symlink() or not p.resolve().is_relative_to(root):
            raise ValueError("Symlinks/escaped paths are not bank payloads")
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if actual != PAYLOAD_FILES | {"manifest.json"}:
        raise ValueError("Unlisted bank files forbidden; store gold/scoring outside the bank")


def _check_checkpoint_specs(specs):
    if not isinstance(specs, list) or not specs:
        raise ValueError("Declare at least one checkpoint")
    seen = set()
    for cp in specs:
        if (
            not isinstance(cp, dict)
            or set(cp) != {"id", "kind", "path", "sha256"}
            or not identifier(cp["id"])
            or cp["id"] in seen
            or cp["kind"] not in ("engine", "curriculum_selected")
            or not digest(cp["sha256"])
            or not isinstance(cp["path"], str)
            or not cp["path"]
        ):
            raise ValueError("Invalid/duplicate checkpoint declaration")
        seen.add(cp["id"])


def _load(cp, codec):
    path = Path(cp["path"])
    if cp["kind"] == "engine":
        model, meta, _ = load_for_evaluation(path, codec, expected_sha256=cp["sha256"])
        stage, step = meta["stage"], meta["step"]
    else:
        model, meta = load_selected(path, codec, expected_sha256=cp["sha256"])
        stage, step = "curriculum_sft", meta["selection"]["step"]
    # Do not propagate selection scores/labels/history/paths from checkpoint metadata.
    descriptor = {
        "id": cp["id"],
        "kind": cp["kind"],
        "sha256": cp["sha256"],
        "stage": stage,
        "step": step,
        "model_config": model.config.to_dict(),
        "parameter_count": model.parameter_counts()["total"],
    }
    return model, descriptor


def build_bank(
    output,
    questions,
    codec,
    checkpoints,
    *,
    generation,
    trial_seeds,
    candidates_per_question,
    batch_size,
    device="cpu",
    precision="float32",
    log=emit,
):
    """No references/scoring callback/opaque metadata inputs are accepted here."""
    validate_questions(questions)
    validate_grid(candidates_per_question, batch_size, trial_seeds)
    _check_checkpoint_specs(checkpoints)
    if (
        not isinstance(generation, GenerationConfig)
        or not generation.do_sample
        or generation.max_new_tokens < 1
        or generation.eos_token_id != codec.eos_token_id
        or generation.pad_token_id != codec.pad_token_id
    ):
        raise ValueError("Sample banks require sampled generation with codec EOS/PAD and budget>=1")
    generation.validate_vocab(len(codec))
    if precision not in ("float32", "float16", "bfloat16") or (
        torch.device(device).type == "cpu" and precision == "float16"
    ):
        raise ValueError("Unsupported bank precision/device")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    codec.save(output / "tokenizer")
    atomic_text(output / "questions.jsonl", "".join(canonical_json(q) + "\n" for q in questions))
    descriptors = []
    records = 0
    with (output / "candidates.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for cp in checkpoints:
            log("bank_checkpoint_loading", checkpoint_id=cp["id"])
            model, descriptor = _load(cp, codec)
            model.to(device).eval()
            descriptors.append(descriptor)
            for q in questions:
                prefix = encode_reasoning_prompt(codec, q["question"])
                for trial in trial_seeds:
                    for start in range(0, candidates_per_question, batch_size):
                        count = min(batch_size, candidates_per_question - start)
                        ids, mask = left_pad(
                            [prefix] * count,
                            codec.pad_token_id,
                            model.token_embedding.weight.device,
                        )
                        draw = generation_seed(cp["sha256"], q["question_id"], trial, start)
                        log(
                            "bank_batch_started",
                            checkpoint_id=cp["id"],
                            question_id=q["question_id"],
                            trial_seed=trial,
                            candidate_start=start,
                            batch_size=count,
                        )

                        def progress(step, token_ids, active):
                            if (step + 1) % 32 == 0:
                                log(
                                    "bank_decode_progress",
                                    step=step + 1,
                                    active_rows=int(active.sum()),
                                )

                        with torch.autocast(
                            torch.device(device).type,
                            dtype=getattr(torch, precision),
                            enabled=precision != "float32",
                        ):
                            result = generate(
                                model,
                                ids,
                                mask,
                                config=generation,
                                seed=draw,
                                use_cache=True,
                                on_token=progress,
                                return_logprobs=True,
                            )
                        for row in range(count):
                            valid = result.generated_mask[row]
                            tokens = result.generated_ids[row][valid].tolist()
                            probs = result.generated_logprobs[row][valid].tolist()
                            record = {
                                "checkpoint_id": cp["id"],
                                "question_id": q["question_id"],
                                "trial_seed": trial,
                                "candidate_index": start + row,
                                "generation_seed": draw,
                                "batch_start": start,
                                "batch_size": count,
                                "batch_row": row,
                                "prompt_ids_sha256": config_digest(prefix),
                                "generated_ids": tokens,
                                "generated_logprobs": probs,
                                "finish_reason": result.finish_reasons[row],
                            }
                            record.update(
                                describe_output(tokens, probs, record["finish_reason"], codec)
                            )
                            stream.write(canonical_json(record) + "\n")
                            records += 1
                        stream.flush()
                        log(
                            "bank_batch_finished",
                            records_written=records,
                            checkpoint_id=cp["id"],
                            question_id=q["question_id"],
                            trial_seed=trial,
                            candidates_completed=start + count,
                        )
            if file_sha256(cp["path"]) != cp["sha256"]:
                raise RuntimeError("Checkpoint changed during bank generation")
            del model
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
    manifest = {
        "schema_version": 1,
        "work_package": "WP08.1",
        "scope": "model_outputs_only_no_gold",
        "split": "validation",
        "tokenizer_fingerprint": codec.fingerprint,
        "question_ids": [q["question_id"] for q in questions],
        "checkpoints": descriptors,
        "trial_seeds": trial_seeds,
        "candidates_per_question": candidates_per_question,
        "batch_size": batch_size,
        "generation": generation.to_dict(),
        "precision": precision,
        "runtime": _runtime(device),
        "implementation": package_fingerprint()["files"],
        "logprob_protocol": LOGPROB_PROTOCOL,
        "batching_protocol": BATCHING_PROTOCOL,
        "records": records,
        "files": {
            name: {
                "sha256": file_sha256(output / name),
                "size_bytes": (output / name).stat().st_size,
            }
            for name in sorted(PAYLOAD_FILES)
        },
    }
    manifest["fingerprint"] = config_digest(manifest)
    # Commit marker last. A partial/interrupted directory is never a readable bank.
    atomic_text(output / "manifest.json", canonical_json(manifest) + "\n")
    bank = SampleBank(output)
    log(
        "sample_bank_committed",
        bank_directory=str(output),
        fingerprint=bank.fingerprint,
        records=records,
    )
    return bank


class SampleBank:
    """Read and validate once; policy observations contain no gold or scorer outputs."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        m = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if (
            not isinstance(m, dict)
            or set(m) != MANIFEST_KEYS
            or type(m["schema_version"]) is not int
            or m["schema_version"] != 1
            or m["work_package"] != "WP08.1"
            or m["scope"] != "model_outputs_only_no_gold"
            or m["split"] != "validation"
            or m["logprob_protocol"] != LOGPROB_PROTOCOL
            or m["batching_protocol"] != BATCHING_PROTOCOL
            or config_digest({k: v for k, v in m.items() if k != "fingerprint"}) != m["fingerprint"]
        ):
            raise ValueError("Invalid bank schema/commit/fingerprint")
        validate_grid(m["candidates_per_question"], m["batch_size"], m["trial_seeds"])
        if set(m["files"]) != PAYLOAD_FILES or not digest(m["tokenizer_fingerprint"]):
            raise ValueError("Bank payload allowlist/tokenizer mismatch")
        _check_files(self.root)
        for name, spec in m["files"].items():
            p = self.root / name
            if (
                set(spec) != {"sha256", "size_bytes"}
                or not digest(spec["sha256"])
                or not integer(spec["size_bytes"])
                or p.stat().st_size != spec["size_bytes"]
                or file_sha256(p) != spec["sha256"]
            ):
                raise ValueError("Bank payload checksum/size mismatch")
        codec = ByteBPETokenizer.load(
            self.root / "tokenizer", expected_fingerprint=m["tokenizer_fingerprint"]
        )
        questions = [
            json.loads(s)
            for s in (self.root / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        validate_questions(questions)
        if [q["question_id"] for q in questions] != m["question_ids"]:
            raise ValueError("Bank question order mismatch")
        policy = GenerationConfig.from_dict(m["generation"])
        policy.validate_vocab(len(codec))
        if (
            not policy.do_sample
            or policy.max_new_tokens < 1
            or policy.eos_token_id != codec.eos_token_id
            or policy.pad_token_id != codec.pad_token_id
        ):
            raise ValueError("Bank generation policy mismatch")
        if (
            m["precision"] not in ("float32", "float16", "bfloat16")
            or not isinstance(m["runtime"], dict)
            or set(m["runtime"]) != RUNTIME_KEYS
        ):
            raise ValueError("Bank runtime allowlist mismatch")
        runtime = m["runtime"]
        if (
            not isinstance(runtime["device"], str)
            or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", runtime["device"])
            or not isinstance(runtime["torch"], str)
            or any(
                runtime[k] is not None and not isinstance(runtime[k], str)
                for k in ("cuda", "gpu_name")
            )
            or not integer(runtime["num_threads"], 1)
            or any(
                type(runtime[k]) is not bool for k in ("deterministic", "matmul_tf32", "cudnn_tf32")
            )
        ):
            raise ValueError("Invalid runtime scalar metadata")
        if (
            not isinstance(m["implementation"], dict)
            or not m["implementation"]
            or any(
                not re.fullmatch(r"[a-z0-9_/]+\.py", k) or not digest(v)
                for k, v in m["implementation"].items()
            )
        ):
            raise ValueError("Invalid implementation provenance")
        seen = set()
        for cp in m["checkpoints"]:
            if (
                set(cp) != CHECKPOINT_KEYS
                or not identifier(cp["id"])
                or cp["id"] in seen
                or not digest(cp["sha256"])
                or cp["kind"] not in ("engine", "curriculum_selected")
                or cp["stage"] not in ("pretrain", "sft", "curriculum_sft")
                or not integer(cp["step"])
                or not integer(cp["parameter_count"], 1)
            ):
                raise ValueError("Invalid bank checkpoint descriptor")
            architecture = ModelConfig.from_dict(cp["model_config"])
            if architecture.vocab_size != len(codec):
                raise ValueError("Bank model vocabulary mismatch")
            seen.add(cp["id"])
        if not seen:
            raise ValueError("Empty checkpoint inventory")
        rows = [
            json.loads(s)
            for s in (self.root / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        expected = len(questions) * len(seen) * len(m["trial_seeds"]) * m["candidates_per_question"]
        if not integer(m["records"], 1) or len(rows) != expected or m["records"] != expected:
            raise ValueError("Incomplete bank grid")
        index = 0
        groups = []
        for cp in m["checkpoints"]:
            for q in questions:
                prefix = encode_reasoning_prompt(codec, q["question"])
                if len(prefix) + policy.max_new_tokens > cp["model_config"]["context_length"]:
                    raise ValueError("Bank prefix/budget exceeds declared context")
                for trial in m["trial_seeds"]:
                    candidates = []
                    for k in range(m["candidates_per_question"]):
                        row = rows[index]
                        index += 1
                        start = (k // m["batch_size"]) * m["batch_size"]
                        size = min(m["batch_size"], m["candidates_per_question"] - start)
                        identities = {
                            "checkpoint_id": cp["id"],
                            "question_id": q["question_id"],
                            "trial_seed": trial,
                            "candidate_index": k,
                            "batch_start": start,
                            "batch_row": k - start,
                            "batch_size": size,
                            "generation_seed": generation_seed(
                                cp["sha256"], q["question_id"], trial, start
                            ),
                            "prompt_ids_sha256": config_digest(prefix),
                        }
                        if (
                            not isinstance(row, dict)
                            or set(row) != CANDIDATE_KEYS
                            or any(
                                row[key] != value or type(row[key]) is not type(value)
                                for key, value in identities.items()
                            )
                        ):
                            raise ValueError(
                                "Candidate identity/order/batch/seed mismatch or forbidden fields"
                            )
                        derived = describe_output(
                            row["generated_ids"],
                            row["generated_logprobs"],
                            row["finish_reason"],
                            codec,
                        )
                        if (
                            type(row["answer_valid"]) is not bool
                            or type(row["format_valid"]) is not bool
                            or not integer(row["token_count"], 1)
                            or any(
                                type(row[key]) not in (int, float)
                                for key in ("logprob_sum", "logprob_mean")
                            )
                            or any(row[key] != value for key, value in derived.items())
                        ):
                            raise ValueError(
                                "Candidate answer/length/logprob aggregates disagree with tokens"
                            )
                        if row["token_count"] > policy.max_new_tokens or (
                            row["finish_reason"] == "max_new_tokens"
                            and row["token_count"] != policy.max_new_tokens
                        ):
                            raise ValueError("Candidate length/finish budget mismatch")
                        candidates.append(
                            CandidateObservation(
                                **{
                                    k: (
                                        tuple(row[k])
                                        if k in ("generated_ids", "generated_logprobs")
                                        else row[k]
                                    )
                                    for k in CandidateObservation.__dataclass_fields__
                                }
                            )
                        )
                    groups.append(
                        PolicyObservation(
                            cp["id"],
                            q["question_id"],
                            q["question"],
                            q["difficulty"],
                            trial,
                            tuple(candidates),
                        )
                    )
        self._manifest = m
        self._groups = tuple(groups)
        self._stamps = {
            p: (p.stat().st_size, p.stat().st_mtime_ns)
            for p in (self.root / name for name in PAYLOAD_FILES | {"manifest.json"})
        }

    def assert_unchanged(self):
        _check_files(self.root)
        if any((p.stat().st_size, p.stat().st_mtime_ns) != s for p, s in self._stamps.items()):
            raise RuntimeError("Bank files changed after verification")

    @property
    def fingerprint(self):
        self.assert_unchanged()
        return self._manifest["fingerprint"]

    @property
    def manifest(self):
        self.assert_unchanged()
        return copy.deepcopy(self._manifest)

    def observations(self):
        self.assert_unchanged()
        return self._groups


def load_config(path):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    d = tomllib.loads(source)
    fields = {
        "experiment": {"name"},
        "reproducibility": {"seed", "deterministic", "num_threads"},
        "artifacts": {"root"},
        "bank": {"candidates_per_question", "batch_size", "trial_seeds", "precision"},
    }
    if (
        set(d) != {"schema_version", "generation", "checkpoints", *fields}
        or type(d["schema_version"]) is not int
        or d["schema_version"] != 1
    ):
        raise ValueError("Invalid sample-bank config")
    for name, keys in fields.items():
        if not isinstance(d[name], dict) or set(d[name]) != keys:
            raise ValueError(f"Unknown/missing {name} config fields")
    if not identifier(d["experiment"]["name"]):
        raise ValueError("Invalid experiment name")
    r, b = d["reproducibility"], d["bank"]
    if (
        not integer(r["seed"], 0, 2**32 - 1)
        or type(r["deterministic"]) is not bool
        or not integer(r["num_threads"], 1)
    ):
        raise ValueError("Invalid seed/determinism")
    validate_grid(b["candidates_per_question"], b["batch_size"], b["trial_seeds"])
    if b["precision"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("Invalid bank precision")
    if set(d["generation"]) != {"max_new_tokens", "do_sample", "temperature", "top_k", "top_p"}:
        raise ValueError("Unknown generation fields")
    policy = GenerationConfig(**d["generation"])
    if not policy.do_sample or policy.max_new_tokens < 1:
        raise ValueError("Bank generation must sample with a positive token budget")
    _check_checkpoint_specs(d["checkpoints"])
    for cp in d["checkpoints"]:
        cp["path"] = str((path.parent / cp["path"]).resolve())
    if not isinstance(d["artifacts"]["root"], str) or not d["artifacts"]["root"]:
        raise ValueError("Invalid output root")
    d["artifacts"]["root"] = str((path.parent / d["artifacts"]["root"]).resolve())
    return LoadedConfig(d, {"sample_bank": source}, path.resolve())


def run_builder(config, questions, tokenizer, *, device="cpu", output_root=None):
    loaded = load_config(config)
    d = loaded.data
    q = [json.loads(s) for s in Path(questions).read_text(encoding="utf-8").splitlines()]
    codec = ByteBPETokenizer.load(tokenizer)
    seed_everything(**d["reproducibility"])
    env = inspect_environment(device)
    if not env["validation"]["ok"]:
        raise RuntimeError(env["validation"]["errors"])
    if output_root:
        d["artifacts"]["root"] = str(Path(output_root).resolve())
    d["execution"] = {
        "device": device,
        "questions_sha256": file_sha256(questions),
        "questions": str(Path(questions).resolve()),
        "tokenizer": str(Path(tokenizer).resolve()),
    }
    with Run(loaded) as active:

        def log(event, **fields):
            emit(event, **fields)
            active.log(event, **fields)

        policy = GenerationConfig(
            **d["generation"], eos_token_id=codec.eos_token_id, pad_token_id=codec.pad_token_id
        )
        bank = build_bank(
            active.path / "artifacts/bank",
            q,
            codec,
            d["checkpoints"],
            generation=policy,
            trial_seeds=d["bank"]["trial_seeds"],
            candidates_per_question=d["bank"]["candidates_per_question"],
            batch_size=d["bank"]["batch_size"],
            device=device,
            precision=d["bank"]["precision"],
            log=log,
        )
        atomic_text(
            active.path / "artifacts/bank_build_report.json",
            canonical_json(
                {
                    "work_package": "WP08.1",
                    "status": "completed",
                    "bank_fingerprint": bank.fingerprint,
                    "records": bank.manifest["records"],
                    "scope": "Foundation build only, not a multi-seed feasibility report",
                }
            )
            + "\n",
        )
    return active.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", type=Path, default=Path("configs/experiments/wp08_sample_bank_smoke.toml")
    )
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output-root", type=Path)
    a = p.parse_args()
    run = run_builder(
        a.config, a.questions, a.tokenizer, device=a.device, output_root=a.output_root
    )
    emit("bank_build_finished", run_directory=str(run))


if __name__ == "__main__":
    main()
