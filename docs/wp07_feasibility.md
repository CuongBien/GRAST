# WP07.6.1 — Operand copying, arithmetic and sampled coverage

WP07.5 measured low generated-answer accuracy despite low SFT validation loss.
This phase checks whether the existing 46M model can memorize a small explicit task,
and whether sampling its original SFT checkpoint exposes correct answers on validation.
It does not train a new corpus baseline or choose confidence thresholds.

## Fixed protocol

`configs/evaluation/wp07_feasibility.toml` declares the source SFT500 SHA256, seed42,
sampling settings, data sizes, update cap and evaluation schedule before execution.
The runner saves the selected question IDs and hashes of implementation files before
loading the model. Config, environment, Git metadata, source integrity and live JSON
logs accompany every run. The original weights are checked again after execution.

Coverage uses 30 WP03 **validation** pilot questions: 10 per difficulty, selected by
SHA256(seed, question_id), independently of answers or model output. Each receives
16 sampled continuations, temperature0.8/top-k40/top-p0.95, max256 new tokens, FP32.
Eight copies of one prompt form a generation batch. Each batch gets a deterministic
SHA256-derived seed; candidates use successive private-generator draws. Batch size,
row order and seeds are recorded because changing them can change samples.

The runner writes all model-only predictions before opening validation references.
Observed coverage@16 is the fraction of questions with at least one exact correct
native `<final>` answer among those 16 candidates. It is not majority accuracy or an
estimate of unlimited-sampling capability. Invalid answers remain in the denominator.
All 480 continuations, token IDs, finish reasons, final extraction, per-question answer
counts and level summaries are retained. No test file is read.

The overfit diagnostic starts separately from the same SFT weights, with a fresh
AdamW optimizer/scaler/scheduler/RNG and step0. It does not resume the source training
state. There are 32 authored operand pairs: 16 additions and16 subtractions, each
paired with a copy-only question, giving **64 train examples**. Eight other pairs
supply **16 authored validation examples**, with no pair overlap. Operands40..99
avoid overlap with WP03 one-operation easy tasks (operands1..30). This range and the
new copy prompt mean that a zero-shot copy failure also includes instruction/format
transfer; it alone does not establish an architectural copying limitation.

Copy gold is an ordered `a,b` pair in native final markers, scored with a strict
integer-pair parser rather than numeric-v1 (which is inappropriate for two operands).
Arithmetic final answers use the unchanged numeric-v1 primary scorer. A separate
strict first-equation parser counts correct source operands/operator and correct
arithmetic within the emitted equation. These counts are not a general CoT verifier.

Each update consumes all64 train examples once, shuffled reproducibly, B8/G8.
The fixed budget is200 successful updates, BF16, LR3e-4, no weight decay, cosine to
0.1×LR, clipping1.0; the bounded attempt limit handles possible skips. Evaluations at
0/25/50/100/200 record teacher-forced loss and **free-generated** greedy answers,
FP32, max80 tokens, on train and authored validation. No evaluation token is fed
back as a generated prefix. The budget is not extended if memorization fails.
The diagnostic checkpoint stays separate and is not promoted to the baseline.

## Running

Use the verified WP03 dataset directory and WP04 tokenizer directory:

```powershell
uv run --no-sync python -u scripts/check_reasoning_feasibility.py --device cuda --dataset-root PATH/TO/datasets --tokenizer PATH/TO/tokenizer
```

`--part coverage` or `--part overfit` executes one component; `--output-root` changes
the artifact destination. All paths are explicit; no checkpoint auto-discovery.
Coverage runs before training when using the default `--part all`.

For an offline CPU pipeline check, including fresh optimizer, one bounded update,
generated-answer scoring and saved checkpoint:

```powershell
uv run --no-sync python -u -m llm_scratch.research.feasibility_acceptance
```

This fixture uses a tiny model and six sampled continuations; its output is not
evidence about the trained 46M model's reasoning ability.

## Interpreting the gate

Perfect train EM only establishes memorization on64 examples. Authored held-out EM
is a small transfer probe; it does not replace WP03 validation. Coverage can reveal
that sampling recovers some correct answers while still leaving too few cases for a
confidence study. Record these outcomes before deciding whether WP07.6.2 should
prepare a copy/add/sub curriculum or whether WP08 can proceed with the current model.
The WP07.5 held-out test remains frozen and is not used to choose that recipe.
