# GRAST

Repository cho nghiên cứu LLM from scratch theo Hướng C: reasoning và test-time compute.

**Trạng thái: WP01+WP02 baseline; WP03 data và WP04 tokenizer đã triển khai.** Đã có config,
seed, metadata/logs, checkpoint RNG; dataset preparation, benchmark toán, answer extraction,
split/dedup, fixed pilot và byte BPE tự viết. Model hoàn chỉnh, training loop, inference và stopping chưa implement.
Smoke WP02 chỉ dùng RNG/ma trận nhỏ; WP03 chưa đánh giá chất lượng model.

WP02 đã nghiệm thu CPU, RTX 4060 và Colab A100; xem [evidence](docs/wp02_validation.md).
Baseline được đánh dấu bằng tag `wp02-baseline`; WP03 phát triển trên branch riêng.
Scope WP03: [handoff](docs/wp03_handoff.md); cách chạy: [WP03 data pipeline](docs/wp03_data.md).
WP04: [tokenizer và API chung](docs/wp04_tokenizer.md), [nghiệm thu](docs/wp04_validation.md).
WP05 đang chia thành 5 phase: [kế hoạch kiến trúc](docs/wp05_architecture.md).
WP05.1 có model config, embedding, RoPE và RMSNorm; WP05.2 có causal MHA và padding masks.
WP05.3 có SwiGLU và pre-norm decoder block với residual connections.
Full decoder/LM head/loss chưa triển khai. [Báo cáo WP05.3](docs/wp05_3_validation.md).

Nguồn kế hoạch: [LLM From Scratch — Hướng C | Project HQ](https://app.notion.com/p/4093badafe55810db45bfcc26b766de9)
(bản cập nhật 31/10/2026; hạn đồ án 01/12/2026).

## Hai giai đoạn

1. **Đồ án:** *When Does Confidence-Gated Test-Time Scaling Fail? A Study of
   Reliability and Compute Efficiency in Small Language Models.* Nghiên cứu confidence
   theo độ khó/checkpoint, fixed-K và stopping, calibration, token cost và online KV profiling.
2. **NCKH, sau 01/12/2026:** *Learning Memory-Constrained Adaptive Test-Time Sampling
   Policies under Shared GPU Budgets.* Chỉ mở khi workload đồng thời thật sự tạo memory
   contention; RL không nằm trong critical path đồ án và chưa có implementation trong repo.

Mục tiêu tiếp theo là hoàn thiện decoder-only Transformer, generation
và KV cache bằng PyTorch. Model khoảng 30–60M là mục tiêu cần feasibility gate,
chưa phải cấu hình đã được kiểm chứng. Không dùng `AutoModelForCausalLM` làm kiến trúc chính.

## Bắt đầu

Cài [uv](https://docs.astral.sh/uv/getting-started/installation/), rồi chạy trong repository:

```powershell
uv sync --locked --extra cpu --extra data
uv run --no-sync grast smoke --config configs/experiments/local_smoke.toml
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
uv build
```

Chuẩn bị dữ liệu thật và fixed pilot:

```powershell
uv run --no-sync grast prepare-data --config configs/data/wp03.toml
# Các lần sau có thể thêm --offline để dùng lại cache đã xác minh.
uv run --no-sync grast train-tokenizer --config configs/tokenizer/train.toml
```

Python mặc định là 3.11 trong `.python-version`. `uv.lock` cố định dependency đã resolve.
Package phân phối tên `grast`, import bằng `llm_scratch`. Extra `cpu` cài PyTorch 2.9.1
CPU; dùng `uv sync --locked --extra cuda --extra data` cho CUDA 12.8, rồi chọn config `local_cuda.toml`.
Hai extra loại trừ nhau. Dùng `--no-sync` khi chạy để giữ đúng runtime đã chọn.
Colab dùng PyTorch có sẵn và ghi phiên bản thực tế, xem [WP02](docs/reproducibility.md).

## Cấu trúc

```text
.github/                 CI, Dependabot cho Actions, issue và PR templates
configs/                 data/source lock, model, tokenizer, pretrain, sft, sampling, experiments
src/llm_scratch/
  data/                  tokenizer, datasets, collator, difficulty
  model/                 config, layers, attention, transformer, cache
  training/              trainer, checkpoint, optim
  inference/             generate, sampling, profiling
  reasoning/             formatting, answer_extraction, voting, confidence, calibration, stopping
  research/              sample_bank, policy_replay, modal_analysis, error_analysis
  evaluation/            benchmarks, metrics, latency_vram
  utils/                 seed, logging
scripts/                 prepare_data, train_tokenizer; model/training còn placeholder
tests/                   package, reproducibility, data/math, answer extraction, BPE và adapters
notebooks/               Colab A100 preflight và WP02 smoke; chưa chạy training
experiments/             manifest thực nghiệm; chưa có kết quả
artifacts/               runs, checkpoints, tokenizers, sample_bank, results (gitignore)
reports/                 figures, paper
docs/                    roadmap, data contracts, GitHub workflow
```

Xem [phạm vi và lộ trình](docs/roadmap.md), [data contracts](docs/data_contracts.md),
[GitHub workflow](docs/github_workflow.md), [WP02 reproducibility](docs/reproducibility.md)
và [hướng dẫn đóng góp](CONTRIBUTING.md).

## Quy tắc nghiên cứu

- Gold label chỉ dùng training/evaluation, không làm policy input.
- Calibration fit trên validation, báo cáo trên held-out test.
- Phân biệt pass@K/coverage@K với majority-vote accuracy.
- Replay offline không được trình bày là latency/VRAM thực đo.
- Không đánh dấu Done khi thiếu bằng chứng acceptance; không xem scaffold là kết quả nghiên cứu.
