# GRAST

Repository cho nghiên cứu LLM from scratch theo Hướng C: reasoning và test-time compute.

**Trạng thái: WP01 skeleton + WP02 configuration/reproducibility.** Đã có config,
seed, run metadata/logs, checkpoint RNG và smoke test tái lập. Tokenizer, model,
training loop, inference và thuật toán stopping vẫn chưa implement.
Smoke chỉ dùng RNG và ma trận float32 nhỏ, không phải kết quả thực nghiệm model.

WP02 đã nghiệm thu CPU, RTX 4060 và Colab A100; xem [evidence](docs/wp02_validation.md).
Baseline được đánh dấu bằng tag `wp02-baseline`; chuẩn bị WP03 trên branch riêng.
Chi tiết chuyển tiếp: [WP03 handoff](docs/wp03_handoff.md).

Nguồn kế hoạch: [LLM From Scratch — Hướng C | Project HQ](https://app.notion.com/p/4093badafe55810db45bfcc26b766de9)
(bản cập nhật 31/10/2026; hạn đồ án 01/12/2026).

## Hai giai đoạn

1. **Đồ án:** *When Does Confidence-Gated Test-Time Scaling Fail? A Study of
   Reliability and Compute Efficiency in Small Language Models.* Nghiên cứu confidence
   theo độ khó/checkpoint, fixed-K và stopping, calibration, token cost và online KV profiling.
2. **NCKH, sau 01/12/2026:** *Learning Memory-Constrained Adaptive Test-Time Sampling
   Policies under Shared GPU Budgets.* Chỉ mở khi workload đồng thời thật sự tạo memory
   contention; RL không nằm trong critical path đồ án và chưa có implementation trong repo.

Mục tiêu sau này là tự viết tokenizer, decoder-only Transformer, attention, generation
và KV cache bằng PyTorch. Model khoảng 30–60M là mục tiêu cần feasibility gate,
chưa phải cấu hình đã được kiểm chứng. Không dùng `AutoModelForCausalLM` làm kiến trúc chính.

## Bắt đầu

Cài [uv](https://docs.astral.sh/uv/getting-started/installation/), rồi chạy trong repository:

```powershell
uv sync --locked --extra cpu
uv run --no-sync grast smoke --config configs/experiments/local_smoke.toml
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
uv build
```

Python mặc định là 3.11 trong `.python-version`. `uv.lock` cố định dependency đã resolve.
Package phân phối tên `grast`, import bằng `llm_scratch`. Extra `cpu` cài PyTorch 2.9.1
CPU; dùng `uv sync --locked --extra cuda` cho CUDA 12.8, rồi chọn config `local_cuda.toml`.
Hai extra loại trừ nhau. Dùng `--no-sync` khi chạy để giữ đúng runtime đã chọn.
Colab dùng PyTorch có sẵn và ghi phiên bản thực tế, xem [WP02](docs/reproducibility.md).

## Cấu trúc

```text
.github/                 CI, Dependabot cho Actions, issue và PR templates
configs/                 model, tokenizer, pretrain, sft, sampling, experiments (TOML)
src/llm_scratch/
  data/                  tokenizer, datasets, collator, difficulty
  model/                 config, layers, attention, transformer, cache
  training/              trainer, checkpoint, optim
  inference/             generate, sampling, profiling
  reasoning/             formatting, answer_extraction, voting, confidence, calibration, stopping
  research/              sample_bank, policy_replay, modal_analysis, error_analysis
  evaluation/            benchmarks, metrics, latency_vram
  utils/                 seed, logging
scripts/                 entry points dự kiến; chạy hiện tại sẽ báo skeleton và exit 1
tests/                   package, config, RNG/checkpoint và fresh-process reproducibility
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
