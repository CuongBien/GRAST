# Phạm vi và lộ trình

Nguồn: [Project HQ](https://app.notion.com/p/4093badafe55810db45bfcc26b766de9),
research focus cập nhật 31/10/2026. Tài liệu này ánh xạ kế hoạch vào repository;
không thay đổi ngày hoặc trạng thái task trong Notion.

## WP01 — repository skeleton

- Dựng package `src/llm_scratch` và module placeholders theo HQ.
- Có pyproject, lockfile, Python pin, lint/format, smoke tests, build và CI.
- Có quy trình GitHub branch/PR/issue và hướng dẫn settings.
- Có thư mục configs, scripts, Colab scaffold, artifacts, experiments, reports.
- Ghi data contract và ranh giới hai giai đoạn.

## WP02 — configuration và reproducibility

Config model/tokenizer/training/generation/experiment, seed Python/NumPy/PyTorch,
metadata Git/config/seed/runtime, log, run directories và checkpoint RNG.
Smoke so sánh hai tiến trình mới và tiếp tục RNG từ checkpoint, không chạy model/training.
Các default trong config là draft/debug, không phải kết luận về feasibility hay hyperparameter.
Xem [WP02](reproducibility.md) để biết cách chạy và giới hạn nghiệm thu.
Các mục tiêu model/nghiên cứu bên dưới vẫn là công việc tương lai.

WP02 đã đạt nghiệm thu local và Colab A100 ngày 05/11/2026.
Evidence được kiểm tra với source fingerprint và lưu manifest trong repository;
tag `wp02-baseline` chốt WP01+WP02 trước các thay đổi WP03.

## WP03 — Dataset & Benchmark Design (đã triển khai)

Scope chốt ngày 05/11/2026: ưu tiên TinyStories cho pretraining ban đầu; bộ toán có độ khó
điều chỉnh; answer normalization; train/validation/test và dedup; GSM8K bổ sung;
pilot cố định. Nghiệm thu cần manifest, script tạo/tải dữ liệu, answer-extraction tests
và fixed pilot. Không dùng test set chọn threshold.
Đã có pipeline nguồn thật, numeric-v1, split/dedup và fixed pilot 300 câu.
Xem [WP03 data](wp03_data.md) và [validation](wp03_validation.md) để tái lập/kiểm tra.

## WP04 — Tokenizer from Scratch (đã triển khai)

Byte BPE tự viết, Unicode lossless, tám control tokens, vocab/merges/config/checksum,
API chung cho pretraining/SFT/inference và thống kê trên dữ liệu WP03.
Chỉ fit TinyStories train; held-out dùng kiểm tra codec cố định.
Xem [thiết kế và cách chạy](wp04_tokenizer.md), [nghiệm thu](wp04_validation.md).

## WP05 — Transformer Architecture (đã đạt nghiệm thu kiến trúc)

WP05.1 config/embedding/RoPE/RMSNorm → WP05.2 causal MHA/masks →
WP05.3 SwiGLU/decoder block → WP05.4 decoder/LM head/loss/parameter count →
WP05.5 overfit và causal-isolation của model hoàn chỉnh.
Mỗi phase có test/báo cáo/commit riêng; chỉ chốt WP05 sau gate cuối.
Xem [kế hoạch và contract](wp05_architecture.md), [kết quả phase 1](wp05_1_validation.md)
và [kết quả phase 2](wp05_2_validation.md), [phase 3](wp05_3_validation.md), [phase 4](wp05_4_validation.md),
[phase 5 / gate cuối](wp05_5_validation.md). Đã overfit fixture cố định, kiểm tra loss,
gradients, causal/padding isolation và checkpoint continuation; chưa pretrain corpus.

## WP06 — KV Cache & Autoregressive Inference (đã đạt nghiệm thu)

WP06.1 cache storage/transactions → WP06.2 prefill/token decode/RoPE/masks →
WP06.3 greedy/temperature/top-k/top-p/EOS → WP06.4 cached/uncached equivalence →
WP06.5 online profiling TTFT/latency/throughput/VRAM tại L=128/512/1024/2048.
Từng phase có báo cáo/commit riêng; đã có equivalence và profiling online thực đo.
Xem [kế hoạch và contracts](wp06_inference.md), [phase 1](wp06_1_validation.md),
[phase 2](wp06_2_validation.md), [phase 3](wp06_3_validation.md),
[phase 4](wp06_4_validation.md). Đã có cached/uncached generation với greedy/temperature/
top-k/top-p/EOS, logits/tokens equivalence và RoPE/context guards.
[WP06.5](wp06_5_validation.md) lưu raw timing/VRAM và protocol RTX 4060 Laptop,
B1/N16/FP32, L128/512/1024/2048, explicit inference context 2064; baseline 1024 giữ nguyên.

## Giai đoạn 1 — đồ án, hạn 01/12/2026

*When Does Confidence-Gated Test-Time Scaling Fail? A Study of Reliability and
Compute Efficiency in Small Language Models.* Không tuyên bố confidence stopping là thuật toán mới.

| Mốc HQ | Công việc và gate dự kiến |
| --- | --- |
| W1: 30/10–05/11 | Proposal 2 trang, thiết kế, tokenizer, unit tests khởi tạo |
| W2: 06/11–12/11 | Decoder, pretrain baseline, generation, cached/uncached logits, profiling L=128/512/1024/2048 |
| W3: 13/11–19/11 | CoT SFT, feasibility gate, sample bank K=16, majority, confidence baseline |
| W4: 20/11–26/11 | Offline replay, calibration, modal saturation, accuracy–cost, ít nhất 5 ca lỗi, paper/demo |
| Buffer: 27/11–30/11 | Chạy lại, sửa lỗi, kiểm tra nộp |

Critical path: tokenizer → decoder/causal masking → pretraining → checkpoint → CoT SFT
→ K candidates → answer extraction → benchmark → paper/demo.

Baseline tối thiểu: greedy, fixed-K majority, consensus early stop, confidence threshold.
CGES-style nếu đủ thời gian. K dự kiến: 1, 2, 4, 8, 16. Benchmark toán phân mức khó
là chính; GSM8K có thể chỉ là stress test.

Metrics dự kiến: accuracy, pass@K/coverage@K, majority accuracy, ECE, Brier,
premature-stop, generated tokens và online latency/VRAM. Chỉ fit calibration trên validation.

Go/no-go: model giải được tỷ lệ bài toán đủ phân tích và có confidence signal đúng/sai
để kiểm định. Không mặc định model nhỏ đạt accuracy cao trên GSM8K.

## Giai đoạn 2 — NCKH, sau 01/12/2026

*Learning Memory-Constrained Adaptive Test-Time Sampling Policies under Shared GPU Budgets.*

Chỉ khởi động memory RL khi đo online xác nhận nhiều query thật sự tranh chấp VRAM.
Sau đó mới xét controller, action masking/admission control và baseline supervised/heuristic.
Không có RL/controller implementation hoặc dependency trong bước 1.

## Các quyết định còn mở

Kích thước/context model, GPU budget và threshold/calibration cần kiểm chứng
ở task tương ứng. Các config draft/debug của WP02 chỉ cung cấp contract và default ban đầu.
