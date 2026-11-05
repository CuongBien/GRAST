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

## WP03 — Dataset & Benchmark Design (chỉ chuẩn bị)

Scope chốt ngày 05/11/2026: ưu tiên TinyStories cho pretraining ban đầu; bộ toán có độ khó
điều chỉnh; answer normalization; train/validation/test và dedup; GSM8K bổ sung;
pilot cố định. Nghiệm thu cần manifest, script tạo/tải dữ liệu, answer-extraction tests
và fixed pilot. Không dùng test set chọn threshold.
Xem [WP03 handoff](wp03_handoff.md) để biết checklist. Chưa triển khai WP03.

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

Vocab, kích thước/context model, GPU budget, dataset và threshold/calibration cần kiểm chứng
ở task tương ứng. Các config draft/debug của WP02 chỉ cung cấp contract và default ban đầu.
