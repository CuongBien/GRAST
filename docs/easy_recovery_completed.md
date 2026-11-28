# Easy recovery — kết quả đã audit, 28/11/2026

Đã hoàn tất một vòng easy SFT đã khóa trước:1.000 updates trên Kaggle,
không mở thêm vòng train. Model inference được chọn ở **step900**, full checkpoint
`latest.pt` ở step1000. Đã kiểm tra checksum, source/data/tokenizer pins,
implementation hashes, checkpoint-owned history/optimizer/cursor, best-weight digest
và quy tắc chọn checkpoint bằng selection16, không dùng gate chọn winner.

| Hạng mục | Kết quả |
| --- | --- |
| Model | 46.148.096 parameters |
| GPU / precision | Tesla T4 / FP16; dùng cuda:0,2 GPU nhìn thấy, không DDP |
| Runtime | Python3.13.16, PyTorch2.11.0+cu130, CUDA13.0 |
| Successful updates / AMP skips | 1.000 /0 |
| Thời gian run theo lifecycle | Khoảng8 phút18,5 giây, gồm evaluation/checkpoint/gate |
| Supervised tokens đã học | 767.837 |
| Selection validation loss, step0→1000 | 0,097597→0,006419 |
| Checkpoint chọn | Step900:16/16 final và canonical trace đúng trên selection |
| Latest step1000 | 15/16 trên selection; không thay thế best900 |
| Gate cuối, baseline→selected | 6/48→45/48 (12,5%→93,75%), final và verified trùng counts |

Validation loss ở bảng chỉ trên16 câu selection. Gate là48 câu development validation
khác, chấm sau khi chọn checkpoint; không phải test accuracy.

| Nguồn gate | Baseline | Selected900 |
| --- | --- | --- |
| WP03 | 0/4 | 2/4 |
| Authored | 6/44 | 43/44 |
| Tổng | 6/48 | 45/48 |

Gate nghiêng authored; macro accuracy theo hai nguồn là73,86%, thấp hơn micro93,75%.
WP03 chỉ4 câu nên không đủ kết luận về toàn bộ WP03 hay reasoning nói chung.
Verified yêu cầu trace chuẩn sau chuẩn hóa whitespace, không phải judge reasoning tổng quát.

## Quyết định trước WP09

Capability screen **đạt** theo từng nguồn. Mixed-outcome screen **chưa đạt**:
còn3 câu sai, ít hơn mức15 đã khóa trước. Kết quả này yêu cầu điều chỉnh thiết kế
độ khó và tăng số nhóm đánh giá trước calibration; không phải lý do train thêm vòng SFT.
Không hạ screen để gọi benchmark hiện tại đủ, không tune threshold theo gate hoặc test.

Tiếp theo là **WP09.1: khóa benchmark/phân tầng độ khó và chia calibration/evaluation
theo question/group**. Dùng validation để làm bounded feasibility probe cho benchmark;
giữ nhóm/câu trong cùng partition qua mọi seed/checkpoint và tách phần dùng thiết kế benchmark
khỏi phần đánh giá sau cùng. Sau khi khóa protocol, thu sample bank mới cho selected900
để so sánh greedy, fixed-K, consensus và confidence policies trên cùng mẫu.
Gate48 hiện tại là evidence development, không tự chuyển thành bộ test độc lập.

## Artifacts và kiểm tra độc lập

Model được bố trí tại `artifacts/checkpoints/easy-recovery-1000-selected-900/`:

- `selected_model.pt`: inference export step900, SHA256
  `6b24c29a7be8b4700bbd92b6e1d87e88e64e670fe2378437b18dc97abd8d4a97`.
- `latest.pt`: full training checkpoint step1000; optimizer/scheduler/scaler/RNG/cursor/history
  đã kiểm tra cấu trúc và counters. Không chạy exact resume trên local vì runtime khác Kaggle.
- `evidence/`: ZIP gốc và26 file summary đã giải nén, gồm report, gate predictions,
  loss curves, metadata và logs.

Đã chấm lại stored predictions bằng tokenizer/references đã khóa. Đồng thời nạp model
và sinh lại greedy FP32 trên GPU local cho baseline và selected, xác nhận cùng gate metrics.
Đây là inference audit, không chạy thêm training và không đổi recipe/threshold.
Receipt đầy đủ: `experiments/easy_recovery_completed_20261128.json`;
raw local audit nằm tại `artifacts/easy-recovery-audit/20261128/`.
