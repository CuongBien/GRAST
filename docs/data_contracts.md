# Data contracts dự kiến

Nguồn: [Project HQ — Research focus](https://app.notion.com/p/4093badafe55810db45bfcc26b766de9).
Đây là hợp đồng thiết kế cho bước tiếp theo; chưa có serializer hoặc validator.
Không có dữ liệu thực nghiệm trong repository.

## Sample bank — giai đoạn 1

Mỗi record đại diện một candidate của một question tại một checkpoint và decoding config.

| Trường bắt buộc từ HQ | Ý nghĩa |
| --- | --- |
| `question_id` | ID ổn định để join benchmark và chia split |
| `difficulty` | Mức khó theo benchmark/protocol |
| `checkpoint` | Định danh checkpoint có thể truy vết |
| `sample_order` | Vị trí candidate trong thứ tự sampling |
| `seed` | Seed dùng để sinh candidate |
| `decoding_config` | Snapshot config decoding |
| `generated_text` | Văn bản model sinh |
| `extracted_answer` | Đáp án được trích, cần quy ước lỗi/thiếu ở bước triển khai |
| `is_correct` | Nhãn evaluation; tuyệt đối không làm policy input |
| `token_logprobs` | Log-probability theo generated token, khi có |
| `generated_tokens` | Số token sinh, theo protocol đếm cần chốt |

Gold answer và `is_correct` chỉ được dùng training/evaluation. Khi triển khai replay,
policy chỉ được thấy prefix candidates đã quan sát, không thấy kết quả tương lai hoặc gold label.
Định dạng lưu, dtype, null handling, indexing và versioning cần chốt trước khi ghi sample bank.
Manifest phải truy vết được split, config, code commit, checkpoint và seed.

## Online profiling — bảng riêng

Các trường HQ yêu cầu: `batch_size`, `concurrent_requests`, `context_length`, KV bytes,
peak VRAM, prefill/decode/total latency. Khi triển khai cần chuẩn hóa tên trường,
đơn vị bytes/milliseconds, thiết bị, dtype, warm-up và quy tắc GPU synchronization.

Replay offline chỉ cung cấp cost proxy như generated tokens và số candidates đã dùng.
Không được gán latency/VRAM thực đo từ replay. Kết quả online cần có protocol và log riêng.

## Evaluation boundaries

Calibration chỉ fit trên validation; test giữ held-out. Báo cáo pass@K/coverage@K
(ít nhất một candidate đúng) riêng với majority accuracy (đáp án được vote đúng).
Các định nghĩa tie, invalid answer và premature-stop phải được chốt khi triển khai metrics.
