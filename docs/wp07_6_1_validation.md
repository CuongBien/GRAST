# WP07.6.1 — Reasoning feasibility ngày 20/11/2026

Đã triển khai và chạy chẩn đoán copy/arithmetic cùng sampled coverage trên checkpoint
SFT500 thật. Coverage@16 có đáp án đúng ở5/30 câu validation, nhưng chỉ5/480 candidates
đúng và cả năm đều có phép tính sai trong reasoning. Chưa có cơ sở chuyển baseline
này thẳng sang nghiên cứu confidence với tuyên bố model reasoning đã đủ tốt.

## Nguồn và protocol

Implementation/config được commit trước lượt đo tại
`0b26d6817232d5baee566cc7b23f0dc4d29bfb52`, nhánh
`feat/wp07-6-1-reasoning-feasibility`. Git sạch khi run bắt đầu.
Protocol SHA256:
`0f9bae11c1cfffea68e6d5fb19b23c88455d4c44fb651f4253713699533be4aa`.
Source SFT500 SHA256:
`5f75909261c79e9f7f0c6e37a7e9d76ba5670be8b4be5133b6e148e0cd06ff42`.

Model46.148.096 parameters/tokenizer WP04/context1024 giữ nguyên. Local RTX4060
Laptop, PyTorch2.9.1+cu128/CUDA12.8, deterministic, TF32off, một CPU thread.
[Protocol/cách chạy](wp07_feasibility.md),
[evidence JSON](../experiments/wp07_6_1_validation_20261120.json).

Run: `artifacts/runs/20261120T145114388704Z-wp07-6-1-reasoning-feasibility-seed42-2d13c96a`.
Không đọc held-out test, không fit threshold, không đổi sampling sau khi nhìn kết quả.

## Coverage trên validation

30 câu WP03 validation, mỗi mức10 câu, chọn bằng hash seed/ID độc lập với gold.
Checkpoint SFT gốc sinh16 candidates/câu, FP32, B8, temperature0,8/top-k40/top-p0,95,
budget256. Raw predictions được lưu trước khi mở references. Tất cả480 mẫu có final
và format hợp lệ, kết thúc EOS; không mẫu nào hết budget. Tổng31.111 generated tokens.

| Mức | Câu có ≥1 final đúng /10 | Observed coverage@16 | Candidates đúng /160 |
| --- | ---: | ---: | ---: |
| Easy | 3/10 | 30% | 3/160 |
| Medium | 2/10 | 20% | 2/160 |
| Hard | 0/10 | 0% | 0/160 |
| Tổng | 5/30 | 16,67% | 5/480 (1,04%) |

Mỗi câu covered chỉ có đúng một candidate đúng trong16. Đây là observed coverage,
không phải majority accuracy hoặc ước lượng năng lực với ngân sách vô hạn. Greedy cũ
WP07.5 trên **đúng30 IDs này** đúng1/30; chỉ đọc lại artifact validation đã có, không
chạy thêm hoặc thay primary protocol. Không diễn giải30 câu thành kết quả toàn bộ
pilot hoặc held-out test, và không coi coverage16 là mức tăng accuracy của một policy
chọn được đáp án đúng mà không biết gold.

Kiểm tra sau scoring trên **toàn bộ năm candidate final đúng**:

| Đề / đáp án chuẩn | Trace tự sinh | Final |
| --- | --- | ---: |
| 18−21=−3 | `R1 = 11 - 11 = -3.` | −3 |
| 15−17=−2 | `R1 = 17 - 17 = -2.` | −2 |
| 28+14=42 | `R1 = 27 + 21 = 45.` | 42 |
| (13+19)−(1+17)=14 | `R1 = 18 + 13 = 23.`; các bước sau cũng không theo đề, R3=19 | 14 |
| (5+7)+(19−15)=16 | `R1 = 13 + 8 = 16.`; R3=19 | 16 |

Các first equations đều sai cả toán hạng so với đề lẫn phép tính tự viết. Primary
final EM vẫn giữ nguyên; đây là audit sau chấm, không phải metric kiểm chứng toàn bộ
CoT. Có thể lấy mẫu trúng final mà không có trace suy luận đúng.

## Overfit copy và arithmetic

Phép train chẩn đoán dùng64 authored train examples (32 copy,16 cộng,16 trừ),16
authored validation examples từ8 cặp toán hạng khác. Toán hạng40..99; không dùng
WP03 pilot làm training. Copy yêu cầu cặp số theo thứ tự trong final; arithmetic
giữ template R1 và numeric-v1. Hai nhiệm vụ được chấm riêng khi tự sinh greedy FP32.

Optimizer mới từ weights SFT500, BF16/B8/G8, LR3e-4,200 successful updates tối đa,
không weight decay, không resume source optimizer. Mỗi update dùng toàn bộ64 ví dụ
đúng một lần. Các mốc đánh giá cố định0/25/50/100/200; không kéo dài budget để đuổi
theo kết quả. Checkpoint diagnostic không thay baseline SFT500.

Đã chạy đủ200 updates/200 attempts, không skip;320.400 supervised target tokens.
Toàn bộ pipeline coverage + diagnostic mất638,34 giây (~10 phút38 giây), chưa gồm
audit replay sau run. Đây là lượt train diagnostic có giới hạn, không phải corpus SFT v2.

| Update | Train copy /32 | Train arithmetic /32 | Authored val copy /8 | Authored val arithmetic /8 | Train loss | Authored val loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 1 | 0 | 0 | 3,43560 | 3,34100 |
| 25 | 0 | 1 | 0 | 0 | 0,62897 | 0,70913 |
| 50 | 5 | 9 | 0 | 0 | 0,28771 | 0,75608 |
| 100 | 32 | 32 | 1 | 0 | 0,000372 | 0,97629 |
| 200 | 32 | 32 | 1 | 0 | 0,000121 | 0,99448 |

Train đạt100% đáp án tự sinh từ mốc100. Ở mốc200, cả32 arithmetic traces trong train
giữ đúng toán hạng/operator và tính đúng first equation. Trên8 arithmetic validation
mới, chỉ1 trace giữ đúng toán hạng/operator và0 first equations tính đúng; final EM0/8.
Copy validation1/8. Markup cuối cùng hợp lệ ở cả80 câu, nên các lỗi này không phải
chỉ do thiếu final markers. Loss ở bảng là tổng target scope copy+arithmetic của bộ
authored mới, không so trực tiếp với SFT target validation loss0,54306 trước đây.

Gate học thuộc64 mẫu đã đạt. Kết quả không thiết lập generalization; train loss gần0
trong khi authored validation loss tăng và generated accuracy vẫn thấp. Phép đo này
cho thấy model/training stack có thể ghi nhớ các mẫu copy và tính toán nhỏ, chưa xác
định nguyên nhân duy nhất khiến baseline SFT không áp dụng được sang đề khác.

## Quyết định tiếp theo

Ưu tiên WP07.6.2: chuẩn bị curriculum copy → cộng/trừ một bước → nhiều bước, với
train/validation chia theo cặp toán hạng và chọn checkpoint bằng generated validation
EM. Tăng độ phủ dữ liệu và kiểm tra conditioning trên câu hỏi trước khi chi thêm
ngân sách cho full WP08 sample bank/confidence. Đây là hướng tiếp theo từ kết quả
validation, chưa phải một recipe được chứng minh sẽ cải thiện model. Chưa train dài
hoặc triển khai curriculum đó trong WP07.6.1.

## Kiểm chứng và giới hạn

785 tests local passed, trong đó15 tests mới. Ruff lint/format và build wheel/sdist
qua; distribution không chứa cache/runs/weights. Installed-wheel CPU smoke chạy ngoài
source tree, tạo run/config/log và checkpoint mới, sinh/scoring đủ samples, một update
giới hạn; đây là fixture kỹ thuật, không phải bằng chứng chất lượng46M.
[CI implementation](https://github.com/CuongBien/GRAST/actions/runs/44757513512)
xanh trên Windows và Ubuntu.

GPU replay kiểm tra lại8 candidates trên câu easy đầu và8 candidates trên câu hard
đầu của selection: cùng seed/batch/policy cho **token IDs và finish reasons chính xác**.
Tái nạp standalone diagnostic checkpoint và sinh lại batch8 train đầu cho kết quả
chính xác với batch cuối run. Đây là spot-check16 sampled +8 diagnostic continuations,
không phải replay toàn bộ480 mẫu hoặc bảo đảm mọi hardware. Source SFT SHA giữ nguyên.

Artifacts gồm config/protocol/environment/Git/log, raw/scored480 candidates, selected
IDs và greedy comparison, five-correct-final trace audit, authored data, các mốc
generated evaluations, PNG/SVG và `overfit/diagnostic-final.pt`. ZIP tóm tắt
`dist/grast-wp07-6-1-feasibility-20261120-2d13c96a.zip` chứa dữ liệu chẩn đoán/log/curves,
không chứa weights; SHA256 và CRC check nằm trong evidence JSON.

Không có corpus SFT v2, model lớn hơn, tokenizer mới, confidence threshold hoặc thay
test protocol trong phase này. Các quyết định tiếp theo phải dựa trên train/validation;
kết quả test WP07.5 vẫn là lần đo đã đóng băng.
