# WP07.5 — Checkpoint capability và pilot ngày 20/11/2026

Pipeline checkpoint/text/pilot đã chạy đủ trên hai checkpoint thật. **Khả năng giải
toán của baseline hiện còn yếu:** SFT đúng 2/150 validation và 1/150 test, trong khi
format CoT trên test đúng 150/150. Loss teacher forcing thấp chưa chứng minh model
tự dùng đúng toán hạng hoặc tính được đáp án.

WP07 có đủ checkpoints, loss curves, text samples và pilot artifacts để nghiệm thu
chức năng. Feasibility reasoning cho nghiên cứu confidence chưa được thiết lập từ
baseline greedy này; không coi kết quả format là thành công về toán.

## Protocol và nguồn

Code/config được commit trước lần đo test thật tại
`6b1511a3d7100e7b181870f6f385e0f533b70a31`, nhánh
`feat/wp07-5-checkpoint-capability`. Git sạch khi run bắt đầu.
Protocol SHA256:
`9a1e3674eef80b86724cf63891ab97e0536efe02960fac3d05eef0535845b9ac`.
[Protocol/API](wp07_capability.md), [evidence JSON](../experiments/wp07_5_validation_20261120.json).

Hai model cùng architecture 46.148.096 parameters, context1024 và tokenizer WP04.
User train trên T4/FP16, Torch2.11.0+cu130. Evaluator reload weights trên RTX4060
Laptop, Torch2.9.1+cu128/CUDA12.8, FP32, deterministic, TF32off, 1CPU thread.
Đây là inference reload, không restore optimizer hoặc exact training resume.

Pilot WP03 giữ nguyên 150 validation +150 test, mỗi split50 câu mỗi mức. Greedy K=1,
256-token budget, batch8/left-padding, cùng template cho cả hai checkpoints. Không
đổi policy, budget, checkpoint hay cách chấm sau khi thấy kết quả test. Raw predictions
của mỗi checkpoint/split được ghi trước scorer-only reference read tương ứng. Primary EM dùng canonical answer trong
một cặp native final-token IDs; invalid/malformed tính sai. Tổng600 predictions.

## Kết quả

| Checkpoint | Validation EM | Test EM | Test valid final | Test complete format | Test EOS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pretrain2000 | 0/150 (0%) | 0/150 (0%) | 0% | 0% | 88,67% |
| SFT500 | 2/150 (1,33%) | 1/150 (0,67%) | 100% | 100% | 100% |

Pretraining chưa được dạy reasoning markers. Điểm0 dưới template SFT phản ánh cả
format compliance; không diễn giải thành phép đo mọi dạng năng lực toán của model gốc.

| Mức WP03 | SFT validation | SFT test | Mean generated tokens test |
| --- | ---: | ---: | ---: |
| Easy | 2/50 (4%) | 1/50 (2%) | 24,60 |
| Medium | 0/50 (0%) | 0/50 (0%) | 64,70 |
| Hard | 0/50 (0%) | 0/50 (0%) | 110,18 |

SFT không hết budget ở câu nào và đều kết thúc EOS. Validation có149/150 final/format
hợp lệ, test150/150. Lỗi chính ở phép đo này là nội dung đáp án, không phải truncation
hoặc parser bỏ qua đáp án đúng. Mỗi difficulty chỉ50 câu/split; không ngoại suy ra
toàn bộ miền toán hoặc GSM8K.

Pretrain sinh22.704/21.994 tokens ở validation/test; SFT10.027/9.974. Trung bình SFT
test66,49 tokens/câu. Đây là token counts của protocol, không phải speedup benchmark.

## Kiểm tra kết quả thấp

Tái tính teacher-forced loss trên toàn bộ300 SFT validation examples/18.971 targets
bằng FP16 local được **0,5430529416**, sát report T4 **0,5430620898** (delta~9,15e-6).
Không thực hiện optimizer update. Loss mô tả dự đoán token khi đã có prefix reasoning
tham chiếu; chuỗi tự sinh có thể sai từ toán hạng/bước đầu.

Sau primary run, kiểm tra ba câu easy validation đầu. Single/batch, cached/uncached
và FP16/FP32 đều sinh cùng chuỗi sai trên ba ca đó. Chẩn đoán này chỉ dùng validation,
giữ nguyên primary protocol/results, không kiểm chứng mọi hardware hoặc mọi câu.
Artifact: `validation_generation_audit.json`.

Ví dụ nhận `16 - 8` nhưng sinh `R1 = 20 - 20 = -11`, final `-11`; nhận `30 + 29`
nhưng sinh `R1 = 20 + 20 = 30`, final `30`. Model hoàn thành markup nhưng thay toán
hạng và tạo phép tính sai.

## Năm ca validation minh họa

Chọn sau scoring: hai ca easy sai đầu, hai medium sai đầu, một hard sai đầu theo thứ
tự file. Đây là minh họa, không phải sample đại diện để ước lượng tỷ lệ. Full question,
IDs/traces nằm trong `representative_validation_failures.json`.

| Ca | Gold | Model final | Điểm thấy trong trace |
| --- | ---: | ---: | --- |
| Easy:16−8 | 8 | −11 | Thay hai toán hạng thành20, rồi tính sai cả biểu thức tự viết |
| Easy:30+29 | 59 | 30 | Sinh20+20=30 |
| Medium:(13+19)−(1+17) | 14 | −13 | Lặp13+13=23 cho R1/R2, không theo definitions |
| Medium:(16×5)+(1+7) | 88 | 240 | Trace R3=220 nhưng final240, các bước trước cũng sai |
| Hard:(11−12)/(14×7)−(18+18) | −3529/98 | −131/13 | Thay nhiều toán hạng, không giữ rational của các bước |

Validation easy có answer30 xuất hiện17/50, −13 là11/50 và −11 là8/50. Đây là phân bố
output quan sát, không xác định được nguyên nhân huấn luyện. Chưa sửa model, tăng
budget, voting hoặc fit threshold để nâng điểm. Các bước cải thiện/selection tiếp
theo cần dựa trên validation, giữ test ngoài quá trình fit.

Đọc cả ba ca final EM đúng cũng cho thấy hạn chế: đề10+20 và24+6 ở validation,
đề8+22 ở test đều nhận trace **`R1 = 20 + 20 = 30`**, final30. Final trùng gold nhưng
toán hạng và phép tính trong trace sai. EM vẫn được giữ đúng định nghĩa đã khóa;
quan sát thủ công này không phải metric CoT tự động và không chứng minh reasoning đúng.

## Checkpoint reload và text

Checkpoint SHA, metadata, finite weights, tokenizer, tied head, history hashes đều
qua kiểm tra. Hash hai source files giữ nguyên sau evaluation/audit. Cached/uncached
check8tokens/story có token IDs chính xác; max absolute logit errors pretrain8,58e-6,
SFT5,72e-6, trong tolerance WP06.

Mỗi checkpoint có ba story continuations từ prefixes authored và seed/policy định
trước. Cả hai sinh được văn bản tiếng Anh kiểu truyện; đọc mẫu vẫn thấy đổi nhân vật/
đồ vật và mất mạch lạc. Pretrain chuyển box sang rock/dress, rồi apple/ladder; SFT có
chuyển Tom sang Anna. Đây là minh họa generation, không là quality benchmark, và
không tham gia pilot EM.

Chỉ có weights cuối pretrain2000 và SFT500 được cung cấp. Curves có đầy đủ history
hai stages; capability nội bộ các mốc50/100/250 chưa đo được.

## Kiểm thử và artifacts

**770 tests pass** local, gồm29 capability cases mới; lint/format/build pass. Wheel
cài riêng và chạy fixture pipeline ngoài source: pass. CI Windows/Ubuntu của
implementation cũng pass. Tests kiểm gold separation, IDs/hashes/strata, rational EM,
invalid-as-wrong denominator, cache/padding/replay, checkpoint/history/tied/RNG guards,
no-test-subset rule, full runner và export PNG/SVG/CSV. Figures đã kiểm trực quan.

Primary run:
`artifacts/runs/20261120T083644233178Z-wp07-5-checkpoint-capability-seed42-85ecc9db`.
`artifacts/` giữ report/protocol/metrics, stage-loss và EM figures, mỗi checkpoint có
raw/scored JSONL, text/history/inventory và failure cases.

Summary ZIP không chứa weights:
`dist/grast-wp07-5-capability-20261120-85ecc9db.zip` (931.788bytes), SHA256
`69592bcb55d9758c8c683acdf1fbfb6b27a7267bd2116b3ff80bef24fe117137`.
ZIP CRC đã kiểm. Evidence JSON khóa artifact hashes, lineage, runtime và audit.
Training evidence từ user T4 và evaluator local được phân biệt; chưa có
GSM8K/sampling/calibration result.
