# WP07.6.3 — Nhận curriculum model và đo validation ngày25/11/2026

Đã nhận, xác minh model/summary và chạy đánh giá. **Copy đã học tốt, tính toán vẫn yếu**:
additional validation đúng32/32 copy,5/32 single và0/32 multi; WP03 pilot validation
đúng5/150 so với2/150 của SFT500 trước curriculum. Không đánh đồng macro selection
39,58% với accuracy reasoning; macro đó gồm copy100%.

[Protocol/cách chạy](wp07_curriculum_evaluation.md),
[receipt training](../experiments/wp07_curriculum_completed_20261125.json),
[evidence evaluation](../experiments/wp07_6_3_validation_20261125.json).

## Lượt train đã nhận

User chạy đủ2.800 updates/2.800 attempts,0 AMP skips,1.315.393 supervised targets,
400/1.200/1.200 windows theo ba stage. Một TeslaT4 cuda:0, FP16, Torch2.11.0+cu130/
CUDA13.0/Python3.13.16. Hai T4 nhìn thấy trong environment không có nghĩa dùng DDP.
Events ghi thời gian31 phút46,83 giây; run ID dùng UTC24/11, kết thúc25/11 giờ Bangkok.

Validation loss2,15654674 →0,11452071; train loss trung bình100 updates cuối0,11712172.
Model được chọn ở **step2.200**, validation loss tại mốc đó0,13948403 và macro generated
EM19/48=39,58%: copy16/16, single2/16, multi1/16. Mốc2.400/2.800 hòa macro nhưng giữ
mốc2.200 theo quy tắc earliest tie. Không đổi lựa chọn sau khi đo thêm.

Summary archive SHA256:
`d6d3714d7e9ab1cc3abf2eefd86ab4351ce0734dfab3415226a20cae208ef5fc`.
Selected file SHA256:
`d2ea45abc8f81b516552c184d75c595b95fa2bd1abdc260538e497da4b627dc8`.
Weight-content digest:
`e0ac6861efd2be5c2db9bf84e3370bf2101999a96377e5544e0295779e5bb4b7`.

Weights finite/shape/tied head/vocab8192/tokenizer49187f… đã qua loader; selection và
contract khớp summary. Audit tính lại mọi generated score ở15 mốc, kiểm history/counters/
target counts/loss scope và earliest-best. Dataset giữ fingerprint0a891f….

Hai summary được tải của các lượt6be2e236 và0db55cf3 có toàn bộ checkpoint-owned
history, generated-validation summaries và reported best-weight digest giống nhau.
Đây là đối chiếu summary; chưa có file weights của lượt đầu để kiểm tensor trực tiếp.

File được bố trí tại
`artifacts/checkpoints/wp07-curriculum-2800-selected-2200/selected_model.pt`, evidence/
summary/provenance cạnh model. Export inference-only không chứa optimizer để full resume.

## Protocol evaluation đã chốt

Code/config commit `bbac14afa867b170441b3b0f050101d3dbe02f1e`, Git sạch khi bắt đầu.
Protocol SHA256:
`115e6bde4ffb464397980e88b51a29e836b87049852e6ceadd8857a529bf4b53`.
Run: `artifacts/runs/20261124T184655921268Z-wp07-6-3-curriculum-validation-seed42-fad81ac2`.
Artifact timestamp dùng UTC; ngày báo cáo dùng Asia/Bangkok.

RTX4060 Laptop, Torch2.9.1+cu128/CUDA12.8, FP32, deterministic/TF32off/thread1.
Greedy/B8/max256, cùng BOS-question-LF-think template và native final scorer cũ.
Không train thêm, không đổi budget/scorer/checkpoint, không đọc test, không fit threshold.

Replay48 câu selection khớp **toàn bộ token IDs/finish reasons** với Kaggle và cùng
metrics. Cached/uncached spot-check8 tokens khớp IDs, logits trong tolerance, max
absolute error4,77e-6. Các kiểm tra này hỗ trợ inference reload đúng trên scope đã đo;
không bảo đảm bitwise mọi câu hoặc mọi hardware. Input model/summary SHA giữ nguyên.

## Additional validation ngoài selection groups

Chọn96 câu theo hash(seed, ID) sau khi loại toàn bộ group xuất hiện trong selection48.
Các scope vẫn là validation, đã được dùng đo teacher-forced loss trong training;
không gọi đây là held-out test. Không cộng96 với pilot150 thành một sample độc lập.

| Task | Tổng đúng | Authored | WP03 bridge |
| --- | ---: | ---: | ---: |
| Copy | 32/32 (100%) | 32/32 | — |
| Single | 5/32 (15,63%) | 3/16 | 2/16 |
| Multi | 0/32 | 0/16 | 0/16 |

Tất cả96 có valid answer/format/EOS, không hết budget. Macro37/96=38,54% gồm32 câu
copy đúng; arithmetic-only là5/64, không phải38,54% toán đúng.

Posthoc strict first-R1-equation audit trên32 single: parse32/32, giữ đúng nguồn
operands/operator32/32, tự tính đúng5/32; cả5 final đúng cũng có source equation đúng.
Đây là kiểm tra formatter một bước cộng/trừ nguyên, không phải general CoT verifier.

## WP03 pilot validation150 và baseline cũ

| Mức | SFT500 trước curriculum | Selected curriculum2200 |
| --- | ---: | ---: |
| Easy | 2/50 | 5/50 |
| Medium | 0/50 | 0/50 |
| Hard | 0/50 | 0/50 |
| Tổng | 2/150 (1,33%) | 5/150 (3,33%) |

Baseline lấy predictions validation WP07.5 đã đóng băng, kiểm SHA và chấm lại trên
cùng IDs/references; không chạy lại baseline. Có9 pilot prompts từng nằm trong
selection48 (4easy/4medium/1hard). Trên141 câu còn lại: mới4/141=2,84%, cũ2/141=1,42%.
Các mức ngoài selection là46easy/46medium/49hard. Đây vẫn là validation.

Tất cả150 outputs mới có answer/format/EOS hợp lệ, không hết budget, tổng9.897 tokens.
Paired comparison:5 câu mới đúng,2 câu cũ đúng nay sai,0 câu giữ đúng ở cả hai model.
Mức tăng quan sát nhỏ; không tuyên bố significance hoặc cải thiện ổn định trên toàn miền.

First-equation audit trên50 easy:49/50 giữ đúng operands/operator,5/50 tính đúng,
và cả5 final đúng có source equation đúng. Các câu đúng hiện có trace đúng một bước,
khác với các final trúng tình cờ của SFT500 đã ghi ở WP07.5.

| Ví dụ validation | Output mới | Quan sát |
| --- | --- | --- |
| 16−8=8 | `R1 = 16 - 8 = 11.`; final11 | Đọc đúng số, tính sai |
| 30+29=59 | `R1 = 30 + 29 = 47.`; final47 | Đọc đúng số, tính sai |
| 5−18=−13 | `R1 = 5 - 18 = -13.`; final−13 | Equation/final đúng |
| Authored89−58=31 | `R1 = 89 - 58 = 31.`; final31 | Đúng ở cặp ngoài selection |

Quan sát ủng hộ rằng lỗi chép operands đã giảm nhiều, còn việc tính toán tổng quát
chưa được học tốt. Không xác định một nguyên nhân duy nhất của training hoặc kết luận
kiến trúc không có khả năng tính toán từ phép đo này. Loss teacher forcing thấp chưa
đảm bảo chuỗi tự sinh đúng.

## Nghiệm thu và bước tiếp theo

Đã xác minh user train đủ WP07.6.2 và hoàn tất WP07.6.3 validation diagnostics.
848 tests local passed,11 tests mới, Ruff/build passed, installed-wheel pipeline chạy
ngoài source tree. [CI implementation](https://github.com/CuongBien/GRAST/actions/runs/45463281009)
xanh Windows/Ubuntu. Model, source SFT500, main và WP02 tag không bị thay.

ZIP `dist/grast-wp07-6-3-validation-20261125-fad81ac2.zip` chứa config/protocol/log,
raw/scored predictions, audit, baseline snapshot, CSV/PNG/SVG và source summary; không
weights. SHA/CRC và27 artifact hashes nằm trong evidence JSON. Biểu đồ đã kiểm tra hiển thị.

Chưa triển khai confidence/calibration hoặc kết luận greedy thấp đồng nghĩa sampled
coverage thấp. Bước tiếp theo phù hợp là validation sample bankK16 để đo coverage,
majority và số ca đúng/sai thực tế trước khi chọn threshold. Chưa chạy bank đó trong
phase này, và không dùng test để chọn recipe hoặc threshold.
