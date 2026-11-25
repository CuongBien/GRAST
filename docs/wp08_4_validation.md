# WP08.4 — Chốt feasibility và benchmark, 26/11/2026

**Đóng WP08 về nghiệm thu kỹ thuật.** Có bank tái sử dụng được,2.880 ứng viên có
thứ tự/metadata/log-prob, replay8 policy/1.440 quyết định và báo cáo feasibility.
Gold nằm ngoài policy observations và được scorer dùng sau khi quyết định commit.

**Gate về năng lực để chứng minh lợi ích confidence stopping chưa đạt.** Kết quả
hiện tại phù hợp làm failure analysis; chưa đủ để khẳng định hiệu quả calibration/
adaptive stopping với model và benchmark này. Không tự chạy train dài hoặc fit threshold.

## Quyết định benchmark

| Thành phần | Vai trò đã chốt | Căn cứ/phạm vi |
| --- | --- | --- |
| WP03 easy, một phép +/−, toán hạng1–30 | Ưu tiên làm primary diagnostic | Đã có canonical-verified samples, nhưng policy accuracy vẫn thấp; cần nâng capability trước efficacy claims |
| WP03 medium/hard | Negative controls và error analysis | Thiếu verified reasoning; mọi policy replay đều0 đúng trên pilot ở hai tier này |
| GSM8K | Giữ supplementary stress test; chưa chọn làm primary | Probe nhỏ ít final đúng; không mở rộng K16 GSM8K lúc này |
| Bank WP08.2 và replay WP08.3 | Giữ nguyên để tái sử dụng, phân tích thất bại | Đã xác minh lại toàn bộ hashes/provenance; không chọn lại câu/seed/policy sau khi xem output |

Đây là **posthoc descriptive review** trên development validation đã được xem,
không phải gate/cutoff thống kê đã preregister. Quyết định gắn với fingerprint bank
và replay hiện tại; code trả requires-new-review cho snapshot khác. WP03 easy là
hướng diagnostic ưu tiên, chưa phải benchmark đã chứng minh model đủ tốt.

## Bằng chứng tổng hợp

| Phép đo | SFT500 | Selected2200 |
| --- | --- | --- |
| Arithmetic K16 observed coverage toàn pilot, mean3 seeds | 21,1% | 20,0% |
| Easy K16 observed coverage, mean3 seeds | 46,7% | 56,7% |
| Canonical-verified candidates /1.440 | 0 | 28, đều ở easy |
| Majority K16 final đúng /90 question-trials | 1 | 1 |
| Unanimous-min2 token savings so với fullK16 | 0,72% | 1,43% |
| GSM8K greedy supplementary probe: final đúng /16 | 1 | 0 |

Các hàng có objectives/decoding budgets khác nhau. Không so GSM8K greedy accuracy
trực tiếp với arithmetic oracle coverage để xếp hạng dataset. Ba seed arithmetic dùng
lại cùng30 câu;90 question-trials không phải90 câu độc lập. Majority chọn đáp án
còn cách xa số trial có một correct candidate trong bank. Không gọi oracle là policy.

Nguồn: [WP08.2](wp08_2_validation.md), [WP08.3](wp08_3_validation.md).
Review kiểm lại từng artifact checksum từ hai receipt versioned, xác minh lại bank,
recompute replay decisions/history và đối chiếu các fingerprint trong reports.
Không sửa hoặc thu lại bank/replay. Token savings là accounting của prefix,
không phải tốc độ/VRAM thực đo hoặc chi phí GPU đã thu hồi từ bank cũ.

## Supplementary GSM8K probe

[Protocol](wp08_feasibility.md), [manifest đã khóa](../experiments/wp08_gsm8k_probe_v1.json).
Nguồn là500 internal validation rows từ official train; không mở test payloads.
Chọn16 IDs bằng hash trước generation, không dùng gold/predictions để chọn.
Prompt+BOS/think+budget256 vừa context1024 cho cả500 câu, không truncate hoặc loại
câu theo kết quả. Hai checkpoint cũ, greedy cached, FP32/batch8, seed42.

| Kết quả trên cùng16 câu | SFT500 | Selected2200 |
| --- | --- | --- |
| Final exact match | 1/16 (6,25%) | 0/16 |
| Answer-valid / format-valid | 15/16 /15/16 | 15/16 /15/16 |
| EOS / hết budget256 | 16/16 /0 | 16/16 /0 |
| Generated tokens | 689 | 890 |

Một final đúng của SFT500 không đi kèm rationale nhất quán: output tham chiếu R2
chưa được định nghĩa và gán hai kết quả khác nhau cho cùng phép tính. Đây là kiểm
tra tay của đúng một ca, không phải một bộ đánh giá rationale GSM8K tổng quát.
Không gọi final EM là tỷ lệ reasoning đúng. Giữ invalid cases trong mẫu số16.

Probe chỉ có16 câu/greedy, không multi-seed/K16 GSM8K, không ước lượng full benchmark
hoặc chứng minh checkpoint nào tốt hơn trên GSM8K. Việc ưu tiên easy chủ yếu dựa vào
arithmetic/replay đã đo; probe này bổ sung quan sát, không dùng để fit policy.

## Các ca lỗi và handoff

Artifact có **7 ca trên7 question IDs khác nhau**, chọn theo quy tắc cố định từ
outputs đã commit: hai final đúng/trace unverified của arithmetic, hai voting misses
dù đã thấy correct candidate, một early wrong answer có correct candidate phía sau,
một GSM8K correct final/rationale mâu thuẫn và một GSM8K wrong final. Đây là các ví dụ
diagnostic, không phải sample ngẫu nhiên để ước lượng tần suất lỗi.

Bước tiếp theo nên là một nhánh nâng feasibility của easy trước efficacy/calibration:
kiểm tra phép tính một bước và chất lượng rationale, nếu cần chuẩn bị train trên Colab
theo quy trình của user. Sau thay đổi recipe/model/data, khóa một pilot validation mới
trước khi đo; giữ kết quả cũ làm baseline thất bại. Không chọn subset những câu hiện
tại model đã làm đúng, không chọn seed tốt nhất, không dùng test để chọn threshold.
Có thể tiếp tục triển khai công cụ confidence để diagnostic, nhưng chưa gán efficacy
claim cho các kết quả hiện tại.

## Verification và artifact

Full suite **921 tests passed**,14 tests mới;14 focused tests chạy lại sau thay đổi
review guard cũng đạt. Ruff/build/diff checks đạt; installed wheel chạy toàn bộ tiny
review pipeline ngoài source tree. Source distribution đã kiểm tra loại runtime
artifacts/weights. Có tests cho checksum/provenance, stable question-only ranking,
context eligibility, frozen probe tampering, test-origin rejection và scoring chỉ
sau cả hai prediction files. Không train model thật trong WP08.4.

Implementation commit: `821c0f67ea703943ffc4fc1b2daa36de74ec7ce5`, Git sạch khi chạy.
Run UTC: `artifacts/runs/20261125T192957792987Z-wp08-4-feasibility-review-seed42-5365f7f3`.
Probe fingerprint: `52a75efb827d355f6bff301ee6f754aab1205e5f49e27ca125b2f0a1d09f8aa9`.
Bank nguồn vẫn `6941830812e5fe91ba583553556cbc68a10ecf9b84dcad2bb5850c97cab1c809`,
replay nguồn vẫn `7fa3de8093ae73094a74c82b95e74cc7545858c2df05f65c965f1fee242e6a20`.
Đã audit lại source artifacts sau khi probe kết thúc.

[Receipt/file hashes](../experiments/wp08_4_validation_20261126.json).
ZIP local `dist/grast-wp08-4-feasibility-20261126-5365f7f3.zip`,252.551 bytes,
SHA256 `c255fef43b97abcc1dac75ecb62777abdeb300573708e83192f7af67e1cd35be`.
Gồm raw/scored GSM8K predictions, source audit, report,7 failure cases, source figures,
probe manifest, config/metadata/log và export script; không chứa weights/bank nguồn.
