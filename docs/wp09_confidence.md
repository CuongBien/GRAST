# WP09 — Confidence-Gated Sampling

## Các phase đã thống nhất

| Phase | Phạm vi | Đầu ra |
| --- | --- | --- |
| WP09.1 | Benchmark, design probe, partition/protocol | Manifest đã khóa; gate khả năng nghiên cứu calibration |
| WP09.2 | Bank mới cho selected900, tối đa K16/3 seeds | Ordered bank, question-only observations, scorer-only references |
| WP09.3 | Greedy riêng, sampled fixed-K majority, consensus | Accuracy–token baselines và replay xác định |
| WP09.4 | Agreement/entropy/log-probability và calibration | Prefix confidence, reliability/Brier/ECE; không fit evaluation |
| WP09.5 | Threshold stopping và CGES-style có định nghĩa rõ | Policy/threshold đóng băng bằng calibration validation |
| WP09.6 | Evaluation và phân tích thất bại | Accuracy–token trade-off, uncertainty theo nhóm, ca dừng sớm sai |

Không thêm vòng SFT. Không truyền gold cho stopping policy. Greedy không thay bằng
sampled K1; cùng câu/nhóm giữ một partition qua mọi seed, prefix và checkpoint.
Invalid giữ trong denominator và không được tự kích hoạt stop. Token savings từ replay;
chỉ báo cáo latency khi có đo online.

## Trạng thái WP09.1 — 28/11/2026

[Pilot WP09.2 đã chạy](wp09_pilot.md):960 candidates/20 câu/3 seeds,
0 candidates đúng, greedy0/20; giữ diagnostic_only, không tự mở full bank/calibration.

**Đã khóa benchmark ở chế độ diagnostic_only. Chưa đạt screen để tuyên bố có workload
phù hợp calibration/efficacy.** Probe cố định48 câu cho kết quả0/48 final đúng,
format48/48 hợp lệ; không đổi rule hoặc tìm thêm vòng probe sau kết quả này.
Đây là phát hiện trên design validation mới, không phải test accuracy.

Source: selected900, SHA256
`6b24c29a7be8b4700bbd92b6e1d87e88e64e670fe2378437b18dc97abd8d4a97`.
Config trước probe: `configs/evaluation/wp09_design.json`, commit
`c532ca5623a9053fde307d55045f60957cc1b34d`.

| Design stratum | Cấu trúc | Đúng /8 |
| --- | --- | --- |
| cross-short | Một phép +/−, toán hạng1..30 và31..99, prompt ngắn | 0 |
| cross-medium | Cùng khoảng số, prompt medium | 0 |
| boundary-100 | Một phép, toán hạng80..119 và1..99 | 0 |
| three-digit | Một phép, hai toán hạng100..199 | 0 |
| two-step-cross | Hai phép: cross-range rồi toán hạng100..199 | 0 |
| two-step-three-digit | Hai phép, toán hạng100..199 | 0 |

Sau kết quả0/48 đã chạy control debug trên8 câu gate cũ, khớp chính xác generated IDs
đã audit và đạt6/8 như trước. Control chỉ kiểm tra đường load/generation; không tham gia
chọn nhóm benchmark. Không suy rộng93,75% trên gate cũ sang workload mới.
Probe nhỏ này gợi ý hạn chế khái quát ngoài phân phối cũ, chưa đủ để kết luận nguyên nhân
hay khẳng định model đã memorize.

## Rule và partition đã khóa

Trước probe, rule chọn hai strata có accuracy gần0,5 nhất trong các nhóm có1..7/8
câu đúng và format-valid≥75%; tie theo tên. Nếu thiếu nhóm đủ điều kiện, điền bằng
cùng thứ tự khoảng cách và đánh dấu diagnostic_only. Cả6 nhóm không đủ điều kiện;
fallback chọn **boundary-100** và **cross-medium**. Việc chọn hai nhóm bằng tie-break
không có nghĩa chúng tốt hơn các nhóm còn lại.

Đã tạo200 câu main, cân bằng hai strata, không lọc theo dự đoán:

| Partition | Số câu | Vai trò tương lai |
| --- | --- | --- |
| calibration_fit | 60 (30/stratum) | Fit confidence cho current-prefix winner |
| calibration_threshold | 40 (20/stratum) | Chọn threshold theo tiêu chí đã khóa |
| evaluation | 100 (50/stratum) | Báo cáo sau khi đóng băng policy |

48 câu design tách riêng. Toàn bộ200 câu main **chưa được sinh dự đoán hoặc chấm điểm**.
Các partition trên đều là development validation tổng hợp mới; official test không được đọc.

Đã loại6.621 cặp toán hạng có trong toàn bộ curriculum train/validation và recovery
train/selection/gate. Với từng node của biểu thức, evaluate hai vế thành integer pair,
canonicalize không thứ tự, bỏ qua operator để giữ cả +/− và các hoán vị trong một nhóm.
Hash reservation gán fact vào design/fit/threshold/evaluation theo tỷ lệ20/30/20/30.
Mọi fact của một câu phải cùng reservation. Không dùng lại fact giữa các câu, strata
hoặc partition; kiểm tra cả semantic ID, prompt, reference và checksum.
Đây là chống overlap với các corpus đã audit, không bảo đảm chưa từng có exposure trong pretraining.

## Protocol cho các phase sau

- Bank: model selected900, seeds42/43/44, tối đa16 candidates/câu, temperature0,8,
  top-p0,95, không top-k truncation, tối đa128 new tokens, FP32; greedy sinh riêng.
- Fixed K={1,2,4,8,16}; consensus counts={2,3,4}; minimum samples2.
- Majority tie: earliest occurrence. Confidence chỉ đọc prefix đã quan sát.
- Fit calibrator trên calibration_fit; chọn threshold trên calibration_threshold;
  mục tiêu giảm token với accuracy drop tối đa2 điểm phần trăm so fixed-K16 trên phần chọn threshold.
  Đây là tiêu chí empirical để chọn, không bảo đảm trên evaluation.
- Báo cáo uncertainty theo question/fact group, gom mọi seed/prefix cùng nhóm.

WP09.2 nên bắt đầu bằng **pilot nhỏ trên calibration_fit** để kiểm tra stochastic
sampling có tạo được cả ca đúng và sai hay không. Nếu vẫn không có đủ outcome diversity,
giữ nhánh diagnostic và báo No-Go cho fit/calibration; không ép fit bằng dữ liệu một lớp,
không mở evaluation để cứu thiết kế và không train thêm model.

## Tái lập và artifacts

```powershell
python -u -m llm_scratch.research.confidence_design `
  --config configs/evaluation/wp09_design.json `
  --curriculum-root artifacts/wp07-6-2/curriculum_data `
  --recovery-root artifacts/easy-recovery-v1/data `
  --tokenizer artifacts/runs/20261105T154939888754Z-wp04-byte-bpe-seed42-4102afe0/artifacts/tokenizer `
  --checkpoint artifacts/checkpoints/easy-recovery-1000-selected-900/selected_model.pt `
  --output artifacts/wp09-1/design-v1 --device cuda
```

Output phải là thư mục mới. `verify(root, expected_fingerprint)` kiểm tra payload,
question/reference projection, groups và partition. Question-only files dùng schema
QUESTION_KEYS; labels/rationale/expressions nằm riêng trong references.
Output hiện tại: `artifacts/wp09-1/design-v1/`; manifest đã versioned tại
`experiments/wp09_protocol_v1.json`. Local source snapshot và metadata lưu hash đúng
implementation đã chạy probe. Kit ZIP portable nằm trong `dist/`, không chứa weights.

Verification local:956 tests qua, Ruff lint/format qua, wheel/sdist build qua;
đã verify benchmark từ wheel cài ngoài source tree. Cả48 design generations kết thúc
EOS; scorer mới cũng chấm control cũ đúng6/8. Receipt:
`experiments/wp09_1_validation_20261128.json`.
