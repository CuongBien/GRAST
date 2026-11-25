# WP08.2 — Pilot validation và sample bank, 25/11/2026

**Đã hoàn tất thu mẫu và kiểm tra WP08.2.** Pilot có30 câu,10 mỗi difficulty;
hai checkpoint ×3 trial seeds ×16 ứng viên = **2.880 ứng viên**. Bank đã commit,
được reader xác minh đầy đủ và có thể dùng lại cho WP08.3. Reasoning nhiều bước
vẫn yếu; chưa thiết lập gate feasibility cho toàn WP08 hoặc quyết định GSM8K.

Protocol được khóa trước khi sinh mẫu trong [WP08.2](wp08_pilot.md) và
[manifest versioned](../experiments/wp08_pilot_v1.json). Loại9 câu thuộc toàn bộ
group đã dùng checkpoint selection; chọn30 câu bằng hash từ141 câu đủ điều kiện.
Không xếp hạng câu theo gold hoặc predictions. Đây là development validation,
đã dùng theo dõi validation loss và có greedy evaluation trước đây.

## Kết quả tại K16

Mỗi ô coverage là trung bình ba seed; khoảng trong ngoặc là min–max giữa seed.
Mỗi difficulty có cùng10 câu được chạy lại ba lần. Không xem30 question-trials
của một difficulty là30 câu độc lập. Candidate0 là sampled, không phải greedy.

| Checkpoint | Easy answer coverage | Medium answer coverage | Hard answer coverage | Toàn pilot |
| --- | --- | --- | --- | --- |
| SFT500 | 46,7% (40–60%) | 16,7% (10–30%) | 0% | 21,1% (16,7–23,3%) |
| Selected2200 | **56,7% (40–80%)** | 3,3% (0–10%) | 0% | 20,0% (13,3–30,0%) |

Coverage là có ít nhất một final đúng trong prefix đã lưu, không phải accuracy của
một policy chọn đáp án. Không dùng kết quả này thay cho majority accuracy hoặc
unbiased pass@k estimator. Không chọn seed tốt nhất để báo cáo.

| Checkpoint | Final đúng /1.440 ứng viên | Đạt canonical trace verification | Verified coverage toàn pilot |
| --- | --- | --- | --- |
| SFT500 | 27 (1,875%) | 0 | 0% |
| Selected2200 | 29 (2,014%) | **28** | **18,9%** |

Selected2200 có17/30 easy question-trials được covered và verified: theo seed42/43/44
là4/10,8/10,5/10. Medium có một question-trial covered ở seed43, không được verified;
mẫu đó có phép tính trung gian sai. Hard không có final đúng trong bank này.

Verification đã khóa là reasoning khớp canonical WP03 trace sau khi chuẩn hóa
whitespace, cùng final đúng và format hợp lệ. SFT500 không có mẫu đạt tiêu chí
này; các ví dụ correct/unverified đã kiểm tra có chép sai operands và tính sai.
Không suy rộng rằng mọi trace không khớp template đều sai: lời giải tương đương
có thể chưa được checker này xác minh. Raw examples và canonical traces nằm riêng
trong artifact, không nằm trong policy observations.

Nhìn trên final coverage toàn pilot, selected2200 không tăng so với SFT500 trong
lượt đo này. Bằng chứng tích cực tập trung ở easy và canonical verified samples;
chưa có bằng chứng giải đúng nhiều bước ở medium/hard. Pilot nhỏ và chỉ có ba seed,
không đưa ra kết luận thống kê về mức cải thiện tổng quát.

## Dữ liệu đã lưu và kiểm tra

- Sampling FP32, temperature0,8/top-k40/top-p0,95, batch8, budget256, trial seeds42/43/44.
- Mỗi checkpoint có180 batch và1.440 records; toàn bank có180 question/checkpoint/seed
  groups,16 ordered candidates/group. Mọi token/log-prob/seed/batch metadata được giữ lại.
- SFT500 sinh92.993 token, selected2200 sinh93.098 token; tổng **186.091 token** gồm EOS,
  loại masked tails. Không có ứng viên hết budget256; vẫn giữ các mẫu lỗi format/answer.
- Run kéo dài khoảng33 phút51s trên RTX4060 Laptop/Torch2.9.1+cu128/CUDA12.8, FP32.
  Đây là thời gian thu bank, không phải profiling latency/VRAM của stopping policy.
- Reader xác minh checksum, grid, thứ tự, seed, tokenizer, answer extraction và log-prob
  aggregates. Package source fingerprint ở đầu run khớp implementation trong bank cuối run.
- Model inputs SFT500/selected2200 giữ nguyên SHA. Không huấn luyện model thật.

Run:
`artifacts/runs/20261125T120930824002Z-wp08-2-feasibility-pilot-seed42-333b2fdc`.
Implementation commit: `b01f5dee40af7f8ef0d144bb5829954b392cd9ba`, Git sạch khi bắt đầu.
Bank fingerprint:
`6941830812e5fe91ba583553556cbc68a10ecf9b84dcad2bb5850c97cab1c809`.

Outputs: bank tự chứa tokenizer, `pilot_report.json`, per-seed/per-difficulty metrics,
`prefix_coverage.jsonl`, `trace_audit.jsonl`, `trace_examples.json`, `coverage.csv`,
`coverage.png/svg`, export script, logs/config/metadata/environment. Các reference và
scored labels được ghi ngoài bank, sau khi bank commit. Không mở test thật, không fit
threshold/confidence/calibration/stopping và không quyết định benchmark trong phase này.

**887 tests local passed**,8 tests mới; Ruff và build wheel/sdist đạt. Đã kiểm tra
distribution không chứa runtime artifacts/weights; installed wheel chạy pilot smoke
ngoài source tree. [CI implementation](https://github.com/CuongBien/GRAST/actions/runs/45613716185)
xanh trên cả Ubuntu và Windows, gồm smoke WP08.2 với fixture offline.

[Receipt và file hashes](../experiments/wp08_2_validation_20261125.json).
ZIP local `dist/grast-wp08-2-pilot-20261125-333b2fdc.zip`,2.007.567 bytes,
SHA256 `68c85c40477b05d01adf610760ff8ed869fed38affff8129a0871cee2e45958e`.
ZIP chứa toàn bộ bank/raw artifacts và không chứa weights; raw bank giữ local.

Tiếp theo là **WP08.3: offline replay trên cùng prefix bank**. Mọi quyết định của
policy phải được tạo từ observation không gold/future candidates; scorer chỉ đánh
giá sau quyết định. Không fit ngưỡng trên test hoặc dùng oracle coverage làm policy.
