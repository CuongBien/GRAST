# WP08.2 — Frozen validation pilot

Protocol khóa trước khi sinh mẫu: **30 câu WP03 validation, 10 easy/10 medium/10 hard**,
SFT500 và curriculum selected2200, trial seeds42/43/44, K16, batch8, FP32,
temperature0,8/top-k40/top-p0,95, budget256 token. Tổng **2.880 ứng viên**; không train.
Không tự thay seed, batch size, precision hoặc budget sau khi xem kết quả.

Pilot fingerprint: `66de0297d66192347e4e71cd8d766f7249e929524a2e9832375c88ef765ec59f`.
Bản khóa có IDs/protocol/source fingerprints nằm ở
[`experiments/wp08_pilot_v1.json`](../experiments/wp08_pilot_v1.json).
Đây là development validation: dữ liệu đã dùng theo dõi validation loss và có
đánh giá greedy trước đây, không phải một test set chưa từng được xem.

## Chọn và khóa câu hỏi

Preparer kiểm tra WP03 manifest, tokenizer, curriculum corpus và selected-model SHA.
Tái tạo48 selection IDs từ contract của checkpoint, kiểm tra ID khớp chính xác,
rồi loại toàn bộ các group tương ứng. Trong150 câu pilot validation có9 câu thuộc
những group này; còn141 câu đủ điều kiện. Chọn một câu mỗi group theo thứ tự
hash `['wp08-pilot-v1', selection_seed, question_id]` trong từng difficulty.
Không dùng đáp án, loss, correctness hoặc predictions để xếp hạng câu hỏi.

Corpus loader đọc/kiểm tra đầy đủ curriculum validation rows, bao gồm targets, để
xác minh dữ liệu đã dùng huấn luyện. Hàm chọn câu chỉ nhận question-only rows,
mapping prompt→group và tập group bị loại; không nhận targets. Không đọc reference
JSONL, predictions cũ hoặc test trong bước chuẩn bị. Bank builder cũng chỉ nhận
question-only rows. Scorer mới đọc references và cây biểu thức sau khi bank commit.

```powershell
uv run --no-sync python -u scripts/run_feasibility_pilot.py prepare
uv run --no-sync python -u scripts/run_feasibility_pilot.py run --device cuda
```

`prepare` ghi `artifacts/wp08-2/pilot-v1/{questions.jsonl,manifest.json}` và từ chối
overwrite. Lượt `run` chỉ nhận pilot đã khóa; đổi generation, K, seed, batch,
checkpoint digest hoặc exclusion sẽ bị từ chối. Paths/run names không thuộc
portable experiment identity; cấu hình nguồn và paths thực tế vẫn được snapshot.
Có thể chỉ định `--config`, `--pilot-root`, `--output-root`. Muốn đổi protocol cần
một pilot version mới; giữ bank cũ để có thể kiểm tra kết quả đã báo cáo.

Smoke offline, không cần dữ liệu/weights thật:

```powershell
uv run --no-sync python -u -m llm_scratch.research.pilot_acceptance
```

Smoke tạo/training fixture rất nhỏ, khóa pilot hai lần và yêu cầu fingerprint exact;
chạy3 câu/2 checkpoints/2 seeds/K2 rồi chấm riêng24 ứng viên. Đây là kiểm tra kỹ thuật.

## Outputs và cách đọc metrics

Run có config/metadata/environment/log, `pilot_manifest.json`, bank chuẩn WP08.1,
`external-scoring`, `trace_audit.jsonl`, `prefix_coverage.jsonl`, `pilot_report.json`.
Logs luôn in ngay khi bắt đầu/kết thúc batch và mỗi32 decode steps. Bank giữ đủ16
ứng viên/câu/trial theo thứ tự, cùng token count/log-probs và seeds. Không gửi gold,
scoring labels hoặc checkpoint selection scores vào policy observations.

Coverage ở K1/2/4/8/16 là **observed prefix coverage**: có ít nhất một đáp án đúng
trong các ứng viên đầu đã lưu. Không phải unbiased pass@k estimator; K1 sampled
không phải greedy. Báo theo từng checkpoint/seed/difficulty, kèm mean/min/max giữa
ba seed. Ba seed tái sử dụng cùng30 câu;90 question-trials không phải90 câu độc lập.
Không chọn checkpoint hoặc seed tốt nhất để thay cho kết quả đã khóa.

Trace audit dùng tiêu chí bảo thủ: reasoning trước native `</think>` phải khớp
`math_reasoning(expression)` sau khi gộp whitespace. `verified_correct` còn yêu cầu
final đúng và format hợp lệ. Vì canonical reasoning tính chính xác toàn bộ cây,
đáp án tình cờ đúng với trace sai không được tính vào verified coverage. Một lời giải
khác có thể đúng nhưng không khớp template: ghi **unverified**, không kết luận sai.
Report lưu cả answer coverage và verified coverage, cùng format rate/budget exhaustion.

WP08.2 không fit confidence, threshold, calibration, stopping hoặc quyết định GSM8K.
WP08.3 sẽ dùng cùng bank cho các policy; WP08.4 mới chốt báo cáo feasibility/benchmark.
Generated tokens là chi phí token, không phải latency hoặc VRAM thực đo.
