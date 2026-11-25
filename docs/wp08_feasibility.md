# WP08.4 — Feasibility review và supplementary GSM8K probe

Đã hoàn tất review, xem [kết quả và quyết định chốt WP08](wp08_4_validation.md).

Review kiểm tra lại checksum/provenance của WP08.2/3 và dùng nguyên bank/replay đã
khóa. Không thay số liệu, không chọn lại pilot hoặc threshold. Receipt versioned dùng
canonical JSON fingerprint để không phụ thuộc LF/CRLF; runtime artifacts dùng SHA256
bytes/kích thước. Bank reader và replay reader tiếp tục xác minh schema và quyết định.

Config: `configs/evaluation/wp08_feasibility.toml`.

```powershell
uv run --no-sync python -u scripts/review_feasibility.py prepare
uv run --no-sync python -u scripts/review_feasibility.py run --device cuda
# CI/offline tiny fixture:
uv run --no-sync python -u -m llm_scratch.research.review_acceptance
```

## Probe đã khóa trước khi generation

GSM8K WP03 có500 internal validation rows từ **official train**. Chỉ đọc
`gsm8k/validation.jsonl`; không mở official/internal test payloads. Kiểm manifest,
payload SHA/size/count, source/split/upstream_split và unique IDs. Nguồn gốc/split
đã khóa ở WP03; không tải nguồn hoặc thay revision trong phase này.

Chọn16 câu bằng hash `['wp08-gsm-probe-v1',20261126,question_id]` trong các câu đủ
context. Điều kiện là prompt+BOS/think+budget256 không vượt context nhỏ nhất của
hai checkpoint (1024). Không truncate, không chọn lại sau khi xem output. Cả500 câu
đều đủ context trong phiên chuẩn bị thực tế. Hai checkpoint SFT500/selected2200,
greedy cached, batch8, FP32, seed42, max_new_tokens256; tổng32 predictions.
Đây là probe nhỏ, không phải bank K16/multi-seed GSM8K hoặc full-benchmark estimate.

Manifest và question-only JSONL ở `artifacts/wp08-4/gsm16-v1`; `prepare` từ chối
overwrite. Bản khóa versioned: [wp08_gsm8k_probe_v1.json](../experiments/wp08_gsm8k_probe_v1.json).
Fingerprint `52a75efb827d355f6bff301ee6f754aab1205e5f49e27ca125b2f0a1d09f8aa9`.

File nguồn GSM8K kết hợp question/answer nên loader phải parse cả row. Selector và
generator chỉ nhận projection question-only; không xếp hạng theo gold/predictions.
Metadata operation_count/reasoning_steps=0 là **unknown**, không phải khẳng định bài
không cần phép tính. Difficulty=ungraded. Scorer mới tạo reference projection sau khi
predictions của cả hai checkpoint đã ghi xong. Final được chấm numeric-v1 theo cùng
native final extraction; rationale GSM8K không được canonical-trace verified.

## Cách diễn giải và quyết định

GSM8K greedy probe và arithmetic sampled coverage@16 có budget/policy khác nhau;
không so hai tỷ lệ như cùng một loại accuracy hoặc dùng chúng để xếp hạng dataset.
Review benchmark chủ yếu dựa vào thất bại arithmetic/replay đã đo đầy đủ ở WP08.2/3;
GSM8K probe bổ sung quan sát và kiểm tra readiness, không fit hay chọn threshold.

Assessment là **posthoc descriptive review** cho đúng snapshot bank/replay WP08 hiện
tại. Kết luận không phải gate số học/statistical cutoff đã preregister. Code gắn nhận
định vào hai fingerprint nguồn; dữ liệu khác trả requires-new-review, không tự động
cho phép/loại một model mới. Không có auto-training hoặc auto-calibration.

Tách hai gate: artifact/pipeline acceptance có thể hoàn tất trong khi capability chưa
đủ để kiểm chứng lợi ích confidence stopping. Bank vẫn dùng được cho failure analysis.
Role ưu tiên là WP03 easy làm diagnostic chính; medium/hard là negative controls,
GSM8K giữ supplementary stress. WP03 easy cũng chưa phải benchmark chứng minh hiệu
quả, cần nâng capability và khóa pilot mới trước efficacy/calibration claims.

Run xuất config/metadata/log, source evidence audit, probe manifest, raw/scored GSM8K
predictions và feasibility_report.json. Mọi lỗi checksum/source provenance hoặc đổi
probe/config làm run fail. Không dùng test để chọn benchmark, recipe hoặc threshold.
