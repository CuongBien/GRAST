# WP09.2 — bounded stochastic pilot

Tiếp theo [WP09.1](wp09_confidence.md), chỉ chạy pilot trên calibration_fit;
không tự thu full bank hoặc fit calibrator. Model giữ selected900, không thêm SFT.

## Protocol khóa trước generation

Implementation/plan commit: `02080bd5b20ee967f820e96ae94e71763ed74564`.
Plan: `experiments/wp09_pilot_v1.json`, fingerprint
`5be360669114a460de9aa114309bdf3093e10a888884ffbe1c4c99fc0b10b4f5`.

- Chọn20 câu bằng hash:10 boundary-100 và10 cross-medium, chỉ calibration_fit.
- 3 seeds42/43/44,16 candidates/câu/seed:960 candidates; batch8, FP32.
- Temperature0,8, top-p0,95, top-k0, tối đa128 new tokens; greedy20 câu sinh riêng.
- Ngân sách tối đa125.440 generated tokens gồm sampled+greedy, chưa tính prompt.
- Không đổi danh sách, generation settings hoặc budget theo kết quả giữa chừng.

Screen dựa trên **prefix winner** từ n≥2, không chỉ oracle coverage@16:
ít nhất5 question groups có prefix final đúng,5 có prefix sai và5 có prefix đúng kèm
canonical trace. Mỗi group chỉ đếm một lần, dù lặp nhiều seeds/prefixes. Các tập group
đúng/sai có thể giao nhau. Đây chỉ là necessary engineering screen, không chứng minh
cỡ mẫu đủ để calibration. Trace match là kiểm tra bảo thủ theo rationale chuẩn hóa whitespace.

Nếu screen qua: chỉ cân nhắc mở rộng calibration_fit, chưa tự chạy evaluation/full bank.
Nếu không qua: giữ diagnostic_only, không ép fit calibrator bằng dữ liệu một lớp hoặc quá ít
nhóm đúng. Không dùng threshold/evaluation để cứu thiết kế, không train thêm model.

## Generation và scoring tách biệt

`prepare` xác minh benchmark fingerprint rồi chọn câu chỉ từ questions/index, không dùng
correctness. Input folder chỉ chứa plan, question-only rows và tokenizer; từ chối thêm
reference/label files. Generation CLI không nhận đường dẫn benchmark hoặc references.

Bank tái sử dụng storage format WP08.1 (`model_outputs_only_no_gold`), còn wrapper plan,
run và report gắn WP09.2. Mỗi candidate lưu thứ tự, private batch seed, generated IDs,
raw-model FP32 natural-log probabilities, answer extraction và token count. Gold/scoring
nằm ngoài bank. Manifest cuối cùng mới commit một bank đầy đủ; partial bank không được đọc.

Greedy có file riêng và hash trong generation commit, không coi sampled K1 là greedy.
Sau khi commit generation, scorer chỉ lấy references của20 câu pilot. Plurality nhận
`observation.prefix(n)` trước khi gắn correctness; tie theo earliest occurrence,
invalid giữ trong denominator. Prefix labels không được đưa ngược vào policy observation.

## Chạy lại

```powershell
python -u -m llm_scratch.research.confidence_pilot prepare `
  --benchmark artifacts/wp09-1/design-v1 --output artifacts/wp09-2/pilot-inputs-v1

python -u -m llm_scratch.research.confidence_pilot generate `
  --inputs artifacts/wp09-2/pilot-inputs-v1 `
  --plan-fingerprint 5be360669114a460de9aa114309bdf3093e10a888884ffbe1c4c99fc0b10b4f5 `
  --checkpoint artifacts/checkpoints/easy-recovery-1000-selected-900/selected_model.pt `
  --output-root artifacts/wp09-2/runs --device cuda

python -u -m llm_scratch.research.confidence_pilot analyze `
  --inputs artifacts/wp09-2/pilot-inputs-v1 `
  --plan-fingerprint 5be360669114a460de9aa114309bdf3093e10a888884ffbe1c4c99fc0b10b4f5 `
  --run <run_directory_in_log> --benchmark artifacts/wp09-1/design-v1 `
  --output <new_analysis_directory>
```

Luôn in batch/decode progress và ghi run logs. Output phải là thư mục mới;
không append vào một bank dang dở. Để replay offline, nạp SampleBank từ
`<run>/artifacts/bank/`; dùng canonical references bên ngoài để scoring.
Raw bank, references, weights và ZIP chỉ lưu local; protocol và receipt được versioned.

## Kết quả pilot ngày28/11/2026

Đã sinh đủ960 candidates và20 greedy completions trên RTX4060 Laptop, FP32,
giữ nguyên model selected900 và plan đã khóa. Generation lifecycle khoảng5 phút,
gồm load model, commit/verify bank và greedy; không phải phép đo latency của stopping policy.

| Chỉ số | Kết quả |
| --- | --- |
| Candidate final/verified correct | 0/960 |
| Observed coverage@16 | 0/60 question–seed trials, trên20 câu |
| Fixed-K16 winner correct | 0/60 trials |
| Prefix winner đúng từ n≥2 | 0/900 prefix observations |
| Greedy final/verified correct | 0/20 |
| Numeric answer / format / EOS hợp lệ | 960/960 cho cả ba chỉ số |
| Sampled generated tokens | 23.356 |
| Screen mở rộng calibration_fit | Không đạt |

Ở K16,39/60 trials có unanimous agreement nhưng đều sai; ở K2 là51/60.
Agreement là raw vote share, chưa phải xác suất đã calibrated. Không xem60 trials
hoặc900 prefixes là số câu độc lập. Kết quả chỉ mô tả pilot trên calibration_fit.

Đã chấm lại bằng wheel cài ngoài source tree, report khớp hoàn toàn; đồng thời
evaluate lại biểu thức symbolic cho cả960 candidates. Đây là sai đáp án thực sự,
không phải lỗi parse/truncation. Ví dụ private pilot:105−13 được sinh thành12 thay vì92;
89−1 thành8 thay vì88. Không mở rộng suy luận thành kết luận về mọi dạng toán.

**Quyết định: diagnostic_only_no_calibrator_fit.** Không tự thu full bank9.600 candidates,
không fit calibrator, không chấm calibration_threshold/evaluation và không thêm SFT.
Bank đủ cấu trúc để replay; trên bank hiện tại, oracle coverage bằng0 nên policy chỉ
chọn từ các candidate answers không thể cải thiện accuracy. WP09.3 có thể tiếp tục
diagnostic replay/cost và false consensus, nhưng chưa có cơ sở cho claim efficacy/calibration.

Run: `artifacts/wp09-2/runs/20261128T120857964590Z-wp09-2-stochastic-pilot-seed42-b7375a52`.
Bank fingerprint:
`8752dbb5c275be1eeaed2b4d5cc431c4a9967cb34b03072f4506a5cf633e28b7`.
Analysis: `artifacts/wp09-2/analysis-v1/`; receipt:
`experiments/wp09_2_validation_20261128.json`.
