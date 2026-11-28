# WP09.4A — confidence diagnostics

Ngày 29/11/2026 hoàn tất chẩn đoán trên bank WP09.2 đã khóa, tiếp nối
[WP09.3](wp09_baselines.md). **Có prefix signals, reliability diagram và error cases;
chưa có calibrator hoặc threshold được fit.** Không sinh thêm mẫu, train hoặc mở
calibration_threshold/evaluation.

## Phạm vi và provenance

20 câu calibration_fit, 2 strata cân bằng, 3 seeds, K16: 60 câu–seed, 960 prefixes
n=1..16 (900 prefixes từ n≥2). Mọi tín hiệu mô tả current-prefix plurality winner;
tie theo earliest occurrence. Chỉ có 20 câu độc lập, không coi seed/prefix là câu mới.
Greedy riêng của WP09.3 không trộn vào phân bố phiếu sampling này.

Config khóa công thức, bin và prefix hiển thị:
`configs/evaluation/wp09_confidence_diagnostics.json`.
Implementation/config commit trước replay:
`8a866572b5cd5345ec4d00b8bb7cd17bb31b17ae`.
Định nghĩa được khóa sau khi đã biết kết quả WP09.2/3; đây là phân tích diagnostic,
không phải preregistration trước khi thấy pilot.

Bank fingerprint:
`8752dbb5c275be1eeaed2b4d5cc431c4a9967cb34b03072f4506a5cf633e28b7`.
Trace fingerprint:
`935d3b2bfe06b81a8a7ddd0f5b79b802a3c90784ea69ae069010e7142d42e2b9`.

## Tín hiệu quan sát được

`reasoning/confidence.py::prefix_signals` chỉ nhận một `PolicyObservation` prefix
không có gold. Driver gọi lần lượt `group.prefix(n)`; hàm không nhận đường dẫn bank,
references hoặc callback scoring. Mọi mẫu phía sau n đều nằm ngoài input của hàm.

- **Agreement:** top_votes / n; invalid abstain nhưng vẫn tính trong n. Không xem
  vote share là xác suất đúng đã calibrated. Một câu trả lời hợp lệ ở n=1 luôn đạt 1.
- **Answer entropy:** H = −Σ pᵢ ln(pᵢ), với pᵢ tính trên toàn bộ n phiếu cho từng
  canonical answer và một bucket invalid. H_norm = H/ln(n) khi n>1, n=1 trả 0.
  Toàn bộ invalid cũng có entropy 0, nhưng agreement 0 và winner null.
- **Log-probability:** raw model FP32 log-softmax, natural log, không phải likelihood
  của phân bố đã temperature/top-p filter. Chỉ gồm token sinh thật, kể cả EOS và token
  reasoning/format, loại padding. Đây không phải xác suất đáp án đúng.
- Lưu riêng trung bình sequence logprob sum, trung bình các candidate logprob mean
  (mỗi mẫu trọng số bằng nhau), pooled token logprob mean (chuỗi dài trọng số lớn hơn),
  trung bình candidate mean trong winner class và điểm của representative đầu tiên.
  Không exponentiate thành correctness probability hoặc ghép thành confidence tùy ý.

`create_trace` commit feature rows + SHA + implementation fingerprint **trước khi**
scorer đọc references. `load_trace` xác minh payload allowlist, hashes và tính lại
đúng tất cả prefix từ bank để từ chối sửa feature dù đã tính lại checksum. Scored
rows và gold chỉ nằm ngoài bank/trace. Không dùng độ dài/phiếu của future samples.

## Kết quả

Final-answer exact match là 0/960 prefixes, 0/900 ở n≥2; tất cả numeric/format đã được
kiểm tra ở WP09.2. Mỗi hàng dưới có 60 câu–seed trên cùng 20 câu; các mean lấy trung
bình seed trong câu trước, rồi trung bình đều các câu.

| Prefix K | Mean agreement | H (nats) | H/ln(K) | Unanimous nhưng sai | Agreement ≥0,9 nhưng sai |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 0,9250 | 0,1040 | 0,1500 | 51/60 | 51/60 |
| 4 | 0,9292 | 0,1493 | 0,1077 | 46/60 | 46/60 |
| 8 | 0,9313 | 0,1690 | 0,0813 | 43/60 | 43/60 |
| 16 | 0,9167 | 0,1985 | 0,0716 | 39/60 | 44/60 |

Mốc agreement ≥0,9 là cutpoint mô tả cố định, không phải stopping threshold được chọn.
H_norm giảm một phần do mẫu số ln(K) tăng; H thô thực tế tăng. Không diễn giải đường
H_norm giảm thành bằng chứng model ngày càng đúng hoặc uncertainty được calibrated.

Ở K16, mean logprob/token trung bình các candidate là −0,0121548; winner-class mean
là −0,0091645. Điểm token cao vẫn đi cùng đáp án sai; các số này gồm cả nội dung lặp
và định dạng. Chưa có nhóm câu đúng để kiểm tra tín hiệu nào phân biệt đúng/sai tốt hơn,
và không suy ra nguyên nhân từ bảng mean này.

Năm error cases khác câu được chọn theo thứ tự bank cố định, n≥2, agreement≥0,9,
đáp án sai; lưu toàn bộ observed features trong `analysis/error_cases.jsonl`.
Ví dụ 105−13 → 12 thay vì 92 và 89−1 → 8 thay vì 88. Labels chỉ được gắn sau trace.

## Reliability diagram và giới hạn một lớp

Chỉ raw agreement nằm trên trục confidence; không biến entropy/logprob thành
correctness probability. Reliability dùng 10 bins cố định `[0.0,0.1)`, …, `[0.9,1.0]`,
rightmost bin chứa score 1. Empty bins lưu null, không vẽ như accuracy 0.
Mỗi quan sát tại K có trọng số 1/(số câu × số seeds của câu); mass bin, confidence,
accuracy, ECE và Brier đều theo cùng trọng số. Tính riêng từng K và từng stratum,
không pool các prefix thành thử nghiệm độc lập. Không có CI dân số từ pilot này.

| K | Raw-agreement ECE | Raw-agreement Brier |
| --- | ---: | ---: |
| 2 | 0,9250 | 0,8875 |
| 4 | 0,9292 | 0,8823 |
| 8 | 0,9313 | 0,8841 |
| 16 | 0,9167 | 0,8625 |

Đây là **descriptive metrics của score chưa calibrated trên calibration_fit**, không
phải đánh giá held-out calibration. Vì mọi winner đều sai, ECE bằng mean agreement
và Brier bằng trung bình có trọng số của agreement². Chỉ cần hạ mọi score về 0 cũng đạt
ECE/Brier 0 trên
tập này; điều đó không chứng minh khả năng hiệu chuẩn hoặc generalization.
AUROC để null với trạng thái `undefined_single_outcome_class`; không fit một mô hình
calibration hằng và không chọn policy/threshold từ bảng này.

Hai chart PNG/SVG: `reliability_diagnostic` (reliability + bin mass), `prefix_signals`
(agreement/accuracy, entropy chuẩn hóa, raw logprob theo n). CSV chứa summary của
tất cả 16 prefixes và toàn bộ bins, gồm counts/unique questions/mass.

## Chạy lại và bàn giao

```powershell
python -u -m llm_scratch.research.confidence_diagnostics `
  --config configs/evaluation/wp09_confidence_diagnostics.json
```

Cần bank và question-only inputs WP09.2, references 20 câu nằm riêng. Không cần
weights/GPU. Config khóa SHA/fingerprint và bind question IDs/checkpoint hash với
pilot plan calibration_fit. Run luôn in log, tạo config snapshot và metadata.
Output `artifacts/trace/` chưa có labels; `artifacts/analysis/` chứa scoring.

Source run:
`artifacts/wp09-4a/runs/20261128T175853526636Z-wp09-4a-confidence-diagnostics-seed42-72b58308`.
Tên run theo UTC; ngày local là 29/11/2026. Wheel cài ngoài source tree chạy lại
với checkpoint loading bị cấm: trace manifest, features, report, scored rows,
error cases và cả hai CSV khớp từng byte. Chart đã được kiểm tra trực quan.
Local verification: 979 tests qua trong 211,21 giây, gồm 13 tests mới cho signal,
reliability, gold boundary và trace integrity; Ruff lint/format và package build qua.
ZIP cũng đã được giải nén và replay lại từ wheel với kết quả khớp từng byte.
Receipt: `experiments/wp09_4a_validation_20261129.json`; kit portable trong `dist/`
chứa bank/input/outputs/report/config, references tách riêng và không có weights.
Để tái lập trace fingerprint, dùng package đúng implementation commit đã ghi.

**WP09.4A đạt engineering acceptance; real-fit/efficacy gate chưa đạt.**
Cập nhật29/11: [WP09.4B đã implement và nghiệm thu synthetic](wp09_calibration.md);
không cần chờ positive thật để hoàn thiện code WP09.4B/5. Bước quyết định
tiếp theo là khóa hướng nghiên cứu: giữ workload này để báo cáo false consensus và
compute cost, hoặc thiết kế một workload mới có positive/negative ở mức nhóm câu
độc lập. Mọi thay đổi benchmark phải có protocol riêng; không dùng threshold hay
evaluation đã giữ lại để cứu gate, và không tự thêm vòng SFT.
