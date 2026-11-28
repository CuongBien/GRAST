# WP09.3 — diagnostic baseline replay

Ngày 28/11/2026 đã replay greedy riêng, fixed-K và unanimous-prefix trên bank
[WP09.2](wp09_pilot.md). **Engineering acceptance đạt; efficacy/calibration gate vẫn
không đạt.** Không sinh thêm candidate, không train, không fit calibrator và không
chấm calibration_threshold/evaluation.

## Input và phương pháp

Bank có 20 câu calibration_fit (10 boundary-100, 10 cross-medium), seeds 42/43/44,
K16; tổng 960 sampled candidates. Greedy có 20 completions riêng, không coi sampled
K1 là greedy. Mọi sampled policy dùng cùng thứ tự candidate, cùng checkpoint selected900.

Config: `configs/evaluation/wp09_baselines.json`; implementation/config commit trước
replay: `a14bc34236c721a8d58bcd2a35dfd967adb82f29`.
Grid K đã có trong kế hoạch WP09; định nghĩa unanimous-min2/3/4 dưới đây được chốt
**sau khi biết kết quả pilot WP09.2**, không tuyên bố là preregistration trước pilot.
Không chọn phương án thắng hoặc sửa grid theo kết quả replay.

- Fixed-K = plurality của K mẫu, K ∈ {1,2,4,8,16}. Tie chọn lớp xuất hiện trước.
- Unanimous-minN: đọc lần lượt prefix, dừng khi n≥N và tất cả n câu trả lời đồng ý,
  N ∈ {2,3,4}; nếu không thì dùng đủ 16. Đây không phải run-length consensus:
  một prefix đã bất đồng sẽ không trở lại unanimous bằng cách bỏ mẫu cũ.
- Invalid abstain nhưng vẫn nằm trong denominator; không tự kích hoạt stop.
- Commit quyết định trước khi scorer đọc references. Policy chỉ nhận prefix không
  có gold; future answers và batch costs chỉ dùng sau đó để chẩn đoán.

Có 480 sampled decisions (8 policies × 60 câu–seed) và 20 greedy decisions.
Báo cáo trung bình các seed trong từng câu trước, rồi lấy trọng số bằng nhau cho
20 câu. Không so tổng token của greedy 20 lượt với sampled 60 lượt.

## Kết quả thực đo từ bank

Các cột token là **trung bình trên câu**, đã trung bình seed. Tất cả phương án đều
0% accuracy: greedy 0/20, mỗi sampled policy 0/60. Số câu độc lập vẫn chỉ là 20.

| Policy | Mẫu TB | Prefix tokens | Batch tokens | Tiết kiệm prefix / batch so K16 | Dừng sớm sai |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy riêng | 1,00 | 24,35 | 24,35 | 93,74% / 93,74% | — |
| Fixed-K1 | 1,00 | 24,32 | 194,73 | 93,75% / 49,97% | — |
| Fixed-K2 | 2,00 | 48,67 | 194,73 | 87,50% / 49,97% | — |
| Fixed-K4 | 4,00 | 97,40 | 194,73 | 74,98% / 49,97% | — |
| Fixed-K8 | 8,00 | 194,73 | 194,73 | 49,97% / 49,97% | — |
| Fixed-K16 | 16,00 | 389,27 | 389,27 | 0% / 0% | — |
| Unanimous-min2 | 4,10 | 98,70 | 223,30 | 74,64% / 42,64% | 51/60 |
| Unanimous-min3 | 5,60 | 135,13 | 232,93 | 65,29% / 40,16% | 48/60 |
| Unanimous-min4 | 6,80 | 164,07 | 239,13 | 57,85% / 38,57% | 46/60 |

Ở mỗi lần consensus dừng sớm, observed vote share bằng 1,0 nhưng đáp án sai.
Đây là **raw agreement**, chưa phải xác suất đúng đã hiệu chuẩn. Năm câu lỗi khác
nhau được lưu ở `analysis/error_cases.json`; ví dụ 105−13 → 12 (đúng: 92),
89−1 → 8 (88), 109−53 → 26 (56). Labels chỉ có trong analysis, ngoài bank/replay.

Không có trường hợp dừng sớm làm mất đáp án đúng phía sau, và không có trường hợp
full-K16 đúng nhưng early-stop sai: cả hai số đếm đều 0. Cần phân biệt điều này với
51/48/46 **wrong early stops**: oracle coverage@16 bằng 0/60 nên chờ thêm cũng không
thể lấy được đáp án đúng từ bank này. Không quy toàn bộ sai số cho stopping policy.

## Cách tính chi phí và giới hạn

1. **Prefix tokens:** valid generated tokens gồm EOS của các mẫu policy đã đọc.
   Đây là counterfactual sequential proxy, giả sử chỉ sinh đến prefix đó.
2. **Batch tokens:** tính toàn bộ valid generated tokens của mỗi native batch đã
   khởi chạy. Bank được sinh theo batch 8, nên dừng sau 2 mẫu vẫn phải trả batch 8.
   Fixed-K1/2/4 được tính theo cùng layout này; không có nghĩa triển khai online
   của chúng bắt buộc phải sinh thừa. Đổi batch size có thể đổi RNG/completions.
3. **Padded generation slots:** batch size × completion dài nhất trong từng batch
   đã khởi chạy. K16 trung bình 390,67 slots/câu; unanimous-min2 là 224,00.
   Đây là proxy cho layout decode, không phải FLOPs, latency hay VRAM đo online.

Greedy dùng batching qua câu (8/8/4) của lượt sinh riêng; trung bình 26 padded slots.
Prompt tokens của prefix/batch được lưu riêng trong report và per-question rows;
không cộng vào generated-token savings. Các tỷ lệ là ratio of means so full-K16
trên đúng tập câu. Chưa mô hình hóa KV/prefill hoặc overhead policy.

Chi phí acquisition đã trả: 23.356 sampled generated tokens + 487 greedy tokens.
Replay không hoàn lại chi phí này. Không tuyên bố tốc độ hoặc chất lượng tốt hơn
từ việc bỏ bớt các câu trả lời đều sai. Không extrapolate pilot thành test accuracy,
không xem 60 seeds/trials hoặc các prefix là 60 câu độc lập.

## Chạy lại và bằng chứng

```powershell
python -u -m llm_scratch.research.confidence_baselines `
  --config configs/evaluation/wp09_baselines.json
```

Cần bank/input kit WP09.2 ở các đường dẫn trong config, không cần model checkpoint
hoặc GPU. Log in tiến độ và run directory; mỗi run mới có config snapshot/metadata,
`artifacts/replay/` chưa gắn labels và `artifacts/analysis/` sau scoring. Config khóa
fingerprint bank/plan/generation và SHA của greedy/references, không tự tìm bank mới.

Source run:
`artifacts/wp09-3/runs/20261128T131048991509Z-wp09-3-diagnostic-baselines-seed42-f6261240`.
Bank fingerprint:
`8752dbb5c275be1eeaed2b4d5cc431c4a9967cb34b03072f4506a5cf633e28b7`.
Sampled replay fingerprint:
`8c592ba7dc87e2626b8bfcea48cb77ec2d1a28d496f7c4cf3c734cd694ae3eeb`.

Wheel cài ngoài source tree chạy lại với checkpoint loading bị cấm: quyết định,
report, per-question rows, CSV và error cases khớp từng byte. Kiểm thử bao phủ
gold-free decision boundary, batch charging, future-sample independence, chuẩn hóa
theo câu, greedy riêng và từ chối quyết định greedy bị sửa.
Local: 966 tests qua trong 292,33 giây; Ruff lint/format và wheel/sdist build qua.
ZIP đã được giải nén sang thư mục mới và replay lại, kết quả khớp từng byte.

Receipt: `experiments/wp09_3_validation_20261128.json`. Portable replay kit trong
`dist/` chứa input bank, question-only pilot inputs, references tách riêng, outputs,
config, report và chart; không chứa weights. Raw bank/labels/replay và kit giữ local.

WP09.3 hoàn thành trong phạm vi diagnostic. WP09.4/5 chưa được mở theo hướng fit
calibrator/threshold trên dữ liệu một lớp. Muốn đo efficacy cần một workload có cả
đáp án đúng và sai trên các nhóm câu tách biệt, rồi khóa lại protocol trước evaluation.
