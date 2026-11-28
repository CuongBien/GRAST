# WP09.5 — confidence policies và threshold freeze

Ngày29/11/2026 đã hoàn tất phần engineering: raw/calibrated confidence threshold,
CGES-style, chọn và lưu/nạp threshold trên synthetic validation, rồi replay trên
evaluation fixtures tách biệt. **Model thật chỉ diagnostic; chưa có real calibrator,
real threshold được chọn hoặc held-out model evaluation.**

Implementation/config commit trước nghiệm thu:
`88b238f5c7030255fcb1d9b3d7df8248371cfcf7`.
Config: `configs/evaluation/wp09_policies.json`; protocol
`prefix-threshold-cges-fixed-capacity-unseen-slots-v1`.
Đây là engineering tiếp theo [WP09.4B](wp09_calibration.md), không bị chặn bởi pilot
thật thiếu positive. Hyperparameters/grid cố định sau các kết quả WP09.2–4;
không tuyên bố preregistration trước pilot hay efficacy mới.

## Policy và score semantics

`reasoning/confidence_stopping.py` nhận observation chỉ có prefix hiện tại.
Mỗi policy có min_draws2/4, max_draws16 và threshold thuộc {0,50;0,75;0,90;0,95;0,99}.
Rule dừng dùng score≥threshold; nếu không đạt thì fallback khi hết budget.
Latest candidate phải có numeric answer hợp lệ để kích hoạt confidence stop;
invalid vẫn tốn draws/tokens và không tự kích hoạt dừng. All-invalid đến cuối trả null.

| Kind | Score | Đáp án trả về |
| --- | --- | --- |
| raw_threshold | Current-prefix agreement, chưa calibrated | Plurality, tie earliest |
| calibrated_threshold | P(current-prefix winner đúng), từ artifact WP09.4B | Plurality, tie earliest |
| cges_lns | Pseudo-posterior từ response LNS và fixed slots | Observed answer có posterior cao nhất, tie earliest |

Calibrated policy pin artifact fingerprint và context checkpoint/tokenizer/domain.
Không dùng synthetic calibrator với model context. Prefix calibrator dự đoán winner
của cả prefix, **không** dùng nó như confidence của từng response trong CGES.

### CGES-style: nguồn và adaptation

[CGES v1, Algorithm1–2](https://arxiv.org/html/2511.02603v1) dùng likelihood Cₜ khi
response khớp giả thuyết đáp án, và (1−Cₜ)/(M−1) cho giả thuyết khác; normalize và dừng
theo score lớn nhất. LNS geometric lấy exp(mean token logprob). Đây là nguồn cho
scoring routine; repo không tái lập toàn bộ paper hoặc sử dụng kết quả paper làm
bằng chứng hiệu quả cho model GRAST.

Adaptation riêng cho free-form numeric answers:

- M=17 slots cố định trước chạy, không phải số unique answers đọc từ full bank.
  M lớn hơn budget16 nên luôn còn unseen slots.
- Bắt đầu uniform prior. Khi thấy answer mới, gán một slot chưa quan sát; mọi unseen
  slot giữ cùng likelihood của các mismatch trước đó. Không biết tên đáp án tương lai.
- Sau khi factor phần likelihood chung, score log của answer a là tổng
  `log(Cₜ*(M−1)/(1−Cₜ))` cho các response đã sinh a. Unseen slot có score log0.
  Denominator bao gồm từng observed answer và `(M−observed_classes)` unseen slots.
- Cₜ là exp(raw-model mean token logprob) cho cả response gồm EOS/format/reasoning,
  loại padding; clip vào [1e−6,1−1e−6] và tính normalization trong log space.
- Invalid response không thêm evidence nhưng vẫn tính chi phí/budget. Nếu unseen
  slots có score cao hơn mọi observed answer, không bịa đáp án chưa sinh; budget
  fallback chọn observed answer tốt nhất hoặc null nếu chưa có numeric answer.

Đây là **online fixed-capacity adaptation**, không phải theorem-backed posterior
correctness. M17 là giả định hữu hạn cho miền đáp án số tự do; LNS chưa calibrated,
correct answer có thể vắng mặt, và assumptions của paper chưa được xác nhận cho bank.
Một lớp observed không bị normalize thành confidence1 chỉ vì chưa thấy lớp khác.

## Synthetic threshold selection và evaluation

Calibrator synthetic được fit lại theo spec/fit fixtures WP09.4B, có 128 fit groups.
Sau đó tạo **64 synthetic threshold groups**,3 seeds/group,K16. Mỗi nhóm thuộc một
trong4 family tự dựng: easy-correct, sai đầu rồi sửa, luôn sai tự tin và minority-correct
có LNS cao. Token counts/logprobs đều authored fixtures, không phải model measurements.

Grid30 confidence policies +8 sampled baselines tạo7.296 threshold decisions.
Chỉ selector được đọc threshold correctness; policy không đọc gold/future samples.
Criterion khóa trước: trong từng kind, chọn mean consumed-prefix tokens thấp nhất
với accuracy≥fixed-K16 accuracy−0,05. Tie: accuracy cao hơn, threshold cao hơn, rồi ID.
Nếu không có policy đạt, export trạng thái fallback fixed-K16. Đây là ràng buộc
**net accuracy**, không bảo đảm không có từng câu bị dừng sai.

`frozen_selection.json` lưu policy đầy đủ, rule, threshold replay fingerprint,
metrics SHA và calibrator fingerprint; có fingerprint bao phủ payload. Loader pin
fingerprint và expected scope, kiểm tra kind/calibrator. Chỉ **sau khi lưu/nạp freeze**
mới tạo64 synthetic evaluation groups không giao fit/threshold. Evaluation chỉ chạy
8 baselines +3 policy đã chọn:2.112 decisions; không chạy grid để chọn lại bằng evaluation.

| Policy đã khóa trên synthetic threshold | Synthetic eval accuracy | Mẫu TB | Tiết kiệm prefix / batch4 |
| --- | ---: | ---: | ---: |
| Fixed-K16 | 50% | 16 | 0% / 0% |
| Raw threshold min4/t75 | 50% | 5,00 | 68,74% / 68,74% |
| Calibrated threshold min2/t50 | 50% | 14,91 | 6,85% / 5,61% |
| CGES-style LNS min2/t90 | 50% | 2,75 | 82,81% / 68,76% |

Không suy rộng bảng synthetic sang model thật. Ví dụ CGES-style có48/192 avoidable
wrong early stops so fixed-K16; đồng thời cứu48 minority-correct trials nên net
accuracy vẫn50%. Aggregate accuracy che khuất sự đánh đổi giữa các family.
Artifact/calibrator chỉ có domain synthetic; group IDs/prefixes không là dữ liệu dân số
độc lập chỉ vì được đặt tên khác nhau. Scored rows chứa difficulty/future-correct
diagnostics để phân tích **sau** committed decisions.

Selection fingerprint:
`6688b0e1dd0d0adbbd44f94893d35d1736d0dbf73bba38774ec2781c9e74e40c`.
API nạp:

```python
from llm_scratch.research.confidence_policies import load_selection

policies = load_selection(
    "<run>/artifacts/synthetic/frozen_selection.json",
    expected_fingerprint="6688b0e1dd0d0adbbd44f94893d35d1736d0dbf73bba38774ec2781c9e74e40c",
    expected_scope="synthetic_only",
)
```

Loader cũng có scope `model_validation` gắn `calibration_threshold` cho một lượt thật
đủ điều kiện sau này; artifact synthetic hiện tại bị từ chối khi expected scope là model.

## Bank thật:29 phương án, không chọn threshold

Giữ nguyên bank20 calibration_fit questions/3 seeds/K16. Commit480 sampled baseline
decisions và20 greedy completions riêng, cùng1.200 confidence decisions **trước khi**
đọc canonical references20 câu. Replay10 raw-threshold +10 CGES-style policies trong
grid đã khóa. **10 calibrated policies không chạy** vì chưa có real calibrator;
không thay bằng synthetic artifact hoặc gọi raw scores là calibrated.

Có29 phương án,1.700 decisions; tất cả accuracy0%. Greedy có20 observations,
mỗi sampled policy có60; metrics trung bình seeds trong câu rồi đều theo20 câu.
Không chọn phương án thắng trên pilot để mở evaluation hoặc gọi preset là learned threshold.

| Phương án minh họa, không phải policy được chọn | Mẫu TB | Tiết kiệm prefix / batch8 | Dừng sớm sai |
| --- | ---: | ---: | ---: |
| Unanimous-min2 (WP09.3) | 4,10 | 74,64% / 42,64% | 51/60 |
| Raw min2/t90 | 3,80 | 76,55% / 42,64% | 54/60 |
| Raw min4/t99 | 6,80 | 57,85% / 38,57% | 46/60 |
| CGES-style min2/t90 | 2,17 | 86,49% / 49,97% | 60/60 |
| CGES-style min4/t99 | 4,02 | 74,88% / 49,97% | 60/60 |

Bank không có correct candidate nên avoidable-stop regret so full-K16 bằng0, dù
wrong early stops rất nhiều. CGES-style score cao đi cùng đáp án sai; không báo cáo
score đó là xác suất đúng hay claim cải thiện chất lượng. Prefix cost là sequential
counterfactual; batch cost tính đủ native batch đã khởi chạy, gồm phần chưa đọc.
Prompt tokens báo riêng, padded slots chỉ proxy. Full acquisition đã trả;
replay không hoàn lại chi phí cũ và chưa đo latency/VRAM online.

## Nghiệm thu, chạy lại và bàn giao

```powershell
python -u -m llm_scratch.research.confidence_policies `
  --config configs/evaluation/wp09_policies.json

# Engineering acceptance không cần bank/checkpoint/GPU:
python -u -m llm_scratch.research.confidence_policies `
  --config configs/evaluation/wp09_policies.json --synthetic-only
```

Config nguồn WP09.3/4B được pin bằng canonical JSON fingerprints. Logs luôn in tiến độ;
Run lưu config snapshot/metadata. `synthetic/` có calibrator, threshold/evaluation replay,
analysis và frozen selection. `real/` có baseline/confidence replay và comparison29 methods.
Gold chỉ ở scored analysis; replay payload chỉ có manifest/decisions. Loader tính lại
từng quyết định từ đúng prefix để từ chối sửa row dù attacker tính lại hashes.

Source run:
`artifacts/wp09-5/runs/20261128T193357318646Z-wp09-5-confidence-policies-seed42-a17f0e5b`.
Tên run dùng UTC; local date29/11/2026. Real confidence replay fingerprint:
`6f95a50e302c9dfd9d4fdbc5e0ed38bebe88037f9d6973500345c4a7bc5703e2`.
CSV accuracy–token và chart PNG/SVG có riêng cho synthetic evaluation và real diagnostic.
Charts đã được kiểm tra trực quan; adaptive figure height giữ29 labels dễ đọc.
Local:1.001 tests qua trong229,04 giây, gồm8 tests mới WP09.5; Ruff lint/format và
wheel/sdist build qua. Wheel ngoài source tree và ZIP giải nén sang thư mục mới đều
chạy khi cấm checkpoint loading, replay/selection/reports khớp từng byte.
Receipt: `experiments/wp09_5_validation_20261129.json`; portable kit trong `dist/`,
không có weights. Dùng đúng implementation commit để tái lập source/replay fingerprints.

**WP09.5 engineering đã đạt.** WP09.6 có thể tổng hợp toàn bộ evidence/ca lỗi và báo cáo
giới hạn; không đánh dấu real-calibration/held-out efficacy đã nghiệm thu. Phần thật
vẫn chưa fit calibrator, chưa chọn threshold trên40 reserved questions và chưa mở100
evaluation questions, không train hoặc sinh thêm model samples.
