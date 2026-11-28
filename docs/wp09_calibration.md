# WP09.4B — calibration engine và synthetic verification

Ngày 29/11/2026 đã hoàn tất phần triển khai calibration: fit logistic bằng NumPy,
dự đoán từ observable prefix, save/load artifact và kiểm thử dữ liệu tổng hợp hai lớp.
**Engineering acceptance đạt; chưa fit calibrator trên model thật, chưa chọn threshold.**
Việc thiếu positive trong pilot không ngăn nghiệm thu implementation.

## Contract và thuật toán

Module: `reasoning/calibrators.py`; runner: `research/confidence_calibration.py`.
Implementation/config commit trước nghiệm thu:
`f3202408977a945267aee89f69bf2322deae753e`.
Config: `configs/evaluation/wp09_calibration.json`, protocol
`group-weighted-l2-logistic-prefix-v1`. Đây là multifeature logistic calibration,
không tuyên bố là thuật toán mới hay bản sao một phương pháp calibration trong paper.

Feature order cố định:

1. Agreement của current-prefix winner.
2. Answer entropy chuẩn hóa H/ln(n), có invalid bucket như WP09.4A.
3. Mean raw-model logprob/token của các candidate thuộc observed winner class.
4. ln(n)/ln(16), chỉ dùng kích thước prefix đã quan sát.

Prefix n=1 và prefix không có canonical winner không đủ điều kiện, trả `None`;
không gán confidence giả cho abstention. `features_from_prefix` chỉ đọc observation
đã cắt, không nhận labels, references hoặc future candidates. Domain hỗ trợ n=2..16.

Fit tối thiểu hóa weighted binary log loss + (λ/2)‖β‖² với λ=0,01; không regularize
intercept. Center/scale được học **chỉ từ calibration_fit**, với cùng sample weights
như objective; feature có độ lệch chuẩn <1e−12 dùng scale=1. Newton FP64 có analytic
gradient/Hessian và Armijo backtracking; tối đa100 iterations, gradient∞≤1e−8.
Sigmoid ổn định với logits lớn. Không hội tụ thì từ chối export artifact.

Trọng số: đều theo group → đều theo question trong group → đều theo seed trong
question → đều theo prefix trong seed. Câu dài hoặc nhiều seed/prefix không tự tăng
trọng số nhóm. Duplicate question–seed–prefix và một question thuộc nhiều group bị
từ chối. Trong pilot thật hiện tại, mỗi question là một group đã tách ở WP09.1.

`fit_calibrator` chỉ nhận examples có partition `calibration_fit`. Từ chối
`calibration_threshold`, `evaluation`, synthetic assessment hoặc trộn partition.
Yêu cầu ít nhất5 groups có positive và5 có negative; một group có thể thuộc cả hai
nếu winner thay đổi giữa prefix/seed. Đây là necessary engineering screen, không
bảo đảm đủ cỡ mẫu thống kê hoặc thay thế gate reasoning-trace của WP09.2.
Một lớp bị từ chối, không export calibrator hằng rồi gọi đó là hiệu quả calibration.

## Artifact và prediction boundary

Artifact JSON bất biến lưu feature order/protocol, fitted center/scale, coefficients,
intercept, fixed spec, convergence, fingerprint của fit examples và danh sách
group/question để kiểm tra overlap. Không lưu gold answers hoặc correctness theo câu.
Fingerprint bao phủ toàn bộ payload. Loader kiểm tra schema, feature order, tham số
hữu hạn/scale dương/convergence và fingerprint do caller khóa; không ghi đè file cũ.

`CalibrationContext` phân biệt `synthetic` với `model_validation`; model context bind
checkpoint SHA và tokenizer fingerprint. Prediction yêu cầu context khớp artifact.
Calibrator synthetic bị từ chối khi caller cung cấp model context; không tự fallback.
`predict_inputs` chỉ chấp nhận identity + observable features, từ chối gold/unknown
fields. Predictions được commit trước khi gắn correctness để tính assessment metrics.
`require_disjoint_assessment` kiểm tra cả group và question không giao với fit.

Ví dụ nạp artifact synthetic đã kiểm chứng:

```python
from llm_scratch.reasoning.calibrators import CalibrationContext, load_calibrator

calibrator = load_calibrator(
    "<run>/artifacts/synthetic/calibrator.json",
    expected_fingerprint="4c69505427e0f78804d7b588834e39d02c143948da33a8e865811ca719b7e09e",
)
# observation phải là prefix; không truyền labels hoặc references.
confidence = calibrator.predict_prefix(
    observation, context=CalibrationContext("synthetic")
)
```

Đây chỉ là API example cho synthetic context. Khi có model-validation data đủ điều
kiện, dùng `fit_calibrator` với context checkpoint/tokenizer thật và protocol đã khóa;
không dùng artifact synthetic để quyết định stopping của model thật.

## Nghiệm thu synthetic

Fixture `authored-two-outcome-overconfidence-disjoint-groups-v1`: 128 fit groups và
128 assessment groups không giao, 3 seeds, prefixes2/4/8/16, mỗi split1.536 rows.
Mỗi split có44 positive/84 negative groups. Features và labels được **tự dựng để kiểm
thử**, không phải generations, model outputs hoặc benchmark kết quả GRAST.
Seed20261129, λ, grid và fixture đã khóa trước run, không chọn bằng assessment metrics.

| Synthetic held-out metric | Raw agreement | Calibrated |
| --- | ---: | ---: |
| Brier | 0,435656 | 0,203343 |
| Log loss | 7,930035 | 0,591552 |
| ECE,10 fixed bins | 0,478394 | 0,019146 |

Accuracy của authored examples là34,375% và không đổi: calibrator thay score, không
đổi đáp án. Metrics theo cùng group weights; prefixes không phải mẫu độc lập.
Log loss clamp xác suất ở endpoint bằng1e−15 khi tính log; raw score1 trên label0
làm raw log loss rất lớn. Các số trên chỉ chứng minh pipeline hoạt động trên fixture
overconfidence đã dựng, không chứng minh cải thiện calibration hoặc accuracy model thật.

Optimizer hội tụ sau4 Newton updates, gradient∞≈8,20e−10. Artifact fingerprint:
`4c69505427e0f78804d7b588834e39d02c143948da33a8e865811ca719b7e09e`.
Predictions sau save/load và sau fit lặp khớp exact trên môi trường xác định.

## Pilot thật: kiểm tra điều kiện, chưa fit

Runner xác minh bank/plan calibration_fit đã khóa, commit trace observable mới rồi
mới đọc references20 câu. 900 eligible prefixes từ n≥2: 0 positive rows/groups,
900 negative rows,20 negative groups. Report ghi
`blocked_insufficient_positive_negative_groups`; **không có real calibrator artifact**.

Không sinh thêm mẫu, train, fit model calibrator, chấm calibration_threshold/evaluation
hoặc tìm stopping threshold. Runner nghiệm thu luôn chỉ scan điều kiện trên bank thật;
API fit đã sẵn sàng cho một lượt thực nghiệm có protocol/dữ liệu phù hợp về sau.
Raw bank/trace giữ gold-free; labeled fit examples nằm ngoài bank và trace.

## Chạy và bàn giao

```powershell
# Synthetic verification và kiểm tra điều kiện của pilot thật đã có local:
python -u -m llm_scratch.research.confidence_calibration `
  --config configs/evaluation/wp09_calibration.json

# Chỉ nghiệm thu implementation, không cần bank, weights hoặc GPU:
python -u -m llm_scratch.research.confidence_calibration `
  --config configs/evaluation/wp09_calibration.json --synthetic-only
```

Luôn in log và lưu config snapshot/metadata. Source config WP09.4A được khóa theo
canonical JSON fingerprint, độc lập với CRLF/LF. Calibrator/reports trong
`<run>/artifacts/synthetic/`; trace/examples/support của model thật trong
`<run>/artifacts/real-eligibility/`. Không ghi output vào bank.

Source run:
`artifacts/wp09-4b/runs/20261128T190607709203Z-wp09-4b-calibration-engine-seed42-e9253802`.
Tên run theo UTC; ngày local29/11/2026. Wheel cài ngoài source tree, cấm checkpoint
loading, chạy lại khớp từng byte cho calibrator, predictions, metrics, real support
và trace. Chart PNG/SVG đã được kiểm tra trực quan.
Local: 993 tests qua trong207,98 giây, gồm14 tests mới cho calibrator/runner;
Ruff lint/format và wheel/sdist build qua. ZIP đã được giải nén sang thư mục mới và
chạy lại từ wheel, kết quả khớp từng byte trên cùng môi trường.
Receipt: `experiments/wp09_4b_validation_20261129.json`.
Kit portable trong `dist/` gồm bank/question-only inputs, references tách riêng,
source outputs và portable configs; không có weights. Dùng đúng implementation
commit để tái lập source/trace fingerprints; không bảo đảm bitwise giữa mọi BLAS/OS.

**WP09.4B đã hoàn tất phần engineering. [WP09.5 đã implement và kiểm thử](wp09_policies.md)**
policy/threshold trên fixtures có hai outcome classes. Bank thật tiếp tục chỉ
cho diagnostic raw-score replay; nghiệm thu efficacy của calibration/stopping vẫn
phụ thuộc dữ liệu có cả đúng và sai, không dùng reserved evaluation để cứu gate.
