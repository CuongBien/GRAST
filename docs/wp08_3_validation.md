# WP08.3 — Offline replay, 26/11/2026

**Đạt gate kỹ thuật WP08.3:** tám policy,1.440 quyết định trên180
question/checkpoint/seed groups, dùng nguyên bank WP08.2. Hai lần replay cho bytes,
history, token cost và fingerprint giống hệt nhau. Không sinh mẫu thật mới hoặc tải
model weights. Quyết định không nhận gold/future candidates; scorer đọc references
sau khi mọi policy đã commit. Năng lực reasoning và hiệu quả các heuristic vẫn thấp.

[Protocol](wp08_replay.md), [config đã khóa](../configs/experiments/wp08_replay.toml),
[receipt/file hashes](../experiments/wp08_3_validation_20261126.json).
Đây là development validation30 câu dùng lại qua ba seed, không phải90 câu độc lập.
Không fit threshold/calibration hoặc chọn policy theo gold trong phase này.

## Accuracy và token cost

Accuracy là final-answer exact match, gồm invalid/abstention trong mẫu số.
Fixed-K chọn mode/plurality và phá hòa bằng answer class xuất hiện đầu tiên.
K1 là sampled candidate0, không phải greedy.

| Policy | SFT500 đúng /90 | Selected2200 đúng /90 | Token tiết kiệm SFT500 | Token tiết kiệm selected2200 |
| --- | --- | --- | --- | --- |
| K1 | 0 | 2 | 93,77% | 93,72% |
| K2 | 0 | 2 | 87,54% | 87,47% |
| K4 | 0 | 1 | 75,12% | 75,00% |
| K8 | 0 | 3 | 50,04% | 49,89% |
| K16 | 1 | 1 | 0% | 0% |
| Unanimous-min2 | 1 | 1 | 0,72% | 1,43% |
| Unanimous-min4 | 1 | 1 | 0% | 0% |
| Vote80-min4 | 1 | 1 | 0% | 0% |

Tiết kiệm là giảm **tổng generated token_count của prefix đã quan sát** so với
đủ16 mẫu của cùng tập trial. Gồm EOS, loại masked tails, giữ chi phí invalid samples.
Không phải trung bình tỷ lệ từng câu, không phải latency/VRAM/speedup và không lấy
lại chi phí GPU đã dùng thu toàn bộ bank ở WP08.2.

Selected2200 K8 đạt3/90 so với K16 đạt1/90 trên pilot này. Không chọn K8 làm kết luận
tối ưu hoặc suy rộng rằng tăng K luôn làm accuracy giảm: số ca đúng rất ít và nhiều
cấu hình đang dùng cùng dữ liệu development. Per-seed/per-difficulty được giữ trong
report/CSV; tất cả8 policy đều0 đúng ở medium/hard trong lượt đo này.

## Coverage khác với đáp án được policy chọn

Bank K16 có final đúng ở19/90 SFT500 question-trials và18/90 selected2200
question-trials. Majority K16 chỉ chọn đúng1/90 ở mỗi checkpoint. Scorer ghi18 và17
ca tương ứng đã thấy ít nhất một final đúng nhưng mode cuối cùng vẫn sai.
Không dùng oracle coverage làm accuracy của policy có thể triển khai.

Unanimous-min2 dừng sớm2/90 trial SFT500 và4/90 trial selected2200; số draw trung bình
là15,689 và15,378. Unanimous-min4 và Vote80-min4 không có lần dừng sớm nào trước16.
Những luật này gần như dùng đủ bank trên dữ liệu hiện tại.

Selected2200 có hai trial Unanimous-min2 dừng sai ở draw2 dù phần bank phía sau có
final đúng. Đây là diagnostic hậu kiểm wrong_with_later_correct; fixed-K16 cũng sai
trong cả hai trial đó. Không diễn giải thành bằng chứng rằng chỉ cần chạy tiếp thì
majority chắc chắn sửa được đáp án, hoặc early stopping đã làm mất hai đáp án đúng
so với fixed-K16. Policy không được đọc những nhãn/candidate tương lai này.

Report chỉ chấm final answer, không gọi mode answer là một reasoning trace được
verified. Trace audit của WP08.2 vẫn là phân tích riêng. Kết quả hiện tại cho thấy
bank coverage còn cách xa accuracy chọn đáp án và các agreement heuristic này
tiết kiệm rất ít token; chưa xác lập lợi ích của confidence-gated stopping.

## Verification và artifacts

**907 tests local passed**,20 test mới; Ruff/build/diff checks đạt. Wheel được chạy
smoke ngoài source tree; sdist/wheel không chứa cache, runs, weights/token packs.
Smoke offline có48 quyết định và repeated fingerprint exact. Có oracle tests cho
ties, invalid denominator, abstain, threshold boundary4/5, budget fallback, exact
token costs và posthoc diagnostics.

Noninterference tests kiểm tra: policy chỉ nhận prefix; thay future samples không
đổi quyết định đã dừng; chặn đọc reference/scored files trong create_replay; đọc gold
chỉ sau commit; thay gold chỉ đổi score. Replay reader từ chối thiếu marker, file gold,
order/grid/history/cost/field giả mạo, kể cả khi sửa lại checksums.

Implementation commit: `43201740a2086e3c896d4cd5a224174d9499076d`, Git sạch lúc chạy.
Run UTC:
`artifacts/runs/20261125T175551439263Z-wp08-3-offline-replay-seed42-daf75519`.
Replay fingerprint:
`7fa3de8093ae73094a74c82b95e74cc7545858c2df05f65c965f1fee242e6a20`.
Bank nguồn vẫn có fingerprint
`6941830812e5fe91ba583553556cbc68a10ecf9b84dcad2bb5850c97cab1c809`;
manifest và mọi payload hash khớp receipt WP08.2, không bị sửa.

Run chứa decisions/history và manifest, repeated replay, external scored report,
CSV, PNG/SVG accuracy–token cost, export script, repeat proof, config/metadata/log.
ZIP local `dist/grast-wp08-3-replay-20261126-daf75519.zip`,263.584 bytes,
SHA256 `f65e78158ad621083d2ce98f55dea219b938c23488bd8b1b9869486a031b6ce2`.
ZIP không chứa weights hoặc bank nguồn; cần giữ kèm bank WP08.2 để replay lại.

Tiếp theo là **WP08.4 — tổng hợp feasibility và quyết định benchmark/GSM8K**.
Chưa mở test hoặc quyết định thay benchmark/recipe trong WP08.3.
