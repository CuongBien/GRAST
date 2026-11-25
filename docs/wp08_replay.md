# WP08.3 — Offline policy replay

Đã chạy và kiểm tra replay; xem [báo cáo WP08.3](wp08_3_validation.md).

Replay dùng nguyên bank WP08.2, fingerprint
`6941830812e5fe91ba583553556cbc68a10ecf9b84dcad2bb5850c97cab1c809`.
Không sinh lại ứng viên hoặc tải weights. Cùng một question/checkpoint/trial seed
luôn dùng cùng prefix đã lưu, theo candidate_index0..15.

```powershell
uv run --no-sync python -u scripts/replay_policies.py
# Smoke offline tự tạo tiny bank:
uv run --no-sync python -u -m llm_scratch.research.replay_acceptance
```

Config `configs/experiments/wp08_replay.toml` khóa bank/reference SHA và8 policy:

| Policy | Quy tắc |
| --- | --- |
| fixed-k1/2/4/8/16 | Quan sát đủ K mẫu rồi chọn đáp án nhiều phiếu nhất |
| unanimous-min2 | Từ2 mẫu, dừng khi tất cả mẫu đã thấy cùng đáp án hợp lệ; cap16 |
| unanimous-min4 | Như trên, nhưng tối thiểu4 mẫu |
| vote80-min4 | Từ4 mẫu, dừng nếu top_votes / tất cả mẫu đã thấy >=4/5; cap16 |

Đây là các heuristic cố định trước lượt replay, không search threshold hoặc fit
confidence/calibration trên gold. Tỷ lệ phiếu không được xem là xác suất trả lời đúng.
Development validation và kết quả WP08.2 đã được xem trước phase này; không gọi đây
là đánh giá trên dữ liệu chưa từng được xem. WP08.4 mới quyết định benchmark.

## Contract bỏ phiếu và dừng

Vote dùng canonical numeric-v1 `extracted_answer` khi `answer_valid=True`.
Format-valid là diagnostic riêng; answer-valid có thể vẫn vote khi format không chuẩn.
Mẫu không có đáp án abstain khỏi danh sách phiếu, nhưng vẫn tính trong mẫu số
đồng thuận, số draw và token cost. Không bỏ các câu thiếu/invalid answer khỏi accuracy.

Tên majority ở đây dùng theo self-consistency: chọn mode/plurality, không bắt buộc
hơn50% phiếu. Hòa chọn answer class xuất hiện sớm nhất; representative_index là mẫu
đầu của class thắng. Không sắp xếp numeric để phá hòa, không random, không dùng gold
hoặc log-prob để chọn đáp án khi hòa. Không có phiếu hợp lệ thì abstain.

Agreement kiểm tra ngay sau mỗi mẫu, bằng phép nhân số nguyên để so threshold4/5.
Nếu đến cap mà chưa đạt agreement, chọn mode hiện tại và ghi budget_exhausted.
Fixed-K vẫn tiêu thụ đủ K, kể cả prefix nhỏ đã đồng thuận. K1 là sampled candidate0,
không phải greedy baseline. Không trộn48 ứng viên từ ba seed thành một trial.

## Tách policy và scorer

`replay_one` giữ group đầy đủ trong driver và lần lượt gọi
`evaluate_prefix(group.prefix(n), policy_spec)`. Policy chỉ nhận observation bất biến
với n ứng viên đầu và spec cố định. Không có gold, correctness, references path,
scored labels hoặc future candidates trong observation. Policy không đọc file, không
nhận bank handle và không thay RNG toàn cục. Đây là API boundary cho code hợp tác,
không phải sandbox chống Python cố tình truy cập filesystem.

Driver ghi `decisions.jsonl`, gồm answer, số draw, representative_index, stop reason,
consumed token count và history từng prefix; ghi manifest cuối cùng bằng atomic replace.
Reader kiểm tra fingerprint, checksum, grid/order và tái tính toàn bộ quyết định/history
từ bank/spec. Từ chối replay dở dang, nhãn gold, field lạ hoặc chi phí bị sửa.

**Mọi quyết định của cả8 policy phải commit trước lần đọc references đầu tiên.**
Scorer sau đó mới mở external reference JSONL đã kiểm SHA và phủ đúng30 question IDs.
Score output nằm ngoài bank và ngoài replay root. Đổi gold chỉ đổi scores; decision
bytes/fingerprint giữ nguyên. Thay các mẫu ở tương lai của một quyết định đã dừng
không làm đổi quyết định/prefix history của nó. Các điều này có kiểm thử riêng.

## Metrics và giới hạn diễn giải

Báo answer exact-match accuracy, abstention, mean draws và consumed generated tokens
theo checkpoint/policy/seed/difficulty. Ba seed dùng lại cùng30 câu; báo per-seed và
mean/min/max, không gọi90 question-trials là90 câu độc lập. Không chọn policy tốt nhất
từ pilot để thay cho toàn bộ bảng kết quả đã định.

Token cost là tổng generated token_count của prefix đã tiêu thụ, gồm EOS, loại masked
tails; invalid samples vẫn tốn token. So với đủ16 mẫu trên cùng trial, report token
savings fraction. Chi phí thu toàn bộ bank đã phát sinh trong WP08.2; offline replay
không tiết kiệm lại GPU đã dùng. Không gọi tỷ lệ token tiết kiệm là speedup/latency/VRAM.

Hai diagnostic chỉ tính sau khi policy quyết định:

- wrong_with_later_correct: policy sai/abstain và có mẫu final đúng ở phần còn lại
  của bank đến K16; áp dụng cả fixed-K nhỏ và agreement, không tự động suy ra một
  policy cụ thể chắc chắn sẽ chọn được mẫu đúng nếu chạy lâu hơn.
- wrong_despite_observed_correct: đã quan sát ít nhất một final đúng nhưng vote chọn sai.

Oracle coverage@16 được ghi để so sánh tiềm năng bank, không phải policy khả thi.
Replay hiện chấm final answer; không gọi mode answer là một reasoning trace đã verified.
Không fit threshold, không đọc test, không chạy lại model hoặc quyết định GSM8K.

Outputs dưới `artifacts/runs/`: config/source snapshots, metadata/log,
`artifacts/replay/{manifest.json,decisions.jsonl}` và
`artifacts/evaluation/{scored.jsonl,report.json}`. Logs in ngay theo tiến độ group.
