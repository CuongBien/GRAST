# WP07.6.3 — Nghiệm thu model curriculum và validation

Nhận `selected_model.pt` và summary ZIP của lượt user train đủ recipe WP07.6.2.
Evaluator chỉ đọc validation; không train, chọn lại checkpoint hoặc fit threshold.
Config `configs/evaluation/wp07_curriculum_selected.toml` khóa weights/summary/baseline
SHA256 trước khi đo. Loader kiểm finite tensors, model/tokenizer/tied head và checksum
của selected weights; archive audit kiểm counters, token counts, loss scope, toàn bộ
generated predictions/scores và quy tắc macro EM cao nhất, hòa giữ mốc sớm hơn.

Model gốc đã được bố trí tại
`artifacts/checkpoints/wp07-curriculum-2800-selected-2200/selected_model.pt`.
Summary/provenance nằm trong `evidence/` cạnh model. Không ghi đè SFT500 trước curriculum.
Selected export dùng inference; không có optimizer để exact resume training.

Ba scope được đo với cùng model được chọn:

- Replay48 câu đã dùng selection: kiểm tra inference reload ở FP32 so với predictions
  lưu trong summary. Chênh lệch token được ghi rõ; không đổi checkpoint để khớp kết quả.
- Additional validation96:32 câu/task, cùng strata authored/WP03 như selection,
  loại **toàn bộ groups** xuất hiện trong selection48 trước khi chọn theo hash(seed, ID).
  Group là unordered pair/leaf multiset cho authored, semantic AST cho bridge WP03.
- WP03 fixed pilot validation150:50/easy/medium/hard. Báo cáo cả150 và subset không
  trùng prompt selection48; số câu overlap được công khai. Đối chiếu artifact SFT500
  validation WP07.5 đã đóng băng trên cùng IDs, không sinh lại baseline hoặc đọc test.

Các scope đều là validation. Toàn bộ validation đã được dùng đo teacher-forced loss
trong training; additional96 chỉ tách khỏi bộ dùng chọn checkpoint bằng generated EM.
Không gọi đây là held-out test hoặc cộng các scope thành một accuracy chung độc lập.

Generation greedy/B8/FP32/max256, BOS-question-LF-think prefix cũ, cached với left
padding và private seed42. Trước đo có spot-check8 tokens cached/uncached. Model-only
predictions được ghi trước scoring; pilot references chỉ được mở sau lưu predictions.
Numeric final dùng scorer WP07.5, copy dùng strict ordered integer-pair parser.
Invalid/malformed giữ trong denominator. Final đúng không tự chứng minh trace đúng.

```powershell
uv run --no-sync python -u scripts/evaluate_curriculum.py --device cuda --dataset-root PATH/TO/wp03-datasets --curriculum-root artifacts/wp07-6-2/curriculum_data --tokenizer PATH/TO/wp04-tokenizer
```

Mỗi run tạo config/source/Git/environment/log/protocol; giữ archive audit, IDs,
raw/scored predictions cho cả ba scope, replay differences, cache equivalence,
comparative metrics và các ca pilot đúng/sai. Kiểm lại hash weights/summary sau chạy.
CLI không có option test hoặc adaptive policy. Bản smoke kiểm thử dùng tiny authored
checkpoint và fixture; không coi điểm fixture là chất lượng model46M.
