# WP07.5 — Checkpoint capability và fixed pilot

Evaluator so sánh model sau pretraining và sau CoT SFT bằng cùng câu hỏi, template,
generation budget và cách chấm. Kết quả cần đọc cùng text model sinh ra: validation
loss của teacher forcing cho biết khả năng dự đoán token khi đã có prefix tham chiếu,
còn pilot đo đáp án của chuỗi model tự sinh.

## Checkpoints và nguồn dữ liệu

Inventory mặc định trong `configs/evaluation/wp07_capability.toml` gồm hai checkpoint
thật do user cung cấp: pretraining 2.000 updates và SFT 500 updates. Mỗi entry khóa
label, stage, step, SHA256 và path. Loader kiểm architecture, tokenizer, tensor finite,
tied embedding/head, history hash và counters; chỉ nạp weights để inference. Optimizer,
scaler và training RNG không được restore. Khởi tạo model không dịch RNG của caller.

SFT checkpoint đã sắp tại `artifacts/checkpoints/wp07-cot-sft-math-500/final.pt`, SHA256
`5f75909261c79e9f7f0c6e37a7e9d76ba5670be8b4be5133b6e148e0cd06ff42`.
Provenance và summary giữ cạnh weights; [evidence SFT hoàn tất](../experiments/wp07_sft_completed_20261120.json)
ghi 500 updates, 0 skips, 253.174 targets, validation loss 0,5430620898 và target PPL
1,7212694825. Parent SHA khớp checkpoint pretraining 2.000 đã nhận trước đó.

History bên trong checkpoints cung cấp curves toàn bộ 2.000/500 updates. Capability
hiện so hai mốc có weights thật. Muốn so các mốc nội bộ 50/100/250 hoặc pretrain
100/500/1.000, thêm chính file checkpoint đó vào inventory; history loss không tái tạo
được weights trung gian. SFT parent phải nằm trong inventory để kiểm lineage.

Pilot giữ nguyên WP03: 150 validation và 150 test, mỗi split 50 câu mỗi mức
easy/medium/hard. Manifest, IDs, file hashes, counts và strata được kiểm trước chạy;
duplicate IDs/questions bị từ chối. Đọc đúng `pilot/*.questions.jsonl` và scorer-only
`pilot/*.references.jsonl`. GSM8K chưa nằm trong phép đo pilot arithmetic này.

## Protocol đã định trước

Primary là greedy K=1, tối đa 256 generated tokens, batch 8, FP32, seed 42,
deterministic và TF32 off. Không sampling/voting/confidence/calibration/threshold ở
primary pilot. Snapshot và protocol digest ghi cả implementation hashes, dataset,
tokenizer, checkpoint SHA, template, batching, decoding và answer protocol.

Prefix dùng đúng `encode_reasoning_prompt` WP04/SFT: BOS, nguyên question, newline,
`<think>`. Các câu được left-pad theo thứ tự file, batch size cố định. Valid prompt
và generated tokens liền nhau; RoPE dùng physical columns theo contract WP06. Không
truncate prompt, giảm budget hoặc đổi context khi vượt giới hạn.

Sampler chỉ nhận question schema; không nhận gold/rationale. Predictions gồm raw
generated IDs/text, finish reason, question ID/difficulty và prompt-ID hash, được lưu
trước khi scorer đọc references. Không lấy câu hỏi/prefix/gold để tìm đáp án dự đoán.

Primary exact match yêu cầu đúng một cặp **special token IDs** `<final>`/`</final>`
theo thứ tự trong continuation, canonical numeric-v1 trùng gold. Không đoán số cuối,
không fallback bare number và không sửa output thiếu markers. Decimal/fraction tương
đương được normalize chính xác bằng rational; malformed/invalid là sai, vẫn nằm trong
denominator. Literal marker strings bằng byte tokens không thay thế framing IDs.

`answer_valid_rate` đếm final numeric hợp lệ; `format_valid_rate` kiểm thêm có token
trước `</think>`, adjacent final block và EOS kết thúc. Final hoàn chỉnh ở
token budget cuối có thể chấm đúng đáp án nhưng thiếu EOS nên format chưa hoàn chỉnh.
Scorer không xác minh logic của từng dòng reasoning. Báo cáo overall và từng mức gồm
EM, valid/format/EOS/budget-exhausted rates và generated-token counts.

## Chạy và artifacts

```powershell
uv run --no-sync python -u scripts/check_capability.py
uv run --no-sync python -u scripts/evaluate_capability.py --dataset-root artifacts/runs/20261105T112412535097Z-wp03-data-pilot-seed42-fe698f01/artifacts/datasets --tokenizer artifacts/runs/20261105T154939888754Z-wp04-byte-bpe-seed42-4102afe0/artifacts/tokenizer --device cuda --split both
```

Installed CLI tương đương `python -u -m llm_scratch.evaluation.capability`. Default
split chỉ là validation. `--split both` chạy validation/test trong cùng invocation
với protocol đã snapshot; không dùng test để chọn checkpoint, budget hoặc threshold.
Config primary được commit trước lần đo test thật. `--diagnostic-limit N` chỉ cho
validation và report ghi diagnostic subset, không được thay thế full pilot.

`check_capability.py` dùng tiny model và authored offline pilot 3 câu/split để kiểm
pipeline/CI; không đọc checkpoint hoặc pilot thật. Các checkpoints fixture có weights
đã forward/backward, history thật của fixture, nhưng không đại diện chất lượng model.

Mỗi checkpoint có raw predictions JSONL, scored JSONL, metrics, tối đa 5 failure
examples đầu theo thứ tự pilot và text samples. Ba story prefixes authored được sample
với temperature 0,8/top-k 40/top-p 0,95, seed riêng cố định; không tính vào pilot EM.
Cached/uncached logits và generated IDs được đối chiếu trên 8 tokens của story prefix
cố định ở từng checkpoint trước pilot. Full pilot dùng cached generation.

Run giữ config/source/Git/runtime/protocol/environment, checkpoint inventory/history,
`stage_losses.csv`, `stage_loss_curves.png/.svg`, `pilot_metrics.csv` và
`pilot_exact_match.png/.svg`, `capability_report.json`. Mỗi stage có panel loss riêng, giữ đúng dataset/target scope;
không diễn giải chênh lệch TinyStories loss và CoT loss thành chênh lệch accuracy.
Model load, từng batch, decode progress và scorer đều in log trực tiếp.

## Điều kiện đọc kết quả

Đối chiếu EM cùng answer-format rate: pretraining chưa được dạy reasoning markers,
nên điểm dưới template SFT phản ánh cả format compliance và final-answer accuracy.
Một pilot nhỏ cố định không đại diện toàn bộ miền toán hoặc GSM8K. Source checkpoint
được train trên T4; runtime evaluator được ghi riêng, không gọi inference reload trên
GPU khác là exact training continuation. Voting, confidence calibration và threshold
selection tiếp tục ở các work package sau, chỉ fit trên validation.
