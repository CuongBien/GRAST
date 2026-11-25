# WP08 — Feasibility Test & Sample Bank

WP08 chia thành bốn phase. Mỗi phase có kiểm tra và báo cáo riêng; đạt WP08.1
chưa đồng nghĩa reasoning đã đủ tốt hoặc đã đạt nghiệm thu toàn WP08.

| Phase | Phạm vi | Gate |
| --- | --- | --- |
| WP08.1 | Schema, generation scores, lưu/đọc bank, tách gold, smoke | Bank có thứ tự, kiểm tra toàn vẹn và lặp lại được trong điều kiện cố định |
| WP08.2 | Khóa pilot validation; nhiều seed/checkpoint, tối đa16 ứng viên/câu | Đo coverage theo độ khó và tính đúng của trace; lưu bank cố định |
| WP08.3 | Offline replay trên các prefix của cùng bank | Các policy dùng cùng ứng viên, không nhận gold/future samples |
| WP08.4 | Báo cáo feasibility và quyết định benchmark | Có bằng chứng giữ GSM8K hay ưu tiên bài dễ; tách validation/test |

WP08.2 đã hoàn tất [protocol pilot đã khóa](wp08_pilot.md) và
[kết quả2.880 ứng viên](wp08_2_validation.md). WP08.3 đã có
[offline replay và kết quả1.440 quyết định](wp08_3_validation.md).
WP08.4 đã [chốt feasibility/benchmark](wp08_4_validation.md): đủ nghiệm thu kỹ thuật
WP08; chưa đủ capability để khẳng định lợi ích confidence stopping. Ưu tiên easy
diagnostic, giữ GSM8K supplementary stress; không fit threshold trên test.

WP08.1 chỉ dùng hai câu tự soạn để kiểm tra kỹ thuật. Chưa chọn pilot, threshold,
calibration hoặc benchmark theo kết quả smoke. Pilot WP08.2 cần loại các prompt/group
đã dùng selection trong WP07.6 trước khi xem đáp án; không dùng test để chọn threshold.

## Chạy một lệnh kiểm tra

```powershell
uv run --no-sync python -u scripts/check_sample_bank.py
```

Lệnh này tạo checkpoint cực nhỏ từ fixture offline, hai câu, hai trial seed và hai
ứng viên/câu/checkpoint: **16 records/bank**. Sinh hai bank và yêu cầu fingerprint,
toàn bộ tokens, log-probs, finish reasons và thứ tự trùng nhau hoàn toàn. Lệnh luôn
in progress logs và lưu config snapshot, Git/package metadata, environment, log,
bank và report dưới `artifacts/runs/`. Fixture không đo năng lực model thật.

Kiểm tra ngắn checkpoint thật bằng GPU local, với tokenizer WP04 đã xác minh:

```powershell
uv run --no-sync python -u scripts/check_sample_bank.py --device cuda --real-tokenizer artifacts/runs/20261105T154939888754Z-wp04-byte-bpe-seed42-4102afe0/artifacts/tokenizer
```

Config mặc định `configs/experiments/wp08_sample_bank_smoke.toml` trỏ tới selected2200
đã kiểm tra SHA256. Chế độ thật dùng một trial seed, hai câu, hai ứng viên: **4 records/bank**,
lặp hai lần, tối đa64 token/ứng viên. Không train lại model; không tự tải weights.
Không diễn giải số đúng/sai trên hai câu kỹ thuật thành feasibility.

## Tạo bank từ input đã khóa

```powershell
uv run --no-sync python -u scripts/build_sample_bank.py --config configs/experiments/wp08_sample_bank_smoke.toml --questions PATH_TO_QUESTIONS_JSONL --tokenizer PATH_TO_TOKENIZER --device cuda
```

Input là JSONL question-only, đúng các trường: `question_id`, `question`, `source`,
`split`, `difficulty`, `operation_count`, `reasoning_steps`, `prompt_length`,
`prompt_word_count`. WP08.1 chỉ nhận validation, ID và nội dung không trùng.
Thêm `canonical_answer`, `reference`, `correct` hoặc bất kỳ trường lạ nào sẽ bị từ chối.
Builder không có tham số references hay scoring callback. Nguồn câu hỏi vẫn cần
được kiểm duyệt: schema không thể phát hiện đáp án được viết ngay trong văn đề.

Config khai báo checkpoint theo thứ tự, gồm ID, loại `engine` hoặc
`curriculum_selected`, path và SHA256 bắt buộc. Checkpoint phải khớp tokenizer,
kiến trúc và điều kiện finite của loader WP07. Snapshot/run metadata ở bên ngoài
bank chứa paths, Git commit/dirty, config và thông tin môi trường. Bản bank tự chứa
tokenizer, model config, checkpoint digest và hash toàn bộ source package đã dùng.
Không đưa selection scores/history hoặc nhãn evaluation vào manifest bank.

## Contract lưu trữ và seed

Một bank hoàn chỉnh chỉ gồm:

```text
bank/
  manifest.json
  questions.jsonl
  candidates.jsonl
  tokenizer/
    vocab.json
    merges.json
    tokenizer_config.json
    tokenizer_manifest.json
```

`manifest.json` ghi cuối cùng bằng atomic replace, đóng vai trò commit marker.
Không overwrite/append/resume bank trong phase này. Thư mục dở dang không có
manifest không đọc được. Reader xác minh checksum/kích thước, tokenizer fingerprint,
grid đầy đủ, thứ tự, seed, batch layout, context/budget, EOS, answer extraction,
token count và log-prob aggregates. File lạ, symlink hoặc payload bị sửa đều bị từ chối.
Checksum là kiểm tra toàn vẹn dữ liệu, không phải chữ ký chống giả mạo có chủ đích.

Thứ tự cố định: checkpoint → question → trial seed → `candidate_index=0..K-1`,
với `1<=K<=16`. Một batch chỉ gồm nhiều bản sao của cùng câu hỏi. Mỗi batch có
generator riêng, seed được dẫn xuất từ checkpoint SHA, question ID, trial seed và
batch start bằng SHA256. Record lưu trial seed, generation seed thực tế, batch start,
batch size thực tế và batch row. Python/NumPy/PyTorch được seed trong runner.

Các row trong batch dùng chung generator; không gọi trial seed là seed độc lập
của từng ứng viên. Thay batch size/K có thể đổi chuỗi lấy mẫu. **Sinh Kmax một lần,
sau đó dùng prefix đã lưu**; không sinh lại K nhỏ để so sánh policy. Candidate0 là
lần lấy mẫu đầu, không phải greedy. Greedy baseline sẽ được khai báo riêng.
Không cam kết bitwise giữa GPU, phiên bản PyTorch, precision hoặc batch layout khác.

## Ý nghĩa điểm số

`generate(..., return_logprobs=True)` trả `generated_logprobs` có cùng shape với
`generated_ids`; mặc định `None`, giữ API cũ. Mỗi giá trị là natural log của xác suất
token từ **logits gốc của model**, tính `log_softmax` bằng FP32 trước temperature,
top-k/top-p. Đây không phải xác suất dưới phân phối đã lọc và không phải xác suất
model trả lời đúng. Không biến điểm số này thành confidence đã hiệu chỉnh.

Bank lưu từng log-prob, tổng (`math.fsum`) và trung bình trên token hợp lệ. EOS được
tính; phần tail bị mask sau EOS bị loại. PAD nếu thực sự được model lấy mẫu vẫn
được tính là token sinh ra. Output có số token lệch budget hoặc log-prob không finite
bị từ chối. `extracted_answer` dùng numeric-v1/native final extraction chung WP07;
answer-valid và format-valid được lưu riêng, không ngầm bỏ candidate sai format.

## Tách policy và đánh giá

```python
from llm_scratch.research.sample_bank import SampleBank

bank = SampleBank("PATH_TO_BANK")
observation = bank.observations()[0].prefix(4)
# observation chỉ có prompt/ID/difficulty/seed và bốn ứng viên đầu.
```

Observations là dataclass bất biến, token/log-prob dùng tuple. Prefix không mang
tham chiếu tới các ứng viên tương lai. Không có gold, correctness, selection scores
hay đường dẫn scoring trong observation. Đây là API boundary cho policy hợp tác,
không phải cơ chế cô lập filesystem đối với mã Python cố tình truy cập gold.
WP08.3 [gọi policy chỉ với prefix](wp08_replay.md), commit quyết định trước scorer.

Chấm riêng sau khi commit bank:

```powershell
uv run --no-sync python -u scripts/score_sample_bank.py --bank PATH_TO_BANK --references PATH_TO_REFERENCES_JSONL --output PATH_OUTSIDE_BANK
```

Reference có đúng ba trường `question_id`, `split="validation"`, `canonical_answer`
(numeric-v1 đã chuẩn hóa), phủ chính xác mọi question ID. Scorer không sửa bank,
không nhận output nằm bên trong bank và không fit policy/threshold. Đổi reference
chỉ đổi nhãn ngoài bank; bytes/fingerprint/observations giữ nguyên. Chi phí replay
token sau này không được gọi là latency/VRAM thực đo.
