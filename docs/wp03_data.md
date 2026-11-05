# WP03 — Dataset & Benchmark Design

Pipeline đã triển khai: TinyStories subset, GSM8K bổ sung, arithmetic benchmark,
answer extraction, split/dedup và pilot cố định. Không có tokenizer/model/training loop.

## Chạy

```powershell
uv sync --locked --extra cpu --extra data
uv run --no-sync grast prepare-data --config configs/data/wp03.toml
```

Nếu dùng environment CUDA hiện có, sync bằng `--extra cuda --extra data` thay cho CPU.
Data preparation chạy CPU; không cần GPU. Extra `data` cài PyArrow, phiên bản thực tế pin
trong `uv.lock`. Không có Hugging Face remote-code execution hoặc model dependency mới.

Lần đầu tải khoảng 264 MB source: một train shard TinyStories, validation shard và hai
file JSONL GSM8K. Từng file được kiểm tra full SHA-256 và byte size trước khi vào cache
content-addressed ở `artifacts/datasets/cache/`. Cache hit cũng được kiểm tra checksum.

Sau khi cache đủ, tạo lại hoàn toàn offline:

```powershell
uv run --no-sync grast prepare-data --config configs/data/wp03.toml --offline
```

CLI in run directory và đường dẫn manifest. Lỗi cache thiếu/hỏng khiến run fail; không
báo thành công với dữ liệu giả hoặc thiếu nguồn. Entry point tương đương là
`python scripts/prepare_data.py --config configs/data/wp03.toml` sau khi cài package.

`--math-only --offline` chỉ dùng cho kiểm tra nhanh procedural benchmark/CI;
manifest ghi rõ scope `math-only`, không phải nghiệm thu đầy đủ WP03.

## Sources và split policy

Lockfile: [sources.lock.json](../configs/data/sources.lock.json).

| Dataset | Revision | Lựa chọn ban đầu và split |
| --- | --- | --- |
| TinyStories | `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` | 10.000 nonempty stories đầu tiên trong một train shard; 2.000 từ official validation chia thành internal validation/test 1.000/1.000 |
| GSM8K | `3101c7d5072418e28b9008a6636bde82a006892c` | Official train 7.473 chia thành train 6.973 / validation 500; official test 1.319 giữ held-out |
| Arithmetic v1 | Code/config của GRAST | Mỗi mức 500 train / 100 validation / 100 test, tổng 2.100 câu |

TinyStories dùng original published Parquet, không dùng bản V2-GPT4. Subset là lựa chọn
prefix có chủ đích cho lần chạy đầu, không phải sample đại diện toàn corpus. Muốn tăng
budget phải thay đổi config; vượt số hàng của shard hiện tại sẽ fail và cần mở rộng nguồn
đã khóa. Counts sau lọc/dedup luôn ghi theo output thực tế, không giả định đủ số hàng sạch.

Nguồn và license metadata: [TinyStories dataset card](https://huggingface.co/datasets/roneneldan/TinyStories)
ghi `cdla-sharing-1.0`; [OpenAI GSM8K](https://github.com/openai/grade-school-math)
cung cấp JSONL question/answer và MIT license. File nguồn được giữ ngoài Git.

TinyStories giữ Unicode/text gốc sau strip; ID/dedup dùng NFKC, casefold và collapse
whitespace. JSONL escape U+0085/U+2028/U+2029 để giữ một record trên một dòng cho
reader dùng `splitlines()`; JSON decoding vẫn khôi phục đúng text Unicode.
Official held-out text thắng khi trùng với train: loại bản train.
GSM8K giữ rationale gốc và thêm `canonical_answer` lấy từ marker `####`.
Validation chỉ lấy từ official train; không chuyển official test vào train/validation.

Split ranking dùng SHA-256 của seed/namespace/stable ID, không dùng đáp án, accuracy
hoặc kết quả model. Thứ tự input không thay đổi split assignment. Pilot cũng chọn bằng
hash stable IDs và seed, không chọn theo câu model làm đúng/sai.

## Difficulty và pilot

| Mức | Số phép tính | Dependency depth (`reasoning_steps`) | Đề | Toán hạng | Operators |
| --- | --- | --- | --- | --- | --- |
| easy | 1 | 1 | short | 1–30 | +, - |
| medium | 3 | 2 | medium | 1–20 | +, -, * |
| hard | 5 | 3 | long | 1–20 | +, -, *, / |

Ba tham số operations/depth/prompt_length cấu hình độc lập trong giới hạn cây nhị phân:
`depth <= operations <= 2**depth - 1`. `reasoning_steps` là số tầng phụ thuộc tính toán,
không phải số bước CoT hoặc độ khó nhận thức đã đo. Default tiers cùng thay đổi nhiều
trục nên chưa cho phép kết luận nhân quả theo từng trục. Có thể cấu hình ablation một trục.

Generator tạo cây biểu thức có đúng count/depth, dùng phép toán `Fraction` chính xác;
loại phép chia cho 0 và biểu thức có numerator/denominator vượt 1e9. Prompt dùng các
định nghĩa R1/R2/... bằng lời. Medium/long thêm ngữ cảnh không liên quan, không đổi phép tính.
`prompt_word_count` là số từ theo whitespace, không phải tokenizer tokens.

Pilot cố định gồm **150 validation + 150 test = 300 câu**, mỗi mức 50 câu ở mỗi split.
Pilot là subset của các split math tương ứng, không phải split mới được trộn lại.
Cấu hình, seed, IDs, payload hashes và fingerprint khóa được cùng bộ câu cho lần chạy sau.

`pilot/*.questions.jsonl` chỉ chứa câu hỏi và metadata độ khó; không chứa gold, rationale
hay expression AST. `pilot/*.references.jsonl` chứa reference đáp án riêng cho evaluator.
Sampling/policy chỉ đọc questions. Không dùng test để chọn threshold/calibrator;
chọn/fit trên validation và đánh giá cuối trên test.

## Answer protocol numeric-v1

`extract_answer` nhận đúng một `<final>...</final>`, hoặc một suffix `####`, hoặc đáp án
numeric thuần. Marker thiếu/lặp/malformed hoặc rationale không có final được trả `None`.
Không đoán "số cuối cùng" trong rationale. Final block hợp lệ được ưu tiên khi có mặt.

Normalization hỗ trợ số nguyên, thập phân, scientific notation giới hạn, comma-grouping
hợp lệ, dollar prefix, fraction, Unicode NFKC/minus và boxed/LaTeX fraction đơn giản.
Output là integer hoặc fraction tối giản: `0.5`, `2/4` và `1/2` đều thành `1/2`.
So sánh chính xác bằng rational; không có float tolerance hoặc threshold được fit từ test.
Phần trăm, units, nhiều đáp án, biểu thức chưa tính, NaN/inf và chia 0 bị từ chối.
Invalid không bao giờ được chấm đúng chỉ vì reference cũng invalid.

## Output, integrity và nghiệm thu

```text
<run>/
  config.json, config_sources/data.toml, config_sources/sources.lock.json
  metadata.json, run.json, logs/*, result.json
  artifacts/datasets/
    manifest.json, leakage_report.json
    tinystories/{train,validation,test}.jsonl
    gsm8k/{train,validation,test}.jsonl
    math/{train,validation,test}.jsonl
    pilot/{validation,test}.{questions,references}.jsonl
```

Manifest ghi số record/byte/SHA-256 của mỗi JSONL, source pins/license metadata,
levels, split policy, pilot distribution và dedup/audit report. `data_config_sha256`
chuẩn hóa hai runtime paths để cùng config trên máy khác có cùng semantic config hash.
`dataset_fingerprint` hash toàn manifest trước khi thêm chính field fingerprint.
WP02 run snapshot/metadata vẫn ghi config paths thật, Git/source fingerprint và options.
`execution_device=cpu`; runtime inventory ghi Python/NumPy/PyTorch/CUDA/GPU hiện có,
không dùng CUDA availability làm gate của data preparation.

Hai lần chạy trên cùng locked inputs/config tạo manifest và payload byte-identical;
run ID, log timestamps và metadata môi trường vẫn riêng. Manifest versioned ở
[experiments/wp03_dataset_manifest.json](../experiments/wp03_dataset_manifest.json);
run/evidence paths ghi trong [validation report](wp03_validation.md).

Dedup kiểm tra normalized-text overlaps giữa các split và associative/commutative
group của cây +/*. Không phải semantic paraphrase detector. Raw dữ liệu, cache và output
vẫn bị gitignore; source pins/config/code/manifest và kết quả kiểm tra được versioned.
CI kiểm tra full pipeline bằng authored offline fixtures; nghiệm thu dữ liệu thật đã chạy local.
