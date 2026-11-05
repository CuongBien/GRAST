# WP04 — Tokenizer from Scratch

`ByteBPETokenizer` tự triển khai BPE training/encode/decode bằng Python chuẩn;
không gọi thư viện tokenizer. Artifact chung được load lại cho pretraining, SFT và
inference. Chưa khởi tạo neural model hoặc chạy training loop.

## Tái lập

Sau khi chuẩn bị dữ liệu WP03:

```powershell
uv run --no-sync grast train-tokenizer --config configs/tokenizer/train.toml
# Tương đương: uv run --no-sync python scripts/train_tokenizer.py --config configs/tokenizer/train.toml
```

Lệnh tự tìm run WP03 đã hoàn thành, có manifest khớp với
`experiments/wp03_dataset_manifest.json`. Có thể chỉ định `--dataset-root PATH`
đến thư mục chứa các file `tinystories/train.jsonl`, v.v.; checksum vẫn bắt buộc.
`--output-root PATH` đổi nơi ghi run. Không cần GPU hoặc extra tokenizer.

Run có config snapshot, bản TOML và manifest nguồn, metadata Git/runtime, log,
`artifacts/tokenizer/` và `artifacts/tokenizer_report.json`. Bốn file artifact:

- `vocab.json`: token ID, byte hex hoặc tên special token.
- `merges.json`: cặp ID và ID mới theo thứ tự rank.
- `tokenizer_config.json`: thuật toán, pretokenizer, vocab thực tế/đích, specials.
- `tokenizer_manifest.json`: SHA-256 của ba file và fingerprint codec.

Load xác minh checksum, cấu trúc merge/vocab/config và fingerprint. Save từ chối
ghi đè thư mục có sẵn. Truyền `expected_fingerprint` khi cần khóa đúng codec.
Run metadata/timing thay đổi giữa lần chạy; bốn file codec phải giống từng byte.

## Thiết kế đã khóa

`byte-bpe-v1` bắt đầu với toàn bộ 256 byte UTF-8; không normalize hoặc đổi case.
`unicode-groups-v1` tách whitespace, chữ, số, underscore và punctuation thành các
piece; chia piece tối đa 256 byte. Merge không vượt piece hoặc document.
Đây là pretokenizer riêng của dự án, không phải định dạng tương thích GPT-2.

Training đếm tần suất piece, cập nhật pair counts ở các piece bị ảnh hưởng và
chọn pair phổ biến nhất bằng heap. Khi bằng tần suất, chọn cặp token ID nhỏ nhất
theo thứ tự từ điển. Không dùng RNG; đảo thứ tự corpus giữ nguyên artifact.
Merge áp dụng trái sang phải, không chồng lấn; encode dùng merge rank đã học.
BPE tham khảo thuật toán subword trong
[Sennrich et al. (2016)](https://aclanthology.org/P16-1162/); byte alphabet và
framing reasoning ở đây là lựa chọn triển khai của GRAST.

Chỉ fit trên 10.000 bản ghi **TinyStories train** đã đóng băng ở WP03.
Vocab đích 8.192, min frequency 2 được đặt trước khi xem metrics held-out.
Validation/test chỉ đo codec đã đóng băng; không dùng chọn vocab/threshold.
Nếu corpus nhỏ không đủ pair, vocab thực tế có thể thấp hơn đích; không tạo token giả.
Model sau này phải lấy kích thước từ `len(tokenizer)` và lưu fingerprint cùng checkpoint.

## Unicode, unknown và controls

Tám ID mặc định: `<pad>`=0, `<bos>`=1, `<eos>`=2, `<unk>`=3,
`<think>`=4, `</think>`=5, `<final>`=6, `</final>`=7.
Byte ID nằm ở 8–263; merge ID bắt đầu ở 264. `<unk>` được dành chỗ nhưng
mọi chuỗi Unicode hợp lệ đều encode bằng byte fallback, kể cả ký tự chưa thấy.
Lone surrogate không phải UTF-8 hợp lệ và bị từ chối.

Mặc định spelling `<eos>` trong dữ liệu là literal text, không tạo EOS điều khiển.
Chỉ `allowed_special={"<eos>"}` (hoặc `"all"` cho nội dung tin cậy) nhận diện nó.
BOS/EOS là opt-in; các adapter thêm ID trực tiếp để tránh control injection.
Decode mặc định giữ spelling controls; `skip_special_tokens=True` bỏ control IDs,
vẫn giữ spelling literal. ID ngoài vocab bị từ chối. Byte fragments sinh tùy ý có
thể không hợp lệ UTF-8: decode mặc định dùng replacement, `errors="strict"` báo lỗi;
`decode_to_bytes` bảo toàn byte. Encode→strict decode phải khôi phục đúng chuỗi gốc.

## API chung cho các bước sau

```python
from llm_scratch.data.tokenizer import ByteBPETokenizer
from llm_scratch.data.collator import encode_pretraining, collate_tokenized
from llm_scratch.reasoning.formatting import encode_sft_example, encode_reasoning_prompt

tokenizer = ByteBPETokenizer.load("PATH/TO/artifacts/tokenizer")
ids = tokenizer.encode("Xin chào 🌏")
assert tokenizer.decode(ids, errors="strict") == "Xin chào 🌏"
pretrain = encode_pretraining(tokenizer, "A small story.")
sft = encode_sft_example(tokenizer, "What is 2 + 3?", "Add two and three.", "5")
prompt = encode_reasoning_prompt(tokenizer, "What is 2 + 3?")
batch = collate_tokenized([pretrain, sft], tokenizer)
```

Pretraining framing: BOS + literal text + EOS. Reasoning SFT:
BOS + literal question + newline + THINK + literal reasoning + END_THINK + FINAL +
literal answer + END_FINAL + EOS. Inference prompt khớp chính xác prefix SFT đến THINK.
Labels của prompt bị mask `-100`; padding bên phải dùng PAD/attention 0/label `-100`.
Labels chưa shift; causal LM/trainer sau này phải shift đúng một lần.
Adapter không truncate ngầm; vượt `max_length` báo lỗi để caller chọn packing/truncation.
Đây là kiểm tra giao diện token, chưa phải inference qua model.

## Nghiệm thu

Xem [validation](wp04_validation.md),
`experiments/wp04_tokenizer_manifest.json` và `experiments/wp04_validation_20261105.json`.
Metric tokens/character là tổng token ordinary-text chia tổng Unicode code point,
không bao gồm BOS/EOS; không lấy trung bình các tỷ lệ từng document.
Tỷ lệ của các corpus khác nhau chỉ mô tả codec hiện tại, không đo chất lượng reasoning.
CI dùng dữ liệu fixture tự viết, không tải corpus thật.
