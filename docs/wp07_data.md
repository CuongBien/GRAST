# WP07.2 — pretraining data và full session resume

Phase này nối WP03 TinyStories train/validation với tokenizer WP04 đã đóng băng,
packing và training engine WP07.1. Chỉ verify vài updates; corpus baseline/Colab
runner dài thuộc WP07.3, CoT formatting/SFT thuộc WP07.4.
[Nghiệm thu và evidence](wp07_2_validation.md).

## Chuẩn bị và nhận dạng dữ liệu

```powershell
uv run --no-sync python -u scripts/prepare_training_data.py --dataset-root PATH/TO/wp03-datasets --tokenizer PATH/TO/wp04-tokenizer --output PATH/TO/new-pretraining-data --block-size 1024
```

Package entry tương đương: `python -m llm_scratch.data.pretraining`.
Input phải là full WP03 manifest với self-fingerprint đúng. Pipeline chỉ mở hai file
allowlisted `tinystories/train.jsonl` và `tinystories/validation.jsonl`; không đọc
test hoặc pilot. Hash bytes, size, record counts, row source/split và normalized-text
duplicate checks được kiểm tra khi token hóa. Codec load từ WP04 có checksum/fingerprint;
không fit tokenizer hoặc sửa normalization bằng validation.

Output mới có `train.u32`, `validation.u32`, `manifest.json`; dtype little-endian
uint32, portable giữa các thư mục/máy. Manifest ghi source dataset fingerprint,
codec fingerprint/vocab/special IDs, packing protocol, block size, source/token file
hashes, documents/tokens/blocks/supervised-pair counts. Fingerprint không có absolute
paths hoặc timestamp, nên rebuild cùng inputs/config tạo cùng identity.
Manifest viết atomic **sau cùng**; partial preparation không load được. Output đã tồn
tại bị từ chối, không tự overwrite/reuse một preparation dở. Preprocessing luôn in
start/progress/completion logs.

Reader `PackedCorpus` kiểm manifest/token file SHA/size/counts trước khi dùng,
memory-map read-only và copy block tensors ra CPU. Metadata/file stamp được giữ để
phát hiện artifact bị thay đổi sau verification; không chỉnh corpus đang train.
Di chuyển/copy toàn bộ folder giữ identity và resume được; không cần đường dẫn cũ.
Memory mapping dùng [NumPy mode r](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html);
shuffle dùng explicit CPU generator của [torch.randperm](https://docs.pytorch.org/docs/2.9/generated/torch.randperm.html).
Factory calls đặt CPU rõ ràng, kể cả khi notebook đổi default torch device.

## Packing và supervision

Protocol `continuous-bos-eos-overlap1-ignore-bos-target-v1`:

1. Mỗi document encode riêng thành BOS + byte-BPE text + EOS; không truncate text.
2. Nối stream trong **từng split riêng**, không ghép train/validation.
3. Mỗi block dài T, stride T-1 (overlap một token). Block cuối có ít nhất hai tokens,
   right-pad khi thiếu; không drop tail và không mất adjacent next-token pair.
4. Labels unshifted; target BOS và padding là -100, padding mask=0. CausalLM shift
   một lần; mọi non-BOS target trong stream được giám sát đúng một lần mỗi epoch.

Đây là continuous causal pretraining: context **có thể nhìn sang document trước** trong
cùng block. BOS/EOS không phải block-diagonal attention mask. Không gọi đây là
document-isolated packing. Giữa blocks không giữ KV state cho training; overlap chỉ
giữ đủ target pairs, không đảm bảo toàn bộ context của document dài. CoT SFT không
tự đi qua packing policy này; session từ chối stage=sft.

## Thứ tự train và consumed cursor

`TrainingCursor` shuffle block IDs mỗi epoch bằng **private CPU torch.Generator**,
seed dẫn xuất từ corpus identity + shuffle_seed + epoch; không dùng global RNG của
dropout. Shuffle protocol/Torch version, batch size và accumulation G nằm trong contract.
Các epoch đọc đủ mỗi block một lần, không drop last batch/window; window cuối có thể
ít hơn G microbatches hoặc batch size nhỏ hơn. Sau EOF, cursor normalize sang epoch
kế tiếp offset=0. Order hash và counters được validate khi restore.

`peek()` tạo một pending window và CPU batches nhưng **không tăng consumed cursor**.
Chỉ `commit(window)` sau engine update thành công hoặc AMP skip mới tiến cursor;
AMP skip vẫn đã tiêu thụ data/RNG, chỉ optimizer/scheduler không tiến. Exception làm
abort cursor và session failed; nếu optimizer đã đổi trước lỗi thì phải load full
checkpoint cũ, không retry trên weights đã đổi. Pending/stale/foreign windows không
được save/commit tùy ý.

Pipeline hiện là **single-process pull, không worker/prefetch queue**. Checkpoint lưu
consumed state, không phải prefetched position. Không có claim worker/DDP exact resume;
muốn thêm parallel loading sau này phải giữ consumed-window contract này.

## Full checkpoint và validation

```python
train = TrainingCursor(PackedCorpus(data_root, "train"), batch_size=2,
                       accumulation_steps=engine.config.gradient_accumulation_steps, seed=42)
session = PretrainingSession(engine, train, PackedCorpus(data_root, "validation"))
update = session.update()              # Engine luôn in update + data-window logs.
validation = session.validate()        # Toàn validation, fixed sequential order.
session.save("checkpoints/resume.pt")
# New engine/session cùng recipe/runtime, có thể data_root mới sau khi copy folder:
restored_session.load("checkpoints/resume.pt")
```

Engine phải có model vocab/tokenizer fingerprint/context tương thích. Full data_state
gồm session contract và cursor epoch/offset/windows_consumed/order hash. Engine attempts
phải khớp consumed windows. Generic WP07.1 engine load có thêm data validator chạy
**trước model/optimizer/RNG mutation**; mismatch dataset/tokenizer/packing/batch/shuffle/
validation hoặc counter bị từ chối ngay. Model/optimizer/scheduler/scaler và mọi RNG
vẫn theo checkpoint WP07.1; chỉ save tại clean optimizer/data boundaries.

Không gọi `engine.train_update/save/load` riêng để bypass session khi muốn full corpus
resume. Session phát hiện engine attempts/cursor desync. Exact continuation được verify
trên cùng runtime/hardware/precision; không hứa bitwise equality khi đổi GPU/Torch.

Validation có reader/stateless order riêng, không shuffle, không dịch train cursor
hoặc consume dropout RNG. Loss token-weighted theo WP07.1. `validate(max_batches=N)`
cho phép explicit fixed-prefix diagnostic; scope N, batch size, split/identity được
in log. N=None là full validation. Smoke dùng N=2, không trình bày prefix loss là
full validation/corpus quality hoặc dùng nó chọn threshold trên test.

## Verify ngắn

```powershell
# CI: authored offline schema fixtures, không tải source/tokenizer ngoài.
uv run --no-sync python -u scripts/check_training_data.py --device cpu
# Real WP03 + frozen WP04, candidate; chỉ 4 updates + replay sau checkpoint step 2.
uv run --no-sync python -u scripts/check_training_data.py --device cuda --precision bfloat16 --profile candidate --block-size 64 --dataset-root PATH/TO/wp03-datasets --tokenizer PATH/TO/wp04-tokenizer
# Reuse prepared corpus (cùng block size/config), không tokenize lại:
uv run --no-sync python -u scripts/check_training_data.py --device cuda --precision bfloat16 --profile candidate --block-size 64 --prepared-root PATH/TO/pretraining-data
```

Run có config/source/environment/Git metadata, prepared manifest, start/progress logs,
window indices/token-batch hashes, train/validation loss history, resume trace và full
resume.pt/final.pt. Các checkpoint lớn và raw data ở artifacts (gitignore), evidence
nhỏ được version hóa. Data folder có thể upload nguyên vẹn lên Colab ở WP07.3;
hash validation không phụ thuộc source path trên máy local. Luôn in logs trực tiếp;
agent chỉ verify short runs theo yêu cầu user, chưa chạy pretraining corpus dài.
