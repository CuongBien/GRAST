# WP07.3 — Pretraining runner và Colab

Chạy nền trên Kaggle: [notebook và hướng dẫn riêng](wp07_kaggle.md).

Runner đã triển khai; agent chỉ verify chạy ngắn. User chạy corpus baseline trên
Colab. Chưa có bằng chứng training dài hoặc kết quả chất lượng từ A100.

## Recipe khởi đầu

`configs/training/wp07_pretrain.toml` dùng candidate **46.148.096 tied parameters**,
vocab8192/context1024, TinyStories WP03 train10.000 và validation1.000 stories,
tokenizer WP04 frozen. Packing block256, overlap1 và ignore-BOS targets theo
[WP07.2](wp07_data.md); không đọc test. Source tokenizer/dataset hashes được kiểm tra.

Microbatch2, accumulation8, AdamW LR3e-4/weight decay0,1, warmup100 rồi cosine đến
0,1×LR, clipping1, BF16, dropout0. Budget khởi đầu **2.000 successful updates**;
không tuyên bố recipe tối ưu hoặc train đến hội tụ. Tối đa khoảng8,16 triệu next-token
pairs trước khi trừ BOS/tail targets, khoảng2,18 lượt train data. Tail epoch có partial
windows; số targets thực lấy từ `tokens_optimized` và logs.

Validation đầy đủ theo thứ tự cố định, token-weighted trên341.625 targets, trước train,
mỗi100 updates và khi dừng có kiểm soát. Chỉ validation được dùng để xem/chọn recipe.
Không dùng test/pilot hoặc smoke losses để chọn threshold.

## Chạy local và chuẩn bị input

```powershell
uv sync --locked --extra cuda --extra data --extra training
uv run --no-sync python -u scripts/prepare_training_data.py --dataset-root PATH/TO/wp03-datasets --tokenizer PATH/TO/wp04-tokenizer --output NEW_PREPARED_PATH --block-size 256
uv run --no-sync python -u scripts/pretrain.py --config configs/training/wp07_pretrain.toml --prepared-root NEW_PREPARED_PATH --tokenizer PATH/TO/wp04-tokenizer --device cuda --stop-after 4
```

Lệnh installed package: `python -u -m llm_scratch.training.pretrain`. Extra `training`
cung cấp Matplotlib để xuất loss plots; không ảnh hưởng schema WP02. CPU diagnostic:
`python -u scripts/check_pretraining_runner.py --device cpu`. Lệnh này tự tạo authored
offline inputs và đối chiếu run liên tục/stop/resume, tối đa8 successful updates;
không tự khởi chạy budget dài.

`--stop-after N` là **absolute successful update** trong recipe gốc. Không đổi
max_steps hoặc scheduler để smoke. Bỏ flag mới chạy đủ budget. `--precision` override
được snapshot và khóa khi resume. Không có quiet mode: mỗi update/validation/save/
load/start/error đều in JSON flush.

## Checkpoint và phục hồi

Trong mỗi run có `checkpoints/latest.pt`, cập nhật atomic tại clean optimizer
boundaries mỗi100 updates; checkpoints mốc100/500/1000 được giữ riêng và kết thúc có
`final.pt`. Checkpoint mốc cuối nằm ở `final.pt`. Mỗi full checkpoint sau optimizer
initialization khoảng0,52GiB; cần dung lượng Drive cho latest, các mốc và final.

Checkpoint giữ model, Adam slots, scheduler, scaler, RNG Python/NumPy/PyTorch/CUDA,
dataset/tokenizer/packing/shuffle identities, consumed cursor, recipe/implementation
fingerprints và **toàn bộ loss history đã commit**. Lịch sử gồm AMP skips; scheduler
chỉ tiến khi update thành công. Scaler/cursor được checkpoint theo attempt cadence
khi có overflow skips. Validation không dịch train cursor/RNG.

```powershell
uv run --no-sync python -u scripts/pretrain.py --config configs/training/wp07_pretrain.toml --prepared-root NEW_PREPARED_PATH --tokenizer PATH/TO/wp04-tokenizer --device cuda --resume PATH/TO/checkpoints/latest.pt
```

Mỗi resume tạo **run directory mới**, ghi lineage checkpoint vào config/log.
Chỉ checkpoint và input tương thích là đủ; không cần logs hoặc run directory cũ.
Recipe, implementation và runtime signature phải giống nhau; GPU/Torch/CUDA,
thread count/deterministic/TF32 khác không được gọi là exact continuation.
Không tự bỏ optimizer/đổi tokenizer để ép resume. Budget max_steps cũng bị khóa.
Đổi recipe/stage là warm-start riêng cho WP07.4, chưa triển khai ở phase này.

`latest.pt` là transaction authoritative: write temporary cùng thư mục, fsync,
os.replace. CSV/history/figures được dựng lại từ checkpoint khi resume; các dòng log
sau checkpoint cuối chỉ là attempts chưa commit bền vững. Không cộng lặp logs vào
curves. Mất runtime giữa update sẽ replay từ checkpoint hoàn tất trước đó.
Ctrl+C ở boundary sạch có thể ghi tiến độ; interrupt/error trong update không lưu
partial gradients. Lỗi quota/I/O được in rõ và dừng, giữ checkpoint cũ nếu replace
chưa thành công. Drive remote durability phụ thuộc mount/đồng bộ của Colab; atomic
file protocol không phải cam kết chống mọi lỗi dịch vụ lưu trữ.

## Artifacts

`config.json`, `config_sources/`, `metadata.json`, `run.json`, logs JSONL/text,
environment report; `artifacts/dataset_manifest.json`, frozen tokenizer copy,
`history.json`, `losses.csv`, `loss_curves.png/.svg`, `pretraining_report.json`.
Train curve là loss từng effective batch; validation curve là token-weighted loss
trên scope khai báo. Smoke config chỉ validate fixed-prefix2 batches, không diễn giải
thành held-out toàn corpus. `tokens_seen` gồm skipped attempts; `tokens_optimized`
chỉ đếm successful updates.

## Colab

Mở `notebooks/wp07_pretraining_colab.ipynb`, chọn A100, chạy các cells. Notebook dùng
PyTorch sẵn của Colab, in version/GPU và không fallback CPU. Source revision được pin;
checkpoint lưu trực tiếp tại `MyDrive/GRAST/wp07/pretraining/runs/`.

Input local: `dist/grast-wp07-data-block256-17971c0168e4.zip` (~3,18MiB), SHA256
`9f08cdc7ba7c8a4825ce90470fd19fc3ed6da7d6d3bfd742097a389d1aab949d`.
Dataset fingerprint:
`17971c0168e4367b183ebb0a3abc7c1309ffe5095cdc67899bce13ffc4298faa`.
Tokenizer fingerprint:
`49187f5c5a806801ae6fc86cce7a9e6ca26d54d81cc989b361bbc76724abada3`.
ZIP chỉ có train/validation tokens, manifests và frozen tokenizer, không test/weights.
Bundle block64 cũ thuộc diagnostic WP07.2, không dùng cho recipe block256.

Notebook mặc định `SMOKE_ONLY=True`: dừng update4. Sau khi xem log/Drive checkpoint,
đặt `SMOKE_ONLY=False`, điền `RESUME` trỏ latest của smoke và chạy cell train lại.
Không cần đổi budget4 thành2.000. Nếu tắt runtime, chạy lại setup/mount/input cells,
giữ source/recipe/precision và điền checkpoint đã lưu. BF16 ưu tiên; FP16 khi GPU
không hỗ trợ BF16, không thể resume đổi precision giữa chừng.

Subprocess `python -u`, stdout/stderr stream từng dòng. Cell cuối xuất summary ZIP
config/log/curves/report; weights giữ trên Drive cho WP07.4–5. Chạy4 updates thành
công chưa phải corpus baseline hoàn tất.

Tự tạo/xác minh bundle cùng contract:

```powershell
uv run --no-sync python -m llm_scratch.data.training_bundle create --prepared-root NEW_PREPARED_PATH --tokenizer PATH/TO/wp04-tokenizer --output NEW_BUNDLE.zip
uv run --no-sync python -m llm_scratch.data.training_bundle extract --archive NEW_BUNDLE.zip --output NEW_INPUT_DIR --expected-sha256 EXPECTED_SHA256
```

Extractor chỉ cho phép7 payload files cố định và bundle manifest; từ chối paths lạ,
duplicates, symlinks, quá dung lượng, ZIP/SHA corruption; kiểm lại PackedCorpus và
tokenizer sau giải nén. Output phải là thư mục mới, không reuse extraction dở.
