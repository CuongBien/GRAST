# WP07.2 validation — 17/11/2026

**Đạt nghiệm thu phase 2**: immutable pretraining data, deterministic packing/order,
validation và checkpoint khôi phục cả consumed data cursor. [API/contracts](wp07_data.md).
WP07 vẫn còn pretraining/CoT SFT dài trên Colab và capability/pilot ở các phase sau.

## Kiểm thử

**665 tests toàn repository đạt local**. Phase này có 26 cases: 24 CPU + 2 CUDA;
CPU-only CI skip bốn hardware cases toàn repository.

- Target coverage độc lập ở block sizes 3/7/32/128: Unicode bytes, BOS/EOS, overlap,
  tail padding; mỗi non-BOS target được học đúng một lần mỗi epoch.
- Test-file sentinel có trong WP03 manifest nhưng bị cấm mở: pipeline vẫn thành công,
  chứng minh chỉ đọc train/validation. Source/cache checksum, duplicates/overlap giữa
  train/validation, corrupted manifest và artifact mutation sau verification bị chặn.
- Rebuild ở output roots khác nhau có cùng identity; copy/relocate corpus và resume
  trong một **tiến trình Python mới** cho cùng next batch hash/loss.
- Pending peek không tăng cursor; abort/retry giữ batch, foreign/stale commit bị từ
  chối. Partial batch/window và epoch rollover không drop blocks hoặc nhầm counters.
- Validation không advance train cursor/global RNG. Shuffle dùng CPU RNG riêng, CPU
  tensors vẫn đúng khi thay default torch device. Model/vocab/tokenizer/context/stage
  và checkpoint data contract không tương thích bị từ chối trước weights/RNG mutation.
- Actual train next-block IDs/token hashes/loss/LR, optimizer slots/scheduler/scaler/
  model/cursor khớp exact khi resume. Forced FP16 overflow vẫn tiêu thụ data window,
  không tăng optimizer step; checkpoint tiếp tục đúng next window.
- Lỗi sau optimizer nhưng trước commit cursor buộc full checkpoint reload; không
  save trạng thái engine/data bị lệch. Stdout tiếp tục in mọi update/data/validation event.

Ruff lint/format, wheel/sdist, installed-wheel data/full-resume smoke ngoài source
tree đạt. CI chạy authored offline fixtures; không cần tải corpus hoặc tokenizer ngoài.

## Dữ liệu thật đã chuẩn bị và verify

Source implementation commit sạch `ab45104cba56a69fc659e83371291d07c9f30022`.
Python 3.11.16, PyTorch 2.9.1+cu128, CUDA 12.8, RTX 4060 Laptop GPU.
Full WP03 TinyStories train/validation đã được đọc và đối chiếu hashes/counts;
WP04 tokenizer đóng băng, vocab 8192, không refit.

| Split | Documents | Tokens có BOS/EOS | Blocks T=64 | Supervised pairs |
| --- | --- | --- | --- | --- |
| Train | 10.000 | 3.741.797 | 59.394 | 3.731.797 |
| Validation | 1.000 | 342.625 | 5.439 | 341.625 |

Prepared files tổng cộng 16.337.688 bytes, uint32 little-endian. Packing continuous
có cross-document context trong block; không phải document-isolated attention.
Fingerprint prepared data:
`1adc3dd1a9815cdb08ca60f12373e835b4da7ad8e499eef9f4c1234e1e4ecea3`.
Codec fingerprint:
`49187f5c5a806801ae6fc86cce7a9e6ca26d54d81cc989b361bbc76724abada3`.
Hai preparation độc lập tạo cùng corpus manifest/token hashes; resume traces cũng khớp.

## Smoke training/full resume, không phải corpus baseline

4 successful updates, B2/G2, save sau step 2 rồi replay phần còn lại. Dropout=0,1.
Validation cố định **2 batches đầu** (4 blocks), không phải full validation metrics.

| Workload | Model / precision | Initial validation-prefix loss | Final validation-prefix loss | Data + engine resume |
| --- | --- | --- | --- | --- |
| Authored offline fixture | 99.136 params, CPU FP32, T32 | 5,561959 | 5,420898 | Exact |
| Real WP03/WP04 | 46.148.096 params, RTX 4060 BF16, T64 | 9,073520 | 5,465445 | Exact |

Offline fixture có 10 train blocks: checkpoint giữa epoch, replay đi qua partial
window cuối và epoch tiếp theo. Real run dùng toàn prepared train corpus để định
nghĩa permutation, nhưng chỉ tiêu thụ 16 blocks/4 updates cho verify. Validation loss
prefix giảm là quan sát diagnostic, không dùng làm gate chọn recipe hoặc quality claim.
Các runs, histories, resume block/token hashes, configs và environment có trong
[validation manifest](../experiments/wp07_2_validation_20261117.json).

## Artifacts và bước tiếp

Prepared corpus của clean real run:
`artifacts/runs/20261117T121626968367Z-wp07-2-data-resume-seed42-7d058dad/artifacts/pretraining_data`.

Bundle local đã kiểm ZIP CRC và SHA-256 từng entry:
`dist/grast-wp07-data-block64-1adc3dd1a981.zip` (~3,18 MiB).
Bundle có prepared train/validation, manifest và frozen tokenizer; **không có test
data hoặc checkpoint model**. SHA-256:
`3f7ef9d0e477738fa7592e793f3c355cf5eeb367345a98936fc1554f1cba4152`.
Block64 là workload verify hiện tại, chưa phải final corpus training recipe.

```powershell
uv run --no-sync python -u scripts/check_training_data.py --device cpu
uv run --no-sync python -u scripts/check_training_data.py --device cuda --precision bfloat16 --profile candidate --block-size 64 --dataset-root PATH/TO/wp03-datasets --tokenizer PATH/TO/wp04-tokenizer
```

Chưa chạy Colab hoặc training corpus dài. WP07.3 sẽ bổ sung runner/notebook pretraining,
periodic checkpoint lưu bền, loss curves và recipe theo GPU budget để user train.
Mọi lệnh tiếp tục in log trực tiếp. CoT SFT/pilot reasoning không nằm trong phase này.
