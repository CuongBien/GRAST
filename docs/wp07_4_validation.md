# WP07.4 — Local CoT SFT acceptance, 20/11/2026

Đã triển khai data normalization, independent-example prompt masking, pretrained
warm-start, SFT runner/checkpoints/curves và notebooks Colab/Kaggle luôn stream log.
**SFT dài500updates do user chạy; generation/pilot capability thuộc WP07.5.**

Source commit `db0da482c3516d99b3fa6c3b6e6f566a65bf2c2f` trên `feat/wp07-4-cot-sft`.
Run evidence khởi tạo với Git clean tại commit này; notebooks pin cùng source.
[Evidence JSON](../experiments/wp07_4_validation_20261120.json),
[hướng dẫn/API](wp07_sft.md).

## Input đã xác minh

User pretraining checkpoint chuyển từ root vào
`artifacts/checkpoints/wp07-pretrain-tinystories-2000/final.pt`; SHA256
`27cf2d9cb9774b0023d4db09e2735fd666452f5a945468bc71722bb3c69b5309`.
Kiểm trực tiếp weights/metadata/history và summary: step/attempt2000,8.130.916targets,
46.148.096parameters; final val1,5673957146677044, perplexity4,7941465963,
last100 train-loss mean1,4478156856. Source T4/FP16/Torch2.11.0+cu130/CUDA13.
Parent giữ nguyên hash sau các lượt SFT verification.

Math SFT từ WP03:1.500train/300validation,500/100 mỗi mức; exact Fraction rationales,
frozen codec WP04. Prompt tokens209.570/41.880 được mask; target tokens94.915/18.971.
Lengths61–387 dưới cap512; không truncate/drop. Không đọc test/pilot hoặc GSM8K mặc
định. Dataset fingerprint
`78f779a7d367a91f505332a804edbe40a0286ed5fb932269103c18819d35395b`.

## Tests/package/notebooks

Full suite **728 pass**, sau đó thêm hai tests input/launch helpers Kaggle đều pass:
**730 unique local cases**. Trong đó28 cases SFT mới:27CPU,1CUDA-FP16.
Có độc lập shifted-CE reference, prompt/padding masks, batch-row isolation,
Unicode/literal marker IDs, exact arithmetic rationale/GSM8K official-train gates,
no-test/pilot/default-GSM8K reads, payload/mask corruption và no truncation/reuse.
Warm-start không restore parent RNG/Adam/cursor; incompatible metadata/tensor/NaN/tied
weights bị chặn trước mutation. CPU/FP16-CUDA resume exact và restart process mới
không còn parent checkpoint/run cũ vẫn phục hồi full state/history/curves.

Lint/format, wheel/sdist build pass. Wheel cài vào target riêng rồi chạy acceptance
ngoài source tree: pass. Sdist không chứa cache/run payloads hoặc weights.
ZIP round-trip/corruption/path guards pass; bundle thật được extract/verify lại.
Notebook cells compile; **thực thi chính input-discovery và command cells của Kaggle**
với ZIP/expanded layouts, fresh SFT/resume sau khi xóa parent checkpoint: pass.
Chưa thực thi SFT notebook trong Colab/Kaggle; local helper tests không thay execution
evidence trên những dịch vụ đó. Curves PNG đã kiểm tra trực quan.

## CPU fixture diagnostic

`python -u scripts/check_sft.py --device cpu`: tiny29.088parameters/dropout0,1,
byte-only264-token codec,10train/4validation authored examples. Parent có đúng một
pretraining update fixture, không phải corpus baseline. Recipe6updates, verify4,
compare full4 với stop2→resume4. Train targets332; validation92targets:
`5.537389620490696 → 5.463809640511222`. Full engine/data/RNG state và per-attempt
batch/loss/LR/AMP trace exact; common validation losses exact.

Run: `artifacts/runs/20261119T195129202416Z-wp07-4-sft-smoke-seed42-548c6711`.
Run directory dùng UTC theo contract WP02; ngày báo cáo dùng timezone user20/11.

## Checkpoint thật — RTX4060/FP16

```powershell
python -u scripts/check_sft.py --config configs/training/wp07_sft.toml --prepared-root artifacts/wp07-4/sft_data --tokenizer artifacts/runs/20261105T154939888754Z-wp04-byte-bpe-seed42-4102afe0/artifacts/tokenizer --init-checkpoint artifacts/checkpoints/wp07-pretrain-tinystories-2000/final.pt --device cuda --precision float16 --stop-after 4 --split-step 2
```

Parent T4/Torch2.11 nạp **weights-only** vào model cùng kiến trúc/tokenizer trên
local Torch2.9.1+cu128/CUDA12.8. Optimizer/scaler/scheduler/RNG/cursor mới của SFT,
không gọi chuyển runtime này là exact pretraining resume.

46.148.096parameters, FP16,1GPU RTX4060Laptop, deterministic/TF32off/1CPU thread.
Recipe500updates/warmup25 giữ nguyên; chỉ4successful updates,0skips,32examples,
2.037supervised targets. Full held-out SFT validation300examples/18.971targets:
**7,3907369986 → 6,5285369991**.

Full4 và stop2→resume4 có model/Adam/scheduler/scaler/cursor/RNG exact; từng
batch/token hash/loss/LR/AMP trace và validation tại mốc chung exact. Extra validation
ở stop2 nằm trong history resume, không làm dịch RNG/cursor. Full checkpoint sau4
updates553.913.243bytes. Không báo smoke throughput là benchmark hoặc ngoại suy T4.

Run: `artifacts/runs/20261119T195128946697Z-wp07-cot-sft-math-seed42-826208f8`.
Loss SFT trên reasoning/final/math khác TinyStories pretraining loss; không so trực
tiếp6,53 với1,57 để kết luận model tốt/xấu hơn. Chưa đo generated-answer accuracy.

## Handoff

Notebooks: `notebooks/wp07_sft_colab.ipynb`, `notebooks/wp07_sft_kaggle.ipynb`.
Bundle `dist/grast-wp07-sft-math-78f779a7d367.zip`,410.509bytes, SHA256
`100296172c05aff2bb99a4cc314cadf6dece5c65e0d7d7bb846a323794e7296e`.
Parent `final.pt` là input riêng; source SHA được notebook kiểm trước fresh SFT.
Mặc định smoke4. Train dài: `SMOKE_ONLY=False`; tiếp tục smoke bằng `RESUME` trỏ
**SFT latest.pt**, không phải pretrained final.pt. Nếu RESUME rỗng thì SFT mới từ
parent. Đọc [hướng dẫn](wp07_sft.md) cho Drive/Kaggle input và source/runtime contracts.
