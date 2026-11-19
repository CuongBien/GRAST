# WP07.3 — Local runner acceptance, 19/11/2026

Cập nhật20/11/2026: đã nhận/verify checkpoint user train đủ2.000updates trên T4/FP16;
[evidence mới](../experiments/wp07_pretraining_completed_20261120.json) ghi SHA256,
loss/perplexity/token counts và nguồn summary. Báo cáo local4updates phía dưới giữ
nguyên phạm vi lịch sử; corpus baseline không còn pending. CoT SFT/pilot vẫn chưa đóng.

Đã triển khai và verify runner pretraining, checkpoint định kỳ/mốc/final, loss
history/CSV/PNG/SVG và notebook Colab lưu trực tiếp Drive, luôn stream log.
**Corpus baseline 2.000 updates trên Colab vẫn pending user chạy.** Chưa có CoT SFT,
text-quality acceptance hoặc pilot reasoning; toàn WP07 chưa đóng.

Implementation commit: `aa4be36b33ca025701421ef48715a22b43c4e8db`, nhánh
`feat/wp07-3-pretraining-runner`. Notebook pin commit này. Các run evidence bắt đầu
với Git clean ở cùng commit. Configs/commands/recovery: [hướng dẫn](wp07_pretraining.md).
Machine-readable evidence: [manifest](../experiments/wp07_3_validation_20261119.json).

## Kiểm chứng

- **689 tests pass** local, gồm24 cases mới của runner (22CPU,2CUDA). Có restart bằng
  Python process mới và xóa run directory cũ, vẫn khôi phục đầy đủ model/optimizer/
  scheduler/scaler/data cursor/RNG/history và dựng lại curves.
- Lint/format và wheel/sdist build pass. Installed wheel chạy acceptance ngoài source
  tree, không chỉ import. Sdist không chứa cache, runs, token payloads hoặc weights.
- Drive write-failure simulation giữ checkpoint trước đó; interrupts không lưu
  partial updates. CUDA FP16 overflow test lưu/restore skipped attempts, scaler/cursor/
  history; attempt cap ngăn smoke lặp dài khi không có successful update.
- ZIP tests từ chối checksum/path/duplicate corruption. Bundle thật đã giải nén và
  xác minh lại PackedCorpus/tokenizer. Notebook code cells compile; live subprocess
  stdout/stderr và Drive mount/resume cells đã được kiểm tra cấu trúc, chưa execute A100.

## CPU diagnostic

Lệnh `python -u scripts/check_pretraining_runner.py --device cpu`: authored fixture,
tiny99.136 parameters, block32/B2/G2, dropout0,1. Budget recipe6, dừng4; so full4 với
run2→checkpoint→resume4. Qua cả exact engine/data/RNG state và per-attempt batch hashes,
loss/LR/AMP trace. Validation chỉ fixed-prefix2 batches,123 targets:
`5.561959266662598 → 5.3860450372463315`.

Run: `artifacts/runs/20261119T105126959000Z-wp07-3-runner-smoke-seed42-5cf24cb3`.
Đây là verification trên authored fixture, không phải corpus-quality result.

## RTX 4060 Laptop — candidate và dữ liệu thật

PyTorch2.9.1+cu128/CUDA12.8, BF16, deterministic,1CPU thread, TF32off.
Candidate46.148.096 parameters; block256/B2/G8, recipe2.000 updates/warmup100 giữ
nguyên, chỉ chạy full4 và so với run2→checkpoint→resume4.
Lệnh:

```powershell
python -u scripts/check_pretraining_runner.py --config configs/training/wp07_pretrain.toml --prepared-root artifacts/wp07-3/pretraining_data --tokenizer artifacts/runs/20261105T154939888754Z-wp04-byte-bpe-seed42-4102afe0/artifacts/tokenizer --device cuda --precision bfloat16 --stop-after 4 --split-step 2
```

Full held-out validation:1.000 stories,1.344 packed blocks,341.625 targets.
Loss `9.166838798802107 → 7.66257223474312` sau4 successful updates;16.274 supervised
training targets,64 blocks đã tiêu thụ. Full/resumed model/Adam/scheduler/scaler/cursor/
RNG exact, từng batch/loss/LR/AMP trace exact và validation ở các mốc chung exact.
Validation thêm ở điểm dừng2 trong nhánh resume được giữ rõ trong history, không làm
thay đổi model/data/RNG. Không so chúng như hai lịch measurement giống hệt nhau.

Run: `artifacts/runs/20261119T105146024625Z-wp07-pretrain-tinystories-seed42-db0c908f`.
Full checkpoint đo được553.911.515 bytes (~0,516GiB) sau4updates. Đây không phải báo
cáo profiling throughput/VRAM; không ngoại suy số đo RTX4060 thành tốc độ A100.

## Handoff Colab

Notebook: `notebooks/wp07_pretraining_colab.ipynb`; bundle:
`dist/grast-wp07-data-block256-17971c0168e4.zip`,3.330.568 bytes.
SHA256 `9f08cdc7ba7c8a4825ce90470fd19fc3ed6da7d6d3bfd742097a389d1aab949d`.
Chỉ train/validation và frozen tokenizer WP04; không có test set hoặc weights.
Train10.000 stories/3.741.797tokens/14.674blocks; validation1.000stories/342.625tokens/
1.344blocks. Token payload hashes giữ nguyên WP07.2; block manifest fingerprint đổi
thành `17971c0168e4367b183ebb0a3abc7c1309ffe5095cdc67899bce13ffc4298faa` vì block256.

Mặc định notebook smoke4. Để train tiếp: đặt `SMOKE_ONLY=False`, điền `RESUME` trỏ
latest trên Drive, giữ nguyên source/recipe/precision/runtime tương thích. Summary ZIP
chứa config/metadata/log/report/curves; weights/milestones giữ trên Drive cho WP07.4–5.
Chưa có A100 execution evidence, checkpoint corpus hoàn chỉnh hoặc pilot result.
