# WP07.1 validation — 17/11/2026

**Đạt nghiệm thu phase 1**: training engine/AdamW/scheduler/AMP/gradient accumulation,
mandatory stdout logging và engine checkpoint continuation. WP07 chưa đóng; xem
[5 phases và contracts](wp07_training.md). Corpus training dài do user chạy trên Colab.

## Kiểm thử

**639 tests toàn repository đạt local**; 27 cases mới gồm 22 CPU và 5 CUDA.
CPU-only CI skip ba hardware cases toàn repository, gồm FP16 overflow test mới.

- Update với hai microbatches có 4 và 1 supervised pairs khớp independent combined
  batch reference; không mean-of-means. Zero-supervision và partial windows đúng.
- AdamW unique tied Parameter coverage, matrix/norm decay groups; warmup/cosine
  endpoint semantics và successful-update budget được kiểm tra.
- FP32/BF16 loss/gradients finite và diagnostic loss giảm; FP16 master weights giữ FP32.
- Forced FP16 gradient overflow không đổi weights/optimizer step/scheduler, giảm
  scaler; checkpoint save/load sau skip vẫn tiếp tục đúng.
- Save step 2 rồi train tiếp so exact model/Adam slots/scheduler/scaler/loss/LR,
  có dropout và Python/NumPy/NumPy Generator/CPU/CUDA RNG. Có thêm kiểm thử khôi phục
  trong **một tiến trình Python mới**.
- Incompatible config/runtime/counters bị từ chối; checkpoint chỉ ở clean update
  boundaries, error recovery cần reload. Validation token-weighted không tạo gradients,
  đổi RNG hoặc tăng progress. Stdout log luôn hiện mỗi update/validation/checkpoint.

Ruff lint/format, wheel/sdist và installed-wheel smoke/resume đạt. Code cells trong
Colab notebook đã parse/lint/format; **chưa có bằng chứng notebook chạy trên A100**.
Local CUDA verify không được gọi là Colab acceptance.

## Smoke ngắn trên commit sạch

Source implementation commit `dda47201a032442f13668f2d1eacf4b5aa88a260`, Git clean
trước mỗi run. Python 3.11.16, PyTorch 2.9.1+cu128, CUDA 12.8, RTX 4060 Laptop GPU.
6 successful updates, accumulation 2, dropout 0,1; checkpoint sau update 2 và replay
4 updates còn lại. Các run này không có overflow tự nhiên; overflow/backoff được
kiểm tra bằng forced-gradient unit case.

| Device / precision | Parameters | Initial diagnostic eval loss | Final diagnostic eval loss | Resume |
| --- | --- | --- | --- | --- |
| CPU / FP32 tiny | 100.160 | 5,705545 | 4,091048 | Exact |
| RTX 4060 / FP16 tiny | 100.160 | 5,705499 | 4,035496 | Exact |
| RTX 4060 / BF16 candidate + WP04 codec | 46.148.096 | 8,976757 | 2,693684 | Exact |

Candidate dùng authored pretraining/SFT fixture WP05 được encode bằng tokenizer
WP04 thật. Learning rate 0,003 ở recipe này là diagnostic smoke, không phải hyperparameter
đã chọn cho corpus. Train loss ghi trước update, diagnostic eval loss ghi sau update,
trên cùng fixture. **Không phải held-out validation curve, text-quality hoặc pilot result.**

Run directory chứa stdout-equivalent file events/log, config snapshots, environment,
Git/source/runtime metadata, resume.pt/final.pt và loss_history.json/CSV. Weights/Adam
checkpoints lớn không commit vào repository. Report/hashes/history được phiên bản hóa
trong [validation manifest](../experiments/wp07_1_validation_20261117.json).

## Colab và bước tiếp

Notebook: [wp07_training_smoke.ipynb](../notebooks/wp07_training_smoke.ipynb).
Chọn GPU runtime, ưu tiên A100, chạy tất cả cells. Process train dùng Python `-u`;
notebook forward từng dòng stdout/stderr ngay lập tức, không dùng quiet/capture-all.
Notebook tiny không cần download corpus hoặc weights; có zip/download artifact cell.

```powershell
uv run --no-sync python -u scripts/check_training_loop.py --device cpu
uv run --no-sync python -u scripts/check_training_loop.py --device cuda --precision float16
uv run --no-sync python -u scripts/check_training_loop.py --device cuda --precision bfloat16 --profile candidate --tokenizer PATH/TO/wp04-tokenizer
```

WP07.2 tiếp tục data/packing/shuffle/sampler cursor và validation split có identity/hash.
Opaque data_state ở phase 1 chỉ chứng minh diagnostic cursor round-trip; chưa nghiệm thu
corpus exact resume. WP07.3/4 sẽ chuẩn bị runners/notebooks train dài, periodic Drive
checkpoints, corpus curves và CoT SFT để user chạy. WP07.5 sẽ tổng hợp checkpoint capability,
text samples và fixed pilot reasoning từ artifacts thực; không suy ra các kết quả đó
từ loss giảm trên smoke fixture.
