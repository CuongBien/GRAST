# WP07 — Pretraining & CoT SFT theo phase

User chốt ngày 17/11/2026: chia nhỏ implementation; train dài do user chạy trên
Colab. Agent chuẩn bị và verify smoke ngắn, **mọi lệnh train phải in log trực tiếp**.
Không coi diagnostic fixed-batch loss là kết quả corpus hoặc pilot reasoning.

| Phase | Phạm vi | Gate | Trạng thái |
| --- | --- | --- | --- |
| WP07.1 | Engine, AdamW, warmup/cosine, AMP, clipping, accumulation, mandatory log, engine-state checkpoint | Token-weighted gradients đúng; FP32/BF16/FP16 chạy được; engine resume exact | Đã triển khai/verify local |
| WP07.2 | WP03/WP04 data pipeline, deterministic train order/packing, validation, full resume với data cursor | Dataset/tokenizer identities khóa; checkpoint khôi phục cả vị trí đọc; không test leakage | Đã triển khai/verify local |
| WP07.3 | Pretraining runner/config/Colab, periodic checkpoints, train/validation loss curves | Local smoke ổn; notebook log luôn hiển thị; user chạy corpus baseline | Đã nhận/verify checkpoint 2.000 updates T4 |
| WP07.4 | Chuẩn hóa prompt/reasoning/final, CoT SFT từ pretrained checkpoint, Colab/Kaggle runner | Prompt mask/target formatting đúng, validation và milestone checkpoints; user train dài | Đã triển khai/verify local; SFT dài pending |
| WP07.5 | Phân tích checkpoint nhiều giai đoạn, curves, text generation, fixed pilot reasoning | Checkpoint reload; text samples; pilot result/artifact manifest và evidence từ Colab | Chưa triển khai |

WP07 chỉ đóng khi có model checkpoint thực từ pretraining/SFT, curves và pilot.
Việc chạy corpus dài được giao cho user; nếu chưa có artifacts thì ghi pending thay
vì báo là đã train. Đã hoàn tất WP07.1–2: [phase 1](wp07_1_validation.md),
[data/resume contract](wp07_data.md), [phase 2](wp07_2_validation.md).
WP07.3: [runner, checkpoint và Colab](wp07_pretraining.md),
[local acceptance](wp07_3_validation.md). Ngày20/11 đã nhận và verify
[corpus checkpoint 2.000 updates](../experiments/wp07_pretraining_completed_20261120.json).
WP07.4: [CoT SFT/data/warm-start/notebooks](wp07_sft.md),
[local acceptance](wp07_4_validation.md). SFT dài và pilot vẫn pending.

## WP07.1 — engine API

```python
from llm_scratch.training.config import TrainingConfig
from llm_scratch.training.trainer import TrainingEngine
from llm_scratch.utils.seed import seed_everything

rng = seed_everything(42)
# Tạo/load model bình thường với FP32 master weights, rồi .to('cuda').
engine = TrainingEngine(model, TrainingConfig(precision="bfloat16"), generator=rng)
record = engine.train_update([microbatch_a, microbatch_b])
validation = engine.evaluate(validation_batches)
engine.save("checkpoints/step.pt", data_state={"next_window": 1})
# Khởi tạo engine cùng recipe/runtime và model/tokenizer contract:
cursor = restored_engine.load("checkpoints/step.pt")
```

Batch gồm integer `input_ids`, unshifted `labels` (-100 để bỏ prompt targets) và
optional binary `attention_mask`. Engine tự chuyển tensors sang model device;
label shift đúng một lần trong CausalLM WP05. Không bật inference KV cache khi train.
Một update nhận 1..G microbatches, hỗ trợ partial window cuối. Engine không tự chọn
train records, pack/truncate hoặc quyết định dataset splits; các việc này thuộc WP07.2.

Loss của update là tổng CE trên **supervised next-token pairs** chia tổng số pairs
cả window. Các pairs cần cả hai endpoints valid theo mask. Không lấy mean của các
microbatch means: prompt masks/padding/sequence lengths khác nhau có số targets khác.
Microbatch không có supervision được skip (không tiêu thụ dropout RNG); window rỗng
supervision bị từ chối trước optimizer update. Validation cũng token-weighted.

AdamW giữ mỗi unique trainable Parameter đúng một lần, kể cả tied embedding/LM head;
matrix weights có decay, norm/bias vectors không decay. Dùng foreach=False/fused=False
cho portable diagnostic baseline. Warmup W áp `(i+1)/W` cho update index i<W; sau đó
cosine từ base đến min_lr_ratio ở update cuối. Scheduler tiến sau successful optimizer
update, không theo microbatch hoặc AMP-overflow attempt. Config max_steps là **successful
optimizer updates**; attempts/tokens_seen và tokens_optimized được ghi riêng.

FP32 master weights cho tất cả precision policies. FP32 không autocast; BF16 dùng
autocast không scaling; FP16 CUDA dùng torch.amp.GradScaler. Unscale đúng một lần sau
cả accumulation window, rồi clip gradients và step/update scaler.
[PyTorch AMP accumulation/clipping contract](https://docs.pytorch.org/docs/2.9/notes/amp_examples.html).
FP16 overflow skip optimizer và scheduler, giảm scale; vẫn ghi attempt/tokens_seen.
Nonfinite loss hoặc gradients khi không có scaler là error, không silent step. Sau
training exception, engine yêu cầu load checkpoint trước khi train/save tiếp.

## Logging bắt buộc

Engine luôn `print(JSON, flush=True)` khi initialize, bắt đầu/kết thúc mỗi update,
validation, save/load checkpoint và lỗi. Không có quiet flag. Có loss, số supervised
tokens, LR đang dùng, gradient norm trước clip, skipped/AMP scale và elapsed throughput.
Đây là supervised-token throughput của engine, không phải inference benchmark.
Optional sink callback ghi đồng thời file logs; không thay thế stdout.

Runner chạy `python -u`, notebook dùng Popen đọc từng dòng stdout/stderr rồi print
flush=True, không capture toàn bộ output rồi mới in. Long runners ở các phase sau
phải giữ contract này; checkpoint milestones và validation cũng phải xuất log.

## Checkpoint foundation và giới hạn resume

WP02 atomic envelope vẫn giữ format_version=1. Engine payload có training_schema_version=1,
model/optimizer/scheduler/scaler, training config, step/attempt/token counters, runtime
signature, opaque `data_state`, cùng Python/NumPy/global CPU/CUDA RNG và NumPy Generator.
Model extra_state giữ model config/tokenizer fingerprint; tied weights được restore
bằng loader WP05. Chỉ save ở optimizer boundaries sạch, không có pending gradients.
Không hỗ trợ checkpoint giữa microbatch/accumulation window ở phase này.

Resume kiểm recipe/runtime và scheduler counters trước khi restore RNG; model/optimizer
load lỗi khiến engine failed để không train trên partial state. Exact test dùng cùng
device/precision/Torch/hardware/flags/thread count và dropout, không hứa bitwise equality
giữa CPU/GPU hoặc runtime khác. Source Git/packages được ghi ở run metadata. Checkpoint
inference có thể đọc model state với CPU mapping mà không restore training RNG.

`data_state` ở WP07.1 là cursor diagnostic do caller cung cấp. WP07.2 đã bổ sung full
`PretrainingSession` với dataset/hash/packing/shuffle/consumed cursor và next-block
verification; dùng [session API](wp07_data.md) để resume data. Không được bỏ data state
hoặc đổi tokenizer/recipe rồi gọi đó là exact continuation. Đổi stage sang SFT là
explicit warm-start recipe ở WP07.4, không resume optimizer pretrain tùy ý.

## Smoke local và Colab

```powershell
uv run --no-sync python -u scripts/check_training_loop.py --device cpu
uv run --no-sync python -u scripts/check_training_loop.py --device cuda --precision float16
uv run --no-sync python -u scripts/check_training_loop.py --device cuda --precision bfloat16 --profile candidate --tokenizer PATH/TO/wp04-tokenizer
```

Config riêng `configs/experiments/wp07_training_smoke.toml`: 6 successful updates,
save sau update 2, rồi resume và so exact model/optimizer/scheduler/scaler cùng loss/LR/
skip/scale của các attempts còn lại. Tiny 100.160 parameters cho CI; candidate 46.148.096
parameters bắt buộc dùng tokenizer WP04 thật. Dropout=0,1 để test RNG continuation.
FP16 scaler backoff được log/replay; diagnostic có attempt cap để không lặp vô hạn.

Artifacts có run/config/source/environment/metadata/log, resume.pt/final.pt và loss
history JSON/CSV. `diagnostic_eval_loss` đo **cùng authored fixture**, không phải held-out
validation hoặc corpus training curves. Không fit trên validation/test hoặc dùng pilot
để chọn threshold trong phase này.

Mở `notebooks/wp07_training_smoke.ipynb` trong Colab, chọn GPU (ưu tiên A100), chạy
cells để clone branch/install package, in môi trường và chạy tiny smoke. BF16 được
chọn khi GPU hỗ trợ, còn lại FP16; không fallback CPU im lặng. Notebook có cell zip/
download logs + checkpoints trước khi tắt runtime. Chưa có Colab execution evidence
trong repository; local CUDA verify và code-cell syntax checks không thay thế chạy A100.
Notebook corpus/Drive checkpointing đã có ở WP07.3; xem `wp07_pretraining_colab.ipynb`.
