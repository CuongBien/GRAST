# WP07.4 — CoT formatting và SFT

Đã nhận/verify checkpoint user train đủ 500 updates trên T4/FP16: 253.174 targets,
validation loss 0,543062 và không AMP skip. [Evidence](../experiments/wp07_sft_completed_20261120.json).
Đánh giá chuỗi model tự sinh tiếp tục tại [WP07.5](wp07_capability.md).

SFT dùng model/tokenizer của pretraining, nạp **trọng số** rồi tạo optimizer/scheduler/
scaler/RNG và data cursor mới. Đây là chuyển stage có chủ đích; không resume optimizer
pretraining. Sau khi đã có SFT checkpoint, `--resume` khôi phục đầy đủ state của SFT.
Agent chỉ verify chạy ngắn; user train đủ budget trên Colab/Kaggle. Mọi lệnh luôn in log.

## Baseline đã nhận

Checkpoint pretraining 2.000 updates nằm ở
`artifacts/checkpoints/wp07-pretrain-tinystories-2000/final.pt`, chuyển từ root theo
yêu cầu user. Không đưa weights vào Git. SHA256:
`27cf2d9cb9774b0023d4db09e2735fd666452f5a945468bc71722bb3c69b5309`.
Đã kiểm trực tiếp checkpoint và summary ZIP: 46.148.096 parameters, 8.130.916 targets,
2.000 attempts/2.000 updates, validation loss cuối1,5673957147, perplexity4,79415.
Runtime Torch2.11.0+cu130/CUDA13.0/TeslaT4/FP16; runner dùng cuda:0, gpu_count2 là số
GPU nhìn thấy, không phải DDP hai GPU. Summary và provenance giữ cạnh checkpoint;
[evidence](../experiments/wp07_pretraining_completed_20261120.json) không chứa weights.

Fresh SFT cho phép chuyển runtime/precision vì chỉ nạp FP32 weights, kiểm tất cả tensor
finite/shape/dtype, model metadata, tokenizer và tied weights trước khi thay đổi model.
Không resize vocab, không fit lại tokenizer. Exact SFT resume vẫn khóa runtime/recipe/
implementation và dataset/cursor/RNG như WP07.2–3. Để tiếp tục **pretraining**, dùng
source pin WP07.3 cũ; source mới có SFT nên implementation fingerprint khác.

## Dataset và template

Default chỉ lấy WP03 `math/train.jsonl` và `math/validation.jsonl`: 1.500 train
(500 mỗi mức easy/medium/hard), 300 validation (100 mỗi mức). Không đọc test/pilot;
không trộn validation vào train. GSM8K giữ vai trò benchmark bổ sung mặc định.
Có option explicit `--sources math gsm8k` nếu sau này quyết định dùng official-train
rationale; manifest ghi nguồn và khi đó không được gọi GSM8K là unseen-domain transfer.
Official test GSM8K không được nhận làm nguồn SFT.

Math reasoning sinh deterministic từ AST WP03 theo đúng thứ tự R1/R2/... trong đề,
tính bằng Fraction và kiểm final answer. Ví dụ `R1 = 2 - 5 = -3.`,
`R2 = R1 / 6 = -1/2.`. Prompt phải khớp render của AST; answer phải khớp canonical
numeric-v1. GSM8K optional tách rationale trước đúng một `####`, normalize final sau
marker. Không suy luận gold từ model hoặc đoán số cuối rationale.

Normalized rows có question_id/source/split/difficulty, prompt/reasoning/final_answer,
input_ids/labels/attention_mask. Shared `reasoning.formatting` WP04 giữ cùng prefix
cho train/inference:

```text
<bos>question\n<think>reasoning</think><final>canonical_answer</final><eos>
```

Giữ nguyên question WP03, gồm instruction numeric answer, và dùng forced `<think>`
prefix nhất quán khi inference. Prompt đến hết `<think>` có labels=-100. Reasoning,
closing think/final markers, answer và EOS là targets. Causal shift đúng một lần trong
model. Literal token strings trong user text vẫn encode như bytes, không thành special
IDs. `<pad>` không là target; attention mask che padding.

Mỗi example là một sequence riêng, không concatenate/pack nhiều câu vào một context.
Dynamic right padding theo batch; không có attention giữa các examples. `block_size`
ở SFT recipe là **max_length cap**, không phải fixed padding length. Vượt cap thì fail
rõ, không cắt prompt/reasoning/final, không drop hàng im lặng. Loader kiểm lại encoding,
mask/framing/final/tokenizer/hash; cursor tái dùng private CPU per-epoch shuffle,
consumed-window commit/abort và full resume đã test ở WP07.2.

Input đã chuẩn bị với cap512: train209.570 prompt tokens +94.915 target tokens;
validation41.880 prompt tokens +18.971 targets; độ dài61–387 tokens. Không truncate/drop.
Fingerprint `78f779a7d367a91f505332a804edbe40a0286ed5fb932269103c18819d35395b`.
Tokenization giữ codec WP04 fingerprint
`49187f5c5a806801ae6fc86cce7a9e6ca26d54d81cc989b361bbc76724abada3`.

## Chạy local

```powershell
uv sync --locked --extra cuda --extra data --extra training
uv run --no-sync python -u scripts/prepare_sft_data.py --dataset-root PATH/TO/wp03-datasets --tokenizer PATH/TO/wp04-tokenizer --output NEW_SFT_DATA --max-length 512
uv run --no-sync python -u scripts/finetune.py --prepared-root NEW_SFT_DATA --tokenizer PATH/TO/wp04-tokenizer --init-checkpoint artifacts/checkpoints/wp07-pretrain-tinystories-2000/final.pt --device cuda --stop-after 4
```

Installed CLI: `python -u -m llm_scratch.training.sft`. `--init-checkpoint` nhận
pretraining checkpoint có step>0; `--resume` nhận full **SFT** checkpoint, hai flags
loại trừ nhau. Resume không cần giữ file pretraining gốc, nhưng provenance/hash của
parent vẫn đi theo checkpoint. Mỗi invocation tạo run mới và ghi lineage.

Config `configs/training/wp07_sft.toml`: candidate46,15M/context1024/dropout0,
max_length512, microbatch2/accumulation4, AdamW LR5e-5/weight decay0,1, warmup25,
cosine đến0,1×LR, clip1, FP16, **500 successful updates** (~2,67 epochs). Đây là recipe
khởi đầu, không tuyên bố tối ưu/hội tụ. Chọn model/recipe bằng validation; không test.
`--precision bfloat16` dành GPU hỗ trợ; override nằm trong snapshot và khóa khi resume.

Full validation 300 examples/18.971 targets trước train, mỗi25 updates và controlled
stop. Loss token-weighted chỉ trên targets, không phải trung bình loss các batch/câu.
Latest checkpoint atomic mỗi25 updates; giữ mốc50/100/250 và final500. Source parent
không bị ghi đè. Các artifacts gồm config/source/metadata/lineage/environment/logs,
frozen tokenizer/dataset manifest, full loss history/CSV/PNG/SVG/report và checkpoints.
`--stop-after 4` dừng sớm trong recipe500, giữ nguyên scheduler, không đổi max_steps.

```powershell
uv run --no-sync python -u scripts/finetune.py --prepared-root NEW_SFT_DATA --tokenizer PATH/TO/wp04-tokenizer --resume PATH/TO/sft/checkpoints/latest.pt --device cuda
uv run --no-sync python -u scripts/check_sft.py --device cpu
```

`check_sft` bounded2..8 successful updates, so run liên tục với stop/resume, full state,
batch/loss/LR/scaler/RNG/cursor và common validation measurements. Offline mặc định
tạo fixture math và một pretraining update tiny; không coi đó là pretrained baseline.
Real verification truyền đủ `--prepared-root`, `--tokenizer`, `--init-checkpoint`, config
candidate và CUDA/precision. Không chạy train dài từ acceptance tool.

## Colab/Kaggle

Bundle `dist/grast-wp07-sft-math-78f779a7d367.zip`, SHA256
`100296172c05aff2bb99a4cc314cadf6dece5c65e0d7d7bb846a323794e7296e`.
Gồm train/validation normalized examples, manifest và tokenizer; không test/weights.
Checkpoint pretraining là input riêng. Bundle extractor kiểm SHA/allowlist/duplicates/
symlinks/size, hashes từng payload, rồi kiểm lại codec và full SFT examples.

`notebooks/wp07_sft_colab.ipynb`: mount Drive, đặt bundle và pretrained `final.pt` tại
`MyDrive/GRAST/wp07/sft/inputs/` hoặc sửa path, output trực tiếp `.../sft/runs/`.
`notebooks/wp07_sft_kaggle.ipynb`: Add Input bundle và pretraining output chứa final.pt;
tự tìm đúng SHA256, không upload/mount tương tác. Chọn GPU/Internet, Save Version →
Save & Run All cho batch session sạch. Output tại `/kaggle/working/grast-wp07-sft/runs/`.
Chỉ một GPU cuda:0 được dùng, kể cả có hai T4 nhìn thấy.

Cả hai mặc định `SMOKE_ONLY=True` dừng4updates, stdout/stderr stream từng dòng và
validation-start/checkpoint-start logs rõ. Sau smoke, đặt False và `RESUME` trỏ latest
**SFT** để tiếp tục500; nếu RESUME rỗng thì bắt đầu SFT mới từ pretrained weights.
Kaggle resume cần Add Input output của version trước và dùng path thực dưới
`/kaggle/input/`; Colab dùng path Drive. Không cần checkpoint pretraining khi resume.
Giữ source pin/runtime/precision/recipe tương thích cho exact continuation.

Native precision selection giống adapter pretraining: capability major>=8 và BF16
supported mới chọn BF16; T4/P100 dùng FP16 kể cả runtime báo có BF16 emulation.

Summary ZIP chỉ có logs/config/metadata/report/curves. Giữ latest/final/milestone weights
cho WP07.5. Agent chưa chạy đủ500 updates trên Colab/Kaggle. SFT loss tốt chưa thay thế
generation/pilot reasoning evaluation; việc đó thuộc WP07.5.

## Khắc phục `ModuleNotFoundError: llm_scratch`

Editable install tạo `.pth` cho Python khởi động sau; kernel notebook đang chạy
không tự nạp lại file này. Setup mới kích hoạt `REPO/src` trong `sys.path`, chuyển
đường dẫn qua `PYTHONPATH` cho subprocess, invalidate import caches và kiểm tra cả
kernel lẫn subprocess. Cần thấy hai dòng `Kernel imports verified:` và
`Subprocess imports verified:` trước khi chạy input/training cells.

Nếu đang mở notebook cũ và setup đã cài package xong, chạy cell sau rồi chạy lại
cell input bị lỗi:

```python
import importlib
import sys

sys.path.insert(0, str((REPO / "src").resolve()))
importlib.invalidate_caches()
```

Nếu chưa có biến `REPO`, chạy setup trước. Với notebook mới, chạy lại setup là đủ
cho editable install mới; nếu kernel đã cache package ở vị trí khác, setup báo rõ
cần restart kernel rồi chạy lại từ đầu. Source pin, recipe và checkpoints SFT giữ
cùng contract; sửa này chỉ thay import bootstrap của notebook.

## Kaggle không tìm thấy pretraining checkpoint

File local `artifacts/checkpoints/wp07-pretrain-tinystories-2000/final.pt` chưa tự
xuất hiện trong `/kaggle/input`. Add Input output của lượt pretraining, hoặc tạo
Kaggle Dataset từ chính `final.pt` (554.303.513 bytes), rồi Add Input dataset đó.
SFT bundle410KB và summary ZIP không chứa weights; không thể dùng chúng thay final.pt.

Cell input mới ưu tiên tìm `final.pt`, rồi nhận các file `.pt` bị đổi tên nếu đúng
SHA256 đã xác minh. Có thể đặt `PRETRAIN_CHECKPOINT` thành path thực trước cell input;
cell không ghi đè lựa chọn đó. Nếu dùng notebook cũ, sửa trực tiếp dòng
`PRETRAIN_CHECKPOINT = ""` trong cell input thành đường dẫn thật dưới `/kaggle/input`.

Không có `.pt`: lỗi nói rõ cần Add Input weights. Có files nhưng hash không khớp:
lỗi riêng báo expected SHA256 và log candidate path/size/hash để kiểm tra đã gắn
đúng bản2.000updates. Fresh SFT vẫn chỉ nhận checkpoint đã xác minh; SFT resume dùng
`RESUME` và không yêu cầu parent pretraining còn tồn tại.
