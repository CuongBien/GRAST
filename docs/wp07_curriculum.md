# WP07.6.2 — Curriculum SFT và đánh giá đáp án tự sinh

WP07.6.1 cho thấy model có thể học thuộc64 mẫu nhưng áp dụng yếu sang cặp số mới.
Phase này chuẩn bị dữ liệu có độ phủ lớn hơn, curriculum tích lũy và notebook Colab.
Agent verify lượt ngắn; user đã train đủ2.800 updates trên Kaggle. Ngày25/11 nhận model/summary và
[đánh giá validation WP07.6.3](wp07_6_3_validation.md). Copy32/32, single5/32, multi0/32
ở additional validation; pilot5/150. Các smoke checks dưới đây xác minh pipeline,
không thay bằng chứng quality từ lượt train và evaluation thực.
Đã có [báo cáo kiểm chứng local và handoff](wp07_6_2_validation.md).
Kaggle: [notebook và hướng dẫn Save & Run All/resume](wp07_curriculum_kaggle.md).

## Dữ liệu và split

Config: `configs/data/wp07_curriculum.toml`. Dữ liệu có **13.976 train /3.361 validation**,
max387 tokens, cap512, không truncate/drop. Tokenizer WP048192 giữ fingerprint
`49187f5c5a806801ae6fc86cce7a9e6ca26d54d81cc989b361bbc76724abada3`.
Dataset fingerprint:
`0a891fda1dc3325a85f73a7022cef14fb6ce1b2d27467c1a7fbd42a7a0d16b56`.

| Task pool | Train | Validation | Nội dung |
| --- | ---: | ---: | --- |
| Copy | 3.780 | 981 | Chép đúng hai toán hạng theo thứ tự vào final `a,b` |
| Single | 6.196 | 1.580 | Cộng/trừ một bước; gồm500/100 WP03 easy |
| Multi | 4.000 | 800 | Authored2/3 phép cộng/trừ; thêm1.000/200 WP03 medium/hard với nhân/chia |

Authored dùng toán hạng31..99, khác miền leaves1..30 của WP03. Copy và single cùng
cặp số không phân biệt thứ tự nằm chung split qua hash(seed, unordered pair), kể cả
đổi operator/đảo toán hạng. Multi chia theo multiset leaves, chung split giữa các
operator/order có cùng bộ toán hạng. Phép cộng trùng theo commutativity được loại;
AST canonical IDs, prompt và question IDs được audit trong/giữa train/validation.

Phần cầu nối WP03 giữ **nguyên1.500 train /300 validation** và semantic split gốc;
không gọi phần này là split theo operand pair. Chỉ đọc hai file math/train.jsonl và
math/validation.jsonl sau kiểm manifest/fingerprint/SHA. Không đọc pilot/test/GSM8K.
Authored không thể có AST/prompt giống WP03 vì miền leaves khác. Không tuyên bố
đã kiểm chứng tương đương đại số tổng quát giữa các biểu thức khác AST.

Mỗi row lưu provenance, task, group, expression, prompt/reasoning/final và token IDs.
Loader tính lại AST answer bằng Fraction, reasoning, prompt và encoding; sai checksum,
mask, split/group, answer hoặc vượt cap đều fail. Special IDs và prompt mask dùng chung
`reasoning.formatting`; mỗi ví dụ độc lập, dynamic right padding khi train, không pack
nhiều câu vào một context. Copy dùng integer-pair scorer; toán dùng numeric-v1.

## Recipe dự kiến cho Colab

`configs/training/wp07_curriculum.toml`: model46.148.096 parameters/context1024,
weights-only từ **SFT500 thật**, SHA
`5f75909261c79e9f7f0c6e37a7e9d76ba5670be8b4be5133b6e148e0cd06ff42`.
Optimizer/scheduler/scaler/RNG/cursors mới; không dùng checkpoint overfit diagnostic
và không tiếp tục optimizer của SFT500.

| Successful updates | Pool được lấy mẫu |
| --- | --- |
| 1–400 | Copy |
| 401–1.600 | Copy + single |
| 1.601–2.800 | Copy + single + multi |

Mỗi pool shuffle theo epoch, không weighted sampling; các bài trước được ôn qua pool
tích lũy. Stage chuyển tại successful step cố định, không dựa vào điểm validation vừa
thấy. Optimizer/LR liên tục giữa stage; mỗi stage có consumed cursor riêng. AMP skip
tiêu thụ data/RNG nhưng không tiến successful step hoặc scheduler.

Microbatch4/accumulation4, effective16 trừ window cuối epoch có thể ngắn hơn. AdamW
LR1e-4/weight decay0,01, warmup50, cosine đến0,1×LR, clip1. Recipe FP16; notebook chọn
BF16 khi GPU hỗ trợ. Đây là recipe khởi đầu để user train, chưa có bằng chứng hội tụ.

## Validation và chọn model

Step0, mỗi200 successful updates và cuối stage: full teacher-forced loss trên3.361
validation examples/112.788 target tokens, cùng **48 câu tự sinh cố định**. Selection
chỉ theo hash(seed, question_id), độc lập với đáp án/output:

- Copy16 authored.
- Single8 authored +8 WP03 easy.
- Multi4 authored ops2 +4 authored ops3 +4 WP03 medium +4 WP03 hard.

Inference prefix chỉ có câu hỏi và forced `<think>`, không có rationale/answer; greedy,
cached, left padding, FP32, budget256. Mọi raw prediction được ghi trước scoring. Final
phải nằm trong đúng một cặp native final IDs; invalid tính sai, không đoán số cuối.

`selected_model.pt` được chọn theo **mean của ba task EM**, hòa điểm giữ step sớm hơn;
không chọn bằng loss. Scope48 câu là lựa chọn checkpoint với độ phân giải còn thô,
không thay kết quả full pilot sau train. Score không xác nhận reasoning trace đúng.
Nếu không model nào tốt hơn step0, export vẫn giữ step0 và log/report ghi không cải thiện;
không coi training chạy hết là quality pass. Chưa fit confidence threshold.

## Checkpoint, resume và inference

`latest.pt` là full checkpoint nguyên tử mỗi100 updates, stage end và controlled stop:
model/optimizer/scheduler/scaler, Python/NumPy/Torch CPU/CUDA RNG, cả ba train cursors,
history, fixed validation predictions và **best weights**. Resume chỉ cần file này
cùng bundle; không cần checkpoint SFT gốc hoặc sibling best file. Recipe, runtime,
source implementation, dataset/tokenizer và selection IDs phải khớp.

Giữ full checkpoint ở stage ends400/1600, final2800. Full latest có thêm một bản best
weights nên lớn hơn checkpoint SFT cũ; output lên Drive cần chừa vài GB cho các mốc.
`selected_model.pt` là **inference-only** export có metadata/weight checksum, không có
optimizer để resume. API:

```python
from llm_scratch.curriculum.evaluation import load_selected
model, metadata = load_selected(selected_path, tokenizer)
model = model.to("cuda").eval()
# Dùng cùng inference.generate và encode_reasoning_prompt như các phase trước.
```

Controlled stop giữa hai lịch evaluation không thêm một vòng validation mới; resume
giữ cùng sequence cập nhật/selection như lượt liên tục. Khi runtime ngắt giữa update,
resume từ latest đã hoàn tất gần nhất; không có checkpoint giữa microbatches.

## File cho Colab

1. Mở `notebooks/wp07_curriculum_colab.ipynb`, chọn GPU.
2. Đặt `dist/grast-wp07-curriculum-0a891fda1dc3.zip` và SFT500 `final.pt` (đổi tên
   `sft500-final.pt`) vào `MyDrive/GRAST/wp07/curriculum/inputs/`, hoặc sửa đường dẫn.
3. Chạy các cells. `STOP_AFTER=None` chạy đủ2.800 updates; notebook clone source pin,
   không thay PyTorch, kiểm bundle/source SHA, chọn precision và stream từng dòng log.
4. Output nằm tại `MyDrive/GRAST/wp07/curriculum/runs/`. Khi ngắt, đặt `RESUME_CHECKPOINT`
   đúng đường dẫn latest.pt và giữ runtime/precision/source pin như cũ.
5. Để đánh giá phase tiếp, cung cấp **selected_model.pt và summary ZIP**. ZIP summary
   không chứa weights; latest.pt chỉ cần nếu tiếp tục training.

Bundle3.059.243 bytes, SHA256:
`02c9e6a9d969365921d00cc119e614295a25f1d1bde53fb7808141f2ffcee217`.
Bundle extractor kiểm allowlist, duplicate entries, symlink, size, SHA từng payload và
toàn bộ Corpus. Existing explicit checkpoint paths được tôn trọng, không tự chọn file khác.

## Lệnh local và smoke

```powershell
uv run --no-sync python -u scripts/prepare_curriculum.py --dataset-root PATH/TO/wp03-datasets --tokenizer PATH/TO/wp04-tokenizer --output NEW_CURRICULUM_DATA
uv run --no-sync python -u scripts/train_curriculum.py --prepared-root NEW_CURRICULUM_DATA --tokenizer PATH/TO/wp04-tokenizer --init-checkpoint artifacts/checkpoints/wp07-cot-sft-math-500/final.pt --device cuda --stop-after 2
uv run --no-sync python -u scripts/train_curriculum.py --prepared-root NEW_CURRICULUM_DATA --tokenizer PATH/TO/wp04-tokenizer --resume PATH/TO/checkpoints/latest.pt --device cuda
uv run --no-sync python -u -m llm_scratch.curriculum.acceptance
```

Offline acceptance dùng model6.832 parameters/dropout0,1, cả ba stage, dừng ở step3,
resume từ một file latest được copy riêng rồi so toàn bộ model/optimizer/scaler/
scheduler/RNG/cursor/metrics với lượt liên tục. Fixture này kiểm chức năng; không
được trình bày là kết quả reasoning của model46M hoặc lượt train2800updates.
