# WP07.6.2 — Curriculum SFT trên Kaggle

Notebook: `notebooks/wp07_curriculum_kaggle.ipynb`. Đây là bản Kaggle của
[curriculum Colab](wp07_curriculum.md), giữ nguyên source pin
`2383ab9a2900dc276eec7305a7e7d8cdfc5f2903`, tokenizer, data và recipe2.800 updates.
Mặc định smoke2 updates; agent chỉ verify adapter local, chưa chạy thực tế Kaggle.

## Chuẩn bị hai input

Tạo một Kaggle Dataset từ hai file sau, hoặc tạo riêng rồi gắn cả hai bằng Add Input:

| File local | Dùng để làm gì |
| --- | --- |
| `dist/grast-wp07-curriculum-0a891fda1dc3.zip` | Data train/validation và tokenizer; khoảng3MB |
| `artifacts/checkpoints/wp07-cot-sft-math-500/final.pt` | Trọng số SFT500 thật; khoảng554MB |

Có thể giữ Dataset private. Không tải pretraining final.pt hoặc checkpoint overfit
WP07.6.1 vào vị trí SFT500. Summary ZIP không chứa weights nên không đủ để bắt đầu train.
Tên file weights có thể đổi; notebook tìm theo SHA256, không chọn chỉ vì tên final.pt.

SHA bundle: `02c9e6a9d969365921d00cc119e614295a25f1d1bde53fb7808141f2ffcee217`.
SHA SFT500: `5f75909261c79e9f7f0c6e37a7e9d76ba5670be8b4be5133b6e148e0cd06ff42`.
Nếu Kaggle tự giải nén ZIP, notebook nhận layout có bundle_manifest.json và kiểm
manifest/payload hashes tương ứng. Nhiều bundle cùng gắn hoặc path tùy chỉnh: điền
`BUNDLE_PATH` rõ ràng trong configuration cell, không sửa các SHA đã khóa.

## Lần đầu: chạy smoke

1. Tạo notebook Kaggle và import file `wp07_curriculum_kaggle.ipynb` qua menu File.
2. Gắn Dataset ở trên bằng **Add Input**. Bật **Internet**, chọn **GPU** trong Settings.
3. Giữ `SMOKE_ONLY=True`, `RESUME=""`; để `SFT_CHECKPOINT=""` cho tự tìm theo SHA.
4. Chọn **Save Version → Save & Run All** và kiểm accelerator của phiên batch.
5. Khi version hoàn tất, mở logs/Output. Cần thấy report `controlled_stop`, step2,
   checkpoints/latest.pt, checkpoints/selected_model.pt và summary ZIP.

Theo [tài liệu Kaggle](https://www.kaggle.com/docs/notebooks), Save & Run All chạy
toàn notebook trong phiên sạch, tách khỏi phiên editor. Output trong /kaggle/working
có thể gắn làm Input cho notebook tiếp. Vì vậy mọi file cần thiết phải được Add Input
trước khi chạy nền; không dựa vào file chỉ có trong phiên tương tác.

Notebook không mount Drive hoặc mở hộp upload/download. Source checkout và bundle
giải nén tạm nằm ngoài working; project commands dùng subprocess mới với PYTHONPATH
trỏ đúng source pin, tránh import nhầm package cũ trong kernel. PyTorch sẵn có được giữ.

## Train đủ recipe

Đổi **`SMOKE_ONLY=False`**, còn `RESUME=""`: bắt đầu curriculum mới từ SFT500.
Recipe cố định400 copy →1.200 copy+single →1.200 copy+single+multi successful updates.
Các stage sau vẫn ôn task trước, không tự tăng/giảm budget theo điểm quan sát.

Mỗi update in loss, LR, gradient norm, AMP scale/skip. Full validation có log tiến độ
mỗi50 batches;48 câu validation tự sinh được đánh giá mỗi200 updates/cuối stage.
Model tốt nhất chọn bằng macro EM của ba task, hòa điểm giữ mốc sớm; không chọn bằng
loss, không dùng test set. Training hoàn tất không tự đồng nghĩa quality đã đạt.

GPU có BF16 native thì chọn BF16; còn lại FP16. Có probe forward/backward thật trước
training, không fallback CPU. Kể cả hai GPU nhìn thấy, runner chỉ dùng **cuda:0**, không DDP.
Không mặc định có A100 hoặc một quota/thời lượng cố định; xem thông tin runtime thực tế.

## Tiếp tục từ output smoke hoặc lượt bị ngắt

1. Add Input **Notebook Output của version trước** và giữ input bundle.
2. Sao chép đường dẫn thật của `runs/.../checkpoints/latest.pt` từ Input panel vào RESUME.
3. Đặt `SMOKE_ONLY=False` và Save & Run All. Khi resume không cần gắn SFT500 gốc.

```python
SMOKE_ONLY = False
RESUME = "/kaggle/input/PATH-THUC-TE/runs/RUN-THUC-TE/checkpoints/latest.pt"
```

Không dùng nguyên placeholder trong ví dụ. Chỉ `latest.pt` chứa full state và best
weights; `selected_model.pt` là inference-only, không resume optimizer. Khi giữ
`SMOKE_ONLY=True` cho resume, adapter chạy thêm tối đa2 successful updates, thay vì
đặt stop target2 thấp hơn checkpoint đã có.

Giữ source pin, runtime image/Torch/CUDA, GPU type/count và precision tương thích.
Resume giữ precision đã lưu, không tự đổi sang BF16 vì GPU mới hỗ trợ. Engine kiểm
model/optimizer/scaler/scheduler/RNG, cả ba cursors, data và implementation. Colab→Kaggle
hoặc đổi GPU/image có thể bị từ chối exact resume; adapter không bỏ state để ép chạy.
Lượt mới từ SFT500 weights được phép dùng runtime khác vì khởi tạo optimizer mới.

`latest.pt` lưu nguyên tử mỗi100 updates, stage end và controlled stop. Runtime bị
kill trước khi Notebook Output được lưu không phải là backup chắc chắn; giữ/tải output
version đã hoàn tất khi có thể. Dừng trong editor sẽ gửi SIGINT cho child, chờ giới hạn
rồi kill nếu cần; resume từ latest hoàn tất trước đó, không từ giữa microbatch.

## Output và file cần gửi lại

Output root: `/kaggle/working/grast-wp07-curriculum/`:

- `console.log`: stdout/stderr của subprocess; GPU/precision/source ở `kaggle_environment.json`.
- `runs/.../`: config, metadata, event logs, history/CSV/PNG/SVG, generated validation.
- `runs/.../checkpoints/latest.pt`: full continuation; khoảng755MB theo model smoke.
- `runs/.../checkpoints/selected_model.pt`: model được chọn cho inference; khoảng201MB.
- Full run còn giữ stage-end400/1600 và final2800; tổng output weights vài GB.
- `RUN-summary.zip`: metrics/logs/curves, không có model weights.

Sau train, giữ **selected_model.pt + summary ZIP** để đánh giá phase tiếp. Cần latest.pt
nếu muốn tiếp tục optimizer. Không lấy model smoke step0 làm bằng chứng curriculum đã
cải thiện; smoke chỉ kiểm tra chạy và checkpoint. [Contracts và quality gate](wp07_curriculum.md).

## Kiểm tra lỗi thường gặp

| Thông báo | Cách xử lý |
| --- | --- |
| No .pt weights found | Add Input chứa SFT500 final.pt; summary ZIP không thay thế weights |
| No checkpoint matches SFT500 | Đang gắn nhầm pretrain/diagnostic/weights khác; dùng đúng SHA nguồn |
| Multiple curriculum inputs | Để một bundle hoặc điền BUNDLE_PATH tới ZIP/layout cụ thể |
| Explicit path does not exist | Chép lại path thật từ Input; path explicit sai sẽ không tự fallback |
| Choose a GPU / GPU probe failed | Chọn accelerator và kiểm runtime image; không chạy tiếp bằng CPU |
| Incompatible config/runtime on resume | Dùng runtime/precision/source tương thích; không đổi recipe để ép checkpoint |

Adapter đã có local tests cho ZIP/expanded/corrupt/ambiguous input, nhận weights theo
hash, smoke/full/resume commands, live logs/cancellation, và pipeline tiny CPU qua
chính helpers notebook. Local GPU probe không thay cho một lần chạy thực tế Kaggle.

Kiểm chứng22/11/2026:36 targeted tests passed (26 tests adapter mới), Ruff lint/format
qua. Helpers đã kiểm bundle thật cả ZIP và expanded, SFT500 theo SHA và header của
checkpoint curriculum step4. RTX4060 local forward/backward probe qua với BF16 cho
fresh run và FP16 khi resume; lệnh smoke resume từ step4 nhắm step6. Chỉ build lệnh
model thật, không launch train dài; luồng launch/resume qua đủ ba stage dùng fixture
tiny CPU. Source training/config khớp pin, không sửa thuật toán hoặc recipe để lên Kaggle.
[Evidence local](../experiments/wp07_curriculum_kaggle_validation_20261122.json).
[CI adapter](https://github.com/CuongBien/GRAST/actions/runs/45038234769) đã xanh trên
Windows và Ubuntu; kết quả này không phải phiên Kaggle.
