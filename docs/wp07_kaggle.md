# WP07.3 — Chạy nền trên Kaggle

Cập nhật20/11/2026: đã nhận checkpoint pretraining T4/FP16 đủ2.000updates, val loss
1,56740 và kiểm trực tiếp weights/metadata/summary; xem
[evidence](../experiments/wp07_pretraining_completed_20261120.json).
Các ghi chú local-only phía dưới mô tả kiểm chứng adapter ngày19/11.
SFT tiếp theo có [notebook và contract WP07.4](wp07_sft.md).

Notebook: `notebooks/wp07_pretraining_kaggle.ipynb`. Bản Colab giữ riêng.
Runner/model/recipe vẫn pin source `aa4be36b33ca025701421ef48715a22b43c4e8db`;
không đổi thuật toán train hoặc budget để chuyển môi trường.

## Lần đầu: smoke

1. Import notebook Kaggle; bật Internet để clone source/cài package, chọn GPU trong
   Settings. Notebook dùng PyTorch/CUDA có sẵn, không cài đè Torch.
2. Tạo Kaggle Dataset từ `dist/grast-wp07-data-block256-17971c0168e4.zip`, rồi Add Input
   dataset đó vào notebook. Có thể giữ dataset private. ZIP nguyên bản hoặc layout
   đã giải nén đều được nhận tự động; SHA256 từng payload và identities được kiểm tra.
3. Giữ `SMOKE_ONLY=True`, `RESUME=""`; chọn **Save Version → Save & Run All**, xác nhận
   GPU của batch session. Notebook chạy từ đầu trong phiên mới, không hiện hộp upload,
   mount Drive hoặc download tương tác. Có thể đóng trình duyệt sau khi phiên nền bắt đầu.
4. Khi version hoàn tất, xem logs và Output. Smoke chỉ4 successful updates; cần thấy
   report `stopped_at_boundary`, loss curves và `runs/.../checkpoints/latest.pt`.

Kaggle mô tả Save & Run All như một phiên độc lập chạy toàn notebook, đồng thời lưu
code/log/output của version: [tài liệu chính thức](https://www.kaggle.com/docs/notebooks).
Run All trong editor không thay thế chế độ này. Kiểm tra accelerator/thời lượng/quota
thực tế ở UI; không mặc định Kaggle cung cấp A100.

## Train đủ budget và resume

Đặt `SMOKE_ONLY=False`. `RESUME=""` sẽ train từ đầu2.000 updates với seed42.
Để tiếp tục smoke, gắn **Output của version smoke** làm Input rồi điền `RESUME` bằng
đường dẫn đầy đủ tới `latest.pt` dưới `/kaggle/input/...`.

```python
SMOKE_ONLY = False
RESUME = "/kaggle/input/previous-notebook-output/grast-wp07/runs/.../checkpoints/latest.pt"
```

Tên mount/path thực tế xem trong Input panel, không chép nguyên placeholder.
Save & Run All bắt đầu sạch nên không dựa vào file chỉ tồn tại trong phiên editor.
Giữ source/recipe/runtime image/GPU count/precision tương thích với smoke. Loader
từ chối exact resume khi Torch/GPU/precision khác, kể cả chuyển từ Colab sang Kaggle;
không tự bỏ optimizer/RNG/data state để ép resume. Có thể train mới trên Kaggle từ seed42.

## GPU và logging

GPU có BF16 native (compute capability major>=8 và Torch hỗ trợ) chọn BF16; T4/P100
chọn FP16. Có forward/backward probe thật trước train để phát hiện runtime image
không hỗ trợ hardware hoặc precision. Không fallback CPU. Hai T4 vẫn chỉ dùng
`cuda:0`; WP07.3 chưa có distributed/multi-GPU training.

Subprocess `python -u` in từng dòng stdout/stderr, đồng thời ghi `console.log`.
Các run vẫn giữ file logs/update/validation/checkpoint events như bản Colab.
Source clone và ZIP extraction tạm nằm ngoài `/kaggle/working`; output không phình
thêm vì chép cả Git repo hoặc token payloads. Sau editable install, sys.path được
cập nhật trong kernel đang mở để import package ngay, không cần restart.

## Output và giới hạn

Tất cả kết quả giữ trong `/kaggle/working/grast-wp07/`:

- `kaggle_environment.json`: Torch/CUDA/GPU/precision và source revision.
- `console.log`, `runs/.../config.json`, metadata/logs.
- `runs/.../checkpoints/latest.pt`, checkpoints mốc và final khi train đủ.
- `runs/.../artifacts/`: tokenizer, dataset manifest, history/CSV/PNG/SVG/report.
- Summary ZIP không chứa model weights; giữ raw checkpoints trong Notebook Output.

Một full run giữ khoảng2,8GB weights; nên có5–10GB chỗ cho các run/checkpoints.
File trong working cần được lưu thành Notebook Output để dùng ở phiên sau; không
coi checkpoint runtime là backup ngoài máy khi session bị kill/fail. Tải output hoặc
gắn output version đã lưu làm input cho WP07.4/5.

Đã kiểm tra local các notebook cells/helpers: ZIP/expanded input, corrupt input,
command smoke/full/resume, native precision selection và CUDA forward/backward probe.
Chưa chạy trên Kaggle/T4/P100; không gọi local verification là Kaggle acceptance.
Ước tính thời gian A100 trước đó không áp trực tiếp cho GPU Kaggle; dùng logs100
updates đầu để ước lượng. [WP07.3 local evidence](wp07_3_validation.md).

Local adapter verification19/11/2026:14 targeted tests pass (13 mới + Colab syntax),
lint/format pass. Thực thi chính helpers của notebook để tìm/giải nén bundle thật,
tạo lệnh và stream subprocess FP16 trên RTX4060: model46.148.096 parameters,
4successful updates, full1.000-story validation loss9,166916→7,663098, checkpoint
được lưu. Đây là local smoke, không phải chạy thực tế trên Kaggle/T4/P100.
[Evidence JSON](../experiments/wp07_kaggle_validation_20261119.json).
