# Easy recovery trên Kaggle

Notebook: [`easy_recovery_kaggle.ipynb`](../notebooks/easy_recovery_kaggle.ipynb).
Dùng nguyên kit/recipe của [bản Colab](easy_recovery.md), không đổi protocol hoặc train thêm vòng:
source selected2200,1.000 successful updates, selection16/gate48 và cùng tokenizer.

## Chuẩn bị và chạy

1. Import notebook vào Kaggle; chọn GPU accelerator.
2. Tạo Dataset và **Add Input** hai file:
   - `dist/grast-easy-recovery-kit-20261127.zip`.
   - `artifacts/checkpoints/wp07-curriculum-2800-selected-2200/selected_model.pt`.
3. Chạy `SMOKE_ONLY=True` trước:2 updates, không chấm gate.
4. Khi smoke qua, đổi `SMOKE_ONLY=False`, chạy lại cell cấu hình và các cell bên dưới
   để train đầy đủ. Lưu notebook version có Output để giữ kết quả trước khi kết thúc phiên.

Kaggle có thể giải nén ZIP khi tạo Dataset. Notebook tự tìm cả ZIP và thư mục đã giải nén,
kể cả layout `/kaggle/input/datasets/<owner>/<dataset>/versions/<n>/`.
Nếu có nhiều bản, đặt `KIT_PATH` là ZIP hoặc thư mục đúng. Có thể chỉ rõ
`SOURCE_CHECKPOINT` khi cần; checkpoint nguồn được kiểm tra bằng SHA256, không dựa vào tên.
Không truyền inference export của vòng mới vào vị trí checkpoint nguồn selected2200.

Kit ZIP SHA256:
`c8b15e13ab936af43a7efc649fedc1191f210fa4d9a097c95bc63bb68ab606cf`.
Manifest khi giải nén SHA256:
`d58c11c093e80035a627dad338956b70514a8ef9dfa8121a54ec3c0e3231af98`.
Không cần Internet nếu runtime đã có PyTorch/NumPy/Matplotlib. Wheel trong kit được cài
offline vào thư mục tạm, giữ PyTorch của Kaggle; project chạy trong subprocess mới để
tránh dùng nhầm module đã import ở kernel trước. Không clone branch đang thay đổi.

## Log, checkpoint và Output

Tất cả stdout/stderr được in ngay và lưu vào
`/kaggle/working/grast-easy-recovery/console.log`. Runner in từng update,
validation/generation batch và checkpoint. Chỉ dùng `cuda:0`; không dùng DDP
hoặc tự tăng effective batch khi có2 GPU. T4 dùng FP16; BF16 chỉ dùng khi GPU hỗ trợ native.
Preflight kiểm tra GPU forward/backward và ghi phiên bản GPU/PyTorch/CUDA thực tế.

Kết quả nằm dưới `/kaggle/working/grast-easy-recovery/`:

- `output_index.json` ghi đúng run, đường dẫn và checksum; không tự chọn run cũ.
- `<run>-summary.zip` chứa config, log, metadata, loss curves và report, **không có weights**.
- `runs/<run>/checkpoints/selected_model.pt`: model inference đã chọn.
- `runs/<run>/checkpoints/latest.pt`: full checkpoint để resume; milestones500/1000 giữ riêng.

Smoke dùng thư mục `smoke_runs/`. Sau full run, tải **summary ZIP + selected_model.pt**
từ Notebook Output để gửi lại kiểm tra trước WP09. `/kaggle/working` là storage của
phiên notebook; cần lưu version có Output, không coi đó là Drive tự lưu qua mọi phiên.

## Resume

Attach Notebook Output đã lưu bằng Add Input, đặt `SMOKE_ONLY=False` và `RESUME`
bằng đường dẫn full `latest.pt`. Vẫn attach kit và checkpoint nguồn selected2200
để xác minh lineage và chấm baseline cuối vòng. Không dùng inference export hoặc
checkpoint curriculum cũ để resume optimizer của vòng này.

Exact resume cần cùng wheel/recipe/data, PyTorch/CUDA, loại GPU, số GPU nhìn thấy và precision.
Đặc biệt Colab1 GPU và Kaggle2 GPU không có cùng runtime contract. Notebook giữ precision
của checkpoint, runner từ chối runtime không tương thích; không âm thầm warm-start.
Nếu interrupt, notebook đợi child dừng hoặc kill khi cần và cố xuất summary từ đúng run;
resume dùng checkpoint đã lưu gần nhất.

## Verification và giới hạn

Tests kiểm tra discovery ZIP/expanded, checksum/path safety, nguồn weights bằng hash,
precision/resume, log trực tiếp, interrupt cleanup, chống chọn stale output và chạy
smoke→resume thật bằng fixture CPU nhỏ. Kit và runner giữ nguyên bản đã verify trước đó.
Verify GPU local dùng cùng notebook helpers và model thật; không phải kết quả chạy trực tiếp
trong phiên Kaggle và không chứng minh improvement sau full training.

Ngày27/11/2026:29 related tests qua, gồm fixture smoke→resume đến step4, kiểm tra
ZIP/expanded inputs, precision, interrupt cleanup và chống stale output. Local GPU
RTX4060/PyTorch2.9.1+cu128 chọn native BF16, chạy đủ2 updates,0 AMP skip,
lưu checkpoint/summary và xác minh resume header; gate thật chưa chấm.
Receipt: `experiments/easy_recovery_kaggle_validation_20261127.json`.
