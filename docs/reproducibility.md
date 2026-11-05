# WP02 — Research Configuration & Reproducibility

WP02 cung cấp hạ tầng run cho kế hoạch hai giai đoạn trong HQ. Chưa implement tokenizer,
model, training loop, generation hoặc policy. Probe chỉ sử dụng RNG và phép nhân ma trận
8×8 float32; không phải thử nghiệm chất lượng hay hiệu năng model.

## Chạy nghiệm thu local

```powershell
uv sync --locked --extra cpu
uv run --no-sync grast smoke --config configs/experiments/local_smoke.toml
```

Một lệnh smoke tạo run directory, config snapshot, log và hai checkpoint RNG. Nó khởi chạy
hai tiến trình Python mới, so sánh kết quả chính xác, đồng thời kiểm tra chuỗi RNG tiếp tục
sau save/load checkpoint. Exit code 0 và `result.json` có `repeat_equal=true`,
`checkpoint_resume_equal=true` là pass. Mỗi lần gọi lệnh tạo thư mục mới, không ghi đè run cũ.

Để so sánh hai lần gọi độc lập, chạy lại cùng lệnh và đối chiếu `probe_sha256` trong hai
`result.json`. Timestamp, đường dẫn run và log không cần giống nhau. Đổi seed qua `--seed 43`
phải tạo probe khác. Seed/device/output override được đưa vào snapshot resolved.

```powershell
uv run --no-sync grast smoke --config configs/experiments/local_smoke.toml --seed 43
uv run --no-sync grast check-env --device cpu --output artifacts/environment_local.json
```

## Local CUDA

```powershell
uv sync --locked --extra cuda
uv run --no-sync grast check-env --device cuda
uv run --no-sync grast smoke --config configs/experiments/local_cuda.toml
```

Extra CPU/CUDA đều pin PyTorch 2.9.1 từ index chính thức; CUDA extra dùng cu128. Chọn
đúng extra trước khi chạy; `uv run --no-sync` giữ môi trường đã sync, không tự đổi runtime.
Driver phải hỗ trợ CUDA runtime tương ứng. Kiểm tra ghi cả driver/GPU từ `nvidia-smi`
và CUDA build/runtime availability từ PyTorch; có GPU trong máy chưa có nghĩa torch dùng được GPU.
Config yêu cầu CUDA/A100 sẽ fail khi không phù hợp, không âm thầm chuyển sang CPU.

## Config và snapshot

File experiment compose bốn component TOML: model, tokenizer, training, generation.
Path tương đối được resolve theo file experiment. Schema version 1 kiểm tra key, type,
giá trị, vocab/head dimensions, K, seed và environment constraints.

`draft_35m`, vocabulary và training defaults là thiết kế/debug chưa kiểm chứng feasibility;
WP02 chỉ validate/snapshot chúng. Không có train hoặc sinh văn bản từ những config này.
Muốn chuyển pretrain sang SFT, đổi component `training` sang `../sft/debug.toml`.

```text
artifacts/runs/<UTC>-<experiment>-seed<seed>-<unique-id>/
  config.json                 snapshot đã resolve và áp dụng override
  config_sources/*.toml       nội dung TOML nguồn tại thời điểm bắt đầu
  metadata.json               Git, config hash, seed, package/source/runtime versions, GPU
  run.json                    running / completed / failed, UTC time, error
  result.json                 kết quả nghiệm thu; chỉ có khi pass
  logs/run.log                log text theo UTC
  logs/events.jsonl           lifecycle events dạng JSONL
  logs/repeat_*.{stdout,stderr}.log
  checkpoints/repeat_*.pt     state primitive/tensor và RNG; không phải model checkpoint
  artifacts/repeat_*.json     draws, matrix output, actual worker determinism flags
```

Run artifacts bị gitignore. Metadata ghi commit và dirty state; nếu repository chưa có
commit thì ghi `commit=null`, `state=unborn`, không tạo SHA giả. Fingerprint từng file Python
thực sự được import giúp nhận diện editable code chưa commit. Snapshot không thay thế việc
commit code: một run chỉ có thể checkout đầy đủ khi commit/source/environment được lưu bền vững.

## Seed và checkpoint

- Seed Python `random`, NumPy legacy global RNG, explicit `default_rng`, PyTorch CPU/CUDA.
- `PYTHONHASHSEED`, OMP/MKL threads và cuBLAS workspace được truyền trước khi worker interpreter
  khởi động. Không tuyên bố đổi `os.environ` có thể đổi hash seed của interpreter đang chạy.
- Bật `torch.use_deterministic_algorithms(..., warn_only=False)`, tắt cuDNN benchmark,
  bật cuDNN deterministic, tắt TF32 và dùng float32 matmul precision `highest`.
- `seed_worker` hỗ trợ DataLoader tương lai; caller vẫn phải truyền một `torch.Generator`
  được seed. Các Generator tự tạo khác phải được quản lý riêng.
- Checkpoint dùng atomic replace cùng filesystem, lưu state dict primitive/tensor và RNG
  Python/NumPy/PyTorch, gồm explicit NumPy generator. CUDA RNG chỉ được lưu khi CUDA đã
  khởi tạo; restore từ chối CUDA topology không phù hợp.
- Load dùng `weights_only=True`, `map_location=cpu`; chưa có model/optimizer/scheduler để
  restore. Sau này caller đưa các `state_dict` tương ứng vào state, không serialize object trainer.

Các phép thử kiểm tra bitwise-equivalent probe trong cùng môi trường, seed, device, versions,
thread setting và thứ tự thực thi. Không bảo đảm bitwise equality giữa CPU/GPU, Windows/Linux,
GPU khác nhau hoặc phiên bản thư viện khác nhau. Training/dropout/data loading cần behavioral
tests riêng khi được triển khai. Xem [PyTorch reproducibility](https://docs.pytorch.org/docs/2.9/notes/randomness.html)
và [NumPy RNG seeding](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html).

## Colab A100

Dùng [colab_runner.ipynb](../notebooks/colab_runner.ipynb), chọn runtime A100 có sẵn trong
tài khoản. Notebook không tự đổi hay mua runtime. Preflight bắt buộc CUDA và GPU name chứa
`A100`; không pass với T4, L4 hoặc CPU. Dùng PyTorch có sẵn; NumPy chỉ cần thay đổi nếu chưa
đáp ứng requirement của package. Ghi phiên bản thực tế; không coi Colab là môi trường
bitwise-identical với lockfile local.

Để chạy lại baseline, checkout tag `wp02-baseline` sau khi clone repository. Hoặc chạy
`uv build` trên đúng baseline, upload `dist/grast-0.1.0.tar.gz` từ local qua setup cell.
Source distribution gồm code, configs và notebook, loại cache/environment/artifact.
Chạy từ sdist không có `.git` sẽ ghi commit là null; giữ archive và đối chiếu package
fingerprint với baseline. Không sửa metadata cũ để gán một commit chưa tồn tại khi chạy.

Notebook chạy preflight rồi chạy smoke hai lần, so sánh hash, và có cell tải evidence archive.
Mặc định output trong `/content` là tạm thời. Muốn dùng Drive, tự mount Drive rồi đặt
`RUN_ROOT` vào Drive; không tạo thư mục giả khi Drive chưa mount. Không đưa token vào notebook.
Lưu archive ra máy/Drive trước khi Colab kết thúc phiên.

Trạng thái kiểm tra thực tế và link evidence được ghi trong [WP02 validation](wp02_validation.md).
