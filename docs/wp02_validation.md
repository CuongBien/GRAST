# WP02 validation — accepted 05/11/2026

WP02 đạt nghiệm thu local CPU/CUDA và Colab A100. Evidence A100 do người dùng chạy,
đã kiểm tra checksum archive, JSON payloads, config snapshots và fingerprint source local.
Tổng hợp versioned: [A100 evidence manifest](../experiments/wp02_a100_validation_20261105.json).

## Local

Kiểm tra CPU thực tế trên Windows, Python 3.11.16, NumPy 2.4.6, PyTorch 2.9.1+cpu.
Máy có NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, driver 610.47 theo `nvidia-smi`.
Có hardware GPU không đồng nghĩa CPU wheel sử dụng được CUDA.

Smoke đã tạo run dưới `artifacts/runs/`, với config, metadata, logs và RNG checkpoints.
Hai fresh processes cùng seed cho output giống nhau; checkpoint continuation giống nhau.
**19 tests pass** trên CPU; sau đó **19 tests pass** trên PyTorch 2.9.1+cu128,
CUDA 12.8, cuDNN 91002 và RTX 4060 local. `uv pip check` xác nhận installed packages tương thích.
Behavioral tests gồm lặp độc lập, khác seed, validation, run failure, strict A100 gate,
RNG resume và checkpoint bị ngắt không làm hỏng file trước đó.

Hai lần gọi CUDA smoke độc lập đã pass, đều tạo hai fresh processes và RNG checkpoints:

- `20261101T050911682172Z-wp02-local-cuda-seed42-8e4d01a1`
- `20261101T051132303457Z-wp02-local-cuda-seed42-74bcd5f3`

Chúng có cùng `probe_sha256`:
`34be15696c496e653f37b9e0f332a0ad65a3533ac59e659fa837b5f1165622f0`.
Run nằm dưới `artifacts/runs/`; runtime metadata, logs và checkpoints là output gitignored.
Đã build wheel/sdist, cài wheel vào environment sạch và import mọi module ngoài source tree.
Đã validate YAML, notebook schema/code và kiểm tra bundle không chứa cache/environment/run artifacts.
Không có kết quả model.

## Colab A100

Notebook người dùng cung cấp:
[CLLM.ipynb](https://colab.research.google.com/drive/1DvLhdsVMx5MvV0XH58pjluhlOn3FgpJf).

Người dùng cung cấp evidence ZIP sau khi chạy thành công ngày 05/11/2026:

- GPU: NVIDIA A100-SXM4-40GB; driver 580.82.07.
- Python 3.13.16, NumPy 2.4.6, PyTorch 2.11.0+cu130, CUDA 13.0, cuDNN 92700.
- Hai run: `20261105T102041285589Z-wp02-colab-a100-seed42-edfa4004`
  và `20261105T102056298173Z-wp02-colab-a100-seed42-159a0390`.
- Cả hai có `status=completed`, `repeat_equal=true`, `checkpoint_resume_equal=true`.
- Bốn worker payloads giống nhau, cùng probe SHA-256:
  `62d8e2f70ba38799fd81914ea5beb4d1b82e74752bc332041ef3a0928d9ec5de`.
- Seed/hash seed 42; deterministic algorithms bật, cuDNN benchmark và TF32 tắt,
  cuBLAS workspace `:4096:8`, một intra-op thread. Inter-op threads được ghi thực tế là 6.
- Config snapshots và toàn bộ fingerprint Python source khớp source local đã review.

Archive được giữ nguyên dưới `artifacts/evidence/` và bị gitignore. SHA-256:
`1e48c0b4fb83d396ce52beb5e098a0cc5b2873885846ce44e0ec081c178f3d43`.
Manifest trong Git đủ để đối chiếu artifact; raw ZIP chứa logs/checkpoints được lưu riêng.
Kiểm tra archive không deserialize các file `.pt`.

Metadata Colab ghi `git.state=unavailable`, `commit=null` vì source được cài từ sdist.
Không sửa ngược metadata lịch sử. Fingerprint package
`293da9ccc4330319e0fde1520262cafde12ea2d29fbd09abe4d472c0a2df7330`
liên kết evidence này với source baseline. Không so probe hash Colab với local vì môi trường khác nhau.

## Baseline Git và chuyển tiếp

Tag đóng WP01+WP02: `wp02-baseline`. Sau commit, chạy lại smoke trên working tree sạch
để metadata mới ghi Git SHA thật; evidence mới nằm trong `artifacts/runs/`.
Annotated tag lưu commit và thông tin smoke sau commit, tránh đưa một commit hash tự tham chiếu
vào chính commit đó. WP03 dùng branch riêng từ baseline và cần acceptance riêng.
