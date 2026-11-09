# WP06.5 validation — 09/11/2026

**WP06.5 đạt nghiệm thu; WP06 hoàn tất trong phạm vi đã chốt.** KV cache/generation
đã đối chiếu ở WP06.1–4; phase này bổ sung profiling online thực đo trên GPU local.
[Protocol đầy đủ](wp06_profiling.md).

## Môi trường và workload

NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB; Python 3.11.16, PyTorch 2.9.1+cu128,
CUDA 12.8. Candidate 46.148.096 tied parameters, checkpoint diagnostic overfit WP05.5.
FP32 weights/KV, explicit MHA, deterministic, TF32 disabled, CPU num_threads=1.
B=1, N=16, greedy, EOS disabled, all-valid synthetic prompts với L128/512/1024/2048.
2 warm-up requests + 5 measured repeats mỗi condition, cached/uncached mode order
đổi theo L. 40 measured requests, 640 token-ready timestamps và 600 next-token intervals.

Primary run bắt đầu từ **commit sạch `81ebadbc3ecfeca7ec67545e49de06966954f6cf`**,
không dùng kết quả nhanh nhất giữa nhiều lần chạy để chọn báo cáo. Dry run trước đó
kiểm protocol; giữa hai runs prompt/generated IDs khớp chính xác, timing có noise.
Git/source/environment và raw samples được phiên bản hóa trong
[validation manifest](../experiments/wp06_5_validation_20261109.json) và
[raw report](../experiments/wp06_5_profiling_20261109.json).

Baseline/checkpoint context **1024** giữ nguyên. Model inference copy có context
**2064 = 2048+16**, chỉ đổi context metadata/bounds và kiểm weights equality trước
precision cast. RoPE theta/architecture/embedding tying không thay đổi. Không sửa
checkpoint trên đĩa hoặc tuyên bố long-context quality.

## Latency và throughput

TTFT và next-token latency tính bằng ms, ghi p50/p95. Decode tokens/s là median
per-repeat B*(N-1)/(tN-t1), loại TTFT; overall tokens/s dùng B*N/(tend-t0).
Đây là warm host-observed streaming generate API, gồm validation/cấp cache/forward/
selection/output bookkeeping và per-token CUDA synchronization. Không gồm model load,
tokenization/H2D hoặc report writes. Không dùng logits-copy acceptance hooks trong timing.

| L | Mode | TTFT p50 / p95 ms | Next-token p50 / p95 ms | Decode tok/s p50 | Overall tok/s p50 |
| --- | --- | --- | --- | --- | --- |
| 128 | Uncached | 43.08 / 45.45 | 39.35 / 44.18 | 24.83 | 24.88 |
| 128 | Cached | 53.32 / 56.32 | 47.35 / 57.19 | 20.82 | 20.79 |
| 512 | Uncached | 41.88 / 43.15 | 42.06 / 46.62 | 23.67 | 23.69 |
| 512 | Cached | 66.76 / 77.76 | 42.99 / 48.24 | 22.90 | 22.06 |
| 1024 | Uncached | 70.32 / 74.18 | 71.82 / 74.70 | 13.89 | 13.90 |
| 1024 | Cached | 86.98 / 95.08 | 41.64 / 56.35 | 23.56 | 21.97 |
| 2048 | Uncached | 186.30 / 202.53 | 188.61 / 214.07 | 5.26 | 5.23 |
| 2048 | Cached | 190.04 / 249.72 | 53.97 / 139.21 | 17.38 | 15.19 |

Ở L2048, median decode throughput cached/uncached khoảng **3,30×**. L128 và L512
chưa cho lợi ích tốc độ trong primary run; không suy rộng cache luôn nhanh hơn.
TTFT cached vẫn phải prefill và cấp/ghi cache. Đuôi latency dao động đáng kể ở L2048;
raw samples giữ nguyên, không loại outliers hoặc hứa timing bitwise-identical.
Đây là PyTorch baseline trên laptop GPU dùng chung desktop, không phải GPU kernel-only
benchmark hoặc optimized serving throughput; không ngoại suy sang A100.

## CUDA memory thực đo

Bảng lấy **max peak qua 5 repeats**, đơn vị MiB = 2^20 bytes. Baseline/end allocated/
reserved và peak delta từng request có trong raw report. Peak counters reset trước
mỗi request, sau warm-up; empty_cache chỉ tại condition boundary ngoài timing.

| L | Uncached peak allocated | Cached peak allocated | Uncached peak reserved | Cached peak reserved |
| --- | --- | --- | --- | --- |
| 128 | 222.01 | 222.17 | 234 | 240 |
| 512 | 259.52 | 263.09 | 282 | 280 |
| 1024 | 360.43 | 359.72 | 498 | 390 |
| 2048 | 706.98 | 708.77 | 1044 | 862 |

KV cache không làm peak allocated tự động giảm: prefill attention vẫn tạo tensors
quadratic theo L, trong khi cache thêm storage riêng. L2048 có K/V capacity payload
**80,625 MiB** (2064 columns FP32, 10 layers/B1/H512), dùng 2063 columns cuối request;
payload này là công thức shape/dtype, được lưu riêng với CUDA allocator samples.
Allocated/reserved là process PyTorch allocator, không phải tổng VRAM toàn GPU,
driver context hay ứng dụng khác. nvidia-smi telemetry lưu ngoài vùng timing.

## Kiểm thử và đóng WP06

**612 tests đạt local**, phase này thêm 26 cases (22 CPU + 4 CUDA): callback không
đổi tokens, timestamps thật monotonic/đủ budget, CUDA memory counters, metric formulas,
context copy/metadata/strict-load invariants, config/order/input identity và stdlib
cProfile/torch.optim compatibility. Không có speed thresholds phụ thuộc hardware.
CPU smoke dùng model nhỏ, VRAM=null. Ruff lint/format, wheel/sdist và installed-wheel
profiling smoke đạt; CI chạy CPU protocol smoke trên Ubuntu/Windows.

Ma trận GPU hoàn tất 8/8 conditions với exact token IDs giữa modes và repeats; context
override/weights equality, checkpoint SHA và source commit đều được lưu. Các gate WP06:

- WP06.1: layer cache/storage/transaction/rollback và bounds.
- WP06.2: cached prefill/token/chunk forward, logits/RoPE/masks/ownership.
- WP06.3: sampling, greedy, EOS, seeded generation.
- WP06.4: actual cached/uncached generation logits/IDs/masks/stop equivalence.
- WP06.5: TTFT/next-token latency/throughput và allocated/reserved/peak thực đo đủ L.

Checkpoint là diagnostic overfit; chưa có corpus pretraining/SFT trainer hoặc held-out
model-quality acceptance. Không dùng phép đo này để tuyên bố accuracy/quality ở context dài.

## Tái lập

```powershell
uv run --no-sync python scripts/profile.py --checkpoint artifacts/runs/20261106T162829823401Z-wp05-acceptance-candidate-seed42-13fcf49c/checkpoints/overfit.pt
uv run --no-sync python scripts/profile.py --config configs/experiments/wp06_profiling_smoke.toml
```

Mỗi run giữ config/model source snapshots, source/effective configs, metadata,
environment, logs, per-condition JSON và report tổng. Checkpoint SHA-256:
`e756d7a9519d1423a7ee554884afdc2b0ebb98a73188bd40a5554179eaa8de56`.
Run chính: `artifacts/runs/20261109T051450654044Z-wp06-5-inference-profile-seed42-2f17621f`.
