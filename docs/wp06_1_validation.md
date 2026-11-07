# WP06.1 validation — 07/11/2026

**Đạt nghiệm thu phase 1**: KV cache storage và transactional append theo layer.
WP06 chưa nghiệm thu cached model inference hoặc profiling; xem [phases và contract](wp06_inference.md).

## Kiểm thử

403 tests toàn repository đạt trên local. WP06.1 có 57 cases:
48 cases chạy CPU, 9 cases bổ sung cần CUDA. CI CPU skip riêng cross-device case
khi không có GPU, không dùng skip làm placeholder cho generation.

- FP16/BF16/FP32/FP64, CPU/CUDA: chunk append 4+1+2+1 khớp concat reference chính xác
  ở mọi layer; committed length chỉ tăng khi mọi layer hoàn tất.
- K đã qua RoPE, V nguyên, masks và explicit positions round-trip; sửa source inputs
  sau update không sửa cache. Noncontiguous projection views được copy đúng.
- Layer thứ tự khác nhau được phép; duplicate, invalid index và incomplete commit bị từ chối.
- Exception sau một phần hoặc sau tất cả layer trước commit giữ nguyên prefix;
  có thể append lại sau abort. Step cũ/forged không thể ghi vào active transaction khác.
- Nested append và reset trong active transaction bị chặn. Capacity overflow không evict.
- Reset đưa used length/bytes về 0, giữ storage pointers và tái sử dụng được.
- Payload đếm từ tensors khớp công thức và storage nbytes thực tế.
- Cache được tạo/cập nhật trong `inference_mode` vẫn reuse được ngoài scope đó;
  không có autograd graph và K/V require gradients bị từ chối.
- Dtype/device/shape/mask/positions sai bị từ chối, không implicit cast/broadcast/transfer.
  Physical capacity và model RoPE position bounds được kiểm tra độc lập.

Ruff lint/format, wheel/sdist và wheel cache integration đạt.
CI Ubuntu/Windows chạy CPU suite và cache storage smoke.

## Probe geometry candidate

Probe riêng CPU và NVIDIA GeForce RTX 4060 Laptop GPU, Python 3.11.16,
PyTorch 2.9.1+cu128, CUDA 12.8. Không khởi tạo neural model trong probe này.

| Thuộc tính | Kết quả |
| --- | --- |
| Layers / batch / heads / head dim | 10 / 2 / 8 / 64 |
| Capacity / dtype | 1024 / FP32 |
| Append chunks | 4 + 1 + 1 |
| Committed physical length trước reset | 6 |
| K/V capacity payload | 83.886.080 bytes = 80 MiB |
| K/V used payload tại length 6 | 491.520 bytes |
| Mask/position capacity payload | 18.432 bytes |
| Tổng tensor capacity payload | 83.904.512 bytes |
| Round-trip K/V, prefix sau abort | Exact |
| Reset/reuse | Storage pointers giữ nguyên |

Payload là kích thước tensors, **chưa phải VRAM allocated/reserved/peak thực đo**.
Không có TTFT, next-token latency hoặc tokens/s ở phase này.
Run paths/report hashes: [validation manifest](../experiments/wp06_1_validation_20261107.json).

## Tái lập và bước tiếp

```powershell
uv run --no-sync python scripts/check_kv_cache.py --device cpu
uv run --no-sync python scripts/check_kv_cache.py --device cuda
```

WP06.2 sẽ nối cache vào attention/CausalLM, xử lý prefill/token decode,
RoPE offsets, masks, model ownership và context boundaries. WP06 chỉ chốt sau
generation equivalence và profiling thực đo ở các phase còn lại.
