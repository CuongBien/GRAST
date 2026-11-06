# WP05.2 validation — 06/11/2026

**Đạt nghiệm thu phase 2**: causal MHA + RoPE và padding masks.
Toàn bộ WP05 tiếp tục chờ decoder, loss và overfit ở các phase 3–5.
[Thiết kế/API và lệnh tái lập](wp05_architecture.md#wp052--causal-mha).

## Kiểm thử

230 tests toàn repository đạt trên local. WP05.2 có 39 cases, gồm 28 CPU cases
và 11 cases bổ sung trên CUDA. CI dùng CPU suite, không giả định có GPU.

- Output, input gradients và gradients của cả bốn projection khớp native SDPA math
  với RoPE reference bằng số phức, ở FP64 trên CPU/CUDA.
- Test tính tay với Q/K bằng 0 xác minh causal averaging chỉ dùng V được phép.
- Thay token tương lai giữ nguyên prefix output; forward prefix khớp prefix của
  full forward. Gradient từ prefix outputs đến future inputs bằng 0.
- Causal mask vẫn đúng khi explicit position IDs đảo thứ tự.
- Right/left/interior padding không làm giá trị padding ảnh hưởng output hợp lệ;
  cả padding chứa NaN/Inf cũng có output/gradients hữu hạn.
- Mẫu toàn padding có output và mọi input/parameter gradient bằng 0, không NaN.
- Finite-difference input gradcheck đạt trên mẫu có padding.
- FP32/FP16/BF16 và autocast đạt; FP16 Q/K scores lớn không gây overflow softmax.
- Dropout tái lập theo seed trong train, bị tắt trong eval; state-dict round-trip,
  noncontiguous input, shapes/count và invalid mask/input được kiểm tra.
- Tokenizer, embedding và collator WP04 tích hợp đúng quy ước mask 1=valid.

Ruff lint/format và build wheel/sdist đạt. CI chạy CPU attention smoke trên cả
Ubuntu và Windows cùng suite. Wheel được kiểm tra forward/backward ngoài source import.

## Probe candidate trên CPU và GPU

PyTorch 2.9.1+cu128 / Python 3.11.16, CUDA 12.8,
GPU NVIDIA GeForce RTX 4060 Laptop GPU. Probe chạy riêng trên CPU và CUDA:

| Thuộc tính | Kết quả |
| --- | --- |
| Input/output shape | `[2, 16, 512]` |
| Heads / head dimension | 8 / 64 |
| Attention parameters | 1.048.576 = `4 × 512²` |
| Chênh lệch prefix khi thay future inputs | Chính xác 0 |
| Chênh lệch output khi thay padding bằng NaN | Chính xác 0 |
| Gradients thông thường | Hữu hạn |
| Gradients future/padding bị cấm | Chính xác 0 |
| Mẫu toàn padding | Output/gradients hữu hạn bằng 0 |

Evidence run/report/môi trường:
[validation JSON](../experiments/wp05_2_validation_20261106.json).
Đây là parameter count của một attention module; chưa phải tổng decoder.
Probe chỉ kiểm tra attention, không có language-model loss hoặc overfit.

## Nhánh và phase kế tiếp

Nhánh phase này: `feat/wp05-2-causal-attention`. Ba nhánh phát triển cũ đã chuyển
từ `codex/` sang `feat/` ở local và GitHub, giữ nguyên commit/upstream tương ứng.
`main` và tag `wp02-baseline` giữ nguyên. Nhánh tự động Dependabot cùng PR #1 giữ nguyên.
Evidence: [branch migration](../experiments/branch_prefix_migration_20261106.json).

WP05.3 tiếp theo triển khai SwiGLU, residual và pre-norm decoder block.
