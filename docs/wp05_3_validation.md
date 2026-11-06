# WP05.3 validation — 06/11/2026

**Đạt nghiệm thu phase 3**: SwiGLU, hai residual connections và pre-norm decoder block.
Full language model, LM head/loss và overfit tiếp tục thuộc phase 4–5.
[Thiết kế, API và cách chạy](wp05_architecture.md#wp053--swiglu-và-decoder-block).

## Kiểm thử

276 tests toàn repository đạt trên local. WP05.3 bổ sung 46 cases:
33 CPU cases và 13 cases bổ sung trên CUDA. Suite attention WP05.2 tiếp tục đạt
sau khi chia sẻ phần validation mask giữa attention và block.

- SwiGLU khớp reference matrix-form với `gate * sigmoid(gate)` về output,
  input gradients và cả ba projection gradients; input và mọi parameter đạt gradcheck.
- Cả pre-norm block khớp reference độc lập dùng native RMSNorm, SDPA math,
  complex RoPE và matrix-form FFN ở FP64, gồm input và mọi parameter gradient.
- Hai nhánh output bằng 0 cho exact residual identity ở token hợp lệ.
  Kiểm tra cả VJP trên CPU/CUDA và **toàn bộ input Jacobian** trên fixture nhỏ:
  diagonal bằng 1 ở valid inputs, bằng 0 ở padding, mọi phần tử ngoài diagonal bằng 0.
- Block input đạt finite-difference gradcheck khi có padding.
- Xếp một hoặc hai block vẫn giữ causal prefix, prefix-only equivalence và
  zero gradient từ prefix output đến future inputs; explicit positions đảo thứ tự
  không phá causal order theo chỉ số chuỗi.
- Padding phải/trái/ở giữa, kể cả NaN/Inf, không đi vào residual hoặc RMSNorm gradients.
  Mẫu toàn padding NaN có output và mọi input/parameter gradient hữu hạn bằng 0.
- FP32/FP16/BF16, autocast, shapes, dropout train/eval, state-dict round-trip và
  noncontiguous input được kiểm tra. FP32 residual stream được giữ dưới autocast.
- IDs/masks từ tokenizer/collator WP04 ghép qua embedding và block, chạy backward thành công.

Ruff lint/format và build wheel/sdist đạt; wheel được kiểm tra forward/backward
ngoài source import. CI chạy CPU suite và decoder-block smoke trên Ubuntu/Windows.

## Probe kích thước candidate

Probe riêng CPU và GPU RTX 4060 Laptop, PyTorch 2.9.1+cu128 / Python 3.11.16 / CUDA 12.8:

| Thuộc tính | Kết quả |
| --- | --- |
| Số block được probe | 1 |
| Input/output shape | `[2, 16, 512]` |
| Heads / head dimension | 8 / 64 |
| FFN intermediate size | 2048 |
| Parameter count | 4.195.328 |
| Công thức | `4*512² + 3*512*2048 + 2*512` |
| Normal block gradients | Mọi parameter hữu hạn, khác 0 |
| Chênh lệch prefix khi thay future inputs | Chính xác 0 |
| Chênh lệch output khi thay padding bằng NaN | Chính xác 0 |
| Future/padding input gradients | Chính xác 0 |
| All-padding output và mọi gradients | Hữu hạn bằng 0 |
| Khi zero hai output projections | Exact identity và identity input VJP trên valid tokens |

Run paths, gradient norms và runtime:
[validation JSON](../experiments/wp05_3_validation_20261106.json).
Run gốc có config nguồn, resolved architecture, log, Git/package fingerprint và script SHA.

Parameter count ở đây là **một block**, không phải decoder đầy đủ hoặc khẳng định
candidate `draft_35m` có đúng 35M tham số. Probe không có LM cross-entropy hoặc overfit.
Phase 4 sẽ ghép embedding/stack/final norm/LM head và tính loss; phase 5 nghiệm thu toàn WP05.
