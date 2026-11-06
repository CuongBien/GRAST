# WP05.1 validation — 06/11/2026

**Đạt nghiệm thu phase 1**, chưa nghiệm thu toàn bộ WP05.
Phạm vi hoàn thành: ModelConfig, TokenEmbedding, RMSNorm và RotaryEmbedding.
[Các phase tiếp theo và contracts](wp05_architecture.md).

## Kiểm thử

- 191 tests toàn repository đạt trên local, gồm 53 cases WP05.1.
- 46 cases phase 1 chạy được trên CPU; local có thêm 7 cases trên CUDA.
- RMSNorm FP64 khớp reference `torch.nn.RMSNorm` cả output và gradients.
- RMSNorm input/scale và RoPE input đạt finite-difference gradcheck.
- RoPE khớp ma trận quay 2×2 độc lập, giữ norm và relative-position dot product;
  explicit position IDs cho chunk khớp encode toàn chuỗi.
- FP32/FP16/BF16 được kiểm tra trên CPU và RTX 4060; reductions ở FP32 tránh
  overflow khi RMSNorm nhận input lớn. Position lớn hơn 2048 không bị ép về BF16
  trước khi tính góc RoPE.
- Embedding nhận IDs từ tokenizer/collator WP04, kiểm tra repeated-token gradients,
  parameter count, out-of-vocabulary IDs và state-dict round-trip.
- Config từ chối giá trị/kiểu sai, unknown keys, missing fields và odd head dimension.

Ruff lint/format, build wheel/sdist đạt. CI chạy CPU suite và primitive smoke trên
Ubuntu/Windows; GPU cases chỉ chạy bổ sung khi CUDA thực sự có sẵn.

## Probe kích thước candidate

PyTorch 2.9.1+cu128, Python 3.11.16; CUDA 12.8 trên NVIDIA GeForce RTX 4060 Laptop GPU.
Probe riêng trên CPU và GPU đều forward/backward thành công với gradient hữu hạn,
khác 0 ở embedding và RMSNorm. Shapes:

| Tensor | Shape |
| --- | --- |
| Input IDs | `[2, 16]` |
| Hidden | `[2, 16, 512]` |
| Rotated heads | `[2, 8, 16, 64]` |

Parameter counts của **các lớp nền được probe**, không phải decoder hoàn chỉnh:
embedding 4.194.304, một RMSNorm 512, RoPE 0.
Sai số tuyệt đối lớn nhất của squared norm sau RoPE: `1.52587890625e-05`
trên cả hai thiết bị, đạt tolerance của probe.

Run paths, môi trường, gradients và report được chép vào
[validation JSON](../experiments/wp05_1_validation_20261106.json).
Run gốc có config nguồn, resolved architecture, Git/package fingerprint và log.
CPU/GPU dùng RNG streams của thiết bị tương ứng, không yêu cầu tensor giống nhau
giữa hai loại thiết bị.

## Giới hạn và phần tiếp theo

Chưa có attention, masking, FFN/residual block, decoder, LM head, CE loss hoặc overfit.
Scalar của probe chỉ dùng để gọi autograd, không phải loss language modeling.
Không gọi kết quả này là bằng chứng model không nhìn future tokens.
WP05.2 tiếp theo triển khai causal MHA và kiểm tra causal/padding masks.
