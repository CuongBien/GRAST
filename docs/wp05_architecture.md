# WP05 — Transformer Architecture theo phase

Mỗi phase có implementation, kiểm thử, báo cáo kết quả và commit riêng.
WP05 chỉ nghiệm thu toàn bộ sau WP05.5; các kiểm tra của phase trước không thay thế
forward/backward, overfit và causal-isolation của model hoàn chỉnh.

| Phase | Phạm vi | Gate nghiệm thu | Trạng thái |
| --- | --- | --- | --- |
| WP05.1 | ModelConfig, token embedding, RoPE, RMSNorm | Reference số học, shapes, gradients, dtype/device, nhận IDs WP04 | Đã triển khai |
| WP05.2 | MHA causal self-attention | Reference attention, causal/padding masks, future-token isolation, không NaN ở hàng bị mask | Chưa triển khai |
| WP05.3 | SwiGLU và pre-norm decoder block | Shapes, gradients qua attention/FFN/residual, identity khi nhánh residual bằng 0 | Chưa triển khai |
| WP05.4 | Decoder-only LM, final norm, LM head, CE loss | Logits, shift labels đúng một lần, ignore -100, tied embeddings, parameter count thực tế | Chưa triển khai |
| WP05.5 | Integration và overfit batch nhỏ | Loss giảm trên batch cố định, gradients hữu hạn, kiểm tra model không nhìn future tokens và không dùng padding | Chưa triển khai |

Lượt triển khai đầu dừng ở WP05.1 để người dùng có thể xem kết quả trước phase attention.
Baseline chọn **MHA**, không yêu cầu GQA ở WP05. Mọi phép attention/block sẽ viết bằng
PyTorch tensors/parameters/autograd; không dùng `nn.Transformer`, `nn.MultiheadAttention`
hoặc model tải sẵn. KV cache/generation và pretraining loop thuộc các bước sau.

## Contract để tích hợp

- IDs từ tokenizer WP04: `[batch, sequence]`; embedding output `[batch, sequence, hidden]`.
- Query/key/value trong attention: `[batch, heads, sequence, head_dim]`.
- `hidden_size % num_heads == 0`; `head_dim` chẵn để áp dụng RoPE cho toàn head.
- RoPE áp dụng riêng lên Q/K bằng cùng position IDs; V không xoay.
- Padding mask từ collator có 1/True là token hợp lệ. Phase attention kết hợp mask
  causal và key padding, xử lý rõ query padding/hàng không có key hợp lệ.
- Decoder block dự kiến pre-norm: residual attention rồi residual SwiGLU; các linear
  projections không bias. Final RMSNorm trước LM head.
- Labels WP04 chưa shift và dùng `-100` cho prompt/padding. Khi triển khai WP05.4,
  loss trong model sẽ so `logits[:, :-1]` với `labels[:, 1:]`; trainer không shift lần hai.
- Kích thước vocab thực tế phải khớp artifact tokenizer; checkpoint model phải ghi
  tokenizer fingerprint và toàn bộ `ModelConfig.to_dict()`.

Candidate `draft_35m.toml` chưa phải parameter count đã đo của model hoàn chỉnh.
SwiGLU và tied/untied head ảnh hưởng tổng số tham số; WP05.4 sẽ tính từ Parameters
thực tế và kiểm tra công thức độc lập trước khi kết luận feasibility.

## WP05.1 — lớp nền tảng

`ModelConfig` là dataclass bất biến, không import torch. Nó đọc được tám trường
model từ WP02, từ chối unknown keys, kích thước/kiểu sai, dropout/epsilon không hợp lệ,
head dimension lẻ và theta không hợp lệ. Hai default kiến trúc mới được ghi tường minh
trong `to_dict()`: `rope_theta=10000.0`, `rms_norm_eps=1e-6`.
Config snapshot WP02 vẫn giữ schema cũ; probe lưu riêng resolved architecture gồm
cả hai default tại `artifacts/model_config.json`.

`TokenEmbedding` dùng Parameter `[vocab, hidden]`, khởi tạo normal std 0.02 và
lookup qua `torch.nn.functional.embedding`, không scale theo hidden size.
PAD có hàng học được bình thường để tương thích weight tying; không dựa vào giá trị
embedding PAD để loại padding. Attention/loss sẽ dùng masks của các phase sau.

`RMSNorm` tự tính `x * rsqrt(mean(x²) + eps) * weight`, không trừ mean/không bias,
theo [Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467).
FP16/BF16 input được tính và nhân scale ở FP32 rồi trả về dtype input;
FP64 được giữ để kiểm tra số học/gradcheck. Không gọi `nn.RMSNorm` trong implementation;
test dùng [PyTorch RMSNorm](https://docs.pytorch.org/docs/2.9/generated/torch.nn.RMSNorm.html)
làm một reference độc lập cho output và gradient FP64.

`RotaryEmbedding` dùng convention cặp chiều liền kề `(0,1), (2,3), ...`, theo phép quay
trong [RoFormer](https://arxiv.org/abs/2104.09864). Với pair i, góc là
`position * theta ** (-2*i/head_dim)`. Không có learned parameter hoặc bảng position học được.
Tần số tính từ integer indices ở mỗi forward; không giữ buffer float có thể bị
quantize khi module chuyển BF16/FP16. Position IDs mặc định `0..T-1`, hoặc truyền
`[T]` / `[B,T]` integer trên cùng device; mọi position nằm trong context đã cấu hình.
Explicit positions hỗ trợ kiểm tra chunk/offset, chưa triển khai KV cache.

Convention cặp liền kề là một phần của kiến trúc GRAST; không giả định checkpoint
dùng convention split-half sẽ tương thích trực tiếp.

## Chạy lại phase 1

Sau khi sync môi trường `cpu` hoặc `cuda` như README:

```powershell
uv run --no-sync pytest tests/test_model_primitives.py -q
uv run --no-sync python scripts/check_model_primitives.py --device cpu
# Nếu môi trường CUDA đã cài:
uv run --no-sync python scripts/check_model_primitives.py --device cuda
```

Probe tạo run/log/config/metadata và báo cáo shapes, parameter counts của các lớp
nền, gradient norms, sai số bảo toàn norm của RoPE. Scalar `probe_objective` chỉ để
gọi backward, không phải LM cross-entropy và không chứng minh overfit.
CUDA được yêu cầu rõ thì không fallback CPU. GPU không phải điều kiện để chạy CI;
GPU cases được chạy bổ sung khi PyTorch thực sự thấy CUDA.

Kết quả: [WP05.1 validation](wp05_1_validation.md).
