# WP05 — Transformer Architecture theo phase

Mỗi phase có implementation, kiểm thử, báo cáo kết quả và commit riêng.
WP05 chỉ nghiệm thu toàn bộ sau WP05.5; các kiểm tra của phase trước không thay thế
forward/backward, overfit và causal-isolation của model hoàn chỉnh.

| Phase | Phạm vi | Gate nghiệm thu | Trạng thái |
| --- | --- | --- | --- |
| WP05.1 | ModelConfig, token embedding, RoPE, RMSNorm | Reference số học, shapes, gradients, dtype/device, nhận IDs WP04 | Đã triển khai |
| WP05.2 | MHA causal self-attention | Reference attention, causal/padding masks, future-token isolation, không NaN ở hàng bị mask | Đã triển khai |
| WP05.3 | SwiGLU và pre-norm decoder block | Shapes, gradients qua attention/FFN/residual, identity khi nhánh residual bằng 0 | Đã triển khai |
| WP05.4 | Decoder-only LM, final norm, LM head, CE loss | Logits, shift labels đúng một lần, ignore -100, tied embeddings, parameter count thực tế | Chưa triển khai |
| WP05.5 | Integration và overfit batch nhỏ | Loss giảm trên batch cố định, gradients hữu hạn, kiểm tra model không nhìn future tokens và không dùng padding | Chưa triển khai |

Mỗi lượt dừng ở phase đã nghiệm thu để người dùng có thể xem kết quả trước phase tiếp theo.
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
- Decoder block pre-norm: residual attention rồi residual SwiGLU; các linear
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

## WP05.2 — causal MHA

`CausalSelfAttention(ModelConfig)` nhận `x [B,T,H]`, optional `attention_mask [B,T]`
và `position_ids [T]` hoặc `[B,T]`. Output giữ shape `[B,T,H]`.
MHA có bốn projection Q/K/V/output không bias, khởi tạo normal std 0.02;
tổng tham số `4 * hidden_size²`. RoPE phase 1 chỉ áp dụng lên Q/K.
Scaled dot-product attention theo [Vaswani et al.](https://arxiv.org/abs/1706.03762)
được tính bằng matmul, scaling, masked softmax, dropout và matmul với V;
implementation không gọi SDPA có sẵn. Test dùng
[PyTorch SDPA math backend](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)
làm reference độc lập cùng RoPE bằng số phức.

Mask phải là bool/integer 0 hoặc 1, cùng device với x; 1/True nghĩa là token hợp lệ,
không phải additive attention bias. Causal rule dựa trên chỉ số chuỗi `key_index <= query_index`,
không dựa trên position IDs. Key padding và query padding đều bị mask.
Position IDs mặc định là chỉ số vật lý của chuỗi; với padding ở giữa chuỗi, caller
có thể truyền explicit positions nếu muốn đánh số theo token hợp lệ.

Padding x được thay bằng 0 trước projection, kể cả khi chứa NaN/Inf; valid activations
được giả định hữu hạn. Hàng không có key hợp lệ nhận finite scores tạm trước softmax,
sau đó mọi xác suất bị cấm được gán 0. Như vậy không tính softmax trên hàng toàn `-inf`;
output query padding và gradients của padding input bằng 0. Mẫu toàn padding có
output/gradients hữu hạn bằng 0. Module không thêm residual; block phase 3 sẽ xử lý residual.

Scores, softmax và weighted sum dùng FP32 cho input FP16/BF16, FP64 cho input FP64.
Autocast bị tắt trong phần này để matmul không tự quay lại FP16; projections vẫn
tuân theo dtype/autocast thông thường của PyTorch. Dropout áp dụng lên probabilities
chỉ trong train mode; eval tắt dropout. Memory attention hiện là `O(B * heads * T²)`;
chưa có KV cache hoặc kernel attention tối ưu.

```powershell
uv run --no-sync pytest tests/test_attention.py -q
uv run --no-sync python scripts/check_attention.py --device cpu
uv run --no-sync python scripts/check_attention.py --device cuda
```

Probe tạo run riêng có architecture snapshot, môi trường, source/Git metadata, log
và `artifacts/attention_report.json`. Kiểm tra future perturbation, future gradients,
padding perturbation và mẫu toàn padding ở kích thước candidate.
Kết quả: [WP05.2 validation](wp05_2_validation.md).

## WP05.3 — SwiGLU và decoder block

`SwiGLU(hidden_size, ffn_hidden_size)` tính
`down(silu(gate(x)) * up(x))`, theo
[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202).
[SiLU](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.silu.html)
là `x * sigmoid(x)`. Ba projection không bias, khởi tạo normal std 0.02;
FFN hoạt động độc lập theo từng token, không trộn chiều thời gian.
`ffn_hidden_size` là chiều intermediate thực tế, không tự nhân hệ số 2/3.
Tổng số tham số FFN là `3 * hidden_size * ffn_hidden_size`.

`DecoderBlock(ModelConfig)` dùng hai RMSNorm riêng và hai residual:

```text
clean_x = clear_padding(x)
a = clean_x + dropout(attention(attention_norm(clean_x), mask, positions))
out = a + dropout(swiglu(ffn_norm(a)))
```

Padding được clear trước norm đầu tiên và sau từng residual. Không để padding NaN/Inf
đi qua identity path hoặc làm hỏng gradients của RMSNorm. Mask validation được dùng
chung giữa attention và block qua `model/masking.py`; quy ước 1=True=valid không đổi.
Valid activations được giả định hữu hạn. Với hai output projection của attention/FFN
bằng 0, output ở token hợp lệ bằng input chính xác và input Jacobian là identity;
padding output/input gradients vẫn bằng 0.

`dropout` trong config áp dụng cho attention probabilities như phase 2 và cho output
của hai nhánh residual; không có dropout nội bộ riêng trong SwiGLU.
Dropout tắt trong eval. Khi autocast với input FP32, residual stream giữ FP32 nhờ
phép cộng với input, còn các linear projections tuân theo autocast. Với module và
input FP16/BF16, output giữ dtype tương ứng; RMSNorm/attention giữ quy tắc tích lũy
FP32 của phase trước. SwiGLU dùng dtype/autocast thông thường của PyTorch.

Một block có `4*H² + 3*H*F + 2*H` tham số: attention + SwiGLU + hai RMSNorm.
Đây chưa phải tổng model: phase 4 sẽ bổ sung embedding, stack, final norm và LM head.
Block nhận/trả `[B,T,H]`, optional mask `[B,T]` và position IDs như attention;
không có KV cache, final norm hoặc loss trong block.

```powershell
uv run --no-sync pytest tests/test_decoder_block.py -q
uv run --no-sync python scripts/check_decoder_block.py --device cpu
uv run --no-sync python scripts/check_decoder_block.py --device cuda
```

Probe ghi run/config/log/metadata, architecture và `artifacts/decoder_block_report.json`.
Nó kiểm tra gradient của mọi parameter ở block thông thường, causal/padding isolation,
mẫu toàn padding và identity output/input Jacobian khi zero hai nhánh.
Test còn kiểm tra tính nhân quả khi xếp hai block; chưa tạo language model đầy đủ.
Kết quả: [WP05.3 validation](wp05_3_validation.md).
