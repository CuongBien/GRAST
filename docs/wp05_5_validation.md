# WP05.5 — nghiệm thu Transformer Architecture, 06/11/2026

**WP05 đạt gate kiến trúc.** Full decoder-only Transformer chạy forward/backward
bằng PyTorch thuần, overfit được batch nhỏ, giữ causal/padding isolation trước và sau train.
Không dùng pretrained model hoặc `nn.Transformer`/`nn.MultiheadAttention`.

## Gate cố định trước thử nghiệm

Recipe: [wp05_acceptance.toml](../configs/experiments/wp05_acceptance.toml),
batch tự viết: [fixture](../experiments/wp05_overfit_fixture.json).
Hai records gồm một pretraining story và một reasoning SFT. Không dùng validation/test
để chọn dữ liệu, hyperparameters hoặc threshold. Các gate:

- Chính xác 200 optimizer steps trên cùng batch, dùng **final step**, không chọn best checkpoint.
- Final loss / initial loss ≤ 0,2; teacher-forced accuracy trên supervised targets ≥ 0,95.
- Gradients hữu hạn mọi bước; các full-model causal/padding audits đạt trước và sau train.
- Model checkpoint round-trip giữ logits chính xác; fingerprint batch/codec/model được khóa.

## Kết quả local

| Thuộc tính | Tiny / CPU | Candidate / RTX 4060 Laptop |
| --- | ---: | ---: |
| Parameters | 100.160 | 46.148.096 |
| Layers / hidden / FFN | 2 / 64 / 128 | 10 / 512 / 2048 |
| Tokenizer | BPE fit trên fixture, vocab280 | WP04 cố định, vocab8192 |
| Input shape | `[2,26]` | `[2,27]` |
| Supervised targets | 26 | 26 |
| Initial loss | 5,638908863 | 8,976395607 |
| Final loss | **0,000239020548** | **0,000025771769** |
| Final / initial | 0,000042387730 | 0,000002871060 |
| Final supervised token accuracy | **100%** | **100%** |
| Optimizer steps | 200 | 200 |

AdamW betas (0,9; 0,95), weight decay 0, norm clipping 1, dropout 0, FP32;
LR tiny 0,003, candidate 0,001, foreach/fused tắt. Python 3.11.16,
PyTorch 2.9.1+cu128, CUDA 12.8; deterministic và TF32 tắt theo WP02.
Teacher-forced accuracy trên batch đã ghi nhớ không phải benchmark reasoning accuracy.

## Kiểm tra tích hợp và tái lập

346 tests local đạt; WP05.5 thêm 19 cases (18 CPU + 1 CUDA bổ sung).
Full model được audit ở ba cutoff trước/sau training: thay suffix không đổi prefix logits
**chính xác**; gradients đến future input embeddings bằng 0 và gradients hợp lệ khác 0.
Padding IDs/gold labels không ảnh hưởng logits; left/right/interior padding hoạt động;
all-padding logits/loss/gradients bằng 0 hữu hạn.

Forward khác shape dùng tolerance FP32 rtol 1e-4 / atol 1e-5, FP64 1e-10 / 1e-12,
do thứ tự cộng của GEMM có thể đổi. Gate same-shape và future gradients vẫn exact-zero.
Audit đối chứng phải thất bại khi cố ý để attention nhìn tương lai hoặc làm PAD logits khác 0.
Kiểm tra zero-target batch và nonfinite gradients dừng trước optimizer step.

Hai fresh processes cùng seed/CPU/runtime cho report/history giống nhau hoàn toàn.
Lưu/tải optimizer + model + RNG rồi cập nhật tiếp khớp từng tensor và history với cập nhật
không ngắt. Final checkpoint chứa step, model, optimizer, RNG và batch fingerprint;
reload giữ logits chính xác. Checkpoint candidate khoảng 528 MiB được giữ trong artifact,
không đưa vào Git. Các run lỗi trong quá trình phát triển giữ failed lifecycle, không phải evidence đạt.

CI Ubuntu/Windows chạy CPU suite và full 200-step tiny acceptance. Ruff lint/format,
wheel/sdist và wheel diagnostic integration được kiểm tra cùng thay đổi.
Evidence versioned: [WP05 validation manifest](../experiments/wp05_validation_20261106.json).

## Tái lập và giới hạn

```powershell
uv run --no-sync python scripts/check_architecture.py --device cpu
uv run --no-sync python scripts/check_architecture.py --profile candidate --device cuda --tokenizer PATH/TO/artifacts/tokenizer
```

Đã hoàn thành các gate WP05.1–WP05.5: embedding/RoPE/RMSNorm, causal MHA,
SwiGLU/residual block, full LM/CE/count và overfit/integration.
Candidate 46,15M chỉ được train trên batch ngắn cố định; chưa nghiệm thu max-context
corpus pretraining, generation, KV cache hoặc chất lượng tổng quát hóa. Những phần đó
thuộc bước sau, không được suy ra từ kết quả overfit này.
