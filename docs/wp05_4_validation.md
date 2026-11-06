# WP05.4 validation — 06/11/2026

**Đạt nghiệm thu phase 4**: full decoder-only LM, final RMSNorm, LM head,
causal cross-entropy và tổng parameter count. Overfit và gate cuối WP05 thuộc phase 5.
[Thiết kế/API và lệnh tái lập](wp05_architecture.md#wp054--full-causal-lm-và-loss).

## Model và số tham số

Candidate có vocab 8192, hidden 512, 10 decoder blocks, 8 heads (head dim 64),
SwiGLU intermediate 2048, tied embeddings, context limit 1024.

| Nhóm Parameters thực | Số tham số duy nhất |
| --- | ---: |
| Token embedding | 4.194.304 |
| 10 decoder blocks | 41.953.280 |
| Final RMSNorm | 512 |
| LM head bổ sung (tied) | 0 |
| **Tổng** | **46.148.096** |

Đối chiếu độc lập: `V*H + L*(4*H² + 3*H*F + 2*H) + H`.
Tên `draft_35m.toml` được giữ để tương thích đường dẫn WP02; model thực là 46,15M,
không phải 35M. Nếu untied, công thức cho 50.342.400; test model nhỏ đo thực cả hai chế độ.
Forward/backward chưa nghiệm thu training feasibility hoặc khả năng giải toán.

## Kiểm thử

327 tests toàn repository đạt trên local, gồm 51 cases WP05.4:
40 CPU và 11 cases bổ sung trên CUDA.

- Mọi block độc lập được gọi theo thứ tự; logits khớp embedding/stack/final norm/head.
- Tied head là cùng Parameter với embedding sau dtype/device conversion, optimizer step,
  state-dict load và `assign=True`; count loại trùng đúng. Untied head có Parameter riêng.
- CE khớp phép tính log-softmax chọn target thủ công, kể cả prompt -100;
  first label không dùng, last logit không đóng góp loss. Gradient của ignored logits bằng 0.
- Loss chỉ tính cặp query/target cùng valid, bỏ left-padding boundary và không nối qua gaps.
  T=1, all-ignored, all-padding hoặc isolated valid tokens trả loss/gradients hữu hạn bằng 0.
- FP16/BF16/FP32 logits, FP32 loss, autocast và FP64 reference được kiểm tra.
- State lưu resolved architecture và tokenizer fingerprint; weights-only và atomic RNG
  checkpoint WP02 round-trip đúng. Config/fingerprint lệch hoặc tied weights mâu thuẫn bị từ chối.
- Ví dụ pretraining/SFT từ tokenizer/collator WP04 tính loss đúng, reasoning inference prefix
  khớp prefix SFT và target reasoning đầu tiên được giám sát đúng vị trí.
- Basic full-model future perturbation không đổi logits quá khứ; dropout train/eval đúng.

Suite chung ban đầu phát hiện CUDA workspace được cấu hình quá muộn khi các test
reproducibility bật deterministic sau các GPU tests khác. `tests/conftest.py` cấu hình
workspace trước torch/CUDA initialization; full suite sau sửa đạt, không tắt deterministic checks.
Ruff lint/format, wheel/sdist và installed-wheel forward/backward đạt.
CI chạy CPU suite và full candidate LM smoke trên Ubuntu/Windows, không tải dataset.

## Probe full candidate với tokenizer WP04

Probe riêng trên CPU và RTX 4060 Laptop, Python 3.11.16 / PyTorch 2.9.1+cu128 / CUDA 12.8.
Tokenizer fingerprint:
`49187f5c5a806801ae6fc86cce7a9e6ca26d54d81cc989b361bbc76724abada3`.

Input `[2,27]` gồm ví dụ pretraining và reasoning SFT được encode bằng codec WP04;
logits `[2,27,8192]`, 28 supervised next-token targets. Trên cả hai thiết bị:
loss hữu hạn và khớp CE thủ công, mọi 92 Parameter objects duy nhất có gradients
hữu hạn/khác 0, count thực khớp công thức, padding logits bằng 0 và prefix không đổi
khi thay future inputs. Loss khoảng 9,0036 là loss của model ngẫu nhiên, không phải kết quả train.

CI-mode probe không tokenizer dùng seeded synthetic IDs và cũng chạy thành công.
Run paths, exact losses/counts, runtime và gradient summaries:
[validation JSON](../experiments/wp05_4_validation_20261106.json).
Không lưu weights lớn trong Git. Chưa chạy optimizer loop, overfit, pretraining hay generation.
