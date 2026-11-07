# WP06.2 validation — 07/11/2026

**Đạt nghiệm thu phase 2**: nối KV cache vào attention và CausalLM, hỗ trợ
prefill, decode từng token hoặc chunk. [API và contract](wp06_inference.md).
Sampling, generated-token equivalence và profiling thuộc các phase tiếp theo.

## Kiểm thử

448 tests toàn repository đạt trên local, gồm 45 cases mới cho cached forward:
31 CPU cases và 14 cases bổ sung trên CUDA. CI chạy CPU suite và cached-forward
smoke trên Ubuntu/Windows; một cross-device case cũ cần CUDA sẽ skip trên CI CPU.

- Cached logits khớp uncached prefix và full forward với chunk 3+1+2+1+1,
  FP64/FP32/FP16/BF16, padding trái/phải, interior gaps và hàng toàn padding.
- Chỉ project token mới; prefix K đã rotate không thay đổi, static storage được reuse.
- Causal mask chữ nhật, per-row custom positions và RoPE offsets đúng; sau custom
  positions, continuation phải tiếp tục truyền positions rõ ràng.
- Giới hạn context/capacity, batch, precision, mask và positions được kiểm tra.
- Cache gắn với model instance và trạng thái weights/metadata. Optimizer update,
  reload, thay parameter, đổi precision hoặc model khác làm prefix cũ bị từ chối.
- Raw storage append không được dùng để âm thầm nối vào model-bound prefix.
  Reset cho phép reuse; cache không giữ model sống sau khi request kết thúc.
- Exception giữa layers hoặc sau LM head rollback cả append, giữ prefix cũ để retry.
  Model thay đổi trong forward cũng bị phát hiện và abort.
- Cached path yêu cầu eval và no_grad/inference_mode; training/loss path WP05 giữ
  nguyên và tiếp tục được regression suite kiểm tra.

Ruff lint/format và wheel/sdist build đạt. Probe import trực tiếp wheel từ ngoài
source tree chạy prefill/decode khớp full forward; source distribution không chứa
local caches, generated runs hoặc weights.

## Probe trên checkpoint thật

Cùng candidate 46.148.096 parameters, checkpoint diagnostic overfit của WP05.5,
tokenizer WP04 và authored pretraining/SFT fixture cố định `[2, 27]`.
Prefill 3 tokens, tiếp theo 1 rồi 2 tokens, sau đó từng token đến hết fixture.
Mỗi chunk so với uncached forward trên toàn prefix tương ứng.

| Thiết bị | Precision | Max absolute logit error | Kết quả |
| --- | --- | --- | --- |
| CPU | FP32 | 7,62939453125e-6 | Passed |
| NVIDIA GeForce RTX 4060 Laptop GPU | FP32 | 4,291534423828125e-6 | Passed |

Tolerance `rtol=1e-4`, `atol=1e-5`. Python 3.11.16, PyTorch 2.9.1+cu128,
CUDA 12.8. Checkpoint SHA-256:
`e756d7a9519d1423a7ee554884afdc2b0ebb98a73188bd40a5554179eaa8de56`.
Tokenizer fingerprint:
`49187f5c5a806801ae6fc86cce7a9e6ca26d54d81cc989b361bbc76724abada3`.

Probe synthetic riêng cho CI có shape `[2, 9]`, không cần tải dữ liệu/tokenizer
hoặc checkpoint. Local CPU max error `1,3113021850585938e-6`.
Run paths, report hashes và kết quả chi tiết nằm trong
[validation manifest](../experiments/wp06_2_validation_20261107.json).

## Tái lập và phạm vi

```powershell
uv run --no-sync python scripts/check_cached_forward.py --device cpu
uv run --no-sync python scripts/check_cached_forward.py --device cuda --checkpoint <wp05-overfit.pt> --tokenizer <wp04-tokenizer-directory>
```

Mỗi lệnh tạo run directory, config snapshot, metadata, environment và JSON report.
Checkpoint chỉ cung cấp model weights/metadata; probe dùng seed của run mới,
không khôi phục optimizer/RNG training.

Đây là teacher-forced logits trên fixture, chưa phải generated-token equivalence,
TTFT, tokens/s hoặc VRAM profiling. Tensor payload trong report không phải VRAM
thực đo. Baseline context vẫn 1024; chưa chạy ma trận L=128/512/1024/2048.
WP06.3 tiếp tục sampling và generation; WP06.4 nghiệm thu generated tokens;
WP06.5 profiling thực đo. Chỉ model mutation được PyTorch theo dõi qua version/
identity và các API load/device/dtype thông thường được hỗ trợ; sửa `.data` hoặc
private storage trực tiếp cần reset cache theo contract.
