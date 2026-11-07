# WP06.3 validation — 08/11/2026

**Đạt nghiệm thu phase 3**: greedy, temperature, exact top-k/nucleus sampling,
EOS và seeded uncached autoregressive generation. [API/contracts](wp06_inference.md).
Cached generation equivalence và profiling còn ở WP06.4–5.

## Kiểm thử

526 tests toàn repository đạt trên local. Phase này có 78 cases: 66 CPU cases và
12 cases bổ sung trên CUDA. CPU-only CI skip hai hardware cases toàn repository:
cross-device cache cũ và wrong-device sampling generator mới.

- Phân phối temperature so với xác suất tính trực tiếp; FP16/BF16/FP32/FP64 inputs.
- Exact top-k với ties giữ ID nhỏ trước, k=0/k=vocab; top-p giữ minimal prefix,
  crossing token và exact threshold boundary. Combined top-k/top-p dùng xác suất
  renormalize sau top-k. Temperature cực nhỏ/lớn vẫn còn finite candidate và giữ rank.
- Empirical frequencies 12.000 draws với seed cố định trên CPU/CUDA nằm trong absolute
  tolerance 0,02 của phân phối `[0.6, 0.3, 0.1]`.
- Greedy raw argmax không draw RNG; sampling dùng private generator, replay đúng,
  generator advance đúng và không advance global CPU/CUDA RNG.
- Known-transition model kiểm tra EOS khác thời điểm theo hàng, PAD=EOS, mixed
  EOS/budget stop, last-valid readout khi left/right padding/gaps, valid generated PAD,
  prompt EOS và finished rows được bỏ trước sampling.
- Zero budget, exact context boundary, invalid mask/vocab/config/seed và model mode
  được kiểm tra. Context overflow bị từ chối trước forward, không truncate.
- Real small CausalLM greedy khớp independent full-prefix reference loop; stochastic
  generation tái lập cùng seed, khác seed đổi outputs trên fixed broad distribution.

Ruff lint/format, wheel/sdist build đạt. Wheel được import ngoài source tree và chạy
greedy/sampled generation với exact replay. Source distribution không chứa local caches,
generated runs hoặc weights. CI thêm generation smoke CPU cho Ubuntu và Windows.

## Probe checkpoint WP05 + tokenizer WP04

Candidate 46.148.096 tied parameters; Python 3.11.16, PyTorch 2.9.1+cu128, CUDA 12.8.
Checkpoint diagnostic overfit WP05.5 và tokenizer WP04 dùng cùng fingerprints như WP06.2.
Authored story prefix `A fox` và reasoning prefix `What is 2 + 3?` được encode bằng codec
thật, reasoning framing dùng chung với SFT. Batch right-padded `[2,13]`, seed 42,
max_new_tokens=16, temperature=0,8, top_k=40, top_p=0,95, EOS=2, PAD=0.
Greedy dùng cùng validated settings nhưng không áp dụng sampling filters.

| Thiết bị | Greedy lengths / stop | Sampled lengths / stop | Replay |
| --- | --- | --- | --- |
| CPU | 7, 14 / EOS cả hai | 7, 14 / EOS cả hai | Exact token IDs/masks/lengths |
| RTX 4060 Laptop GPU | 7, 14 / EOS cả hai | 7, 14 / EOS cả hai | Exact token IDs/masks/lengths |

Trong fixture này, greedy và sampled decode cùng suffix:

```text
 shares story.<eos>
Add 2 and 3.</think><final>5</final><eos>
```

Generated EOS được giữ; tail padding của hàng đầu có mask 0. Đây là dữ liệu diagnostic
đã gắn với ví dụ overfit, **không chứng minh khả năng tổng quát hóa hoặc benchmark accuracy**.
Cùng seed không hứa sampled IDs giống nhau giữa devices/runtime/batch schedules nói chung.
Phiên bản hóa report, policy, hashes và environment trong
[validation manifest](../experiments/wp06_3_validation_20261108.json).

## Tái lập và phạm vi

```powershell
uv run --no-sync python scripts/check_generation.py --device cpu
uv run --no-sync python scripts/check_generation.py --device cuda --checkpoint <wp05-overfit.pt> --tokenizer <wp04-tokenizer-directory>
```

Optional `--seed`, `--policy` và `--output-root`. Mỗi lệnh tạo run directory, WP02
config snapshot, runtime policy source/effective snapshot, metadata, environment, log
và `artifacts/generation_report.json`. `generation.k_values` ở WP02 là số candidate
sequences, không phải vocabulary `top_k`; runtime schema riêng không nhận key đó.

Generation phase này dùng **uncached full-prefix baseline**, không expose custom
positions hoặc cache argument. WP06.4 sẽ tích hợp cache và nghiệm thu logits/tokens
equivalence. Chưa đo TTFT, next-token latency, tokens/s hoặc VRAM; ma trận prompt
L=128/512/1024/2048 thuộc WP06.5. Baseline context vẫn 1024, không tự kéo dài.
