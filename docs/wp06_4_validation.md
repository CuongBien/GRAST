# WP06.4 validation — 09/11/2026

**Đạt nghiệm thu phase 4**: cached autoregressive generation khớp uncached reference
trên các fixtures đã kiểm tra. [API và contracts](wp06_inference.md).
WP06 còn profiling thực đo ở phase 5.

## Kiểm thử

586 tests toàn repository đạt trên local; 60 cases mới gồm 35 CPU và 25 CUDA.
CPU-only CI vẫn skip hai hardware cases cũ: cross-device cache và wrong-device RNG.

- Actual greedy generation: FP64/FP32/FP16/BF16, CPU/CUDA, full/left-right/gapped
  padding; logits close theo tolerance đặt trước và exact generated IDs/bookkeeping.
- Seeded stochastic generation: temperature-only, top-k, nucleus, combined top-k/top-p;
  cùng RNG schedule giữa paths, cached replay và global RNG không thay đổi.
- EOS lệch hàng, EOS ngay prefill, PAD=EOS, model-derived EOS và continued padding;
  so logits chỉ ở active rows, mọi row vẫn so exact output masks/IDs/lengths/reasons.
- Scalar RoPE offset và context boundaries với budget 0/1/5, overflow bị từ chối
  trước model/cache work. Kiểm tra cache positions trực tiếp bằng expected arange.
- Hook K projection xác nhận input lengths `[P,1,...]`, cache masks/positions đúng,
  prefix K nguyên và static storage không đổi. Last sampled token không forward thêm.
- Cache không được tái dùng giữa requests, được giải phóng khi call xong; zero budget
  không allocate/forward, tracing hook cleanup khi head lỗi và call kế tiếp vẫn chạy.
- CPU BF16/CUDA FP16 autocast và invalid use_cache/offset settings được kiểm tra.

Tolerance rtol/atol: FP64 1e-10/1e-12, FP32 1e-4/1e-5, FP16 0,005/0,002,
BF16 0,02/0,01. Không round logits hoặc điều chỉnh tolerance để làm khớp token IDs.
Ruff lint/format, wheel/sdist và installed-wheel cached generation probe đạt.
CI thêm cached/uncached generation acceptance trên Ubuntu và Windows CPU.

## Probe checkpoint thật

Candidate 46.148.096 tied parameters, checkpoint diagnostic overfit WP05.5,
tokenizer WP04, authored story/reasoning prefixes từ WP06.3. Batch `[2,13]`, seed 42,
budget 16, sampling temperature=0,8, top_k=40, top_p=0,95; EOS=2/PAD=0.
Python 3.11.16, PyTorch 2.9.1+cu128, CUDA 12.8, RTX 4060 Laptop GPU.

| Device / RoPE offset | Max absolute active-logit error | Greedy IDs | Sampled IDs |
| --- | --- | --- | --- |
| CPU / 0 | 5,7220458984375e-6 | Exact | Exact |
| RTX 4060 / 0 | 4,291534423828125e-6 | Exact | Exact |
| RTX 4060 / 7 | 5,7220458984375e-6 | Exact | Exact |

Cả greedy và sampled sinh 7/14 valid tokens cho hai hàng, dừng EOS, masks/lengths/
stop reasons giống hệt reference. Cached forward lengths `[13,1,...,1]` (14 calls),
uncached `[13,14,...,26]`. Minimum reference top-2 gap khoảng 1,2439 trong fixture này.
Mỗi probe còn replay cached generation hai lần cùng seed và kiểm tra global RNG.

```text
 shares story.<eos>
Add 2 and 3.</think><final>5</final><eos>
```

Đây là diagnostic gắn với ví dụ đã overfit, **không phải held-out benchmark hoặc bằng
chứng về chất lượng giải toán**. Stochastic equality chỉ khẳng định cho fixtures,
settings, precision, runtime và batch schedule đã ghi; near-ties/filter boundaries
có thể làm numerical-close logits chọn token khác trên input khác. Checker sẽ fail
nếu token/mask/reason khác hoặc logits ngoài tolerance, không ép chúng khớp.

Runs, report hashes, source hashes và environment được lưu trong
[validation manifest](../experiments/wp06_4_validation_20261109.json).
Checkpoint SHA-256:
`e756d7a9519d1423a7ee554884afdc2b0ebb98a73188bd40a5554179eaa8de56`.
Tokenizer fingerprint:
`49187f5c5a806801ae6fc86cce7a9e6ca26d54d81cc989b361bbc76724abada3`.

## Tái lập và bước tiếp

```powershell
uv run --no-sync python scripts/check_generation_equivalence.py --device cpu
uv run --no-sync python scripts/check_generation_equivalence.py --device cuda --checkpoint <wp05-overfit.pt> --tokenizer <wp04-tokenizer-directory>
uv run --no-sync python scripts/check_generation_equivalence.py --device cuda --position-offset 7 --checkpoint <wp05-overfit.pt> --tokenizer <wp04-tokenizer-directory>
```

Run directory chứa config/policy snapshots, metadata, environment, log và
`artifacts/generation_equivalence_report.json`. Shared setup với WP06.3 giữ cùng
tokenizer/SFT prompt framing, không restore optimizer/RNG training.

Acceptance hooks copy logits về CPU và đồng bộ thiết bị, không dùng để đo tốc độ.
Chưa có TTFT, next-token latency, tokens/s hoặc VRAM thực đo. WP06.5 sẽ chạy protocol
riêng tại L=128/512/1024/2048 với context config đủ cho cả prompt và generated tokens;
baseline context 1024 chưa tự mở rộng hoặc thay đổi. Không chốt toàn WP06 trước gate đó.
