# WP06.5 — online inference profiling protocol

Chạy model PyTorch thật và đường `generate` đã đối chiếu ở WP06.4. Đây là warm
streaming request latency trên một thiết bị, không phải offline replay cost hoặc
throughput server nhiều request. [Kết quả thực đo](wp06_5_validation.md).

## Protocol cố định

- Candidate 46.148.096 tied parameters; PyTorch explicit MHA, không compile/fused
  attention substitution. FP32 weights/KV; score/softmax/reductions theo implementation WP05.
- B=1, L=128/512/1024/2048 valid prompt tokens, N=16 new tokens. Greedy, EOS disabled
  để mọi lần đo có cùng lượng output, không lấy tốc độ từ replay.
- Seed 42, deterministic algorithms, TF32 disabled, CPU num_threads=1 theo WP02.
  Prompt IDs dùng private CPU Generator(seed+L), uniform [8,V), BOS=1 ở cột đầu;
  all-valid masks. Đây là synthetic workload cho timing, không phải benchmark chất lượng.
- Cùng prompt/checkpoint cho cached và uncached. 2 full warm-up requests cho từng
  condition, bỏ samples; 5 measured repeats. Mode order đổi theo L: uncached/cached
  ở L128, cached/uncached ở L512, rồi lặp. Không có speed threshold để chọn samples.
- Model load, tokenize/prompt construction, H2D, metadata/report writes nằm ngoài timing.
  Validation, cache allocation, model forward, greedy selection và output bookkeeping
  của request nằm trong timing. CUDA synchronize trước request, tại mỗi token ready,
  và sau request. Không dùng acceptance hooks/copy logits về CPU trong phép đo.

`generate(..., on_token=callback)` gọi observer sau khi token được chọn, append/mask
và stop decision đã xong; callback nhận `(step, token_ids, valid_rows)` read-only.
Profiler chỉ synchronize device và ghi `perf_counter_ns`. Callback không có khi gọi
generation bình thường. Per-token sync/callback có chi phí và cắt asynchronous overlap:
đây là **host-observed streaming latency**, không phải GPU kernel-only timing.

## Định nghĩa metrics

Với t0 trước generate, ti khi token thứ i đã sẵn sàng, tend sau return và sync:

- TTFT = t1-t0, gồm prefill và chọn token đầu, cả cấp cache/validation của request.
- Next-token latencies = t2-t1, ..., tN-t(N-1). Tổng 15 intervals mỗi repeat, 75 mỗi condition.
- Decode tokens/s = B*(N-1)/(tN-t1), loại TTFT.
- Overall tokens/s = B*N/(tend-t0), gồm toàn request.
- p50/p95 dùng linear quantiles. Latency intervals được pool trong từng condition;
  throughput là distribution của ratios từng repeat, không dùng reciprocal p95 latency.
- Raw samples lưu token-ready offsets/request elapsed bằng ns, các latency bằng ms,
  generated IDs và memory snapshots; không chỉ giữ summary.

Cached prefill một lần, decode N-1 lần token mới; uncached forward full prefix mỗi
bước. Last sampled token không forward thêm. Exact generated IDs được so giữa các
repeats/modes ở mỗi L, lỗi thì run fail; không ép logits/token để giữ benchmark xanh.

## Context override có kiểm soát

Baseline/checkpoint WP05 giữ context 1024. Config profiling riêng đặt 2064 = 2048+16.
Loader trước hết strict-load checkpoint theo config gốc. Sau đó tạo model inference
mới, chỉ đổi `context_length` và in-memory extra_state của copy, strict-copy weights
và kiểm tra equality từng Parameter. Không đổi checkpoint trên đĩa hoặc nới global
checkpoint compatibility rules. RoPE theta, heads/layers/dim và embedding tying giữ nguyên.
Report ghi source/effective configs, checkpoint/tokenizer fingerprints và override.
Mở rộng bounds phục vụ đo kernel; không chứng minh model đã được train/tổng quát hóa
ở context 2064. Tổng prompt+output phải trong effective context, không truncate/evict.

## VRAM

Theo [CUDA synchronize](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.synchronize.html),
đồng bộ trước khi chốt timing/memory snapshots. Trước mỗi measured request reset
[peak counters](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.memory.reset_peak_memory_stats.html),
ghi baseline/end allocated/reserved và max allocated/reserved của request.
Peak allocated above baseline chỉ là delta allocator, không quy toàn bộ delta cho KV.

`empty_cache` chỉ ở boundary condition trước warm-up; không gọi trong timed request.
Warm-ups làm allocator/kernel warm, nên baseline reserved có thể lớn hơn allocated.
Allocated là live tensor allocations; reserved bao gồm caching allocator blocks.
Đây là memory của process do PyTorch allocator theo dõi, không phải whole-device
VRAM/driver context/ứng dụng khác. Có nvidia-smi telemetry ngoài timing (nhiệt độ,
power/clock/utilization/memory-used khi được hỗ trợ); không khóa GPU clocks hoặc giả
định laptop GPU được cách ly. CPU smoke trả memory=null, không gọi đó là VRAM.

`cache_payload_formula_bytes` là bytes tính từ shape/dtype, capacity L+N và used
columns L+N-1; được ghi riêng, không thay thế allocator peak thực đo.

## Chạy và artifacts

```powershell
uv run --no-sync python scripts/profile.py --checkpoint PATH/TO/wp05-overfit.pt
# Tương đương, không qua wrapper:
uv run --no-sync python -m llm_scratch.inference.profiling --checkpoint PATH/TO/wp05-overfit.pt
# CI/smoke nhỏ, CPU timing, không phải nghiệm thu GPU matrix:
uv run --no-sync python scripts/profile.py --config configs/experiments/wp06_profiling_smoke.toml
```

Default config `configs/experiments/wp06_profiling.toml`; CLI có `--config`,
`--device cpu|cuda` và `--output-root`. CUDA requested không fallback CPU. Không truyền
checkpoint thì report ghi seeded random weights, không gọi đó là trained checkpoint.
`scripts/profile.py` chuyển CLI vào package subprocess; khi Python cProfile import
tên `profile` từ scripts, file giữ stdlib API để không làm lỗi torch.optim.

Run directory có config/model source snapshots, resolved config, metadata/source Git,
environment, logs, setup, JSON từng condition và report tổng. Lỗi giữa ma trận giữ
partial records và run failed, không xuất final passed report. Timings có noise;
reproducibility nghĩa là cùng protocol/weights/inputs và token IDs, không phải latency
bitwise-identical. Không dùng các số này để suy ra held-out accuracy hoặc A100 performance.
