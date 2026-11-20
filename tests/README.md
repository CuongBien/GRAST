# Test status — WP02–WP06 (các phase đã triển khai)

Sync environment với `uv sync --locked --extra cpu --extra data` (hoặc `cuda` thay `cpu`).
`test_package.py` kiểm tra metadata/importability; `test_reproducibility.py` kiểm tra
config validation, RNG/checkpoint continuation, run failure logging, independent-process
repeatability, khác seed, CUDA preflight failure và Git chưa có commit.
Model được bổ sung kiểm thử theo từng phase WP05.

WP03: `test_answer_extraction.py`, `test_difficulty.py` và `test_datasets.py` kiểm tra
numeric protocol, exact rational answers, operation count/depth/wording, stable splits
không phụ thuộc labels, normalized-text/semantic leakage, checksum/offline cache và
full pipeline/pilot bằng authored Parquet/JSONL fixtures. Không tải dữ liệu thật trong CI.

WP04: `test_tokenizer.py` so với trainer tham chiếu full-recount, kiểm tra merge rank,
Unicode/byte fallback, specials, round-trip, deterministic ties, save/load/checksum và
pretraining/SFT/inference adapters. `test_tokenizer_pipeline.py` chạy CLI với JSONL tự viết,
kiểm tra log/config/artifact và thay held-out text để xác minh codec không đổi.

WP05.1: `test_model_primitives.py` kiểm tra ModelConfig, embedding với IDs WP04,
parameter count và repeated-token gradients, RMSNorm đối chiếu PyTorch FP64 và
finite-difference gradcheck, RoPE đối chiếu ma trận quay độc lập, relative dot products,
explicit positions/chunk equivalence, Unicode IDs, shapes và low precision.
CPU cases luôn chạy; các cases tương ứng chạy thêm trên CUDA khi có thiết bị.
WP05.2: `test_attention.py` so MHA với native SDPA math + complex RoPE về output,
input/parameter gradients; kiểm tra causal prefix, zero future gradients, left/right/interior
padding, fully padded rows và nonfinite padding. Có gradcheck, FP16/BF16/autocast,
dropout train/eval, state-dict, count/shapes và tích hợp IDs/masks WP04.
WP05.3: `test_decoder_block.py` so SwiGLU và cả block với reference độc lập về
output/input/parameter gradients, kiểm tra finite-difference, zero-branch identity/Jacobian,
causal isolation của 1–2 block, padding NaN/Inf, all-padding gradients, dtype/autocast,
dropout, state-dict/count/shapes và tích hợp WP04. Việc chia sẻ mask validation được
kiểm tra cùng suite attention của phase trước.
WP05.4: `test_transformer.py` kiểm tra assembly order, unique parameter counts,
tied/untied head và alias sau dtype/device/load/optimizer step, shifted CE reference,
prompt/padding pair masks, differentiable-zero cases, mixed precision và FP32 loss.
Có checkpoint metadata/config/fingerprint/weights-only round-trip qua atomic checkpoint WP02,
WP04 pretraining/SFT prefix và basic future-token isolation. Overfit thuộc WP05.5.
WP05.5: `test_architecture_acceptance.py` kiểm tra full-model overfit/audits trước-sau,
negative controls phát hiện future leakage/PAD logits lỗi, finite-gradient gate, zero-target
batch, optimizer coverage, exact optimizer/RNG checkpoint continuation, failed lifecycle
và hai fresh-process full-protocol reports giống nhau. Model nhỏ cùng kiến trúc chạy trong CI.
`conftest.py` đặt cuBLAS workspace trước khi CUDA được khởi tạo để GPU cases chạy
được trước/sau các test bật deterministic mode mà không phụ thuộc thứ tự.

WP06.1: `test_kv_cache.py` kiểm tra static storage với exact concat references,
RoPE keys/value/metadata, layer-consistent commit, exception/incomplete abort, bounds,
reset/reuse, payload bytes, noncontiguous inputs, dtype/device/shape validation và
inference-mode lifetime. Một test cross-device được skip khi runtime không có CUDA;
đây là hardware case thực, không phải placeholder cho cached generation.
WP06.2: `test_cached_forward.py` so prefill/token/chunk logits với uncached ở các dtype,
padding layouts và positions; kiểm tra inference-only guards, ownership, stale weights,
dtype/autocast, reset, failure rollback sau head, requests interleaved và weak owner lifetime.
Hook projection xác minh decode chỉ project token mới và past K không đổi.

Các test dự kiến theo HQ, thêm cùng implementation:

- `test_sample_bank.py`: contract, ordering và reproducibility.
- `test_policy_replay.py`: prefix-only observations, không gold leakage.
- `test_confidence.py`: confidence signal, stopping và validation-only calibration.

Không tạo test rỗng/skip cho các mục này để tránh hiểu nhầm là đã nghiệm thu.

## WP06.3 sampling và generation

`test_sampling.py`: distribution/temperature, exact top-k với ties, minimal nucleus
prefix và threshold boundary, combined filtering sau renormalization, extreme finite
settings, statistical frequencies, private RNG replay và input/config errors.
`test_generation.py`: hand-authored transitions, last-valid readout khi padding/gaps,
EOS từng hàng và cả batch, EOS/PAD dùng chung ID, mask/length/budget/context boundaries,
finished rows không tiêu thụ sampling draws và real-model seeded baseline.
CPU/CUDA được kiểm tra; wrong-device RNG case cần CUDA và skip trên CPU-only CI.

## WP06.4 cached/uncached generation

`test_generation_equivalence.py` chạy hai generation paths thật với các precision,
padding layouts, sampling policies và scalar RoPE offsets. So active logits theo
tolerance, exact IDs/masks/lengths/stop reasons; kiểm tra EOS lệch hàng/PAD=EOS,
model-derived EOS, prefill/decode projection lengths, unchanged prefix/storage,
cache masks/positions, request isolation/release, zero budget, context boundary,
autocast, RNG và hook cleanup khi forward lỗi. 60 cases local: 35 CPU + 25 CUDA.

## WP06.5 profiling

`test_profiling.py` kiểm tra callback không đổi generated tokens, timestamps thật
monotonic/đủ budget, CUDA allocator peaks, context-only copy không đổi weights/source
metadata và giữ strict load rules, protocol validation, condition order/input identity,
quantiles/throughput denominators và stdlib cProfile/torch.optim regression.
Không có latency/speedup thresholds phụ thuộc phần cứng. CPU timing có VRAM=null.

## WP07.1 training engine

`test_training_engine.py` kiểm độc lập accumulated gradients/updates với combined
token-weighted reference, decay/tied Parameter dedup, warmup/cosine endpoints, finite
loss và logging, exact continuation với dropout + RNG/Adam slots/scheduler/scaler,
FP16 overflow không tăng scheduler, failed-state recovery, zero supervision/partial
windows, validation và config/runtime/boundary guards. Colab long training chưa được
thay bằng những unit tests này; dataset cursor integration thuộc WP07.2.

## WP07.2 pretraining data/full resume

`test_training_data.py` kiểm target coverage/Unicode/EOS/BOS/tail/overlap, no test-set
reads, checksums/duplicates/relocation/immutability, pending-window commit/abort và
epoch coverage, mismatches bị chặn trước weights/RNG, actual next-batch/loss/Adam
resume, validation không dịch cursor, failures sau optimizer, FP16 skips tiêu thụ
data, model/stage compatibility và fresh-process resume sau khi đổi data-root path.

## WP07.3 pretraining runner

`test_pretraining_runner.py` chạy full runner và restart bằng Python process mới,
so model/Adam/scheduler/scaler/data cursor/RNG/history exact, xóa run cũ vẫn dựng lại
curves/CSV; kiểm periodic/milestone/final, completed no-op resume, recipe/history guards,
Drive write failure, boundary/failed-update interrupts, CUDA FP16 skipped-attempt
durability, input ZIP checksum/path/duplicate guards và notebook syntax/live logging.
Quality/corpus baseline/A100 training chưa được thay bằng các tests này.

`test_kaggle_notebook.py` thực thi helpers từ notebook phát hành, kiểm tra input ZIP/
expanded, corruption, precision selection, launch smoke/full/resume và real CUDA probe.
Không thay thế execution evidence từ Kaggle; adapter không thay core runner/model.

## WP07.4 CoT SFT

`test_sft.py` kiểm reasoning Fraction/postorder, numeric normalization/GSM8K split,
prompt/padding masks với CE reference độc lập, example isolation, Unicode/literal
special strings, no test/pilot/default-GSM8K reads, no truncation/reuse, data guards,
warm-start chỉ weights và fresh optimizer/RNG/cursor, preflight incompatible tensors,
CPU/CUDA-FP16 exact SFT resume, restart tiến trình mới không còn parent checkpoint,
validation token count/RNG/cursor invariance, overlap, bundle corruption và notebooks.
Long SFT/generation/pilot acceptance vẫn cần artifacts thực từ user.

`test_notebook_bootstrap.py` tái hiện kernel chưa xử lý `.pth` của editable install
bằng Python `-S` với site-packages chỉ trên PYTHONPATH: trước setup import fail,
sau activation import thật datasets/SFT bundle/runner ở cả kernel và subprocess.
Kiểm tra idempotence, giữ dependency paths, checkout thiếu và cached package khác
vị trí; cả Colab/Kaggle dùng chính helper trong notebook phát hành.
