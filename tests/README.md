# Test status — WP02–WP05 (các phase đã triển khai)

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

Các test dự kiến theo HQ, thêm cùng implementation:

- Cached/uncached logits và generated tokens sẽ được thêm cùng WP06.2–WP06.4.
- `test_generation.py`: generation boundaries và seed reproducibility.
- `test_sample_bank.py`: contract, ordering và reproducibility.
- `test_policy_replay.py`: prefix-only observations, không gold leakage.
- `test_confidence.py`: confidence signal, stopping và validation-only calibration.

Không tạo test rỗng/skip cho các mục này để tránh hiểu nhầm là đã nghiệm thu.
