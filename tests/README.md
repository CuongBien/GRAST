# Test status — WP02 + WP03

Sync environment với `uv sync --locked --extra cpu --extra data` (hoặc `cuda` thay `cpu`).
`test_package.py` kiểm tra metadata/importability; `test_reproducibility.py` kiểm tra
config validation, RNG/checkpoint continuation, run failure logging, independent-process
repeatability, khác seed, CUDA preflight failure và Git chưa có commit.
Chưa có behavioral test cho model.

WP03: `test_answer_extraction.py`, `test_difficulty.py` và `test_datasets.py` kiểm tra
numeric protocol, exact rational answers, operation count/depth/wording, stable splits
không phụ thuộc labels, normalized-text/semantic leakage, checksum/offline cache và
full pipeline/pilot bằng authored Parquet/JSONL fixtures. Không tải dữ liệu thật trong CI.

Các test dự kiến theo HQ, thêm cùng implementation:

- `test_attention.py`: causal mask, shape và không nhìn future tokens.
- `test_kv_cache.py`: cached/uncached logits trong tolerance, fixed seed.
- `test_generation.py`: generation boundaries và seed reproducibility.
- `test_sample_bank.py`: contract, ordering và reproducibility.
- `test_policy_replay.py`: prefix-only observations, không gold leakage.
- `test_confidence.py`: confidence signal, stopping và validation-only calibration.

Không tạo test rỗng/skip cho các mục này để tránh hiểu nhầm là đã nghiệm thu.
