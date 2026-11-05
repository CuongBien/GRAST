# Research configurations — WP02

Config TOML được compose từ `experiments/` và bốn component: `model`, `tokenizer`,
`training` (ở `pretrain/` hoặc `sft/`), `generation` (ở `sampling/`). Các path component
và artifact root tương đối được resolve theo file experiment, không theo working directory.

- `experiments/local_smoke.toml`: CPU, seed 42, deterministic.
- `experiments/local_cuda.toml`: bắt buộc CUDA, không fallback CPU.
- `experiments/colab_a100.toml`: bắt buộc GPU có tên chứa A100.
- `model/draft_35m.toml`: thiết kế ứng viên, chưa kiểm chứng parameter count/feasibility.
- `tokenizer/bpe.toml`: vocabulary và special-token contract, chưa có tokenizer.
- `pretrain/debug.toml`, `sft/debug.toml`: debug defaults, chưa có training loop.
- `sampling/baseline.toml`: generation defaults, chưa sinh văn bản.

Schema version 1 được kiểm tra trong `llm_scratch.utils.config`: từ chối key không biết,
kiểu sai, seed không hợp lệ, head dimension/vocabulary không khớp, K không hợp lệ, v.v.
Snapshot resolved nằm trong `run/config.json`; file TOML nguồn được chép vào `config_sources/`.
Model/training/generation config hiện chỉ được validate và snapshot; smoke chạy probe RNG
và ma trận 8×8 float32, không khởi tạo model hoặc dùng các hyperparameter để train.
