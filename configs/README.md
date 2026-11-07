# Research configurations — WP02

Config TOML được compose từ `experiments/` và bốn component: `model`, `tokenizer`,
`training` (ở `pretrain/` hoặc `sft/`), `generation` (ở `sampling/`). Các path component
và artifact root tương đối được resolve theo file experiment, không theo working directory.

- `experiments/local_smoke.toml`: CPU, seed 42, deterministic.
- `experiments/local_cuda.toml`: bắt buộc CUDA, không fallback CPU.
- `experiments/colab_a100.toml`: bắt buộc GPU có tên chứa A100.
- `model/draft_35m.toml`: tên file kế thừa WP02; model thực có 46.148.096 tied parameters.
  Training feasibility chưa nghiệm thu.
- `tokenizer/bpe.toml`: vocabulary đích và special-token contract dùng chung WP02/WP04.
- `tokenizer/train.toml`: WP04 train-only TinyStories, kiểm tra round-trip và thống kê codec.
- `pretrain/debug.toml`, `sft/debug.toml`: debug defaults, chưa có training loop.
- `sampling/baseline.toml`: WP02 experiment grid; `k_values` là số candidate
  sequences cho nghiên cứu, không phải top-k vocabulary filtering.
- `sampling/wp06_generation.toml`: runtime policy WP06.3, đọc riêng bằng
  `GenerationConfig`; có greedy/sampling, temperature, `top_k`, `top_p`, EOS/PAD.
  Probe lưu policy nguồn và effective settings riêng bên cạnh WP02 snapshot.

Schema version 1 được kiểm tra trong `llm_scratch.utils.config`: từ chối key không biết,
kiểu sai, seed không hợp lệ, head dimension/vocabulary không khớp, K không hợp lệ, v.v.
Snapshot resolved nằm trong `run/config.json`; file TOML nguồn được chép vào `config_sources/`.
Trong WP02, model/training/generation config chỉ được validate và snapshot; smoke chạy probe RNG
và ma trận 8×8 float32, không khởi tạo model hoặc dùng các hyperparameter để train.
WP05.4 dùng các model fields qua `ModelConfig` để khởi tạo full LM; architecture snapshot
thêm explicit `rope_theta` và `rms_norm_eps`. Xem `docs/wp05_architecture.md`.

WP03 có schema data riêng: `data/wp03.toml` và `data/sources.lock.json`.
Lệnh `grast prepare-data --config configs/data/wp03.toml` tạo corpus/splits/math/pilot
và manifest, không đi qua model-config loader. Xem `docs/wp03_data.md`.

WP05.5 có schema riêng `experiments/wp05_acceptance.toml`, hai profile tiny/candidate,
fixed fixture, optimizer recipe và thresholds đặt trước. Tiny dùng `model/overfit_tiny.toml`;
candidate dùng `draft_35m.toml` với artifact tokenizer WP04 bắt buộc. Xem `docs/wp05_architecture.md`.

WP04 dùng loader riêng cho `tokenizer/train.toml`, compose `bpe.toml` và khóa manifest WP03.
`grast train-tokenizer --config configs/tokenizer/train.toml` tạo codec/metrics trong run riêng.
Không fit trên validation/test. Xem `docs/wp04_tokenizer.md`.
