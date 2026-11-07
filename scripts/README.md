# Entry points

`prepare_data.py` thực hiện WP03 và tương đương `grast prepare-data`.
Xem `docs/wp03_data.md` để cài extra `data` và chạy nguồn thật/offline.

`train_tokenizer.py --config configs/tokenizer/train.toml` thực hiện WP04, tương đương
`grast train-tokenizer`. Xem `docs/wp04_tokenizer.md`.

`check_model_primitives.py --device cpu` (hoặc `cuda`) chạy probe WP05.1,
ghi run/log/metadata và kết quả forward/backward của embedding/RMSNorm/RoPE.
Xem `docs/wp05_architecture.md`; chưa chạy model hoàn chỉnh hoặc overfit.

`check_attention.py --device cpu` (hoặc `cuda`) chạy probe WP05.2 trên kích thước
candidate: forward/backward, future-token isolation, padding/fully-masked rows.

`check_decoder_block.py --device cpu` (hoặc `cuda`) chạy probe WP05.3: gradients
qua attention/SwiGLU/norms, residual identity, causal/padding isolation ở một block.

`check_causal_lm.py --device cpu` (hoặc `cuda`) chạy full model/loss/count probe WP05.4.
Thêm `--tokenizer PATH` để dùng artifact WP04 với ví dụ pretraining/SFT;
không truyền thì dùng seeded synthetic IDs cho CI. Chưa chạy overfit hoặc pretraining.

`check_architecture.py --device cpu` chạy WP05.5 model nhỏ và 200-step overfit gate;
thêm `--profile candidate --device cuda --tokenizer PATH` để dùng full candidate/codec WP04.
Cấu hình/gate: `configs/experiments/wp05_acceptance.toml`; đây là diagnostic fixed-batch loop,
chưa phải pretraining/SFT trainer cho corpus. Checkpoint lớn được lưu trong run, không vào Git.

`check_kv_cache.py --device cpu` (hoặc `cuda`) chạy WP06.1 KV storage probe với
geometry candidate. `--dtype`/`--max-length` chọn payload layout; chưa chạy cached LM
hoặc profiling latency/VRAM. Xem `docs/wp06_inference.md`.

`check_cached_forward.py --device cpu` (hoặc `cuda`) chạy WP06.2 cached/uncached logits
probe. Optional `--checkpoint PATH --tokenizer PATH` kiểm tra candidate đã overfit ở WP05
với fixture được encode qua codec WP04. Chưa chạy sampling/generation/profiling.

Các script model/training bên dưới giữ tên theo HQ và hiện exit 1 với thông báo `skeleton only`.
Chưa có CLI arguments, training, sampling, profiling hay evaluation.

`pretrain`, `finetune`, `build_sample_bank`, `replay_policies`,
`evaluate`, `profile`, `demo` sẽ được triển khai theo critical path và có kiểm tra tương ứng.
