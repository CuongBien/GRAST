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

Các script model/training bên dưới giữ tên theo HQ và hiện exit 1 với thông báo `skeleton only`.
Chưa có CLI arguments, training, sampling, profiling hay evaluation.

`pretrain`, `finetune`, `build_sample_bank`, `replay_policies`,
`evaluate`, `profile`, `demo` sẽ được triển khai theo critical path và có kiểm tra tương ứng.
