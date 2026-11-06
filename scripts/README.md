# Entry points

`prepare_data.py` thực hiện WP03 và tương đương `grast prepare-data`.
Xem `docs/wp03_data.md` để cài extra `data` và chạy nguồn thật/offline.

`train_tokenizer.py --config configs/tokenizer/train.toml` thực hiện WP04, tương đương
`grast train-tokenizer`. Xem `docs/wp04_tokenizer.md`.

`check_model_primitives.py --device cpu` (hoặc `cuda`) chạy probe WP05.1,
ghi run/log/metadata và kết quả forward/backward của embedding/RMSNorm/RoPE.
Xem `docs/wp05_architecture.md`; chưa chạy model hoàn chỉnh hoặc overfit.

Các script model/training bên dưới giữ tên theo HQ và hiện exit 1 với thông báo `skeleton only`.
Chưa có CLI arguments, training, sampling, profiling hay evaluation.

`pretrain`, `finetune`, `build_sample_bank`, `replay_policies`,
`evaluate`, `profile`, `demo` sẽ được triển khai theo critical path và có kiểm tra tương ứng.
