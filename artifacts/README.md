# Local artifacts

`runs/`, `checkpoints/`, `tokenizers/`, `sample_bank/`, `results/` là output cục bộ, bị gitignore.
Chỉ `.gitkeep` và tài liệu này được versioned. Lưu output bền vững ngoài Git
(ví dụ Google Drive), ghi link/checksum trong manifest ở `experiments/`.
WP02 smoke tạo `runs/<UTC-name-seed-unique-id>/` gồm config snapshot, metadata,
logs, checkpoints RNG và kết quả probe. Đây không phải checkpoint model hoặc benchmark.
Xem `docs/reproducibility.md` để biết protocol và cấu trúc cụ thể.
