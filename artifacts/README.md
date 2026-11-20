# Local artifacts

`runs/`, `checkpoints/`, `tokenizers/`, `sample_bank/`, `results/` là output cục bộ, bị gitignore.
Chỉ `.gitkeep` và tài liệu này được versioned. Lưu output bền vững ngoài Git
(ví dụ Google Drive), ghi link/checksum trong manifest ở `experiments/`.
WP02 smoke tạo `runs/<UTC-name-seed-unique-id>/` gồm config snapshot, metadata,
logs, checkpoints RNG và kết quả probe. Đây không phải checkpoint model hoặc benchmark.
Xem `docs/reproducibility.md` để biết protocol và cấu trúc cụ thể.

Checkpoint pretraining2.000updates do user cung cấp đã sắp tại
`checkpoints/wp07-pretrain-tinystories-2000/final.pt`; cạnh đó có `provenance.json`
và `evidence/` từ summary. Hash/evidence nhỏ được versioned trong `experiments/`;
weights không vào Git. Prepared CoT SFT local tại `wp07-4/sft_data/`.

Checkpoint SFT đủ 500 updates tại `checkpoints/wp07-cot-sft-math-500/final.pt`,
có provenance và summary evidence cạnh weights. Evaluator WP07.5 dùng hai checkpoint
pretrain/SFT thật, lưu predictions/score/text/curves trong run mới; không sửa weights.
