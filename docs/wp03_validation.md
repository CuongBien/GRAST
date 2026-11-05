# WP03 validation — 05/11/2026

Pipeline đã chạy đầy đủ trên source thật, rồi tái lập offline từ cache có checksum.
Không có training hoặc đánh giá accuracy của model trong WP03.

## Bằng chứng versioned

- [Dataset manifest](../experiments/wp03_dataset_manifest.json).
- [Validation record](../experiments/wp03_validation_20261105.json).
- [Source lock](../configs/data/sources.lock.json).
- [Protocol và cách chạy](wp03_data.md).

Hai run độc lập:

- `artifacts/runs/20261105T111456959057Z-wp03-data-pilot-seed42-1c1de7b2`
- `artifacts/runs/20261105T111500087337Z-wp03-data-pilot-seed42-32792f68`

Manifest và toàn bộ payload files giống nhau từng byte. Đã đọc lại mọi output JSONL,
kiểm tra record counts, file size và SHA-256; source downloads cũng khớp full SHA-256 đã pin.

Dataset fingerprint:
`79ff92320ebb64e283ce342076c5e570989ef5433d8791231b2d78f370baf485`.

Manifest file SHA-256:
`28517ab4e8a65674d64206d9ff536626b2661d008bd26797d36ec930830281eb`.

## Counts thực tế

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| TinyStories subset | 10.000 | 1.000 | 1.000 |
| GSM8K | 6.973 | 500 | 1.319 |
| Arithmetic (3 mức) | 1.500 | 300 | 300 |

Pilot có 150 validation + 150 test, mỗi mức easy/medium/hard có 50 câu/split.
Questions và references join bằng stable IDs; đã xác nhận questions không chứa answer,
canonical answer hoặc AST. Mọi pilot ID thuộc đúng split nguồn; không chồng validation/test.

Audit không phát hiện normalized-text hoặc procedural semantic-group overlap giữa các split
trên corpus/benchmark đã chọn. Source prefixes hiện tại cũng không có duplicate nội bộ bị loại.
Đây không phải chứng nhận không có paraphrase/near-duplicate theo nghĩa ngữ nghĩa.
Chọn split/pilot không dùng gold answers hoặc model outcomes; không fit threshold trên test.

## Kiểm tra phần mềm

86 tests WP02+WP03 pass. WP03 có 67 cases cho exact numeric extraction/comparison,
invalid/ambiguous markers, Unicode, arithmetic count/dependency depth/wording,
seed isolation, canonical dedup groups, stable label-independent splits, source checksum,
offline cache failures và full pipeline bằng authored fixtures.

Fixture Unicode line separator kiểm tra JSONL round-trip đúng ngay cả với reader dùng
`splitlines()`. Lint và format checks pass. CI dùng authored offline fixtures và math-only
smoke, không tải raw corpus trong mỗi PR. Raw source/output/cache nằm ngoài Git.
