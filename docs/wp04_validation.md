# WP04 validation — 05/11/2026

WP04 đạt nghiệm thu local: BPE tự viết, encode/decode dữ liệu thật, artifact tái lập
và giao diện token dùng chung cho pretraining/SFT/inference.

## Dữ liệu và codec

Fit duy nhất 10.000 TinyStories train từ manifest WP03 fingerprint
`79ff92320ebb64e283ce342076c5e570989ef5433d8791231b2d78f370baf485`.
Corpus training có 8.674.761 Unicode code point / 8.692.527 UTF-8 byte.
8.192 token thực tế = 8 controls + 256 byte + 7.928 merge; min frequency 2.

Tokenizer fingerprint:
`49187f5c5a806801ae6fc86cce7a9e6ca26d54d81cc989b361bbc76724abada3`.
Hai lần train độc lập tạo cả bốn file artifact giống từng byte; SHA-256 và run paths
được ghi trong [manifest](../experiments/wp04_tokenizer_manifest.json) và
[validation JSON](../experiments/wp04_validation_20261105.json).
Artifact local có thể tái tạo bằng [một lệnh](wp04_tokenizer.md); không đưa corpus hoặc
vocab/merges sinh ra vào Git.

## Kiểm tra thực tế

22.892/22.892 document thuộc cả chín split TinyStories/math/GSM8K round-trip chính xác
với UTF-8 strict, không sinh `<unk>`. Có 144 so sánh token IDs trước/sau save-load
trong mỗi run. Các tỷ lệ dưới đây dùng codec cố định, không thêm BOS/EOS.

| Corpus | Train tokens/char | Validation | Test |
| --- | ---: | ---: | ---: |
| TinyStories | 0,4290 | 0,4263 | 0,4263 |
| Math | 0,5001 | 0,5005 | 0,5009 |
| GSM8K question | 0,4590 | 0,4602 | 0,4604 |

Pretraining framing khôi phục text; reasoning prompt khớp prefix SFT.
Đây là kiểm tra token-level, chưa chạy neural model hoặc đo chất lượng generation.

## Kiểm thử và đóng gói

138 tests toàn repository đạt trên local. WP04 bổ sung 52 tests:
trainer đối chiếu bản tham chiếu full-recount trên corpus ngẫu nhiên; ties và thứ tự
corpus; Unicode, emoji, combining marks, byte fallback; special-token literal/opt-in;
ID lỗi, incomplete UTF-8; save/load, checksum và vocab không khớp merges;
padding/labels và SFT/inference prefix; CLI/config/logs và held-out isolation.
Thay riêng validation text rồi train lại tạo codec giống từng byte.
Pipeline từ chối fit file validation/test và dữ liệu sai checksum.

Ruff lint/format và build wheel/sdist được kiểm tra cùng thay đổi.
GitHub Actions chạy suite và đóng gói trên Ubuntu/Windows khi push branch WP04.
CI dùng authored fixtures, không cần mạng tải dataset cho các test WP04.

Tái lập codec dựa trên config, dữ liệu có checksum và môi trường Python được pin.
Không yêu cầu timing, run ID hoặc metadata runtime giống nhau.
Các adapter chưa thực hiện packing/chunking cho context model; bước training sẽ bổ sung
theo context đã chọn và dùng đúng tokenizer fingerprint.
