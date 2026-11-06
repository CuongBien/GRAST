# WP03 — Dataset & Benchmark Design

Scope do người dùng chốt và cho phép thực hiện ngày 05/11/2026.
**Trạng thái: đã triển khai và kiểm tra pipeline dữ liệu local.**
Nhánh `feat/wp03-preparation` bắt đầu từ tag `wp02-baseline`
(đổi prefix từ `codex/` ngày 06/11/2026, giữ nguyên commit).
Xem [cách chạy và protocol](wp03_data.md) cùng [bằng chứng nghiệm thu](wp03_validation.md).

## Baseline đầu vào

WP01+WP02 đã nghiệm thu CPU, local CUDA và Colab A100. Config, seed, metadata,
run logging và RNG checkpoint sẵn để dùng lại. Tokenizer/model/training/generation
vẫn là placeholder. Tag WP02 giữ nguyên; thay đổi WP03 đi qua branch/commit riêng.

Scope khớp task
[khảo sát dữ liệu pretrain và benchmark toán; khóa train/validation/test](https://app.notion.com/4093badafe55819e9bd5ee16a5a0c500).
Task [BPE và round-trip tests](https://app.notion.com/4093badafe5581788c8dfe85c40cfb8b)
phụ thuộc corpus và nằm sau phần data này.

## Phạm vi đã chốt

1. Chọn corpus pretraining, ưu tiên **TinyStories** cho lần chạy đầu.
2. Xây bộ toán điều chỉnh được ba trục: số phép tính, số bước suy luận và độ dài đề.
3. Chuẩn hóa đáp án để so sánh tự động.
4. Chia train/validation/test, kiểm tra trùng lặp và lưu báo cáo.
5. Chuẩn bị **GSM8K làm benchmark bổ sung**.
6. Xác định các mức độ khó và số lượng câu hỏi cho pilot; tạo bộ pilot cố định.

## Sản phẩm và gate nghiệm thu

| Sản phẩm cần triển khai | Tiêu chí nghiệm thu |
| --- | --- |
| Dataset manifest | Source/revision, license, preprocessing, seed, split sizes và fingerprint/checksum được ghi rõ |
| Script tạo/tải dữ liệu | Tạo lại corpus/benchmark/split/pilot theo config, giữ stable IDs và lưu log/metadata bằng hạ tầng WP02 |
| Answer normalization/extraction | Có quy ước đáp án canonical và tests kiểm tra trường hợp hợp lệ, thiếu/sai định dạng, dấu và biểu diễn số được hỗ trợ |
| Split/dedup report | Không chồng lặp stable IDs hoặc câu hỏi đã chuẩn hóa giữa train/validation/test; ghi phương pháp và các trùng lặp phát hiện được |
| Pilot cố định | Mức khó, số câu, seed, stable question IDs, split và fingerprint được khóa trong manifest |
| GSM8K bổ sung | Source revision, cách dùng split và quy tắc answer extraction/evaluation được ghi rõ |

**Không dùng test set để chọn threshold.** Mọi threshold/calibrator chỉ chọn hoặc fit trên
validation; giữ test set cho đánh giá cuối. Gold label chỉ dùng training/evaluation,
không làm policy input. Không tuyên bố GSM8K là gate accuracy bắt buộc của model nhỏ.

## Các quyết định của WP03 v1

- TinyStories: một original Parquet train shard và official validation shard pin revision/SHA;
  10.000 train / 1.000 validation / 1.000 test sau xử lý trên source hiện tại.
- Bộ toán: ba mức operation count/dependency depth/prompt length = 1/1/short,
  3/2/medium, 5/3/long; có thể cấu hình từng trục độc lập trong giới hạn cây.
- Pilot: 50 câu/mức/split validation và test, tổng 300 câu; prompts và references tách riêng.
- Đáp án numeric-v1: canonical exact rational, invalid trả None, không float tolerance.
- GSM8K: official train carve 500 validation; official test 1.319 giữ held-out.

Mức khó là thiết kế cấu trúc, chưa phải độ khó nhận thức đã đo bằng model.
Budget/mức độ khó có thể đổi qua config; khi đổi phải tạo manifest/fingerprint mới.

## Các phần đã triển khai

- Config/data source lock trong `configs/data/`; manifest và validation record trong `experiments/`.
- `src/llm_scratch/data/config.py`, `datasets.py`, `difficulty.py`, `prepare.py` cho pipeline.
- `src/llm_scratch/reasoning/answer_extraction.py` cho đáp án chuẩn hóa.
- `scripts/prepare_data.py` / CLI `grast prepare-data` và tests data/difficulty/answer extraction.

Dataset/output lớn lưu ngoài Git, kèm manifest/đường dẫn/checksum.
Vocab/model size hiện vẫn là draft.
WP03 không bao gồm BPE, decoder, pretraining, CoT SFT, stopping policies hoặc memory RL.

## Bắt đầu từ baseline

```powershell
git switch feat/wp03-preparation
uv run --no-sync grast smoke --config configs/experiments/local_smoke.toml
```

Giữ tag baseline WP02 cố định. Nghiệm thu WP03 là data pipeline/pilot, không phải chất lượng model.
