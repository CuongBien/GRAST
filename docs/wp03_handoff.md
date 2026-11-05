# WP03 — Dataset & Benchmark Design

Scope do người dùng chốt ngày 05/11/2026. **Trạng thái: chỉ chuẩn bị; chưa triển khai WP03.**
Nhánh `codex/wp03-preparation` bắt đầu từ tag `wp02-baseline`. Chưa tải/sinh dữ liệu,
chưa tạo dataset manifest hoặc pilot, chưa viết script data hay answer extraction.

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

## Các quyết định sẽ chốt khi bắt đầu triển khai

- TinyStories: source/revision cụ thể, subset/budget, preprocessing và split policy.
- Bộ toán: templates, phép tính hỗ trợ, định nghĩa từng mức trên ba trục độ khó.
- Pilot: số mức, số câu mỗi mức và phân bố validation/test; chưa gán số lượng mặc định.
- Quy ước đáp án: kiểu số hỗ trợ, canonical form, xử lý invalid/missing và comparison rule.
- GSM8K: revision, sample budget và protocol benchmark bổ sung.

Các giá trị này chưa được chọn hoặc tải về trong bước chuẩn bị.

## Vị trí code dự kiến khi thực hiện WP03

- Config data trong `configs/data/`; manifest versioned và artifact links trong `experiments/`.
- `src/llm_scratch/data/datasets.py`, `difficulty.py` cho data/split/difficulty.
- `src/llm_scratch/reasoning/answer_extraction.py` cho đáp án chuẩn hóa.
- Script chuẩn bị dữ liệu và tests split/dedup/answer extraction sẽ thêm khi triển khai.

Đây là danh sách nơi sẽ sửa, không phải các implementation đã có. Dataset/output lớn
lưu ngoài Git, kèm đường dẫn và checksum. Vocab/model size hiện vẫn là draft.
WP03 không bao gồm BPE, decoder, pretraining, CoT SFT, stopping policies hoặc memory RL.

## Bắt đầu từ baseline

```powershell
git switch codex/wp03-preparation
uv run --no-sync grast smoke --config configs/experiments/local_smoke.toml
```

Chỉ bắt đầu implementation khi người dùng yêu cầu thực hiện WP03. Giữ tag baseline cố định.
