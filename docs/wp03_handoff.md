# Chuẩn bị chuyển sang WP03

WP01+WP02 được chốt tại tag `wp02-baseline`. Evidence CPU, local CUDA và A100 đã đạt
nghiệm thu; config, seed, metadata, run logging và RNG checkpoint sẵn để dùng lại.
Tokenizer/model/training/generation chưa được triển khai. Nhánh chuẩn bị là
`codex/wp03-preparation`, bắt đầu từ baseline, chưa chứa implementation WP03.

## Phạm vi cần chốt

WBS hiện tại trong [Project HQ](https://app.notion.com/p/4093badafe55810db45bfcc26b766de9)
chưa gắn mã WP03. Task tiếp theo theo thứ tự WBS là
[khảo sát dữ liệu pretrain và benchmark toán; khóa train/validation/test](https://app.notion.com/4093badafe55819e9bd5ee16a5a0c500).
Task [BPE và round-trip tests](https://app.notion.com/4093badafe5581788c8dfe85c40cfb8b)
phụ thuộc corpus. Vì vậy đề xuất WP03 là data/benchmark/splits trước khi triển khai BPE.
Đây là đề xuất chuyển tiếp; không tự coi mã WP03 hoặc lựa chọn dataset đã được người dùng chốt.

## Checklist nếu WP03 theo task data trong WBS

1. Chọn corpus pretrain và benchmark toán có độ khó điều chỉnh; GSM8K là phần bổ sung.
   Ghi source, revision, license, ngôn ngữ, kích thước và giới hạn tài nguyên dự kiến.
2. Khóa train/validation/test bằng seed và stable IDs; ghi số lượng, fingerprint/hash,
   split rule và thao tác lọc/chuẩn hóa vào manifest versioned.
3. Kiểm tra trùng lặp/contamination giữa splits và với benchmark; lưu report và các
   trường hợp cần xử lý. Gold label chỉ dùng training/evaluation, không làm policy input.
4. Định nghĩa difficulty và metric cho feasibility gate trước khi đánh giá model.
   Không giả định model nhỏ giải tốt GSM8K.
5. Thêm config data và một lệnh kiểm tra tái lập khi triển khai; dùng run directory,
   metadata và config snapshot của WP02. Dataset/output lớn vẫn lưu ngoài Git.

## Gate nghiệm thu đề xuất

- Có dataset/benchmark manifest chứa source revision, split sizes và fingerprint.
- Cùng input/config/seed tạo lại đúng stable IDs và splits; khác split không chồng lặp.
- Có báo cáo chống leakage và protocol difficulty/feasibility rõ ràng.
- Có test cho hành vi split/normalization có rủi ro, không chỉ kiểm tra file tồn tại.
- Có commit, run logs và artifact link/checksum; cập nhật acceptance khi thực sự đạt gate.

Vocab/model size trong config hiện là draft; feasibility và training budget vẫn cần kiểm chứng.
WP03 chưa bao gồm decoder, pretraining, CoT SFT, stopping policies hoặc memory RL.

## Bắt đầu từ baseline

```powershell
git switch codex/wp03-preparation
uv run --no-sync grast smoke --config configs/experiments/local_smoke.toml
```

Giữ tag baseline cố định. Các thay đổi WP03 sau khi chốt scope đi qua commit/PR riêng.
