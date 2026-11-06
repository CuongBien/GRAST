# GitHub workflow

Remote đã cấu hình: [CuongBien/GRAST](https://github.com/CuongBien/GRAST).
Các file trong `.github/` là cấu hình versioned; settings phía GitHub không tự thay đổi
khi thêm file. Không có workflow train, deploy hoặc publish package ở bước 1.

## Baseline WP01+WP02

Tag `wp02-baseline` chốt skeleton, reproducibility utilities và evidence A100 đã audit.
Commit đầu tiên bootstrap `main`; annotated tag kèm kết quả smoke trên commit sạch.
Không commit raw ZIP/logs/checkpoints; manifest SHA-256 được versioned trong `experiments/`.

```powershell
git show wp02-baseline
git switch --detach wp02-baseline
```

Các work package sau dùng branch và PR từ `main` hoặc baseline đã nghiệm thu:

```powershell
git switch main
git pull --ff-only
git switch -c feat/ten-task
# Thay đổi và chạy checks trong README.
git add <file-da-review>
git commit -m "chore: describe the change"
git push -u origin feat/ten-task
```

Mở PR vào `main` trên GitHub và dùng template có sẵn. Notion vẫn là task tracker chính.

Từ 06/11/2026, mọi nhánh phát triển GRAST dùng prefix `feat/`. Ba nhánh `codex/wp03-preparation`,
`codex/wp04-tokenizer`, `codex/wp05-1-model-primitives` đã được đổi sang tên `feat/`
tương ứng ở local và remote; commit không đổi, upstream đã trỏ tới tên mới.
Nhánh cũ được gỡ sau khi xác minh SHA của tên mới, có lease chống xóa nhầm ref đã thay đổi.
`main` giữ nguyên theo yêu cầu người dùng; tag baseline không thay đổi.
Nhánh Dependabot tự động có PR riêng giữ tên do bot quản lý.
Evidence: `experiments/branch_prefix_migration_20261106.json`.

## CI hiện có

Trigger: push lên `main` hoặc các prefix branch trong `ci.yml`, PR vào `main`, và chạy tay
(`workflow_dispatch`, sau khi workflow có trên default branch).

Hai job matrix chạy Python 3.11 trên Ubuntu và Windows:

1. `uv sync --locked --extra cpu --extra data`.
2. Ruff lint và format check.
3. Package, reproducibility, data/answer-extraction, tokenizer và model-phase tests.
4. WP02 smoke, WP03 math-only smoke và các probe WP05.1/WP05.2/WP05.3 trên CPU;
   data sources dùng offline fixtures trong tests.
5. Build wheel và source distribution.
6. Cài wheel vào environment sạch và import ngoài source tree.

Token chỉ có `contents: read`; không yêu cầu repo secret. Action được pin bằng commit SHA,
Dependabot kiểm tra Actions hàng tuần. Tool uv pin 0.12.19; dependency resolve nằm trong `uv.lock`.
CI xanh kiểm tra package và các phase đã triển khai trên CPU; không xác nhận CUDA/A100
hoặc các phase model/training chưa được thực hiện.

Tham chiếu: [uv GitHub Actions integration](https://docs.astral.sh/uv/guides/integration/github/)
và [actions/checkout](https://github.com/actions/checkout).

## Settings nên bật trên GitHub sau bootstrap

Đây là hướng dẫn thao tác, chưa phải settings đã được áp dụng:

- Đặt default branch là `main`, bật Issues và Actions nếu đang tắt.
- Dùng squash merge và tự xóa branch đã merge.
- Tạo ruleset cho `main`: yêu cầu PR và status checks
  `checks (ubuntu-latest)` / `checks (windows-latest)` sau CI run đầu tiên.
- Với một người phát triển, không yêu cầu một approval từ người khác để tránh tự chặn merge.
- Chặn force push và xóa `main`; kiểm tra ruleset có được hỗ trợ bởi gói/visibility hiện tại.

Không có branch protection nào được kích hoạt chỉ bằng việc commit tài liệu này.
