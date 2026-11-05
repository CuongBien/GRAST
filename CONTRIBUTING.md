# Đóng góp

Notion HQ là nguồn task và acceptance. GitHub lưu code, review và bằng chứng thực thi.
Gắn link task Notion trong issue/PR thay vì tạo hai bảng tiến độ độc lập.

## Quy trình

1. Chọn một task có phạm vi nhỏ, ghi tiêu chí nghiệm thu và dependency.
2. Tạo branch từ `main`, ví dụ `codex/repository-skeleton`, `feat/tokenizer` hoặc
   `fix/causal-mask`. Không tạo branch giai đoạn 2 để làm việc ngoài scope hiện tại.
3. Thay đổi code/docs và chạy các lệnh kiểm tra trong README.
4. Commit có mục đích rõ ràng, ví dụ `chore: scaffold repository and CI`.
5. Mở PR vào `main`, điền template, đính kèm test/log/config/artifact phù hợp.
6. Chờ CI và review; ưu tiên squash merge, xóa branch đã merge.
7. Chỉ cập nhật Done trong Notion khi acceptance đủ bằng chứng.

Repo trống cần bootstrap commit đầu tiên trước khi có base để mở PR.
Xem [GitHub workflow](docs/github_workflow.md) để biết thao tác bootstrap và settings.

## Dữ liệu và thực nghiệm

- Không commit dataset, checkpoint, token, `.env` hay output lớn.
- Lưu artifact ngoài Git; manifest ghi vị trí, config, seed, commit và checksum khi có.
- Khi thêm dependency: cập nhật `pyproject.toml`, chạy `uv lock`, commit cả `uv.lock`.
  Sync với `--extra cpu` hoặc `--extra cuda`, thêm `--extra data` cho WP03;
  chạy `uv run --no-sync ...` để giữ runtime đó.
- Thêm behavioral tests cùng implementation; không dùng test rỗng hoặc skip để tuyên bố đã pass.
- WP02 có PyTorch/NumPy cho reproducibility probes; chưa implement model hoặc tokenizer.
