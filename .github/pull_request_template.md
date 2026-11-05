## Thay đổi và mục đích

<!-- Mô tả hành vi trước/sau. Liên kết issue hoặc task Notion. -->

## Phạm vi nghiên cứu

- Giai đoạn: <!-- 1: đồ án / 2: NCKH sau 01/12 -->
- Bằng chứng: <!-- test, log, config, commit, artifact hoặc N/A kèm lý do -->

## Kiểm tra

- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check .`
- [ ] `uv run --no-sync pytest` (environment đã sync với extra `cpu` hoặc `cuda`)
- [ ] Nếu thay đổi reproducibility: `uv run --no-sync grast smoke --config configs/experiments/local_smoke.toml`
- [ ] `uv build`
- [ ] Không commit dữ liệu, checkpoint hay thông tin bí mật.
- [ ] Nếu có kết quả nghiên cứu: phân biệt replay với online measurement; gold label không vào policy.

## Hạn chế hoặc việc còn lại

<!-- Không đánh dấu task Done nếu acceptance chưa đủ bằng chứng. -->
