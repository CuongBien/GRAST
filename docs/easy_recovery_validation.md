# Easy recovery — sẵn sàng chạy Colab, 27/11/2026

Đã chuẩn bị runner, protocol, data kit và notebook cho **một vòng SFT easy1.000 updates**
trước WP09. Full Colab training **chưa chạy**; chưa chấm gate48 và chưa kết luận improvement.
Xem [cách chạy/protocol](easy_recovery.md).

Code runner được chốt tại `1ff2079f5833345b10ad5f852e3a297f17d072ab` trên
`feat/easy-recovery-colab`; main và tag baseline giữ nguyên.
Notebook cài wheel trong kit có checksum, không lấy source từ branch đang thay đổi.

| Kiểm tra | Kết quả |
| --- | --- |
| Full pytest trước khóa kit | 932 passed,1 deferred checksum-pin test,193s |
| Final affected tests + notebook | 15 passed,10,01s; bao gồm test pin đã hoàn tất |
| Ruff lint/format | Đạt |
| Build wheel/sdist | Đạt; không chứa weights/cache/local runs |
| Wheel ngoài source tree | Tiny fixture full vs stop/resume exact; selected reload; reject sửa best weights |
| Milestone checkpoint | Vẫn lưu khi không trùng kỳ checkpoint theo attempt |
| Notebook | Compile, input hash pin, live output, nhận đúng run dir, cleanup khi interrupt |
| Precision | T4 chọn FP16 kể cả có BF16 emulation; A100/BF16 native được hỗ trợ |

Sau kiểm thử toàn repo, đã sửa điều kiện lưu milestone khi chu kỳ successful updates
không trùng attempts và chạy lại toàn bộ tests liên quan cùng notebook. Không dùng
pytest CPU hoặc local smoke làm bằng chứng rằng model đã đạt gate năng lực.

Local smoke cuối chạy từ **installed wheel ngoài source tree** trên RTX4060 Laptop8GB,
Python3.11.16, PyTorch2.9.1+cu128/CUDA12.8, FP16:

- Model46.148.096 parameters; đủ2 successful updates,0 AMP skips,1.522 supervised tokens.
- Optimizer counters mới đều ở step2; cursor tiêu thụ2 windows, scheduler2 updates;
  weights đã cập nhật, không phải chỉ nạp/export lại checkpoint.
- Chọn checkpoint trên16 câu ở step0:4 final/trace đúng; chưa có generated selection
  ở step2 vì full protocol chỉ đánh giá mỗi100 updates. Best giữ source step0 là đúng quy tắc.
- Đã lưu log, config/metadata, full `latest.pt`, inference export và loss curves.
- Không mở gate payload trong controlled stop. Gold không xuất hiện trong projection
  truyền cho generator. Checkpoint/kit luôn tách weights khỏi gói dữ liệu.

Run: `artifacts/easy-recovery-v1/verified-wheel-runs/20261126T173443990460Z-easy-recovery-v1-seed42-736bb541`.
Timestamp thư mục là UTC; ngày báo cáo theo Asia/Bangkok là27/11/2026.
Git dirty trong metadata do notebook/test handoff còn untracked lúc smoke;
mọi executable package file đã được đối chiếu byte với wheel đóng gói.

Kit: `dist/grast-easy-recovery-kit-20261127.zip` (1.326.953 bytes), SHA256
`c8b15e13ab936af43a7efc649fedc1191f210fa4d9a097c95bc63bb68ab606cf`.
Dataset fingerprint và IDs đã khóa trong
`experiments/easy_recovery_protocol_20261127.json`; receipt/hash của bằng chứng trong
`experiments/easy_recovery_validation_20261127.json`.

Việc còn lại: user chạy notebook trên Colab, gửi summary ZIP + selected model.
Đọc gate theo từng nguồn và counts, rồi chuẩn bị bank mới cho WP09; không tự mở thêm vòng train.
