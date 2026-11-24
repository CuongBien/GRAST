# WP07.6.2 — Curriculum sẵn sàng cho user train, ngày21/11/2026

**Cập nhật25/11:** đã nhận/verify user train đủ recipe; xem [WP07.6.3](wp07_6_3_validation.md).
Báo cáo dưới đây giữ evidence chuẩn bị/smoke ngày21/11, không phải kết quả train dài.

Đã có grouped dataset, ba stage tích lũy, chọn model theo đáp án tự sinh và notebook
Colab có log trực tiếp. **Mới verify execution ngắn; chưa chạy đủ2.800 updates hoặc
chứng minh curriculum cải thiện reasoning.** [Cách dùng/contracts](wp07_curriculum.md),
[evidence JSON](../experiments/wp07_6_2_validation_20261121.json).

## Phiên bản và dữ liệu

Nhánh `feat/wp07-6-2-curriculum-sft`. Notebook khóa code tại
`2383ab9a2900dc276eec7305a7e7d8cdfc5f2903`; bản notebook và smoke CUDA ở
`224c266f23977ea1e117dc4d636923beaea71b64`. Git sạch khi các lượt CUDA bắt đầu.

Dataset fingerprint
`0a891fda1dc3325a85f73a7022cef14fb6ce1b2d27467c1a7fbd42a7a0d16b56`:

| Task | Train | Validation |
| --- | ---: | ---: |
| Copy | 3.780 | 981 |
| Single | 6.196 | 1.580 |
| Multi | 4.000 | 800 |
| Tổng | 13.976 | 3.361 |

Max387/cap512, không truncate/drop. Train494.027 và validation112.788 target tokens.
Authored operands31..99 split theo unordered pairs cho copy/single, leaf multiset cho
multi. WP03 bridge giữ semantic splits gốc1.500/300, leaves1..30. Không đọc test/pilot.
Corpus kiểm hashes, AST, answer, reasoning, masks và groups; không chỉ đếm số rows.

Bundle `dist/grast-wp07-curriculum-0a891fda1dc3.zip`,3.059.243 bytes, SHA256
`02c9e6a9d969365921d00cc119e614295a25f1d1bde53fb7808141f2ffcee217`.
Gồm data/tokenizer, không weights. Input weights bắt buộc là SFT500 thật với SHA
`5f75909261c79e9f7f0c6e37a7e9d76ba5670be8b4be5133b6e148e0cd06ff42`.

## Kiểm chứng chức năng

Local full suite809 passed; sau bổ sung progress validation và xử lý stop notebook,
26 targeted curriculum/notebook tests passed. Tổng hiện tại811 test cases;
[CI candidate cuối](https://github.com/CuongBien/GRAST/actions/runs/44886344914)
xanh trên Windows và Ubuntu. Ruff lint/format, source diff checks và build đều qua.

Fixture tiny6.832 parameters/dropout0,1 đi đủ ba stage, hai updates mỗi stage; dừng
giữa stage ở step3 rồi resume từ **một file latest.pt được copy riêng**. So toàn bộ
model/optimizer/scheduler/scaler/RNG/cursors và metrics với lượt liên tục: exact.
Source checkpoint hoặc sibling best/phase checkpoints không cần cho continuation.
Best selection được test với synthetic predictions để xác nhận dùng macro generated EM
thay loss, tie giữ mốc sớm. Corrupted recipe/cursor/history/best weights đều bị từ chối.

Installed wheel chạy cùng acceptance ngoài source tree tại
`.cache/wp07-6-2-final-wheel-runs/437c20ea/`; có cả bundle verification và selected-model
reload. Wheel/sdist không chứa cache, runs, `.pt/.pth/.ckpt/.u32`. Notebook code cells
compile, hash/source pin khớp; stream in từng dòng stdout/stderr. Bấm stop gửi SIGINT
cho tiến trình con, chờ giới hạn rồi kill nếu chưa dừng; kiểm tra cả hai nhánh.

## Smoke model thật và resume

46.148.096 parameters, RTX4060 Laptop, Torch2.9.1+cu128/CUDA12.8, FP16 autocast với
master weights FP32, TF32off, deterministic, một CPU thread. Không dùng checkpoint overfit WP07.6.1.

- Lượt mới: `artifacts/runs/20261121T133617264503Z-wp07-6-2-curriculum-sft-seed42-c86c74d8`.
- Resume: `artifacts/runs/20261121T133826365321Z-wp07-6-2-curriculum-sft-seed42-24ec4995`.

Lượt đầu chạy full validation3.361 examples/112.788 targets, loss2,15650834. Sinh đủ48
câu validation đã khóa, đều EOS/format hợp lệ, final đúng0/48 ở **step0 của SFT gốc**.
Copy chưa tạo cặp số hợp lệ; không nhầm format compliance với đáp án đúng.

Sau đó train copy2 successful updates, copy latest sang thư mục riêng rồi resume tới
step4. Tổng4 attempts/4 updates,0 AMP skips,1.600 supervised targets. Train losses
5,52278 →5,36386 →4,85706 →4,60824. Các bước đầu, batch fingerprints và history giữ
nguyên sau load; optimizer/scaler/cursor tiếp tục đúng. Controlled stops2/4 không thêm
validation ngoài lịch, nên **không có phép đo quality sau bốn updates**. Best vẫn step0
theo dữ liệu đã đo, report ghi chưa có recorded generated-validation improvement.

Tái nạp `selected_model.pt` và sinh lại batch8 validation đầu: token IDs và finish
reasons chính xác với step0 predictions đã lưu. Source SFT SHA giữ nguyên. Đây là
spot-check inference8 câu và resume execution, không phải whole-GPU exact equivalence
giữa hai lượt corpus2800updates; toàn-state exact comparison thuộc tiny CPU fixture.

Train loss ở mỗi stage thuộc pool khác nhau, còn validation loss dùng toàn bộ fixed
validation. Không diễn giải khoảng cách hai đường thành generalization gap thuần túy;
generated per-task EM là tiêu chí selection. PNG/SVG đã kiểm tra hiển thị; smoke chỉ
có một điểm generated evaluation ở step0 là đúng lịch.

Một attempt trước đó bị mất execution session giữa initial validation, chưa có
optimizer update/checkpoint. Đã xác nhận không còn tiến trình nền, đánh dấu interrupted
và chạy lại hoàn tất. Không tính attempt đó là acceptance success.

## Bàn giao Colab

Mở `notebooks/wp07_curriculum_colab.ipynb`. Đặt bundle và SFT500 weights đổi tên
`sft500-final.pt` tại `MyDrive/GRAST/wp07/curriculum/inputs/`. GPU runtime, `STOP_AFTER=None`
chạy đủ400/1.200/1.200 updates. Notebook giữ PyTorch sẵn có, kiểm hash, chọn BF16 nếu
hỗ trợ hoặc FP16, stream log và lưu lên Drive. Khi resume giữ runtime/precision/source pin.

Sau train cần **selected_model.pt + summary ZIP** cho đánh giá tiếp. Summary không
chứa weights; `latest.pt` là file full resume. Không lấy model smoke step0 làm model
curriculum đã cải thiện. Cần nhận kết quả train dài rồi đo feasibility/pilot trên validation
trước khi triển khai confidence hoặc dùng test để báo cáo cuối.
