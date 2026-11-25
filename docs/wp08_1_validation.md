# WP08.1 — Sample-bank foundation, 25/11/2026

**Đạt gate WP08.1.** Có builder/reader, ordered candidates, per-token raw-model
log-probability, seed/config/checkpoint/source fingerprints, external scorer và
immutable policy observations. Chưa chạy full feasibility pilot, offline stopping
replay hoặc quyết định benchmark cho toàn WP08.

Implementation commit: `94c31685191bf9d261341c42e2fe1b052f1bcdd5`, branch
`feat/wp08-1-sample-bank-foundation`. Hai lượt source smoke dưới đây đều ghi Git sạch.
Hướng dẫn/API: [WP08 contracts](wp08_sample_bank.md).
Receipt có file checksums: [evidence](../experiments/wp08_1_validation_20261125.json).

## Kết quả kiểm tra

| Kiểm tra | Kết quả |
| --- | --- |
| Full pytest local, gồm CPU và CUDA khi áp dụng | **879 passed**, 177,69s; 31 test mới |
| Ruff lint / formatting / diff whitespace | Đạt |
| Build wheel + sdist offline | Đạt; không chứa cache, runs, weights hoặc token packs |
| Installed wheel ngoài source tree | Smoke CPU đạt, 16 records/bank, hai bank exact |
| Source CPU với hai checkpoint tiny | 16 records/bank, 8 groups, 127 generated tokens/bank; hai bank exact |
| Selected2200 thật, RTX4060 Laptop, FP32 | 4 records/bank, 2 groups, 84 generated tokens/bank; hai bank exact |
| Tối đa16 ứng viên và partial final batch | Đạt bằng kiểm thử fixture |
| Checksum/grid/order/seed/aggregate corruption, partial bank | Bị từ chối như thiết kế |
| References hoặc scored files trong bank | Bị từ chối; scorer ghi ngoài bank |
| Đổi gold bên ngoài | Chỉ đổi nhãn scorer; bank fingerprint/observations không đổi |

CPU/source run:
`artifacts/runs/20261125T112528490948Z-wp08-1-sample-bank-smoke-seed42-7845d3a7`.
Fingerprint hai bank:
`66090c6ff68a0d6e8a8452f8b6d5e82424c8c4c79986ba817ddf96f5361b6f07`.

CUDA/source run:
`artifacts/runs/20261125T112513286713Z-wp08-1-sample-bank-smoke-seed42-a2e29ffc`.
Fingerprint hai bank:
`cbfd00679745fe7f3b8d99884afd95c4d41928a223f98d0071fa706697baa98f`.

Mỗi run có config snapshot, metadata, environment report, logs, hai bank hoàn chỉnh
và external scoring. Fixture CPU/wheel tạo checkpoint mới mỗi lần chạy; full checkpoint
digest chứa cả history/elapsed metadata, nên không so fingerprint giữa hai fixture run.
Cam kết ở đây là lặp lại cùng checkpoint/config/runtime/batch layout đã khóa.

## Kiểm tra điểm số trên checkpoint thật

Model46.148.096 tham số, tokenizer WP04 V8192, Torch2.9.1+cu128/CUDA12.8,
RTX4060 Laptop. Đối chiếu bổ sung batch2, budget8, temperature0,8/top-k40/top-p0,95,
seed42, có12 token hợp lệ và4 vị trí tail đã mask:

- Bật/tắt ghi điểm: generated tokens trùng hoàn toàn.
- Cached/uncached: tokens và masks trùng; max absolute log-prob error
  **2,9206275939941406e-6**.
- So với full-sequence teacher forcing, chỉ tính vị trí hợp lệ: max absolute error
  **2,9206275939941406e-6**, dưới `atol=rtol=5e-5` đã dùng khi đối chiếu.
- Trạng thái RNG global CPU/CUDA không đổi. Generator lấy mẫu là private.

Raw result nằm trong `artifacts/real_logprob_equivalence.json` của CUDA run.
Các kiểm thử độc lập trên model nhỏ cũng đối chiếu teacher forcing, RoPE offset,
EOS/tail, sampled PAD, zero budget và trường hợp top-k1 có sampling probability1
nhưng raw-model log-prob vẫn âm. Điểm được ghi từ logits gốc trước sampling filters.

SHA checkpoint selected2200 vẫn là
`d2ea45abc8f81b516552c184d75c595b95fa2bd1abdc260538e497da4b627dc8`.
SFT500 cũng giữ nguyên SHA gốc. Không train lại hoặc sửa weights thật.

## Phạm vi nghiệm thu

Hai câu kỹ thuật không đại diện cho difficulty distribution hoặc reasoning accuracy.
Không mở held-out test thật, không chọn threshold, không fit confidence/calibration,
không đo online latency/VRAM của policy bằng replay. Gold chỉ được đọc bởi scorer
sau khi cả hai bank đã commit; các policy observations không chứa gold/correctness.

Bước tiếp theo là **WP08.2**: khóa pilot validation nhỏ theo độ khó và loại selection
overlap trước khi xem kết quả; sinh tối đa16 ứng viên theo nhiều trial seed/checkpoint.
Dùng bank Kmax đã lưu để so sánh về sau; không sinh lại mỗi K hoặc coi candidate0
là greedy. WP08.3 mới triển khai replay/stopping, WP08.4 mới chốt feasibility/GSM8K.
