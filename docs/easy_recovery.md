# One bounded easy SFT recovery round — trước WP09

Quyết định ngày **27/11/2026**: chạy đúng một vòng easy arithmetic từ selected2200,
user train đầy đủ trên Colab; agent chỉ verify ngắn. WP08 đã đóng về kỹ thuật,
gate năng lực còn yếu. Vòng này không được kéo dài tự động nếu chưa cải thiện;
kết thúc rồi chuyển WP09 với phạm vi chẩn đoán phù hợp kết quả.

## Chạy trên Colab

Mở `notebooks/easy_recovery_colab.ipynb`, chọn GPU runtime. Upload hai input vào
`MyDrive/GRAST/easy_recovery/inputs/`:

- `dist/grast-easy-recovery-kit-20261127.zip`: wheel, recipe, dữ liệu và tokenizer,
  có checksum cố định trong notebook. Không chứa weights; không cần tải repo từ nhánh đang đổi.
- `artifacts/checkpoints/wp07-curriculum-2800-selected-2200/selected_model.pt`:
  giữ tên `selected_model.pt`, SHA256
  `d2ea45abc8f81b516552c184d75c595b95fa2bd1abdc260538e497da4b627dc8`.

Chạy với `SMOKE_ONLY=True` trước. Smoke dừng ở update2 và không chấm gate.
Sau khi smoke qua, đổi `SMOKE_ONLY=False`, chạy lại cell cấu hình và các cell phía dưới.
Full run bắt đầu mới từ selected2200, không nối optimizer từ smoke. Mọi subprocess
in stdout/stderr trực tiếp; trainer in từng update, validation, generation batch và checkpoint.
Notebook giữ PyTorch có sẵn, ghi Python/PyTorch/CUDA/GPU/precision thực tế.
T4 dùng FP16; GPU hỗ trợ BF16 dùng BF16.

Sau khi ngắt kết nối, đặt `RESUME_CHECKPOINT` bằng đường dẫn `checkpoints/latest.pt`
được in trong log và `SMOKE_ONLY=False`. Giữ cùng kit, GPU, phiên bản runtime và precision.
Runner khôi phục model/AdamW/scheduler/scaler/RNG/cursor/history/best weights từ một
full checkpoint; không phụ thuộc các sibling checkpoints của run cũ. Checkpoint nguồn
vẫn cần để xác minh lineage và chấm baseline sau cùng. Inference export không resume optimizer.

Run và checkpoint nằm trên Drive dưới `GRAST/easy_recovery/runs/`;
smoke nằm trong `smoke_runs/`. Notebook tự tải summary ZIP và tải model riêng sau full run.
Gửi lại **summary ZIP + selected_model.pt** để kiểm tra kết quả và chuẩn bị sample bank WP09.

## Protocol đã khóa trước full training

| Thành phần | Giá trị |
| --- | --- |
| Source | selected2200, weights only; optimizer/scheduler/scaler/RNG/cursor mới |
| Model/tokenizer | Giữ nguyên46.148.096 parameters / byte BPE8192 |
| Budget | 1.000 successful updates; không tự thêm vòng |
| Training | AdamW LR1e-4, weight decay0,01, warmup50, cosine min ratio0,1, clip1 |
| Effective batch | 8 examples ×4 accumulation =32; không packing/truncation |
| Context | Dữ liệu tối đa128 tokens; kiến trúc giữ context1024 |
| Loss | Chỉ reasoning/final/EOS; prompt và padding mask |
| Train | 6.196 single-operation examples: WP03 500 + authored5.696, chỉ +/− |
| Sampling | WP03 lặp tham chiếu12 lần trong virtual epoch;6.000 WP03 +5.696 authored |
| Selection | 16 validation questions:4 WP03 +12 authored; step0, mỗi100 và step1000 |
| Generation | Greedy cached, FP32, batch8, tối đa64 new tokens, seed42 |
| Selection rule | Max macro verified EM theo2 nguồn, rồi macro final EM; earliest tie |
| Gate cuối | 48 câu khác:4 WP03 +44 authored; chấm baseline và selected sau khi khóa winner |
| Full checkpoints | latest mỗi100 attempts; giữ milestone500/1000; selected export riêng |

Virtual repeats tăng trọng số WP03, không tăng số câu độc lập. TrainingCursor dùng
private CPU permutation và checkpoint lưu cửa sổ đã tiêu thụ, kể cả AMP skip.
Step0 được chọn nếu các checkpoint sau không cải thiện. Không bắt buộc xuất một model
đã thay đổi weights chỉ để gọi vòng train thành công.

Verified EM yêu cầu final đúng, format hợp lệ và rationale khớp `math_reasoning`
sau chuẩn hóa khoảng trắng. Đây là kiểm tra trace chuẩn của bộ toán tổng hợp,
không phải judge reasoning tổng quát. Loss thấp không thay thế generated accuracy.

Gate năng lực: **từng nguồn** phải đạt final EM≥30% và verified EM≥25%.
Với WP03 chỉ4 câu, gate này rất thô, phải đọc kèm counts. Screen outcome hỗn hợp:
ít nhất15 đúng và15 sai trong48 câu. Hai screen cùng đạt mới gợi ý dùng easy pilot
cho WP09; đây là screen kỹ thuật, không chứng minh đủ cỡ mẫu để calibration.
Nếu gần như đúng hết, cần thiết kế độ khó phù hợp trước calibration; không tự gọi model yếu.
Không đạt thì vẫn chuyển WP09 với mục tiêu chẩn đoán thất bại, không tune thêm vòng theo gate.

## Partition và giới hạn kết luận

Chỉ đọc **train/validation** từ curriculum đã có. Giữ nguyên membership nguồn;
không đọc test và không fit confidence threshold trong vòng này.
Nhóm chống trùng lặp là cặp số không thứ tự, dùng chung cho +/− và thứ tự toán hạng.
Loại khỏi candidate evaluation mọi cặp đã xuất hiện:

1. Trong tất cả curriculum train, bao gồm copy và toán hạng đã evaluate tại **mọi node**
   của cây multi-step, không chỉ leaf operands.
2. Trong48 câu generated selection của checkpoint nguồn hoặc30 câu WP08.2 pilot,
   cũng xét mọi node của các câu đó.

Loại5.998 cặp; còn11 cặp WP03 và64 authored phù hợp. Seed20261127 và ID hash
chọn một câu/cặp, chia selection và gate trước khi generation. Manifest lưu IDs,
pair groups, counts theo nguồn/phép tính và hashes. Train/selection/gate không trùng
ID, prompt hoặc cặp toán hạng. Gate không vào training loss hay checkpoint selection.
Chỉ projection question-only được truyền vào generator; đáp án chỉ dùng scoring sau generation.

Đây là **development validation**: dữ liệu đã từng dùng để đo validation loss;
không khẳng định chưa từng có exposure qua pretraining hoặc các phép chẩn đoán trước.
Gate48 nghiêng authored, WP03 chỉ4 câu, không đại diện full WP03/GSM8K.
Không trình bày score này thành test accuracy hoặc dùng gate/test chọn threshold WP09.
WP09 sẽ cần bank mới và tách calibration/evaluation theo question/group, giữ cùng
question trong một partition ở mọi seed và checkpoint.

Dataset fingerprint:
`bdb0503b93a3f184acb1d0a71dda69bd9227c0ddf57bed189f0911ed2e9eed17`.
Manifest prospective được lưu tại `experiments/easy_recovery_protocol_20261127.json`.
Xem [evidence local và bộ cài đã verify](easy_recovery_validation.md).

## Chuẩn bị lại / local smoke

```powershell
python -u -m llm_scratch.easy_recovery.data `
  --config configs/data/easy_recovery.json `
  --source artifacts/wp07-6-2/curriculum_data `
  --tokenizer artifacts/runs/20261105T154939888754Z-wp04-byte-bpe-seed42-4102afe0/artifacts/tokenizer `
  --checkpoint artifacts/checkpoints/wp07-curriculum-2800-selected-2200/selected_model.pt `
  --prior-questions artifacts/runs/20261125T120930824002Z-wp08-2-feasibility-pilot-seed42-333b2fdc/artifacts/bank/questions.jsonl `
  --output artifacts/easy-recovery-v1/data

python -u -m llm_scratch.easy_recovery.runner `
  --config configs/training/easy_recovery.toml --prepared-root artifacts/easy-recovery-v1/data `
  --tokenizer artifacts/runs/20261105T154939888754Z-wp04-byte-bpe-seed42-4102afe0/artifacts/tokenizer `
  --source-checkpoint artifacts/checkpoints/wp07-curriculum-2800-selected-2200/selected_model.pt `
  --device cuda --precision float16 --stop-after 2
```

`--stop-after` là successful step tuyệt đối, không phải số updates cộng thêm khi resume.
CPU fixture trong `tests/test_easy_recovery.py` chạy liền/stop/resume để so sánh exact
model/optimizer/RNG/cursor/history/best state; tests còn kiểm tra sai reasoning,
checkpoint bị sửa, thiếu gate trong smoke, kit checksum/path safety và gold-free generation.
Chạy thật trên local chỉ2 updates, không chứng minh improvement. Kết quả full Colab còn chờ user.
