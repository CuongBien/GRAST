# WP06 — KV Cache & Autoregressive Inference theo phase

Mỗi phase có implementation, test, báo cáo và commit riêng. WP06 chỉ đạt nghiệm thu
toàn bộ sau cached/uncached equivalence và profiling thực đo ở phase cuối.

| Phase | Phạm vi | Gate | Trạng thái |
| --- | --- | --- | --- |
| WP06.1 | KV storage theo layer, append transaction, masks/positions, reset, bounds/payload | Copy/concat chính xác, không desync layer, rollback giữ prefix, dtype/device/limit đúng | Đã triển khai |
| WP06.2 | Tích hợp attention/model với prefill và token/chunk decode | Cached logits khớp full forward, RoPE offset/mask đúng, model ownership/context checks | Đã triển khai |
| WP06.3 | Greedy, temperature, top-k/top-p, EOS và seeded generation | Sampling/filtering đúng, generated token IDs/masks/context/EOS nhất quán | Đã triển khai |
| WP06.4 | Cached/uncached generation integration | Logits trong tolerance, greedy tokens giống nhau; sampling fixtures cùng RNG kiểm soát, padding/offset boundaries | Đã triển khai |
| WP06.5 | Online profiling tại prompt L=128/512/1024/2048 | TTFT, next-token latency, tokens/s và allocated/reserved/peak VRAM thực đo, protocol và environment đầy đủ | Chưa triển khai |

Đã hoàn thành WP06.1–WP06.4: cached model forward, sampling và hai generation paths
đã đối chiếu logits/token IDs trên fixed fixtures. Không dùng cache payload hoặc
offline replay làm bằng chứng latency/VRAM; profiling thực đo thuộc WP06.5.

## Các quyết định tích hợp đã chốt

- MHA baseline có `num_kv_heads = num_heads`, layout K/V `[B,H,capacity,D]`.
- Key lưu **sau RoPE**, value không xoay; phase attention không được xoay lại cached K.
- Cache theo request/batch, fixed B/device/dtype; không tự chuyển dtype/device, evict,
  sliding window, reorder beam hoặc gộp request. Dùng tuần tự trên cùng execution stream;
  chưa có đồng bộ đa luồng/đa stream.
- Physical cached length bao gồm padding columns. Causal order dựa trên physical token
  order; validity mask và RoPE positions là metadata riêng, không đánh đồng số valid tokens
  với chiều dài cache.
- Default positions khi append là `[committed_length, ..., new_end-1]`. Explicit positions
  được lưu nguyên sau chuyển integer sang int64; caller phải tiếp tục cung cấp khi dùng
  quy ước tùy biến. Foundation không đoán next position từ một history tùy biến.
- Phase 2 đã ràng buộc cache với model/weights phù hợp, eval/inference policy và metadata
  của request qua `CausalLM(..., use_cache=True, cache=...)`.

## WP06.1 — static cache và transaction

```python
import torch
from llm_scratch.model.cache import KVCache

cache = KVCache(model.config, batch_size=2, device="cuda", dtype=torch.float32)
with cache.begin_step(num_tokens, attention_mask=new_mask, position_ids=new_positions) as step:
    for layer_index in range(model.config.num_layers):
        # k_new đã qua RoPE, v_new không xoay, không require gradients.
        k_full, v_full = step.update(layer_index, k_new[layer_index], v_new[layer_index])
        # Dùng k_full/v_full và step.attention_mask cho attention ở layer hiện tại.
    # Entire model/head phải thành công trước khi thoát with và commit.
```

Đây là ví dụ contract storage, chưa phải cached model forward. `step.start/end` là offset
vật lý; `step.position_ids` là positions của chunk mới, `query_mask` là valid mask chunk
mới, `attention_mask` là mask toàn prefix + chunk. Các views mask/positions chỉ đọc khi active;
`start/end` là offsets bất biến của transaction.

Buffers được cấp trước theo capacity, không `cat` để realloc toàn prefix mỗi token.
Một shared `sequence_length` chỉ tăng khi mọi layer đã được ghi đúng một lần.
`get_layer(i)` và cache-level masks/positions chỉ expose committed prefix;
`step.update` trả view gồm cả tail vừa ghi của layer đó để attention có thể tính ngay.
Layer có thể được ghi theo thứ tự bất kỳ; duplicate/out-of-range/missing layers bị từ chối.

Exception hoặc incomplete transaction trong `with` abort metadata commit, giữ nguyên
prefix trước đó. Tail chưa commit có thể còn bytes cũ, nhưng không được expose qua API
committed. Có manual `commit/abort`; explicit commit là điểm commit, không rollback
sau đó. Không mở append lồng nhau hoặc reset khi có active transaction.

`reset()` đưa logical length về 0 và giữ buffers để reuse; không scrub toàn bộ bytes.
Views trả về là borrowed views: caller không sửa và không giữ chúng qua reset/reuse.
K/V đầu vào được copy, nên sửa source tensors sau update không sửa cache.
Không hỗ trợ serialization cache vào model checkpoint hoặc chuyển cache sang request khác.

`max_length` mặc định bằng model context; có thể chọn nhỏ hơn, không lớn hơn.
Append vượt capacity bị từ chối trước ghi. Explicit RoPE IDs phải nằm trong
`[0, model.context_length)`, có thể lớn hơn physical cache capacity; chúng không tăng
capacity. Mask là bool/integer 0/1 `[B,new_tokens]`, positions là int32/int64
`[new_tokens]` hoặc `[B,new_tokens]`, cùng device. K/V phải đúng shape/dtype/device;
không cho broadcasting hoặc implicit cast/transfer dù `Tensor.copy_` vốn hỗ trợ chúng.

Cache không giữ autograd graph và từ chối K/V require gradients. Buffers được tạo thành
ordinary no-grad tensors kể cả khi constructor chạy trong `torch.inference_mode()`,
nên có thể reuse an toàn trong các scope inference/no_grad khác nhau.
Tham chiếu API: [PyTorch inference mode](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.grad_mode.inference_mode.html),
[Tensor.copy_](https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.copy_.html).
Foundation lưu nguyên K/V values; phase attention chịu trách nhiệm projection/padding masking.

## Đếm bộ nhớ và ranh giới profiling

`memory_bytes()` báo **tensor payload**:

- K/V capacity: `2 * layers * B * H * capacity * D * dtype_bytes`.
- K/V used: thay capacity bằng committed physical length.
- Metadata capacity/used: `B * length * (1 byte mask + 8 bytes position)`.

Đếm payload không phải CUDA allocator allocated/reserved/peak hoặc VRAM đo thực.
Reset làm used bytes về 0, capacity vẫn giữ. PyTorch allocator có thể giữ reserved memory
khi Python objects được giải phóng; không suy hiệu suất hoặc VRAM peak từ công thức này.

## Chạy lại phase 1

```powershell
uv run --no-sync pytest tests/test_kv_cache.py -q
uv run --no-sync python scripts/check_kv_cache.py --device cpu
uv run --no-sync python scripts/check_kv_cache.py --device cuda
```

Probe dùng geometry candidate và K đã qua RoPE để kiểm tra round-trip theo layer,
append 4+1+1, prefix sau abort/bounds, formula và reuse storage. Nó không chạy model
forward hoặc đo tốc độ inference. Có `--dtype`, `--max-length`, `--output-root`.
Kết quả: [WP06.1 validation](wp06_1_validation.md).

## WP06.2 — cached model forward

```python
model.eval()  # Khởi tạo/load/chuyển device model bên ngoài inference_mode.
with torch.inference_mode():
    prefill = model(prompt_ids, prompt_mask, use_cache=True)
    cache = prefill.cache
    decoded = model(next_ids, new_token_mask, use_cache=True, cache=cache)
    next_logits = decoded.logits  # [B, new_tokens, vocab_size]
```

`cache=None` khi `use_cache=True` tự tạo cache theo config/model device/effective KV dtype.
Hoặc `model.new_cache(B, max_length=...)` cấp trước capacity nhỏ hơn model context.
Prefill là append đầu tiên; các call sau chỉ nhận token/chunk mới. Mask là **mask của
chunk mới** `[B,new_tokens]`, không phải full-prefix mask. Không truyền thì chunk mới
đều valid; past mask/positions đã được lưu trong cache. Không sửa prefix IDs/masks
đã cache; muốn thay đổi phải reset và prefill lại.

Output thêm `.cache`; output `.logits` chỉ gồm chunk hiện tại. Các API cũ không bật
cache vẫn trả loss/logits và chạy forward/backward như WP05. Cached path yêu cầu toàn
model ở eval, grad disabled (`no_grad` hoặc `inference_mode`) và không nhận labels.
Truyền cache mà không bật `use_cache`, hoặc sai batch/config/device/dtype bị từ chối.

RoPE chỉ áp dụng lên Q/K mới tại `step.position_ids`, mặc định offset bằng physical
committed length. Past K không xoay lần nữa. Causal mask là hình chữ nhật
`key_index <= cache_start + query_index`, kết hợp past/new key validity và query validity.
Điều này khác `tril()` không offset cho một chunk decode. V mới được append nguyên,
padding được clear trong model trước projection. Fully padded rows và prefix toàn
padding theo cùng quy tắc finite-zero của WP05.

Explicit positions được so với uncached full position history trong tests. Sau khi
caller truyền explicit positions, mọi append sau phải truyền rõ positions đến khi reset;
không tự đoán tiếp từ vị trí cuối hoặc số valid tokens. Causal order vẫn là physical order,
kể cả khi positions đảo thứ tự. Tổng physical cached length và mọi RoPE index phải
trong các bounds đã cấu hình; không tự kéo dài context hoặc evict.

Một transaction bao cả blocks, final norm và LM head. Head lỗi hoặc model đổi giữa
forward sẽ abort tail, không commit partial cache. Raw storage append làm mất model binding,
nên không thể thêm K/V ngoài model rồi tiếp tục inference như prefix hợp lệ.

Cache giữ weak reference đến model và signature gồm config, tokenizer metadata,
parameter/buffer identity/version/storage/type/device và module metadata (RoPE/RMSNorm/mode).
Root `to/half/float` hoặc load_state_dict tăng revision để bắt cả dtype round-trip;
optimizer/in-place updates bình thường cũng làm cache stale. Request khác dùng cache riêng,
có thể interleave trên cùng model/stream; model không có global past state.
Reset xóa binding và cho phép prefill request mới. Không ghi trực tiếp vào `.data`, private
storage hoặc borrowed views để bypass version counters; cơ chế này không phải integrity hash
cho tensor contents tùy ý. Model phải được tạo ngoài inference_mode để parameters có version counters.

Effective KV dtype theo [autocast](https://docs.pytorch.org/docs/2.9/amp.html), giữ FP64
khi model là FP64. Cache phải được tạo trong cùng precision policy với decode;
đổi autocast/dtype giữa request bị từ chối, cần reset/tạo cache phù hợp. Không có implicit KV casts.
Tolerances kiểm tra: FP64 rtol/atol 1e-10/1e-12; FP32 1e-4/1e-5;
FP16 0,005/0,002; BF16 0,02/0,01. Test có greedy-independent teacher-forced prefixes,
chưa kết luận sampling/generation equivalence.

```powershell
uv run --no-sync pytest tests/test_cached_forward.py -q
uv run --no-sync python scripts/check_cached_forward.py --device cpu
uv run --no-sync python scripts/check_cached_forward.py --device cuda --checkpoint PATH/TO/overfit.pt --tokenizer PATH/TO/artifacts/tokenizer
```

Probe dùng full candidate. Mặc định seeded synthetic IDs và random weights cho CI;
optional checkpoint WP02 atomic format + tokenizer WP04 dùng fixture pretraining/SFT tự viết.
Checkpoint load với `weights_only=True`, CPU mapping/mmap và strict model metadata;
không restore optimizer/RNG cho inference probe. Report ghi checkpoint SHA-256, codec
fingerprint, chunk errors và cache payload. Chưa đo TTFT/latency/VRAM ở phase này.
Kết quả: [WP06.2 validation](wp06_2_validation.md).

## WP06.3 — sampling và uncached generation

```python
from llm_scratch.inference.config import GenerationConfig
from llm_scratch.inference.generate import generate

model.eval()
policy = GenerationConfig(
    max_new_tokens=16, do_sample=True, temperature=0.8, top_k=40, top_p=0.95,
    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
)
output = generate(model, prompt_ids, prompt_mask, config=policy, seed=42)
tokens = output.generated_ids[0, :output.generated_lengths[0]].tolist()
text = tokenizer.decode(tokens)
```

WP06.3 tạo uncached full-prefix baseline; WP06.4 thêm `use_cache=True` bên dưới.
Không thay mode của model;
caller phải `eval()`, hàm tự disable gradients. Prompt IDs là int32/int64 `[B,P]`
trên model device, mask optional binary `[B,P]`, mỗi hàng phải có ít nhất một valid
token. Output IDs là int64, masks bool. Không thêm BOS hoặc tokenize tự động:
ordinary text dùng codec WP04; reasoning dùng `encode_reasoning_prompt` cùng SFT framing.
Positions theo physical columns; WP06.4 thêm scalar `position_offset`, chưa nhận
arbitrary per-row custom position histories ở generation.

Initial logits đọc tại **last valid prompt token của từng hàng**, hỗ trợ left/right
padding và interior gaps. Generated tokens được append sau physical prompt width;
không compact prompt. Mọi padding column vẫn chiếm context/RoPE index. Yêu cầu
`P + max_new_tokens <= context_length` được kiểm tra trước forward, kể cả nếu EOS
có thể đến sớm. Budget 0 hợp lệ, trả suffix rỗng và không gọi model. Không truncate,
evict hoặc tự giảm budget. Một forward full prefix cho mỗi bước sinh token.

EOS chỉ xét trên token **mới sinh**, không dừng vì EOS có sẵn trong prompt. EOS được
giữ và tính vào `generated_lengths`. Hàng đã EOS không được sampling nữa; các bước
còn lại trong batch append PAD với mask 0. Dừng khi tất cả hàng EOS hoặc hết budget;
EOS tại bước cuối vẫn có reason `eos`. `eos_token_id=None` chỉ dừng theo budget.
PAD có thể trùng EOS. Không tự suppress PAD/BOS/reasoning tokens: nếu model sinh PAD
trong một hàng active, token đó vẫn có mask 1 trừ khi cũng là EOS. Dùng mask để phân
biệt generated PAD và padding sau khi hàng đã dừng.

Output có `sequences`, `attention_mask`, `generated_ids`, `generated_mask`,
`generated_lengths`, per-row `finish_reasons` (`eos`/`max_new_tokens`) và seed.
Suffix tensor dài bằng số bước batch đã chạy, bao gồm padding sau EOS. Decode/evaluate
từng hàng bằng valid generated length, không tính tail padding là output.

Sampling contract:

- Greedy là `do_sample=False`, raw argmax; tie lấy ID nhỏ nhất, không dùng temperature/
  filters hoặc tiêu thụ RNG. Settings vẫn được validate. Không dùng temperature=0
  làm sentinel; mọi temperature phải finite >0.
- Stochastic: temperature → exact top-k → top-p trên phân phối đã renormalize sau top-k.
  `top_k=0` tắt filter, `top_k=1` chỉ giữ argmax; top-k không được vượt vocab.
  Stable sort giữ thứ tự ID tăng khi score bằng nhau, nên ties không làm vượt k.
- Nucleus giữ prefix nhỏ nhất có cumulative probability >= top_p, gồm token đầu tiên
  vượt threshold. Nếu đã đạt đúng threshold thì không thêm token kế. `top_p=1` tắt
  filter; `0 < top_p <= 1`. Luôn giữ ít nhất một finite candidate.
- Filtering trả FP32 (FP64 nếu logits FP64), giữ inputs nguyên, dùng -inf cho candidates
  bị loại. Chấp nhận -inf trong input nếu còn finite candidate; NaN/+inf hoặc all--inf
  row bị từ chối. Extreme finite temperatures được xử lý tránh scalar cast overflow.
- `sample_next_token` yêu cầu explicit same-device `torch.Generator` ở stochastic mode.
  `generate` tự tạo generator riêng theo device/seed; không advance global CPU/CUDA RNG.
  Chỉ active rows theo batch order được sampling, không draw cho finished rows.
  Cùng seed tái lập trong cùng model, precision, device/runtime và batch schedule;
  không hứa sample IDs giống CPU/GPU hoặc giữa batch và từng request riêng.

Implementation dùng stable [sort](https://docs.pytorch.org/docs/2.9/generated/torch.sort.html)
và explicit generator của [multinomial](https://docs.pytorch.org/docs/2.9/generated/torch.multinomial.html).
Không dùng thư viện generation bên ngoài PyTorch.

Runtime policy ở `configs/sampling/wp06_generation.toml`, schema riêng `GenerationConfig`.
`k_values` trong WP02 grid là **số candidate sequences**, không được map vào `top_k`.
Probe mặc định chạy candidate random weights + fixed synthetic IDs không tải external
data. Optional checkpoint WP05 + tokenizer WP04 dùng story/reasoning prefixes tự viết,
lưu generated IDs/masks/decoded text, policy snapshot/hash, checkpoint hash và environment.
Không restore training optimizer/RNG. Đây là diagnostic, chưa phải quality benchmark.

```powershell
uv run --no-sync python scripts/check_generation.py --device cpu
uv run --no-sync python scripts/check_generation.py --device cuda --checkpoint PATH/TO/overfit.pt --tokenizer PATH/TO/artifacts/tokenizer
```

Kết quả baseline: [WP06.3 validation](wp06_3_validation.md).

## WP06.4 — cached generation và equivalence

```python
cached = generate(model, prompt_ids, prompt_mask, config=policy, seed=42, use_cache=True)
reference = generate(model, prompt_ids, prompt_mask, config=policy, seed=42, use_cache=False)
```

`use_cache` phải là bool, mặc định False để giữ baseline WP06.3. Cached call cấp cache
riêng theo B/model/device/effective KV dtype, capacity `P + max_new_tokens`. Prefill
chạy toàn prompt và chọn last-valid readout của từng hàng; sau đó mỗi forward chỉ
nhận một token mới `[B,1]` và mask `[B,1]`. Offset mặc định dùng committed physical
length của KV cache. Khác nhau về số valid prompt tokens không được làm mất padding
columns khỏi cache hoặc thay RoPE convention giữa hai paths.

Một forward sinh một token; khi đã dừng thì không forward token vừa sinh thêm lần
nữa. Nếu chạy S bước thì cache chỉ chứa P+S-1 physical columns, trong khi output có
P+S columns. Đây là request-local cache, không expose trong output hoặc cho phép
external continuation. Call mới luôn prefill/cache mới, kể cả cùng model và seed.
Budget 0 không allocate cache/forward; EOS tại bước đầu không có decode thừa.

EOS/PAD/sampling contract giữ như WP06.3. EOS đã sinh có mask 1 khi được đưa vào
forward kế; padding sau khi hàng dừng có mask 0. Finished rows không được sampling
hoặc reactivated, dù cache vẫn append physical columns cho batch cố định. Logits
của các rows đó có thể khác reference (zero masked query thay vì EOS readout cũ),
nên chỉ **active next-token readouts** được dùng để nghiệm thu logits.

Optional `position_offset` là scalar integer >=0 dùng cho cả hai paths. RoPE index
ở physical column i là `position_offset+i`; prefill và từng append truyền explicit
positions nhất quán khi offset khác 0. Không lấy offset theo số valid tokens, không
đoán lịch sử positions tùy biến. Kiểm tra trước cả allocation/forward:
`position_offset + P + max_new_tokens <= context_length`. Offset không tăng physical
cache length hoặc đổi capacity. Baseline context vẫn 1024; không tự mở rộng context.

`research.generation_equivalence.check_generation_equivalence` chạy generation thật
hai lần với cùng settings/seed, dùng temporary
[forward hooks](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook)
để thu readout logits từng bước và input lengths; hook được remove trong finally.
Đối chiếu toàn sequences, generated IDs/masks, lengths, finish reasons chính xác;
active logits theo tolerance FP64 1e-10/1e-12, FP32 1e-4/1e-5,
FP16 0,005/0,002, BF16 0,02/0,01 (rtol/atol). Kiểm tra forward lengths thật:
cached `[P,1,...,1]`, uncached `[P,P+1,...]`; lưu per-step errors và reference top-2 gaps.

Greedy tokens và seeded stochastic fixtures đạt exact equality trong kiểm thử.
Numerically close logits không tự bảo đảm token equality với mọi near-tie, nucleus
boundary hoặc random draw; không round/ép logits, chỉnh seed hoặc nới tolerance để
che divergence. Fixture thất bại thì checker raise và run lưu failed. Kết quả chỉ
áp dụng cho model/precision/device/runtime/batch schedule đã ghi, không hứa sampling
IDs giống nhau giữa CPU/GPU hoặc arbitrary requests.

```powershell
uv run --no-sync python scripts/check_generation_equivalence.py --device cpu
uv run --no-sync python scripts/check_generation_equivalence.py --device cuda --checkpoint PATH/TO/overfit.pt --tokenizer PATH/TO/artifacts/tokenizer
uv run --no-sync python scripts/check_generation_equivalence.py --device cuda --position-offset 7 --checkpoint PATH/TO/overfit.pt --tokenizer PATH/TO/artifacts/tokenizer
```

Probe dùng shared setup/policy với WP06.3, chạy greedy và sampled paths, rồi replay
cached call cùng seed. Ghi report `artifacts/generation_equivalence_report.json`,
policy/config snapshots, environment, source/checkpoint/tokenizer hashes và log.
CI dùng random candidate weights/synthetic prompts không tải external data; local
dùng checkpoint WP05/tokenizer WP04 trên authored diagnostic prefixes, không dùng test
set hoặc tuyên bố benchmark quality. Trace copy logits về CPU có synchronization:
**không dùng acceptance probe/hook làm profiling**. Kết quả:
[WP06.4 validation](wp06_4_validation.md). WP06.5 mới đo TTFT/latency/throughput/VRAM.

## Protocol profiling cần hoàn tất ở phase 5

L là số prompt tokens hợp lệ; cần định nghĩa rõ số new tokens cố định, B, dtype,
warm-up/repeats, seed, checkpoint, backend và phạm vi timing. Tách TTFT (prefill +
chọn token đầu), latency các decode sau và throughput steady-state; đồng bộ CUDA
ở biên đo, lưu raw samples, không thay latency bằng replay cost proxy.
Tokenizer/host transfers sẽ được ghi rõ là có/không nằm trong timing.

Baseline WP05 có context 1024. Với L=2048 và N new tokens, profiling cần inference
config riêng đủ **L+N** capacity/context (không chỉ 2048). Phase 2/5 phải có override
context/checkpoint metadata minh bạch; không sửa baseline và không tuyên bố chất lượng
ở context dài đã được train. Ghi cả prompt length, cache capacity, used payload và
CUDA allocated/reserved/peak thực đo. Chưa có cơ chế/profiling này ở phase 1.
