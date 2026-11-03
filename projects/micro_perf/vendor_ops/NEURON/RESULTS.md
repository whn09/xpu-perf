# NEURON results

How to run is in [README.md](README.md); how to read these numbers is in
[READING_THE_NUMBERS.md](READING_THE_NUMBERS.md).

## Reference numbers

### Re-validated on Neuron SDK 2.33 (official DLAMI)

The tables below were measured on the beta image. On 2026-11-03 every provider was
re-checked for *numerical correctness* — not just that it runs — on a trn2.48xlarge
from the official Neuron 2.33 multi-framework DLAMI (`aws_neuronx_venv_torch_neuronx_2_14_0`:
torch 2.14.0, torch-neuronx 2.14.3, neuronx-cc 2.28.7631, nki 0.7.0, runtime-lib
2.35.30, collectives 2.35.43, driver 2.31.7 — exactly the SDK 2.33 release content),
with `tools/verify_providers.py`. It builds each provider through the real backend,
compares the device output to an fp32 host reference on the same inputs, and passes
a case only if its error is within 3x of the same computation done on the host in
the case's own dtype. 25 cases across `gemm`, `rms_norm`, `topk` and all three
`flash_attention` providers; collectives were smoke-run (`all_reduce`, `all_gather`,
4 ranks), not numerically checked.

What changed:

- **eager fp8 `gemm` returned wrong products** — rel err 0.65 (e5m2) / 0.68
  (e4m3fn) against an fp32 matmul of the same fp8 values, where rounding the exact
  product to fp8 costs 0.03-0.05. Inputs round-trip bit-exactly and an on-device cast
  to bf16 before the matmul is correct, so it is the eager fp8 matmul itself. The
  `torch` provider now rejects fp8; **discard earlier eager-fp8 rows** (the beta was
  only ever timed, so whether it was already wrong there is unknown). The compiled
  path (`torch_compile`, `XPU_PERF_NEURON_GEMM_COMPILE=1`) is correct: rel 1.7e-3.
- **`tfloat32` `gemm` is now measured.** It was rejected as "an NVIDIA format", but
  TF32 is a Tensor Engine datatype on Trn2/Trn3 at the bf16 rate. It is reached with
  neuronx-cc's `--auto-cast=matmult --auto-cast-type=tf32` (not with
  `torch.set_float32_matmul_precision`, which does nothing here): 106.8 TFLOPS at
  4096³ against 35.4 for fp32, 131.6 TFLOPS (79% of peak) at 32768x1024x8192, rel err
  1.5e-4. The flag is scoped to one `torch.compile` graph per case, because the eager
  graph cache is keyed by shape and dtype but not by compiler flags — exporting it
  would silently make the fp32 cases of the same shapes TF32 too (measured).
- **`rms_norm`/`nkilib` no longer runs token counts that do not split across the
  LNC programs** (`num_tokens: 1`): neuronx-cc 2.28 rejects a 1-program grid at
  LNC=2 with `NCC_EVRF066`. The `torch` provider covers those rows.
- Everything else matches to rounding (bf16 rel 1.66e-3 = the host bf16 floor;
  topk values and indices exact; attention within the host floor).

Trainium3 (`trn3.48xlarge`) is now in the MFU peak table (671 bf16/tf32, 183 fp32,
2517 fp8/MX TFLOPS per chip). **Nothing in this repo has run on Trn3 yet** — run
`tools/verify_providers.py` there before trusting the first sweep.

### Beta-image measurements

All on trn2.3xlarge, one logical NeuronCore (a trn2 chip is 4 logical cores at
the default LNC=2). Read them with [Reading the numbers](READING_THE_NUMBERS.md#reading-the-numbers) —
several are runtime overhead rather than hardware results, and a few of the
shipped workloads measure a reference simulation rather than a kernel.

### Eager runtime

2026-09-01, PyTorch-native image: torch 2.12.1, torch-neuronx 2.12.3.0.1636,
neuronx-cc 2.27.2878.0, nki 0.6.0, host driver 2.30.2.0. The `xla` column is the
table further down, from a second trn2.3xlarge.

| Op | Dtype | Shape | Latency | Metric | MFU | `xla` latency |
|---|---|---|---|---|---|---|
| gemm | fp32 | 1024x4096x4096 | 990.9 us | 34.7 TFLOPS | 77% | 1,462.2 us |
| gemm | fp16 | 1024x4096x4096 | 305.2 us | 112.6 TFLOPS | 68% | 727.9 us |
| gemm | bf16 | 1024x4096x4096 | 276.8 us | 124.1 TFLOPS | 74% | 758.5 us |
| add | bf16 | 1024x1024 | 45.4 us | 138.7 GB/s | ~0% | 711.2 us |
| softmax | bf16 | 1024x1024 | 51.5 us | 81.4 GB/s | ~0% | 612.2 us |
| flash_attention | bf16 | prefill q_len=2048, 8 heads, dim 128 | 539.3 us | 15.9 TFLOPS | 10% | 1,658.6 us (NKI) |
| all_reduce | bf16 | 1024x1024, ws=2 | 105.3 us | 19.9 GB/s bus | n/a | 659.5 us |
| all_gather | bf16 | 1024x1024, ws=2 | 91.1 us | 11.5 GB/s bus | n/a | 1,108.3 us |

Reading the eager-vs-XLA gap: **gemm is the only honest comparison**, and bf16 at
2.7x is large enough to be real (though the two stacks also have different
compiler versions, 2.27 vs 2.23). The small ops are launch-bound on both, so
49 us vs 711 us is graph-cutting cost, not memory bandwidth. `flash_attention` is
native SDPA against a NKI kernel — two implementations, not two runtimes.

**The host driver version did not matter.** The same sweep on a host whose driver
was 2.x.8955.0 against the image's expected 2.30.2.0 — which makes the runtime
log `nrta_tensor_read/write` warnings and fall back to synchronous tensor IO —
agreed to within noise (gemm bf16 279.4 us, add 48.8 us, all_reduce 105.0 us).
That warning looks alarming and is easy to mistake for the cause of a slow run.

### Eager runtime, full sweep

Same host and image, sweeping every workload file in the repo: ~2,995 cases
measured, ~1,100 rejected before reaching the device, two workloads cut short by
their watchdog. Per-workload accounting is in
[Reading the numbers](READING_THE_NUMBERS.md#reading-the-numbers);
`workloads/llm/single_test_ops/ccl_ops.json` is the only file with no runnable
case at all (it asks for `world_size: 8`).

#### Compute-bound peaks

Best case per dtype, over all 208 `gemm` shapes in
`workloads/basic/tensor_gemm_ops/gemm.json`:

| Op | Dtype | Best shape (MxKxN) | Latency | TFLOPS | MFU |
|---|---|---|---|---|---|
| `gemm` | bf16 | 12288x8192x8192 | 11,038.1 us | **149.42** | **90%** |
| `gemm` | fp16 | 32768x1024x8192 | 3,755.8 us | **146.38** | **88%** |
| `gemm` | fp32 | 3968x1024x8192 | 1,802.7 us | **36.93** | **82%** |
| `gemm` | fp8 e5m2 | 16384x8192x8192 | 460,699.0 us | 4.77 | 1.5% |
| `gemm` | fp8 e4m3 | 16384x8192x8192 | 570,968.6 us | 3.85 | 1.2% |
| `moe_gating_gemm` | fp32 | 8192 tokens, 8192 hidden, 128 experts | 784.9 us | 21.89 | 48% |
| `quant_matmul` | "int8" | 73728x8192x8192 | 627,062.4 us | 15.78 | n/a |

The MFU column for the three float `gemm` rows is computed here rather than read
out of the log: that sweep predates `peak_tflops`/`mfu` reaching the basic-gemm
path, so its result blocks carry no `mfu` key (a run today does emit it, and the
fp8 rows are read straight out of it). The denominators are the per-core peaks
from [MFU](READING_THE_NUMBERS.md#mfu) — 166.75 for bf16/fp16, 45.25 for fp32, 324.75 for fp8.

**The two fp8 rows are in the table for completeness and should not be read as
fp8 hardware numbers** — they are the eager path, in an `e4m3fn` that this
chip's Tensor Engine cannot multiply (OCP fp8 matmul starts at NeuronCore-v4).
Compiled, in `e5m2`, the same core does **245.50 TFLOPS at 75.6% of its fp8 peak**. See
[fp8: the sweep measures the eager path, in a dtype this chip cannot multiply](READING_THE_NUMBERS.md#fp8-the-sweep-measures-the-eager-path-in-a-dtype-this-chip-cannot-multiply).

**90% of the dense bf16 peak on one logical core is the headline number for this
chip**, and it is 20 points above the 1024x4096x4096 row in the table above: that
shape is too small to hide the ~60 us dispatch floor, and it is the shape a quick
smoke test uses. `quant_matmul` is not an int8 hardware number on any backend —
see [the quantized ops are a bf16 simulation](READING_THE_NUMBERS.md#the-quantized-ops-are-a-bf16-simulation-on-every-backend).

#### Four-core scaling: what x4 is actually worth

Every per-chip figure in this document and in
[`../GPU/README.md`](../GPU/README.md) is a one-logical-core number times four.
That is load-bearing enough to check per shape rather than at one point, so
`gemm.json` and the reduction/selection files were run twice — once on `--device 0`
and once on `--device 0,1,2,3`, where four *different* cases are in flight at once
and each case's latency is therefore a single-core latency measured with the other
three cores competing for the same HBM — and joined shape by shape.

| Family | Shapes joined | Median 4c/1c latency | p90 | Worst |
|---|---|---|---|---|
| `gemm` bf16 | 208 | **1.004** | 1.170 | 9.011 |
| `gemm` fp16 | 208 | **1.011** | 1.686 | 7.474 |
| `gemm` fp32 | 208 | **1.011** | 1.087 | 5.165 |
| `gemm` fp8 e4m3 | 12 | 1.464 | 1.723 | 1.775 |
| `gemm` fp8 e5m2 | 12 | 1.525 | 2.309 | 2.531 |
| `reduce_max` / `reduce_min` / `reduce_sum` | 33 each | **1.004-1.025** | 1.761-5.256 | 8.332 |
| `topk` | 147 | **1.002** | 1.829 | 11.020 |
| `moe_softmax_topk` | 56 | **1.029** | 4.020 | 8.892 |

**At the plateau, x4 is exact.** The single-core bf16 peak of 150.21 TFLOPS at
30720x8192x8192 becomes 149.59 TFLOPS per core with all four loaded — 0.4% — so
598.4 TFLOPS per chip is a measurement, not an extrapolation. `topk` at its peak
shape scales **4.01x** and `moe_softmax_topk` **4.00x**, which is what retires the
"extrapolated" caveat those two rows used to carry. Memory-bound elementwise work
holds 594-616 GB/s per core, ~2.4 TB/s of the chip's 2.9 TB/s.

`reduce_sum` scales too — 632.73 GB/s per core becomes 634.00 with four loaded, so
**2,536 GB/s per chip**, 87% of the 2.9 TB/s the chip has and within 1.18x of an
H100's measured 2,985.

This was extended to all 64 logical cores of a trn2.48xlarge — same method, one
worker per core — and the plateau holds there too: see
[64 cores at once](#64-cores-at-once).

**Two exceptions.** The first is dtype-specific: fp8 `gemm` (medians 1.464 and
1.525) and fp32 `reduce_sum` (median 1.369) are the only groups whose *median* does
not scale, where every other dtype of every other op sits between 1.000 and 1.025.
For fp8 that is consistent with it timing a software up-cast rather than the matmul
engine — and the compiled path, which *does* reach the engines, scales cleanly
instead: 1.018x worst case over four cores
([details](READING_THE_NUMBERS.md#fp8-the-sweep-measures-the-eager-path-in-a-dtype-this-chip-cannot-multiply)). fp32 `reduce_sum`
is a different thing: its *largest* shape scales perfectly (0.998, the 2,536 GB/s
above), and it is the mid-sized shapes that contend — the same tail described next,
reaching further up the size range for this one op than for any other.

The second is shape: the smallest shape in every family degrades badly under
concurrency, well beyond what the median suggests.

| Worst case | 1 core | 4 cores, per case | Ratio |
|---|---|---|---|
| `gemm` bf16 M=2 K=1024 N=8192 | 70.2 us | 632.3 us | **9.01x** |
| `gemm` fp16 M=4 K=4096 N=4096 | 85.3 us | 637.4 us | 7.47x |
| `topk` bf16 `batch_size` 1024 `dim_size` 128 `k` 8 | 58.4 us | 644.1 us | **11.02x** |
| `moe_softmax_topk` fp32 `num_tokens` 1, 8 experts | 132.9 us | 1,182.1 us | **8.89x** |
| `reduce_max` fp32 1024x2048 | 82.9 us | 690.9 us | 8.33x |

These are all runs dominated by dispatch and synchronisation rather than by
arithmetic — the same ~60 us floor described in
[small-op latencies](READING_THE_NUMBERS.md#small-op-latencies-measure-the-runtime-not-the-chip) — so
they contend on something the large shapes never touch. The practical rule: **do
not multiply a small-shape single-core number by four.** Every per-chip figure
quoted here is from a plateau shape.

One number moved between runs and is worth recording rather than hiding: this
dedicated single-core `gemm.json` run peaked at **37.67 TFLOPS (83.3% MFU)** in
fp32, at 10240x1024x8192, against the 36.93 at 3968x1024x8192 in the table above
from the full sweep. Both are single-core measurements of the same file; the newer
one is what `../GPU/README.md` compares against the H100's 48.72 TFLOPS, since it
is the run that also has the four-core join behind it.

#### Memory-bound ops

A quarter of the chip's 2.9 TB/s is ~725 GB/s, so that is the ceiling to read
these against. Best `mem_bw` per op, with the dtype that reached it:

| Op | fp32 | fp16 | bf16 |
|---|---|---|---|
| `device2device` (a local copy — [see below](#device2device-is-not-a-device-to-device-transfer)) | — | — | **648.7** |
| `reduce_sum` | **639.0** | 314.9 | 315.4 |
| `index_select` | 630.9 | — | **631.6** |
| `embedding` | 629.2 | — | **631.5** |
| `cast` | **616.7** | 594.0 | 593.3 |
| `sub` | **608.6** | 600.3 | 600.0 |
| `add` | 600.6 | **601.9** | 599.9 |
| `mul` | 599.2 | 599.7 | **599.9** |
| `silu` | **587.2** | 466.7 | 471.0 |
| `exp` | 578.8 | 557.8 | **566.0** |
| `div` | 379.0 | **579.9** | 573.7 |
| `log` | 559.1 | **564.6** | 559.4 |
| `sqrt` | 557.9 | **564.5** | 533.6 |
| `softmax` | **495.8** | 291.4 | 295.7 |
| `topk` | **476.3** | 239.7 | 239.4 |
| `rms_norm` | **409.2** | 322.6 | 322.6 |
| `layer_norm` | **259.3** | 172.2 | 172.7 |
| `reduce_max` / `reduce_min` | **176.6 / 177.0** | 97.9 / 97.7 | 97.1 / 97.2 |
| `sin` | **118.9** | 41.8 | 30.6 |
| `cos` | **110.6** | 41.0 | 29.7 |
| `index_add` | **98.6** | 86.1 | 86.3 |
| `gelu` | **83.7** | 42.9 | 42.4 |
| `rotary_embedding` | — | — | 42.8 |
| `moe_gather` | — | — | 9.2 |
| `gather` | **1.4** | — | 0.7 |
| `scatter` | **0.8** | — | — |
| `moe_scatter_dynamic_quant` | — | — | 0.7 |

Five things in that table are lowering quality rather than hardware:

- **`gelu` is 7-14x slower than `silu`** over identical shapes (83.7 vs 587.2 at
  fp32, 42.4 vs 471.0 at bf16). Both are one elementwise pass over one tensor, so
  a 14x gap is the `erf`-based expansion, not bandwidth. If a model can use
  `silu`, that is 14x here.
- **`sin` / `cos` at bf16 are 4x slower than at fp32** (30 GB/s vs 119) — the only
  ops in the sweep that get *worse* with a narrower dtype. This used to carry "and
  the reason `rotary_embedding` lands at 42.8 GB/s", which was wrong and is
  corrected [below](#rotary_embedding-is-not-a-neuron-result-at-all): `cos` and
  `sin` are precomputed when the tensors are created, so no trig runs inside that
  op's timed region at all.
- **`reduce_max`/`reduce_min` at 177 GB/s against `reduce_sum` at 639** is a 3.6x
  gap between three reductions over the same shapes.
- **`div` at fp32 is 379 GB/s where `mul` is 599**, but at fp16/bf16 both are
  ~580-600. The fp32 divider is the outlier, not division as such.
- **`rms_norm` and `layer_norm` report byte-identical fp16 and bf16 latencies**
  (1665.2 us and 778.0/780.2 us): the compiler is emitting the same fp32-accumulate
  kernel for both, so their bf16 rows are not measuring bf16 arithmetic.

`gather` and `scatter` are a different phenomenon again, and `gather`'s is not a
property of the hardware or even of the op — an int32 index instead of an int64 one
makes it 291-731x faster and lands it within 8% of `index_select`. See
[Two index ops are pathologically slow, and one is an int64 index away from
parity](READING_THE_NUMBERS.md#two-index-ops-are-pathologically-slow-and-one-is-an-int64-index-away-from-parity).

**A `gemm` narrow enough to be memory-bound reaches 500-573 GB/s** (M=128,
K=N=8192: 573.3 fp16 / 559.2 bf16 / 499.8 fp32), i.e. 77-79% of the per-core
ceiling. That is the cleanest bandwidth number in the sweep after
`device2device`, because it comes from an op whose operands are read exactly once.

##### `rotary_embedding` is not a Neuron result at all

Its 42.8 GB/s is 5.90% of one core's 725 GB/s ceiling, which reads like the worst
lowering in the table after the index ops. It is not a lowering result. The same
op on an H100 lands at **5.85-5.97% of its own 3.35 TB/s**:

| Case | Trn2 1 core | % of 725 | H100 | % of 3,350 |
|---|---|---|---|---|
| prefill `q_len` 10240 | 42.77 GB/s | **5.90%** | 197.35 GB/s | **5.89%** |
| prefill `cache_len` 5120 + `q_len` 5120 | 42.32 | 5.84% | 195.98 | 5.85% |
| prefill `q_len` 32768 | 42.22 | 5.82% | 199.95 | 5.97% |
| decode `batch_size` 16, `q_len` 1 | 0.99 | 0.14% | 3.80 | 0.11% |
| decode `batch_size` 16, `q_len` 4 | 1.01 | 0.14% | 4.85 | 0.14% |

Two backends whose bandwidths differ by 4.62x giving the same fraction of peak to
one significant figure means the cost is in the op def, not in either stack. Per
chip it is 171.2 against 197.4 GB/s, **1.15x** — exactly the 1.155x memory-bound
bar.

Timing the body piece by piece on one core at `q_len` 10240 (220 MB written; one
full pass at 725 GB/s would be 318 us) says where it goes:

| Step | us | Share | GB/s over the slice |
|---|---|---|---|
| `.contiguous()` on the strided slice | 825.7 | 7.5% | 558.8 |
| **`rotate()`** | **8,591.3** | **78.1%** | 53.7 |
| `copy_` back into the strided slice view | 1,697.5 | 15.4% | 271.8 |
| — same bytes into a contiguous dst (control) | 815.1 | — | 566.0 |
| one elementwise pass, reference (`mul`) | 1,087.4 | — | 424.3 |
| whole body | 10,995.6 | 100% | 42.0 |

`rotate()` is 78% of it, and it is five-plus materialising passes where a fused
kernel is one:

```python
def rotate(qk, cos, sin):                       # core/utils.py:587
    left_part, right_part = qk[:, :, :rope_dim//2], qk[:, :, rope_dim//2:]
    return torch.cat([left_part, right_part], -1) * cos.unsqueeze(1) + \
           torch.cat([-right_part, left_part], -1) * sin.unsqueeze(1)
```

Two `torch.cat` each write a fresh 220 MB, then two multiplies and an add. At the
424 GB/s a single `mul` pass gets, five passes is ~5.4 ms; the measured 8.6 ms says
the concatenations cost more than a plain pass, which is unsurprising for a
last-dim concat with a negation. An H100 pays the same passes, which is why the
percentages match.

The remaining 15% is [the in-place
problem](READING_THE_NUMBERS.md#writes-into-a-strided-slice-view-are-not-in-place), and it is real but
secondary: `copy_` into the strided destination is 1,697.5 us against 815.1 for
the same bytes into a contiguous one, **2.08x**. `packed_qkv` has 96 heads and the
op writes heads 0:88, so the destination view is not contiguous; sizing the buffer
to exactly 88 heads makes the identical write contiguous and confirms the cause
(1,711.6 us strided vs 836.4 contiguous, 2.05x).

The decode rows are a third thing again and are not bandwidth at all:
`vendor_impl_run` loops over batches in Python, so `batch_size` 16 with `q_len` 1
is 16 dispatches writing one token each — 377 us per iteration on a 393 KB tensor.
The H100 is 3.8-4.8x faster there, i.e. at parity per chip, because both are
paying for the loop rather than for memory.

#### Norm, activation and MoE gating ops

2026-09-02, same instance and image. These twelve ops had an `op_defs`
implementation but no workload JSON anywhere in the repo, so nothing had ever
measured them on any backend; `workloads/llm/single_test_ops/norm_ops.json`,
`activation_ops.json`, `moe_gating_ops.json` and `quant_ops.json` now do. Best
`mem_bw`, against the same ~725 GB/s per-core ceiling:

| Op | fp32 | fp16 | bf16 | Peak at |
|---|---|---|---|---|
| `head_rms_norm` | **429.9** | — | 95.7 | 1024 tokens, 96 heads, 80 normalised |
| `qk_rms_norm` | **374.1** | — | 90.8 | 4096 tokens, 80 q / 8 kv heads |
| `swiglu` | **312.1** | 291.1 | 292.1 | 4096 tokens, hidden 4096-8192 |
| `moe_swiglu` | — | — | **306.2** | 10240 tokens, 128 experts, topk 8, ep 8 |
| `add_rms_norm` | — | — | **269.6** | 4096 tokens, hidden 8192 |
| `moe_softmax_topk` | **74.4** | — | — | 32768 tokens, 256 experts, post-softmax |
| `add_rms_norm_dynamic_quant` | — | — | **2.4** | bf16 -> int8 |
| `moe_swiglu_dynamic_quant` | — | — | **2.2** | bf16 -> int8 |
| `swiglu_dynamic_quant` | — | — | **1.6** | bf16 -> int8 |
| `quant_group_gemm_reduce_sum` | — | — | **1.5** | int8 -> bf16, 2.61 "TFLOPS" |
| `head_rms_norm_dynamic_quant` | — | — | **1.1** | bf16 -> int8 |
| `scale_dynamic_quant` | — | — | **0.7** | bf16 -> int8 |

Dashes are dtypes the base op def refuses, not dtypes that failed:
`add_rms_norm`, `moe_swiglu` and every `_dynamic_quant` op accept bfloat16 only,
and `moe_softmax_topk` float32 only. Four things worth reading out of that table:

- **`head_rms_norm` and `qk_rms_norm` are 4.1-4.5x faster at fp32 than at bf16.**
  This is the `sin`/`cos` anomaly again, but much larger and on ops that matter
  more: 429.9 vs 95.7 and 374.1 vs 90.8. `swiglu` over the same token counts shows
  no such gap (312.1 fp32 vs 292.1 bf16), so it is not a general bf16 penalty. The
  two slow ops are the two that normalise a *slice of heads* out of a wider
  `[tokens, total_head_num, head_dim]` tensor, i.e. the ones whose input is a
  strided view.
- **Every `*_dynamic_quant` op is 87-183x slower than its unquantised sibling**,
  on identical shapes, and all six are one unfused helper — see
  [the `*_dynamic_quant` family](READING_THE_NUMBERS.md#the-_dynamic_quant-family-is-90-180x-off-and-it-is-one-shared-helper).
- **Bandwidth peaks around 1024-4096 tokens and then falls back 20-40%.**
  `head_rms_norm` at fp32 goes 429.9 (1024 tokens) -> 394.8 (4096) -> 326.9
  (16384) -> 221.3 (32768); `add_rms_norm` goes 269.6 -> 201.8 -> 175.9. Every norm
  and activation op does this. A sweep that only measures one large batch will
  understate this hardware by a third.
- **`moe_softmax_topk` is launch-bound, not bandwidth-bound.** Latency is flat at
  120-270 us from 1 token to 16384 across all four expert counts, and only starts
  to climb past that. `post-softmax` is consistently 1.5-1.9x cheaper than
  `pre-softmax` (426.2 vs 649.0 us at 32768 tokens / 128 experts), which is
  structural rather than a lowering artifact: `pre-softmax` softmaxes all `E`
  experts, takes topk, then renormalises, while `post-softmax` takes topk first and
  softmaxes only `k` values.

#### Attention

2026-09-02, same instance and image, `single_test_ops/fa_linear_ops.json` through
the eager `torch` (SDPA) provider. Nine of the file's ten cases; the tenth is
`q_len: 4` speculative decode, which this provider rejects on purpose (see
[Known unsupported](READING_THE_NUMBERS.md#known-unsupported)) and which the `nkilib` provider added a
day later does cover — see
[Decode: the `nkilib` provider](#decode-the-nkilib-provider-attention_tkg), which
supersedes the decode rows below. All bf16, `head_dim` 128.

| Mode | q/kv heads | Batch | cache_len | q_len | Latency | TFLOPS | MFU | mem_bw |
|---|---|---|---|---|---|---|---|---|
| prefill | 80/8 (GQA) | 1 | 0 | 4,096 | 9,507.6 us | 36.15 | 21.7% | 19.4 GB/s |
| prefill | 80/8 (GQA) | 4 | 0 | 4,096 | 34,422.1 us | 39.94 | 24.0% | 21.4 GB/s |
| prefill | 80/8 (GQA) | 1 | 0 | 10,240 | 39,969.3 us | 53.73 | **32.2%** | 11.5 GB/s |
| prefill | 80/80 (MHA) | 1 | 0 | 4,096 | 9,520.2 us | 36.10 | 21.7% | 35.2 GB/s |
| prefill | 80/80 (MHA) | 4 | 0 | 4,096 | 34,757.8 us | 39.55 | 23.7% | 38.6 GB/s |
| prefill | 80/80 (MHA) | 1 | 0 | 10,240 | 39,782.3 us | 53.99 | **32.4%** | 21.1 GB/s |
| decode | 80/8 (GQA) | 16 | 4,096 | 1 | 1,976.7 us | 1.36 | 0.8% | 136.2 GB/s |
| decode | 80/8 (GQA) | 64 | 4,096 | 1 | 4,511.3 us | 2.38 | 1.4% | **238.7 GB/s** |
| decode | 80/8 (GQA) | 16 | 10,240 | 1 | 6,066.2 us | 1.11 | 0.7% | 110.7 GB/s |

Read MFU for the prefill rows and `mem_bw` for the decode rows, not both for
both. Prefill is compute-bound (`calc_mem_ratio` is in the hundreds), decode is
memory-bound — one query row against the whole cache does ~10 FLOPs per byte
moved, so a decode MFU of 0.8% is arithmetically unavoidable and says nothing
about the chip. Three things the table does say:

- **GQA and MHA cost the same here, to within 1%.** 9,507.6 vs 9,520.2 us at
  `q_len 4096`, 39,969.3 vs 39,782.3 at 10,240 — and the MHA row is *faster* at
  the longer length. That is the expected result rather than a suspicious one:
  prefill is compute-bound and both configs do identical arithmetic, since GQA
  saves KV *traffic*, not FLOPs. The traffic it saves (168 MB down to 16.8 MB) is
  ~0.2 ms at this core's bandwidth, about 2% of a 9.5 ms latency. **This does not
  demonstrate that `enable_gqa=True` avoids materialising the expanded cache** —
  prefill cannot distinguish the two, because the copy would be hidden under the
  same compute. Decode would show it, and the workload only covers GQA there, so
  there is no MHA decode row to compare against.
- **Prefill MFU improves with sequence length and saturates well below the
  chip.** 21.7% at `q_len 4096` -> 32.2% at 10,240. Batching to 4 buys almost
  nothing (24.0%), which is consistent: at `q_len 4096` one sequence already
  fills the engine, so a batch is four sequential prefills (34,422 ≈ 4 x 9,508 x
  0.905). Nothing here approaches the 90% the same core reaches on a large `gemm`,
  and that gap is the SDPA lowering, not the tensor engine — but it is the *quality*
  of a fused lowering, not a missing one. See the bullet below.
- **Decode reaches 15-33% of this core's memory bandwidth, and that is the real
  finding.** The decode rows move nothing but the KV cache, so `mem_bw` against
  ~725 GB/s is the whole story: 110.7-238.7 GB/s, i.e. 15-33%. For scale, the
  best plain memory-bound op measured on this core is `index_select` at 631 GB/s.
  So decode here is *not* bandwidth-limited — it is leaving a factor of 2.6-5.7x
  of available bandwidth unused, and a fused paged-attention kernel is what would
  reclaim it. Larger batches help (238.7 at B=64 vs 136.2 at B=16) and longer
  caches hurt (110.7 at 10,240), which is the signature of a per-call fixed cost
  being amortised rather than of a bandwidth ceiling.

**The prefill rows are already a fused NKI flash kernel; the decode rows provably
cannot be.** This is worth stating flatly because an earlier version of this
document, and of the two provider docstrings, claimed the eager runtime had no
fused attention kernel at all. It does. `torch_neuronx` rewrites SDPA to a NKI
flash kernel inside its own dynamo backend — `_can_use_nki_flash_attention` in
`torch_neuronx/neuron_dynamo_backend/decompositions.py`, enabled by default via
`TORCH_NEURONX_ENABLE_NKI_SDPA` — for any call satisfying all of:

```
key.shape == value.shape,  attn_bias is None,  dropout_p == 0,
L % 512 == 0,  S % 512 == 0,  D <= 128,  B * H <= 512
```

Setting `TORCH_NEURONX_ENABLE_NKI_SDPA=0` and re-running the 80/8/128 `q_len 4096`
prefill takes it from **9,443 us to 60,500 us — 6.41x**. So 21.7-32.4% MFU *is* the
fused kernel's score, and closing the remaining gap to the H100's 61-69% means a
better kernel rather than a first one.

**And the kernel it rewrites to is `nkilib`'s `attention_cte`.** That is the
obvious candidate for "surely there is a faster NKI kernel than this", so it is
worth settling by identity rather than by benchmark. `decompositions.py:24` does
`from nkilib.core.attention.attention_cte import attention_cte`, `:71` wraps that
same object as `wrapped_flash_fwd`, and `:935` launches it with
`grid = (logical_neuron_cores,)`, `tp_q=True`, `tp_k=True` and KV left at its own
head count so the kernel does the GQA replication itself. Calling the kernel
directly the same way reproduces the SDPA number, at the shape the table publishes:

| Path | Latency | TFLOPS (causal) | % of 166.75 |
|---|---|---|---|
| SDPA — what the op def calls | 7,636.7 us | 44.99 | 27.0% |
| `attention_cte[2]`, `tp_k=True`, kernel-side GQA | **7,407.4 us** | 46.39 | 27.8% |
| `attention_cte` with the `lnc` argument left off | 13,688.3 us | 25.10 | 15.1% |

0.97x, and bit-identical output — it is one kernel, reached two ways. (This probe
expands KV by hand for the SDPA baseline, matching what the op def does, and gets
7,636.7 us where the sweep reports 9,443; the sweep's figure includes the op def's
own tensor handling.)

Two traps that row three is in the table to name. `attention_cte`'s `__getitem__`
takes the **bare int** — `attention_cte[2]`, not `attention_cte[(2,)]`, which
raises `NkiValidationError: NKI only supports LNC 1 or 2, but got (2,)` — and
leaving it off entirely costs **1.85x**, because the kernel then runs on one half of
the LNC2 pair. A hand-rolled NKI provider that skips it would look like evidence
that `attention_cte` is slower than what SDPA gets, and would be measuring half a
core.

So the prefill gap to the H100 is `attention_cte` against cuDNN/FlashAttention at
the same shape. There is no unused kernel to reach for; `swa_fused_cte` and
`attention_segmented_cte` are different problems (sliding-window and segmented
attention), not faster paths for this one. Reproduce with
`tools/probe_attention_kernel.py`.

Every decode row fails that gate twice over and always will: `q_len == 1` can never
satisfy `L % 512 == 0`, and their `B*H` is 1280 and 5120 against a limit of 512.
That is what the 15-33% of bandwidth reflects.

##### Decode: the `nkilib` provider (`attention_tkg`)

The fix is `nkilib`'s `attention_tkg` (token-generation attention), and as of
2026-09-03 it is wired up as a second provider — `ops/nkilib/flash_attention.py`,
provider name `nkilib`. It needs `kv_len % 128 == 0`, which none of
`fa_linear_ops.json`'s decode rows satisfy (`cache_len 4096` + `q_len 1` = 4097), so
it ships with `workloads/fa_decode_tkg.json`: the same rows one token shorter, plus
8,192 and 16,384 to locate the crossover. **Both providers run that file**, so this
is one launch and an exact comparison:

```bash
sudo docker run --rm --privileged -v $HOME/xpu-perf:/xpu-perf \
  -w /xpu-perf/projects/micro_perf -e PYTHONPATH=/xpu-perf/src \
  xpu-perf-beta4:latest python launch.py --backend NEURON --device 0 \
  --report_dir /results --workload vendor_ops/NEURON/workloads/fa_decode_tkg.json
```

2026-09-03, Trn2, one logical core pair, 80/8/128 GQA, all bf16:

| Batch | kv_len | q_len | `torch` (SDPA) | `nkilib` (`attention_tkg`) | Speedup |
|---|---|---|---|---|---|
| 16 | 4,096 | 1 | 866.6 us / 310.5 GB/s | 1,580.1 us / 170.3 GB/s | 0.55x |
| 16 | 8,192 | 1 | 3,764.2 us / 142.8 GB/s | 2,411.4 us / 222.9 GB/s | **1.56x** |
| 16 | 10,240 | 1 | 6,801.3 us / 98.8 GB/s | 2,769.0 us / 242.6 GB/s | **2.46x** |
| 16 | 16,384 | 1 | 15,726.4 us / 68.3 GB/s | 3,841.2 us / 279.7 GB/s | **4.09x** |
| 64 | 4,096 | 1 | 3,239.1 us / 332.3 GB/s | 3,792.3 us / 283.8 GB/s | 0.85x |
| 64 | 10,240 | 1 | 11,467.9 us / 234.3 GB/s | 8,748.5 us / 307.1 GB/s | **1.31x** |
| 16 | 10,240 | 4 | rejected | 3,881.1 us / 173.6 GB/s | new coverage |

- **Neither provider wins everywhere, so keep both.** SDPA's bandwidth *falls* with
  context length (310.5 -> 68.3 GB/s) because the unfused path builds a score matrix
  per step; `attention_tkg`'s *rises* (170.3 -> 279.7 GB/s) as its fixed per-batch
  cost amortises. They cross between 4,096 and 8,192. At 4,096 SDPA is genuinely the
  better choice, at 16,384 it is 4x worse. `--task`/provider selection is how you
  pick, and the answer depends on the context length you actually serve.
- **This is what the "15-33% of bandwidth" gap was.** The best decode row is now
  307.1 GB/s = 42% of ~725 GB/s, up from 238.7. It is still not `index_select`'s
  631 GB/s, so there is more left, but the shape of the curve has changed sign:
  bandwidth no longer degrades with cache length.
- **`q_len 4` speculative decode is newly measurable.** The SDPA provider turns it
  away because `is_causal=True` aligns the mask to the top-left of a non-square
  score matrix and would silently attend to the wrong keys; the `nkilib` provider
  builds the active mask explicitly (`active_mask[k, b, h, s] == 1` iff `k <= s`),
  so the tenth case of the published file finally has a number.
- **`attention_tkg` needs a *paged* prior to be worth anything.** With a flat
  `[bs, 1, s_prior, d]` prior it is 4-8x slower than with pages and loses to SDPA at
  every length measured (0.11-0.48x). The provider therefore always reinterprets the
  linear cache as pages. `block_len` itself barely matters — the kernel's own
  `resize_cache_block_len_for_attention_tkg_kernel` overrides the caller and reduces
  128 to 16 at kv_len 10,240, and all of 128/64/32/16 land within 2%.
- **The kernel's fast MM1 path (QK-swap) is unreachable at this head count.**
  `is_qk_swapped` needs `s_active * q_per_group` to satisfy both `% 32 == 0 or
  32 % it == 0` and `128 % it == 0`; 80/8 GQA gives `q_per_group = 10`, so decode
  offers 10 and fails both. Re-running at 64 query heads makes it eligible and
  changes nothing measurable, so the missing swap is *not* what costs the 4,096 row.

`attention_tkg` is not directly launchable — it takes a `BufferManager` and a
caller-allocated `out`, so the `@nki.jit` entry point is ours. The provider
docstring lists the non-obvious constraints (`qk_in_sb=True` is mandatory when
`fuse_rope=False`; every `alloc_stack` must be inside an open scope; `curr_sprior`
counts the active tokens; q must be pre-scaled by `1/sqrt(d)`; kv heads fold into
the batch). Reproduce the configuration sweep behind these conclusions with
`tools/probe_attention_tkg.py`.

For the same cases measured on an H100 with the same provider code, see
[`../GPU/README.md`](../GPU/README.md).

Collectives, best `bus_bw`. Both columns are **inside one chip** — a
trn2.3xlarge is one NeuronDevice, so nothing here crosses a NeuronLink. For the
cross-chip numbers see
[Multi-chip: trn2.48xlarge](#multi-chip-trn248xlarge).

| Op | ws=2 | ws=4 |
|---|---|---|
| `all_gather` | 64.5 GB/s | 125.1 GB/s |
| `all_reduce` | 37.8 GB/s | 107.1 GB/s |
| `reduce_scatter` | 91.8 GB/s | 101.7 GB/s |
| `all_to_all` | not supported at ws=2 | 54.2 GB/s |
| `device2host` | 14.3 GB/s | 4.2 GB/s |
| `host2device` | 14.0 GB/s | 4.1 GB/s |

- **The host-transfer rows are not comparable across columns.** The ws=4 run
  capped transfers at 1 GiB to stay inside the per-core HBM budget while the ws=2
  run reached 8 GiB, and `device2host` needs the larger sizes to saturate. Four
  ranks also contend for one host DMA path: 4.2 x 4 is in the range of 14.3 x 2.
- `all_to_all` has a ~360 us fixed floor that dominates everything below 8 MB,
  then climbs to a ~40 GB/s plateau (fp32) at 256-512 MB.

**The largest sizes in `workloads/xccl_ops/` do not fit.** Each file sweeps
`batch_size` to 2,097,152 x `dim_size` 1024 — 8 GiB at fp32. One logical
NeuronCore has 24 GiB (96 GiB / 4 at LNC=2, confirmed by the runtime reporting
`total_hbm=25769803776`), and `neuronx-cc` budgets I/O plus an equal scratchpad,
so an 8 GiB buffer asks for 32 GiB:

```
[ERROR] [NCC_EOOM001] Maximum peak HBM usage of 32.00GB exceeds HBM limit of
24.00GB for Trn2. This consists of 16.00GB I/O tensors, 0B intermediate tensors,
and 16.00GB internal (scratchpad) allocations
```

**And an OOM in one rank hangs the run permanently** —
see [Troubleshooting](TROUBLESHOOTING.md#troubleshooting). Cap `batch_size` at 262,144 (1 GiB at
fp32, far past the bandwidth plateau) to avoid it entirely;
`tools/cap_xccl_workloads.py` does this.

### Multi-chip: trn2.48xlarge

2026-09-05, same eager image and driver, on a **trn2.48xlarge**: 16 NeuronDevices
x 4 logical cores = **64 devices** as micro_perf counts them, 96 GiB per chip
(24 GiB per logical core, unchanged), 192 vCPU. Chip *N* owns logical cores
`4N .. 4N+3`, and devices 0-3 and 12-15 hang off NUMA node 1 while 4-11 hang off
node 0.

The NeuronLink topology is a **4x4 torus in row-major order**, from `neuron-ls
--topology`:

```
*––►[ 0 ]◄––►[ 1 ]◄––►[ 2 ]◄––►[ 3 ]◄––*
*––►[ 4 ]◄––►[ 5 ]◄––►[ 6 ]◄––►[ 7 ]◄––*
*––►[ 8 ]◄––►[ 9 ]◄––►[10 ]◄––►[11 ]◄––*
*––►[12 ]◄––►[13 ]◄––►[14 ]◄––►[15 ]◄––*
       (columns wrap too; * = wrap-around link)
```

So chip 0's neighbours are 1, 3, 4 and 12. **Adjacent indices are not adjacent
chips**: 3 and 4 are two hops apart, and that distinction turns out to decide
which collectives are legal, not just how fast they are — see
[Placement](#placement-which-chips-you-pick) below.

Everything here comes from `tools/run_multicard_sweep.sh`, which is separate from
`run_full_sweep.sh` on purpose: 33 of that script's 39 labels are `--device 0` and
it is the reproduction recipe for the single-chip tables above. Summarize a result
tree with `tools/analyze_multicard.py`.

Five things the 3xlarge could not answer are answered below, and one of the five
answers is that a row in the table above does not mean what its name says.

#### `ccl_ops.json` now runs

`workloads/llm/single_test_ops/ccl_ops.json` was the only file in the repo with no
runnable case at all: every case asks for `world_size: 8`. At 8 ranks (cores 0-7,
i.e. chips 0 and 1) it runs, and it is also the first NeuronLink measurement in
the repo. 262,144 x 1024, one rank per core:

| Op | dtype | payload | latency | `bus_bw` | floor at 8 tokens |
|---|---|---|---|---|---|
| `all_gather` | bf16 | 512 MiB | 5.76 ms | **81.6 GB/s** | 148 us |
| `all_reduce` | bf16 | 512 MiB | 11.85 ms | **79.3 GB/s** | 142 us |
| `reduce_scatter` | bf16 | 512 MiB | 6.87 ms | **68.3 GB/s** | 150 us |
| `all_to_all` | bf16 | 512 MiB | 16.46 ms | **28.5 GB/s** | 563 us |
| `all_gather` | **int8** | 256 MiB | **1307.00 ms** | **0.2 GB/s** | 2,407 us |
| `all_to_all` | **int8** | 256 MiB | **2002.03 ms** | **0.1 GB/s** | 6,882 us |

The bf16 `all_reduce` plateau agrees with `nccom-test`, which reports a ~78 GB/s
Ring plateau on the same hardware — two unrelated tools, so treat ~80 GB/s as the
per-rank NeuronLink figure rather than a micro_perf artefact.

**int8 collectives are broken, not merely slower.** 227x on `all_gather` and 122x
on `all_to_all` against bf16, per *element* as well as per byte, with a 2.4 ms
floor at 8 tokens where bf16 pays 147 us — and nothing is logged: no warning, no
exception, no fallback notice. This matters for quantized tensor parallelism, where
the point of int8 activations is to make the all-gather cheaper.

`tools/probe_int8_collective.py` runs a bf16 phase and then an int8 phase of the
same op and samples `neuron-monitor` at 1s underneath, on three signals:
NeuronCore utilization, the 8 worker runtimes' own CPU summed (100% = one saturated
vCPU), and instance-wide user CPU. The series is flat and unambiguous, and
reproduced across three runs:

| phase | NeuronCore util | 8 workers' CPU |
|---|---|---|
| `all_gather` bf16 | brief spike to 15-17% | ~9% of one vCPU |
| `all_gather` int8 | **steady 1.15%** for 10-14 s | 7-11% of one vCPU |
| `all_to_all` bf16 | brief spike to 75-85% | ~6% of one vCPU |
| `all_to_all` int8 | **steady 2.95%** for 15-19 s | ≤32% of one vCPU |

So **nothing is busy**. It is not a host fallback — this is not the fp8 story, where
one vCPU pinned at 100% against an idle device was the whole answer; all eight ranks
together never reach a third of a single vCPU. And it is on the device: utilization
is small but steady and nonzero for the entire int8 phase, in a per-iteration
pattern. Integrating, the int8 phase's total device-busy time is *comparable* to the
bf16 phase's (10 s x 1.15% vs ~2 s x 16%), so this is not 200x more work — it is the
same work spread over 200x the wall clock, i.e. a serialized issue path with the
device idle in between. That makes it a runtime/CCL matter to raise upstream rather
than something a workload or op-def change can fix, and int8 activations should be
assumed to *cost* bandwidth on this stack until it is fixed.

#### What crossing a chip boundary costs

Rank count held at 4, only the placement changed — the comparison a single-chip
instance cannot express. `topo_hop0` is cores 0-3, all inside chip 0. `topo_hop1`
is cores 0, 4, 8, 12: one core in each of chips 0-3, so every ring step crosses a
NeuronLink.

| Op | hop0 (one chip) | hop1 (4 chips) | cost of leaving the chip |
|---|---|---|---|
| `all_gather` | 93.0 GB/s | 36.1 GB/s | **2.58x** |
| `all_reduce` | 105.3 GB/s | 37.6 GB/s | **2.80x** |
| `reduce_scatter` | 73.6 GB/s | 32.4 GB/s | **2.27x** |
| `all_to_all` | 39.9 GB/s | *rejected by the runtime* | — |

bf16, best `bus_bw` over `batch_size` 1 … 262,144. The latency floors are the
same either way (104-111 us), so this is bandwidth, not launch cost.

**`all_to_all` requires a mesh-mappable replica group.** At 4 ranks spread one per
chip the runtime refuses outright:

```
ENC:enc_post_operation  [nec_dev 4] alltoall cannot be supported without Mesh
algorithm, check if Mesh algorithm is supported on the current replica group
size, rank_n(4)
```

It is not a size rule — 8, 16, 32 and 64 contiguous ranks all work. It is a
*shape* rule, and it constrains where an expert-parallel group may be placed, not
how big it may be.

#### world_size scaling, 8 to 64

bf16, best `bus_bw`, contiguous cores, `--numa -1`:

| world_size | chips | `all_gather` | `all_reduce` | `reduce_scatter` | `all_to_all` |
|---|---|---|---|---|---|
| 4 (`topo_hop0`) | 1 | 93.0 | 105.3 | 73.6 | 39.9 |
| 8 | 2 | 84.0 | 79.5 | 68.4 | 28.6 |
| 16 | 4 | 107.6 | 119.2 | 112.7 | 18.6 |
| 32 | 8 | 115.4 | 119.8 | 117.7 | 17.5 |
| 64 | 16 | **166.4** | **146.5** | **142.6** | 15.7 |

`bus_bw` is `2(n-1)/n x algo_size / latency` — bytes crossing *a* link, summed
over the ring, not one link's bandwidth. It is expected to rise with rank count
when the extra ranks bring extra links, so read the table as "the fabric keeps up
to 64 ranks", not "64 ranks are 2x faster than 4". The three ring collectives
scale; the small-message floor grows from 106 us at 4 ranks to ~1.0-1.4 ms at 64,
which is the number that decides whether a small tensor-parallel `all_reduce` is
worth doing at all.

**`all_to_all` goes the other way** — 39.9 GB/s at 4 ranks down to 15.7 at 64,
with the floor climbing 306 us → 5.36 ms. It sends n² messages, so this is the
expected shape, and it is the row to check before assuming an expert-parallel
dispatch will spread across the instance for free.

#### `device2device` is not a device-to-device transfer

Three placements, three identical numbers:

| Label | devices | chips | `mem_bw` |
|---|---|---|---|
| `d2d_intra` | 0,1 | same chip | 642.3 GB/s |
| `d2d_hop1` | 0,4 | adjacent on the torus | 642.5 GB/s |
| `d2d_hop2` | 0,8 | two hops apart | 642.9 GB/s |

Hop distance changes nothing because no hop is taken. `Device2DeviceOp` in
`op_defs/basic_ops/xccl_ops.py` allocates `src` *and* `dst` on the same device and
runs `dst.copy_(src)`; every worker copies within its own HBM. So the **648.7
GB/s** in [Memory-bound ops](#memory-bound-ops) is a local HBM copy at 89% of one
core's 725 GB/s share — a good number, and the right one to compare against `add`
or `cast` — but it is not a peer-to-peer figure and must not be read as one.
Nothing in the repo measures a peer-to-peer copy; the collectives are the only
cross-chip traffic here.

#### 64 cores at once

`conc64_gemm` runs the gemm sweep with one worker per logical core — 64 of them,
all 16 chips loaded — and `conc1_gemm` runs the same shapes on core 0 alone
(`workloads/conc_control/gemm.json`; its header says why it is a subset). A
`ComputeEngine` worker takes cases from a shared queue, so this is 64 *different*
cases in flight and every latency below is still a single-core latency. The ratio
is therefore the answer to "does a core still perform when its 63 neighbours are
busy?".

| dtype | shape (M x K x N) | 1 core | 64 cores | 64c / 1c | 64c TFLOPS | 64c MFU |
|---|---|---|---|---|---|---|
| bf16 | 28672 x 4096 x 4096 | 6,733 us | 6,505 us | **0.97x** | 147.9 | 88.7% |
| bf16 | 20480 x 4096 x 4096 | 4,783 us | 4,765 us | **1.00x** | 144.2 | 86.5% |
| bf16 | 4096 x 8192 x 8192 | 3,805 us | 3,877 us | **1.02x** | 141.8 | 85.0% |
| bf16 | 28672 x 8192 x 8192 | 29,535 us | 28,769 us | **0.97x** | 133.8 | 80.2% |
| fp16 | 20480 x 4096 x 4096 | 5,149 us | 4,723 us | **0.92x** | 145.5 | 87.3% |
| fp32 | 4096 x 8192 x 8192 | 18,189 us | 16,327 us | **0.90x** | 33.7 | 74.4% |
| bf16 | **1** x 4096 x 4096 | 80.9 us | 264.4 us | **3.27x** | 0.1 | 0.1% |
| bf16 | **512** x 8192 x 8192 | 535 us | 910 us | **1.70x** | 75.5 | 45.3% |
| fp16 | **512** x 8192 x 8192 | 533 us | 1,059 us | **1.99x** | 64.9 | 38.9% |
| fp8 e4m3 | 16384 x 8192 x 8192 | 453,977 us | 458,226 us | **1.01x** | 4.8 | 1.5% |

**On the compute-bound plateau there is no concurrency loss at all** — 0.90x to
1.02x, i.e. within run-to-run noise, and several rows come out marginally faster
at 64 cores than at 1. So the per-core numbers in the single-chip tables above are
not a one-core-in-isolation artefact: bf16 gemm holds 147.9 TFLOPS per core with
every core loaded, which is **9.5 PFLOPS of bf16 gemm per instance at 88.7% of the
64-core peak**. "x4 per chip" does extend to "x64 per instance".

**Small shapes are the exception, and they degrade badly.** M=1 costs 3.3x more
latency at 64 cores; M=512 costs 1.7-2.0x. Those cases are host-dispatch-bound
rather than array-bound — 64 workers share 192 vCPUs — so this is contention for
the launch path, not for the fabric or for HBM. It is the same effect as
[small-op latency](READING_THE_NUMBERS.md#small-op-latencies-measure-the-runtime-not-the-chip) showing up as latency, and it
lands exactly on decode-shaped work. One row (fp16 512 x 4096 x 4096, 9.5x) is far
outside the pattern of its neighbours and is more likely a warmup landing inside
the measured window than a real 9.5x.

The elementwise files (`conc64_mem` / `conc1_mem`, `add` `mul` `sub` `cast`, 216
matched cases) say the same thing with a sharper size dependence:

| payload | cases | median 64c/1c | p90 | worst |
|---|---|---|---|---|
| >= 256 MiB | 27 | **1.00x** | 1.87x | 2.33x |
| 16-256 MiB | 48 | 2.00x | 10.81x | 14.29x |
| 1-16 MiB | 48 | 2.09x | 11.33x | 31.84x |
| < 1 MiB | 93 | 3.20x | 14.61x | 28.18x |

Peak per-core `mem_bw` is unchanged for `mul` (601.8 → 602.2 GB/s) and `cast`
(615.4 → 595.7), and loses 6-13% for `sub` (598.9 → 565.0) and `add` (603.9 →
525.8). So **~38 TB/s of elementwise bandwidth per instance** (64 x ~595 GB/s,
82% of 16 x 2.9 TB/s) is a measurement, on the same footing as the per-chip 2.4
TB/s above. Everything below a quarter-gigabyte, though, is 2-3x worse with the
instance loaded — and unlike the gemm case this is *not* only the smallest shapes:
the 16-256 MiB bucket degrades 2x too, where a single core's device time is only a
few hundred microseconds and the per-iteration host work is not negligible against
it.

**float8 is not slow because of CPU contention.** 458 ms at 64 cores and 454 ms on
one idle core with 192 vCPUs free — 1.01x, at the largest shape. So the emulated
float8 path is intrinsically that slow rather than starved. (The 4-core sweep above
does show 1.46-1.53x median on float8, over all 12 of its shapes rather than only
the largest, so smaller float8 shapes may well contend — what is ruled out is
contention explaining the two orders of magnitude.) See
[fp8: a dtype this chip cannot multiply](READING_THE_NUMBERS.md#fp8-the-sweep-measures-the-eager-path-in-a-dtype-this-chip-cannot-multiply).

#### Placement: which chips you pick

Four 8-rank runs of the same `ccl_ops.json`, differing only in the device list.
The first three keep the two chips torus-adjacent and vary the NUMA node; the
fourth breaks adjacency.

| Label | chips | adjacency | NUMA | `all_gather` | `all_reduce` | `reduce_scatter` | `all_to_all` |
|---|---|---|---|---|---|---|---|
| `ccl8_llm` | 0,1 | adjacent | node 1 | 81.6 | 79.3 | 68.3 | 28.5 |
| `numa_node0` | 4,5 | adjacent | node 0 | 81.8 | 79.2 | 68.2 | 28.5 |
| `numa_cross` | 0,4 | adjacent | **crosses** | 81.7 | 79.2 | 68.3 | 28.5 |
| `topo_far8` | 3,4 | **2 hops** | crosses | *fails* | *fails* | *fails* | **63.1** |

bf16 `bus_bw` in GB/s at 512 MiB. The three adjacent placements agree to within
0.3% on every op — including `numa_cross`, whose 8 ranks straddle the two sockets —
so **the NUMA node the chips hang off is irrelevant** to a collective. That is
worth establishing rather than assuming, because the launcher's own default gets
NUMA wrong on this instance (below), and because it means a multi-chip number in
this repo does not have to name its sockets.

It does have to name its chips. A ring cannot be laid over 8 cores
whose two chips are not neighbours, so the runtime falls back to its Mesh
algorithm, and Mesh has a **40 MiB message cap**:

```
ENC:enc_post_operation  [nec_dev 13] Collective operation ALLREDUCE with
total_size 268435456 over group id 0 (rank_n:8 node_n:1) is currently not
supported. Replica-group 0: [0,1,2,3,4,5,6,7]
ENC:enc_post_operation  Collective operation may not be supported on this
replica-group, because message size is larger than 41943040 bytes. Consider
using tiling or increasing buffer size by setting
NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE or NEURON_RT_DBG_MESH_CHANNEL_BUFFER_SIZE
```

So `all_reduce`, `all_gather` and `reduce_scatter` **fail outright** at 512 MiB on
a badly placed group — the small cases still run, which is why the label produces a
report at all — while `all_to_all` not only survives but goes **2.2x faster than on
an adjacent pair** (63.1 vs 28.5 GB/s), Mesh being the better fit for an n²
pattern. Together with the [4-rank `all_to_all` rejection](#what-crossing-a-chip-boundary-costs)
that is one rule stated twice: *which* chips a replica group lands on decides which
collectives are legal and which algorithm they get, and neither the rank count nor
the message size predicts it on its own. A tensor-parallel group placed across
chips 3 and 4 does not run slower, it does not run.

#### NUMA pinning

`run_multicard_sweep.sh` passes `--numa -1` (no CPU pinning) everywhere, because on
this instance the *default* is wrong: with `--numa` unset the launcher starts one
process group per NUMA node and splits the device list between them in index order,
but devices 0-3 and 12-15 are on node 1 while 4-11 are on node 0, so an index-order
split pins most workers to the wrong socket. `NUMA=0` on the script re-enables
pinning for a comparison run. Note `--numa` is only a CPU-affinity knob:
`BaseEngine.__init__` creates one process per device either way, and `numa_num` /
`numa_order` only choose which `numa_cores` those processes are pinned to.

#### Three collectives that had never run

`op_defs/basic_ops/xccl_ops.py` registers eight collectives. Five of them
(`all_gather`, `all_reduce`, `reduce_scatter`, `all_to_all`, `device2device`) had
workload files; **`broadcast`, `p2p` and `all_reduce_h2d` were referenced by no
workload JSON anywhere in the repo**, on any backend. That is a different thing
from the [unsupported table](READING_THE_NUMBERS.md#known-unsupported) below: nobody had found them
missing, because a task name that no file asks for is not a rejection, it is
silence. `workloads/xccl_ops/{broadcast,p2p,all_reduce_h2d}.json` fill the gap —
each file's `_comment` says what the op actually measures and how to read it —
and running them for the first time produced one repo-wide bug and one hard limit.

**`broadcast` was broken on every backend.** `BroadcastOp.summary` had drifted
twice from `BasicOp.summary`: it took `(self, latency_us)` while `Backend.perf`
calls `summary(latency_us, kernel_mapping)` (`core/backend.py:378`), and it
unpacked the result into `(target_dict, env_dict)` while `BasicOp.summary`
returns a single dict. Every case died with `TypeError: BroadcastOp.summary()
takes 2 positional arguments but 3 were given` or, once that was fixed,
`ValueError: too many values to unpack (expected 2)`. Neither had ever been seen
because the op had never once been called. Fixed in place; the per-broadcast
division it does is still correct after the fix, and the comment above it says
why.

With the fix, 8 ranks on chips 0-1, bf16:

| `batch_size` | payload | latency (us) | `algo_bw` = `bus_bw` (GB/s) |
|---|---|---|---|
| 1024 | 2 MiB | 1111.596 | 1.886 |
| 262144 | 512 MiB | 13243.147 | 40.539 |

`broadcast` is the one collective whose `bus_size` equals its `algo_size` — no
`2(n-1)/n` factor — so 40.5 GB/s here is directly comparable to `all_reduce`'s
*algo* bandwidth at the same 8 ranks (79.3 bus → 45.3 algo), not to its 79.3.
Getting a whole 512 MiB payload to 7 other ranks at 40 GB/s rules out the naive
ring, which would be 7 sequential hops of the full payload and so ~1/7 of link
bandwidth: the runtime is pipelining or using a tree. The 2 MiB row is the one
that matters in serving — broadcasting sampled token ids to the other TP ranks is
a tiny-payload latency problem, and 1111 us for 2 MiB says the floor, not the
bandwidth, is the whole cost there.

**`p2p` does not exist on this backend at all.** Not a placement or a capacity
limit — both ranks on one chip, same launch that runs `broadcast` fine, every
size:

```
RuntimeError: Backend neuron does not support send
RuntimeError: Backend neuron does not support recv
```

So the native runtime's `torch.distributed` backend has no point-to-point
primitive, and a real one-way one-link transfer is **unreachable through
`torch.distributed` on Trainium today** — the collectives measure a whole ring,
and `device2device` turns out to be
[a local copy](#device2device-is-not-a-device-to-device-transfer). The three
`p2p_hop*` labels in `run_multicard_sweep.sh` are kept as a one-minute check of
whether a later Neuron release has gained it; the file's real first use is a
multi-GPU box.

**`all_reduce_h2d` runs**, and is the only op in the repo that measures
*overlap*: `dist.all_reduce(async_op=True)` followed by a host-to-device copy
while it is in flight. 8 ranks, bf16, largest case in the file: **167727.201 us,
`algo_bw` 6.404 GB/s, `bus_bw` 8.805 GB/s**. That number is not yet an overlap
verdict and should not be quoted as one, for three reasons written into the
workload file: the op def issues the all_reduce on the *default* group rather
than `op_group`; `async_op=True` is never `wait()`ed, so the timing window
catches an unspecified fraction of the collective; and 8 ranks each copying the
same payload contend for one host DMA path by construction, so the uncontended
`host2device` row is the wrong control. The missing measurement is `host2device`
at 8 ranks — with it, latency near `max(the two)` means the engines overlap and
near their sum means they serialise. `ccl8_h2d` in the sweep produces the first
half.

### XLA runtime

2026-08-31: torch 2.9.1, torch-xla 2.9.0, torch-neuronx 2.9.0.2.15.32035,
neuronx-cc 2.23.6484. The `legacy` column is the pre-refactor branch's own run
from 2026-03-09, kept as a cross-check.

| Op | Dtype | Shape | Latency | Metric | MFU | legacy (2026-03-09) |
|---|---|---|---|---|---|---|
| gemm | fp32 | 1024x4096x4096 | 1,462.2 us | 23.5 TFLOPS | 52% | 1,432.3 us |
| gemm | fp16 | 1024x4096x4096 | 727.9 us | 47.2 TFLOPS | 28% | 698.7 us |
| gemm | bf16 | 1024x4096x4096 | 758.5 us | 45.3 TFLOPS | 27% | 697.4 us |
| add | fp32 | 1024x1024 | 681.6 us | 18.5 GB/s | ~0% | 1,090.4 us |
| add | bf16 | 1024x1024 | 711.2 us | 8.8 GB/s | ~0% | 1,026.7 us |
| softmax | fp32 | 1024x1024 | 655.9 us | 12.8 GB/s | ~0% | 16.8 us (see below) |
| softmax | bf16 | 1024x1024 | 612.2 us | 6.9 GB/s | ~0% | 16.8 us (see below) |
| flash_attention | bf16 | prefill q_len=2048, 8 heads, dim 128 | 1,658.6 us | 5.2 TFLOPS | 3% | not measured |
| all_reduce | bf16 | 1024x1024, ws=2 | 659.5 us | 3.18 GB/s bus | n/a | trn2.48xlarge only |
| all_gather | bf16 | 1024x1024, ws=2 | 1,108.3 us | 0.95 GB/s bus | n/a | trn2.48xlarge only |

`gemm` is the number to check when deciding whether to trust a run: it agrees
with the March baseline to within 2-9%, and it is the only op here where a wrong
device or a pruned graph is unmistakable.

**These are single-run figures and the spread is wide.** Repeating the same
shapes against a warm cache gave gemm fp32 1,258 us, fp16 628 us, bf16 570 us,
all_reduce 1,040 us — 15-25% off, in both directions. Compare orders of
magnitude, not digits: a real fp32 gemm here is "about a millisecond", and tens
of microseconds means a pruned graph or the wrong device. At 1024x1024 `add` and
`softmax` both cost 610-710 us, which is graph-launch time; their GB/s figures
are not meaningful, but their agreeing with *each other* is.

Two cautions on the `legacy` column: **its 16.8 us softmax is not a real
measurement** (two launch-bound ops on identical shapes cannot differ 60x — the
graph was pruned, and that branch run today prunes `gemm` too, reporting ~1,900
TFLOPS), and its collective baselines were taken on trn2.48xlarge. Treat it as a
record of what was run, not a trustworthy baseline.

