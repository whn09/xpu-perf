# Reading the NEURON numbers

The numbers themselves are in [RESULTS.md](RESULTS.md).

## Reading the numbers

### MFU

`mfu` = `calc_flops_power / peak_tflops`, with `peak_tflops` from AWS's published
dense per-chip peaks in `NEURON_CHIP_PEAK_TFLOPS` (`backend_neuron.py`, sourced
from `general/arch/neuron-hardware/trainium2.html`), divided by the logical-core
count `neuron-ls` reports:

| | trn1 (NeuronCore-v2) | trn2 (NeuronCore-v3) |
|---|---|---|
| FP32 | 48 | 181 TFLOPS |
| BF16 / FP16 / TF32 | 191 | 667 TFLOPS |
| FP8 | not published | 1,299 TFLOPS |

So the bf16 denominator on a trn2 is 667 / 4 = **166.75 TFLOPS** at LNC=2 — and
83.375 at LNC=1. The split is read from `neuron-ls` rather than assumed, so check
`logical_neuroncore_config` in the run's `backend` block if a number surprises
you. Sparse peaks (2,563 TFLOPS bf16) are excluded: no micro_perf op feeds a
sparse operand. Three things follow:

- **int8 and fp4 report no MFU at all**, because AWS publishes no peak for them.
  Quote TOPS and leave MFU blank rather than borrowing the fp8 denominator.
  `mxfloat8` *is* scored against the fp8 peak, since the repo's dtype table maps
  it onto `torch.float8_e4m3fn`.
- **A memory-bound op's MFU is correctly near zero and that is not a finding.**
  `add` at 1024x1024 is 0.5 FLOP/byte, so ~0.01% MFU is the ceiling for any
  accelerator. Read `mem_bw(GB/s)` instead.
- There is no clean per-device *bandwidth* denominator the way there is for
  TFLOPS: HBM is a per-chip resource shared by the four logical cores, so one
  core running alone can exceed a naive 1/4 share.

- **On eager, the reported MFU is end-to-end and the ~60 us dispatch floor is
  inside it.** A 1024x4096x4096 bf16 gemm needs 206 us at 166.75 TFLOPS and
  measures 282; charge 60 us to dispatch and the tensor engine is at ~93%, not
  the reported 73%. The gap closes as the shape grows — the 8192 gemm below loses
  under 3 points to it — so a mid-size gemm at 0.70-0.75 is a healthy number
  here, not a shortfall.

Cross-check on the denominator: a standalone probe of a 8192x4096x4096 bf16 gemm
on one logical core reached 143 TFLOPS, 86% of 166.75 — the right shape of number
for a large gemm, which is the reason to believe 166.75 rather than 667 or 83.375.

### Small-op latencies measure the runtime, not the chip

On eager, dispatch costs ~55-65 us per op and that is a floor. `add` at 48.8 us
and `softmax` at 53.8 us are therefore essentially *all* dispatch overhead, and
no smaller number is reachable on this stack. `gemm` at 1024x4096x4096 is the
smallest shape in the tables above where arithmetic clearly dominates. The
measurement behind this, and why `torch.compile` does not fix it, is in
[IMPLEMENTATION.md](IMPLEMENTATION.md#eager-dispatch-costs-55-65-us-per-op-and-that-is-the-floor).

### Sweep wall clock is not a performance comparison

`basic/tensor_gemm_ops/gemm.json` takes ~4,200 s here and ~403 s on one H100.
That 10x is **not** the chips, and it is not visible in any published number: it
is a per-shape warm-up on the eager runtime that lands entirely outside the timed
loop.

`tools/probe_case_overhead.py` replays the real `Backend.perf()` with its phases
instrumented. Per case, in ms, same three cases on both machines:

| | tensors | copies | iters | latency | wall | create | sync | calib | sleep | timed |
|---|---|---|---|---|---|---|---|---|---|---|
| Trn2 idx 0 | 64 M | 4 | 50 | 249 us | **3,223** | 145 | 1,386 | 1,479 | 200 | 14 |
| Trn2 idx 122 | 86 M | 4 | 50 | 832 us | **3,445** | 159 | 1,368 | 1,674 | 200 | 44 |
| Trn2 idx 489 | 188 M | 4 | 50 | 2,036 us | **6,535** | 1,533 | 1,444 | 3,252 | 200 | 107 |
| H100 idx 0 | 64 M | 15 | 10 | 28 us | **253** | 99 | 0 | 52 | 100 | 1 |
| H100 idx 122 | 86 M | 11 | 10 | 553 us | **263** | 123 | 0 | 29 | 100 | 7 |
| H100 idx 489 | 188 M | 5 | 10 | 351 us | **671** | 485 | 0 | 79 | 100 | 4 |

Mean per case: **5.25 s** here against **0.30 s** on the H100.

**The whole gap is a first-touch cost, keyed per shape and cached in-process.**
Repeat one case in the same process and it collapses:

```
idx 0   1st  wall 3,202   sync 1,367   calib 1,475
idx 0   2nd  wall   294   sync     1.2 calib     1.5
idx 0   3rd  wall   291   sync     1.1 calib     1.5
idx 122 1st  wall 3,431   ...
idx 0   4th  wall   293   <- another shape ran in between; still warm
idx 122 2nd  wall   342   <- so the cache is per shape, not last-shape-wins
```

11x, split about evenly between the first `device_synchronize()` after
`create_tensors` (~1.37 s, flat in tensor size) and the first executions of the op
(1.47 s at 64 MiB, 3.25 s at 188 MiB). The same effect exists on CUDA — cuBLAS
picking a kernel — and costs **52 ms**, three orders of magnitude less. With 856
distinct shapes in `gemm.json` and no reuse, 2.9 s x 648 measured cases is ~1,900 s
on its own, and weighting the larger shapes accounts for the rest of the 4,200 s.

**It is not `neuronx-cc`**, despite looking exactly like the compile described
under [A run that looks hung is usually
compiling](TROUBLESHOOTING.md#a-run-that-looks-hung-is-usually-compiling):

* host CPU over the case loop is **8-17% of one core** (the H100 side is 87%).
  A real `neuronx-cc` runs at ~199%. The host is blocked waiting, not compiling.
* it does not survive the process. A fresh container pays 3,204 ms for the same
  case, and mounting a volume at `/var/tmp` catches nothing written to disk.

So it is the runtime preparing and loading a kernel per shape onto the core, held
in process memory only.

Two things it is *not*, both checked because they were the obvious suspects. The
host-side tensor pipeline is the same on both machines — for a 128 MiB fp16 tensor,
CPU fp32 `randn` 272 ms on trn2.3xlarge vs 347 ms on p5.4xlarge, H2D 19.9 ms vs
20.8 ms (~12.5 GB/s both), on-device cast ~1 ms, `clone` <1 ms. And the policy
difference between the two backends is real but small: `backend_neuron.py:595-597`
gives each case a 1 s / 50-iteration timed loop where `core/backend.py:335-336`
gives it 50 ms / 10 iterations, and `:651` sleeps 0.2 s per case against 0.1 s —
together 1-6% of the wall clock here.

**And it runs the other way for small tensors.** `backend_neuron.py:619` caps
`max_data_cnt` at 4, added because a longer clone chain in the HLO makes
`neuronx-cc` take over five minutes. `core/backend.py:337-348` has no such cap: it
builds `floor(1 GiB / tensor_size)` copies, which for the 48-byte operand in
`moe_gating_ops.json`'s smallest case is **22,369,621** of them, to serve 16 op
executions. That one case takes **474 s** on the H100 with 0.02% of it on the GPU,
and it is why the same file is 123 s here against 1,191 s there. So neither
direction of a sweep-duration ratio is a chip comparison; see
[Tiny tensors cost minutes per case on this backend](../GPU/README.md#tiny-tensors-cost-minutes-per-case-on-this-backend).

The consequence for reading this document: latency and MFU rows are unaffected,
because the timed loop is warm by the time it runs. Sweep durations, and the
per-label elapsed times in [Reproduce one row at a
time](REPRODUCE.md#reproduce-one-row-at-a-time), are wall clock and mostly measure this
warm-up. Do not divide one by the other.

### Most `workloads/llm/` cases do not run

| Workload | Cases | Measured | Note |
|---|---|---|---|
| `vendor_test/flash_attention.json` | 588 | **0** | 420 rejected by the base op def, 168 by this backend (paged cache) |
| `vendor_test_demo/flash_attention.json` | 9 | **0** | same, plus 2 `attn_mode=decode` |
| `single_test_ops/fa_ops.json` | 11 | **0** | every case sets `block_size: 512`, i.e. a paged cache. `single_test_ops/fa_linear_ops.json` covers the same head configurations over a linear one |
| `vendor_test/quant_matmul.json` | 736 | 85, of 184 runnable | only the int8 quarter is runnable; the other 552 are `float8`/`mxfloat8`/`mxfloat4` and the base op def rejects them. A 5 h timeout cut the rest |
| `vendor_test/moe_quant_group_gemm.json` | 1380 | **15**, of 276 runnable | same dtype gate, but far worse per case: 15 in 4 h, ~16 min of wall clock each |
| `single_test_ops/ccl_ops.json` | — | **0** | asks for `world_size: 8`; a trn2.3xlarge has 4 logical cores |

That table is about cases this backend cannot run. The mirror-image gap — files
that *do* have a number here but none on the GPU, so no cross-backend ratio can be
quoted — is tabulated in
[`../GPU/README.md`](../GPU/README.md#what-this-table-does-not-cover-yet). The
collectives and `device2device` are hardware-blocked on a single-GPU box; five
files (`xccl_ops/device2host.json`, `xccl_ops/host2device.json`,
`single_test_ops/gemm_ops.json`, `moe_dispatch_ops.json`, `moe_combine_ops.json`)
are not, and are simply not run yet.

The XLA figures above come from `workloads/neuron_smoke/flash_attention.json`, the
only flash_attention workload in the repo that the NKI path can run: no
`block_size`, MHA `[8, 8, 128]`, all-bfloat16. `flash_fwd` takes one contiguous
K/V block, so only prefill is expressible, and it also needs MHA
(`q_head_num == kv_head_num`), `batch_size == 1` and `cache_len == 0`.

The eager `torch` provider is wider than that and no longer mirrors it. It needs
all-bfloat16 and a linear cache, but GQA, batched prefill and single-token decode
all run — `scaled_dot_product_attention(..., enable_gqa=True)` consumes the
unexpanded kv heads directly, so no `repeat_interleave` copy lands in the timed
region. Two case families stay rejected, and for correctness rather than
performance: PyTorch aligns `is_causal` to the **top-left** of a non-square score
matrix, while chunked prefill (`cache_len > 0` with `q_len > 1`) and multi-token
decode both need **bottom-right** alignment. Passing `is_causal=True` anyway does
not fail — it silently attends to the wrong keys.

### The quantized ops are a bf16 simulation, on every backend

`quant_matmul`, `moe_quant_group_gemm`, `moe_quant_group_gemm_combine` and
`quant_group_gemm_reduce_sum` all route through `fake_quant_gemm`
(`core/utils.py`), which casts the int8 operands **to bfloat16**, matmuls, then
scales in fp32. No int8 arithmetic happens anywhere. `grep -rl` across
`vendor_ops/` finds no vendor implementation of any of them, for NEURON or GPU,
so this is what every backend reports.

- Their TOPS figures describe `fake_quant_gemm`, not a quantized datapath.
  `quant_matmul` plateaus around 12-16 "TOPS", ~7-10% of the bf16 peak — about
  what a bf16 matmul with an int8 upcast on both operands and an fp32 scaling
  epilogue should cost.
- **`moe_quant_group_gemm` is worse than a simulation of the wrong dtype.** It is
  a Python `for` loop over experts whose slice bounds are read out of device
  tensors, so it syncs to the host once per expert and recompiles for each
  data-dependent shape. Latency is **2.67-2.84 s across every case measured**,
  1 token to 1024 at `ep_size=4` — a 1024x range with no trend, because none of
  the time is arithmetic (`calc_flops_power` goes 0.000 to 0.012 TFLOPS over the
  same span). Do not quote it as a MoE number for any accelerator. The
  recompilation is also why only 15 of 276 runnable cases finished in four hours:
  ~16 minutes of wall clock per 2.7-second measurement.

### The `*_dynamic_quant` family is 90-180x off, and it is one shared helper

Six ops in the norm/activation table are two orders of magnitude slower than the
unquantised op they wrap, on identical shapes:

| Quantising op | Its bandwidth | Unquantised sibling | Sibling's bandwidth | Ratio |
|---|---|---|---|---|
| `swiglu_dynamic_quant` | 1.6 GB/s | `swiglu` | 292.1 GB/s | 183x |
| `moe_swiglu_dynamic_quant` | 2.2 GB/s | `moe_swiglu` | 306.2 GB/s | 139x |
| `add_rms_norm_dynamic_quant` | 2.4 GB/s | `add_rms_norm` | 269.6 GB/s | 112x |
| `head_rms_norm_dynamic_quant` | 1.1 GB/s | `head_rms_norm` (bf16) | 95.7 GB/s | 87x |

**This is not six problems, and it is not a Neuron problem.** All six —
`scale_dynamic_quant`, `add_rms_norm_dynamic_quant`,
`head_rms_norm_dynamic_quant`, `swiglu_dynamic_quant`,
`moe_swiglu_dynamic_quant` and `moe_scatter_dynamic_quant` — call one function,
`smooth_per_token_dynamic_quant` in `core/utils.py`, and it is written as a chain
of independent full-size tensor ops rather than as one fused pass. Reading it,
a `[T, H]` bfloat16 input produces **six full-size float32 temporaries**:

```
hidden_states.contiguous().view(...).to(torch.float32)   # T1  [T,H] fp32
smoothed_input = torch.mul(hidden_states, smooth_scale)  # T2  [T,H] fp32
smoothed_input.abs()                                     # T3  [T,H] fp32
torch.max(..., -1, keepdim=True)[0].reciprocal()         #     [T,1]
torch.mul(smoothed_input, per_token_scale)               # T4  [T,H] fp32
    .clamp(-max_dtype_val, max_dtype_val)                # T5  [T,H] fp32
    .round()                                             # T6  [T,H] fp32
    .type(dst_torch_dtype)                               # T7  [T,H] int8
```

Two costs follow, and they compound:

- **The declared I/O is not the traffic.** `mem_bw` is computed from the op def's
  `io_bytes`, i.e. what a fused kernel *would* move: 2 bytes in and 1 byte out per
  element, `3TH`. Summing the reads and writes of the chain above gives about
  `55TH` — roughly **18x** amplification, most of it because `.to(torch.float32)`
  doubles the working set on the first line and every later stage pays it.
- **Seven dispatches instead of one.** Each stage is a separate launch, and on the
  eager runtime a launch has a ~60 us floor (see
  [Small-op latencies](#small-op-latencies-measure-the-runtime-not-the-chip)).
  That is a ~420 us floor before any arithmetic.

18x of traffic and 7x of launches do not multiply out to 87-183x on their own, so
some of the remainder is lowering quality on top. That split has **not** been
measured stage by stage — the decomposition above is read off the source, and the
only measured quantity is the end-to-end ratio in the table.

The practical consequences:

- **Do not quote these six as quantisation performance for any accelerator.** They
  measure a helper. If you need a real per-token dynamic-quant number, fuse the
  helper first.
- **The helper is most of the problem but not all of it — that is now measured,
  not assumed.** An earlier version of this section claimed the six "would be
  similarly bad on a GPU"; the same code has since been run on an H100 SXM5, and
  "similarly" was too strong. There, the same helper reaches only **73.4-178.7
  GB/s, i.e. 2.2-5.3% of a 3.35 TB/s HBM peak**, and is 3-14x slower than the
  corresponding unquantised op on the same hardware — so it is genuinely bad on a
  well-served backend, and the shared algorithm is the dominant cause. But a
  logical NeuronCore gets 0.15-0.33% of *its* peak on the same shapes, which is
  another **4-23x down** (median per matched shape, by op). So roughly one order of
  magnitude is the helper and one is Neuron-specific lowering on top. Full table
  in [`../GPU/README.md`](../GPU/README.md#norm-activation-and-moe-ops).
- An earlier version of the helper multiplied `smooth_scale` by `per_token_scale`
  instead of `smoothed_input`, which broadcasts to the same shape and so never
  failed, but made the output independent of `hidden_states`. That is fixed. The
  fix reads a `[num_tokens, hidden_size]` operand where the buggy version read a
  `[1, hidden_size]` one, so it is strictly *more* work — but re-running all
  eleven affected ops measured a per-shape median of **0.98-1.01x**, i.e. the
  change is performance-neutral within noise. Every figure in this section was
  taken before the fix and remains valid as a performance number.
- **Precision here is one significant figure.** At 1-2 GB/s these ops take
  hundreds of ms per case and the run-to-run spread on the peak is about ±25%
  (the re-run above moved individual peaks by up to 20% with the median flat).
  Do not read a difference of 0.2 GB/s between two rows as real.

### fp8: the sweep measures the eager path, in a dtype this chip cannot multiply

> **Superseded on SDK 2.33:** the eager fp8 path below returns numerically wrong
> results (rel err ~0.65) and the `torch` provider now rejects fp8. The analysis of
> *why* eager cannot reach the fp8 datapath still stands; the eager latencies do not.
> See [Re-validated on Neuron SDK 2.33](RESULTS.md#re-validated-on-neuron-sdk-233-official-dlami).

`workloads/basic/tensor_gemm_ops/gemm.json` has `float8_e4m3` and `float8_e5m2`
cases, and this backend's `gemm` accepts them (the base `GemmOp` gates `dtype` to
the four float formats, so no other backend reports them). They run:
`torch.matmul` on `torch.float8_e4m3fn` dispatches to `neuron:0` and returns an
fp8 tensor. **The result is an fp8 storage number, not an fp8 arithmetic number**,
and it lands two orders of magnitude below the 324.75 TFLOPS per-core fp8 peak.

> **This section used to conclude that Trainium2 reaches no fp8 datapath at all.
> That was too broad, and the correction is large enough to lead with.** The
> measurements below are all correct and all of the *eager* path. Compiled, and in
> the encoding the hardware actually implements, one logical core does **245.50
> TFLOPS at 75.6% of its fp8 peak — 1.82x its own bf16, on a nominal 1.95x bar.**
> The per-chip gap to the H100 is therefore about **1.56x**, not 99x. Two separate
> things had to be got wrong at once to see 1.2%, and both are worth understanding
> because they are traps for real workloads too:
> [see below](#the-two-things-that-produce-12).

The measurement that pins down why the eager number looks the way it does, at
4096x4096x4096 on one core:

| What | Latency |
|---|---|
| `matmul(a_bf16, b_bf16)` | 1,041.7 us |
| `matmul(a_e4m3, b_e4m3)` | 84,580 us |
| `matmul(a_e5m2, b_e5m2)` | 64,223 us |
| `matmul(a_e4m3.to(bf16), b_e4m3.to(bf16))` — the control | **49,761.9 us** |

The control is the whole story: casting the operands up to bfloat16 by hand,
outside any matmul, already costs 48× the bf16 matmul. So the fp8 lowering is
*cast up, then run a bf16 matmul*, and the cast dominates — nothing about the 84 ms
is a property of the fp8 engines. What that does *not* license is the conclusion
this section used to draw from it: the eager path reaches no fp8 datapath, the
hardware has one, and [the compiled path reaches
it](#the-two-things-that-produce-12). (Those four numbers come from a probe on a contended machine, so read
them as ratios. The clean figures are below.)

All 24 fp8 cases, on an idle core, MFU against the 324.75 TFLOPS per-core fp8
peak:

| K x N | M | e4m3 latency | e4m3 TFLOPS | e5m2 latency | e5m2 TFLOPS |
|---|---|---|---|---|---|
| 4096x4096 | 1024 | 32,567.8 us | 1.05 | 20,905.1 us | 1.64 |
| 4096x4096 | 4096 | 76,748.5 us | 1.79 | 58,183.0 us | 2.36 |
| 4096x4096 | 8192 | 133,869.5 us | 2.05 | 109,663.6 us | 2.51 |
| 4096x4096 | 16384 | 244,568.8 us | 2.25 | 195,988.7 us | 2.81 |
| 8192x1024 | 1024 | 26,495.7 us | 0.65 | 17,795.4 us | 0.96 |
| 8192x1024 | 4096 | 75,388.6 us | 0.91 | 51,755.6 us | 1.33 |
| 8192x1024 | 8192 | 142,229.6 us | 0.97 | 107,777.1 us | 1.27 |
| 8192x1024 | 16384 | 259,510.5 us | 1.06 | 180,802.7 us | 1.52 |
| 8192x8192 | 1024 | 139,329.7 us | 0.99 | 96,347.2 us | 1.43 |
| 8192x8192 | 4096 | 225,997.0 us | 2.43 | 171,495.6 us | 3.21 |
| 8192x8192 | 8192 | 346,166.1 us | 3.18 | 265,862.5 us | 4.14 |
| 8192x8192 | 16384 | **570,968.6 us** | **3.85** | **460,699.0 us** | **4.77** |

Three things in that table confirm the cast, not the matmul, is being measured.

- **MFU never leaves 0.2-1.5%.** The best fp8 case reaches 4.77 TFLOPS against
  149.42 for bf16 at the comparable 12288x8192x8192 — fp8 is **31x slower** on a
  format whose peak is 2x higher, so it is 63x off relative to its own ceiling.
- **e5m2 beats e4m3 by 20-35% at every one of the 12 shapes.** A `float8 ->
  bfloat16` conversion is exactly the kind of thing that would care: e5m2's 5-bit
  exponent and 2-bit mantissa map onto bf16's 8/7 with a shift and a mask, while
  e4m3 needs its 4-bit exponent rebiased and its subnormals renormalised. This
  bullet used to add "no tensor engine cares how a byte splits its exponent and
  mantissa", offered as proof that no engine was involved. Delete that: Trainium2's
  engines care a great deal, since they implement `f8e5m2` and `f8e4m3` but not the
  `f8e4m3fn` this row is measuring. The conversion reading is still the better one
  — a widening cost 20-35% apart is not an engine 20-35% apart — but it is now one
  reading among several rather than a proof.
- **Latency scales with elements, not with FLOPs.** Going from M=1024 to M=16384
  at 8192x8192 is 16x the arithmetic but only 4.1x the time, because the B operand
  (67 M elements) is re-cast once per call and dominates the small-M cases. That
  is why the TFLOPS column climbs monotonically with M instead of flattening: the
  fixed cast cost is being amortised, which is not how a compute-bound gemm
  behaves.

Practical consequence: `gemm.json` enumerates **856** cases once the fp8 block is
counted (`dtype` x `K.N` x `M` across both blocks), the fp8 ones are last, and
each takes 20-570 seconds. A watchdog sized for the float cases will cut the run
before it reaches them. Run fp8 as a separate workload file if you want these
numbers.

##### The two things that produce 1.2%

Both of these have to be true at once. Fixing either one alone does nothing, which
is why the eager table above is so uniformly flat across both formats.

**1. Trainium2's Tensor Engine cannot multiply `float8_e4m3fn`. That starts at
Trainium3.** The authoritative statement is the `nc_matmul` contract, identical in
the installed `nki` 0.6.0 docstring and in the [online
reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.isa.nc_matmul.html):

> The input `stationary` and `moving` tiles can be one of these supported data
> types: `float8_e4m3/float8_e5m2/bfloat16/float16/tfloat32/float32`. **Starting
> NeuronCore-v4, `float8_e4m3fn` (OCP FP8) is also supported**; `float8_e4m3`
> (legacy) and `float8_e4m3fn` cannot be mixed in the same matmul.

Trn2 is NeuronCore-v3; Trn3 is v4. So the matmul datapath here takes **legacy
`e4m3` and `e5m2`**, and OCP `e4m3fn` is a next-generation feature — which is
exactly what the compiler error below says in its own words. The `double_row`
performance mode repeats the split: v3 and v4 both restrict it to `float8_e4m3` or
`float8_e5m2`, "plus `float8_e4m3fn` on NeuronCore-v4".

Two things make this easy to get wrong, and both misled an earlier revision of this
section into claiming Trn2 *does* support e4m3fn:

* **The dtype symbol exists in our image.** `nki.language` (standalone `nki`
  0.6.0, which the eager image ships *alongside* the older
  `neuronxcc.nki.language`) exposes `float8_e4m3fn`, `float8_e4m3fn_x4`,
  `float8_e5m2_x4`, `float8_e8m0fnu` and `float4_e2m1fn_x4`, where
  `neuronxcc.nki.language` has only `float8_e4m3` and `float8_e5m2`. NKI is one
  language across v2/v3/v4, so **a dtype being nameable says nothing about which
  generation can multiply it** — `nc_matmul`'s contract is where that lives.
* **The [FP8 Quantize
  kernel](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/library/api/fp8-quantize.html)
  defaults to `dtype=nl.float8_e4m3fn` and its page is tagged "relevant for: Trn2,
  Trn3"** — and `nkilib.core.quantization` really is importable in this image with
  that default. But quantization is a scale-and-clip cast, not a matmul, so a Trn2
  tag there does not imply an e4m3fn matmul on Trn2.

What *is* true about the wider fp8 family, and worth knowing separately: [Data
Types](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/data-types.html)
describes NeuronCore v2's fp8 as "8-bit Floating point with configurable range and
precision (cFP8)" over *three* splits — `FP8_e5m2`, `FP8_e4m3` and `FP8_e3m4` —
with v3 supporting all of v2's types. So Trainium's fp8 is configurable, and `e3m4`
exists and is never exercised in this repo; the `e4m3` in that family is the legacy
encoding, not the OCP one.

PyTorch's `torch.float8_e4m3fn` is what CUDA uses and what `TORCH_DTYPE_MAPPING`
gives the workload's `float8_e4m3`, so the OCP encoding is the one the sweep asks
for, and the compiler refuses it by name:

```
[NCC_EVRF051] Data type F8E4M3FN is not supported on TRN1/TRN2. Target TRN3 or
later hardware, or use the --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 flag to cast
F8E4M3FN to F8E4M3.
```

Read that flag's name carefully — `...-fp8e4m3fn-as-fp8e4m3` **casts** the OCP
encoding into the legacy one so the v3 engine can multiply it, and it is labelled
`unsafe` because the two differ in infinities, NaN set and range. It is a numerics
compromise, not a hidden native path. It also does not exist in the `neuronx-cc`
these numbers were taken with (2.27.2878.0 — `compile --help` has no fp8 options at
all), so **the workload's `float8_e4m3` has no route to the tensor engines here by
any means**. `float8_e5m2` has no such split and is supported on v3 directly, which
is why the compiled measurement below is quoted in e5m2.

**A newer SDK is not expected to change this**, unlike most version-scoped claims in
this document: the 2.32 `nc_matmul` reference states the v4 boundary in the same
words as the 0.6.0 docstring shipped here. What a newer SDK could add is the
`--experimental-unsafe-...` cast route, which would give an e4m3fn *interface* on
top of legacy-e4m3 arithmetic rather than an e4m3fn datapath.

**2. The eager path has no fp8 gemm lowering for either format.** This is the part
the cast measurement above was actually detecting, and it applies to e5m2 as much
as to e4m3fn. Under `torch.compile(backend="neuron")` the e5m2 case reaches the
engines; eager never does.

| Shape | Dtype | Eager | Compiled | % of own peak, compiled |
|---|---|---|---|---|
| 2048³ | bf16 | 192.6 us / 89.18 TF | 187.3 us / 91.71 TF | 55.0% of 166.75 |
| 2048³ | e5m2 | 9,274.9 us / 1.85 TF | **133.5 us / 128.71 TF** | 39.6% of 324.75 |
| 2048³ | e4m3fn | 16,115.5 us / 1.07 TF | *compile error* | — |
| 4096³ | bf16 | 1,031.4 us / 133.26 TF | 1,017.0 us / 135.15 TF | 81.0% of 166.75 |
| 4096³ | e5m2 | 64,782.1 us / 2.12 TF | **559.8 us / 245.50 TF** | **75.6% of 324.75** |
| 4096³ | e4m3fn | 95,637.2 us / 1.44 TF | *compile error* | — |

At 4096³ that is **115.7x** from eager to compiled, and **1.82x** over the same
core's bf16 where the nominal fp8/bf16 headroom is 1.95x. Both dtypes reach ~76-81%
of their respective peaks, which is what a working datapath looks like. **The x4 to
a per-chip figure holds here**, unlike for the eager fp8 rows: four concurrent
single-core runs of the compiled 4096³ e5m2 case give 610.7 / 613.3 / 619.2 / 622.5
us against 611.6 us alone, a worst case of 1.018x. So the honest per-chip fp8
number for Trainium2 is **982 TFLOPS**, against the H100's measured 1,527.85 — a
**1.56x** gap where the nominal fp8 peak ratio is 1.52x. There is no fp8 anomaly
left to explain.

Reproduce with `tools/probe_fp8_datapath.py`. Two things it has to do that are easy
to get wrong: `torch.compile(..., dynamic=False)`, because compiling the same
function body at a second shape otherwise specialises it dynamically and the Neuron
compiler rejects `bf16[?,?]` outright; and the failing e4m3fn compile has to run
**last**, because it latches an error that the next device op inherits — a
`torch.randn` three lines later fails with the e4m3fn message still attached.

**What is *not* claimed here.** `gemm.json` is unchanged and its published rows
are still eager, so the table above stands as what a default sweep measures. The
compiled number is reachable from the harness now, but through a separate provider
and a separate workload file — see [Measuring the compiled fp8 path from the
sweep](#measuring-the-compiled-fp8-path-from-the-sweep) below. And the two
sides' fp8 numbers are necessarily in different encodings: the H100's is e4m3fn,
Trainium2's Tensor Engine does not take e4m3fn at all, and Trainium2's has to be
quoted in e5m2 — which `torch._scaled_mm` in turn rejects on CUDA
([details](../GPU/README.md#gemm)). **So the two chips have no fp8 format in common
that both stacks will multiply, and this is a generation fact rather than an SDK
one** — it would take a Trainium3 (NeuronCore-v4) on the Neuron side, or an `e5m2 x
e5m2` path on the CUDA side, to make the fp8 row like-for-like. Worth knowing before
planning a mixed fleet.

One accuracy point that no TFLOPS figure shows: e5m2 has **2 mantissa bits** where
e4m3fn has 3. Real fp8 inference generally wants e4m3 for weights, so 982 TFLOPS
per chip is the throughput of the format that is available on 2.27, not of a
configuration you would necessarily ship. The `--auto-cast-type` route and the cFP8
`e3m4` split (documented for NeuronCore v2/v3, and never exercised in this repo)
are the obvious things to look at if accuracy is the binding constraint.

##### Measuring the compiled fp8 path from the sweep

`tools/probe_fp8_datapath.py` answers "can this chip do fp8" but it is a standalone
script: it does not go through an op def, so it produces no `mfu`, no report file
and nothing an `analyze_sweep.py` run can pick up. The `torch_compile` provider does
the same thing inside the harness.

```bash
sudo docker run --rm --device /dev/neuron0 \
  -e XPU_PERF_NEURON_GEMM_COMPILE=1 -e PYTHONPATH=/w/src \
  -v "$HOME/xpu-perf":/w -w /w/projects/micro_perf xpu-perf-eager:latest \
  python3 -u launch.py --backend NEURON --device 0 --report_dir /tmp/fp8 \
    --workload vendor_ops/NEURON/workloads/gemm_fp8_compiled.json
```

That gives **both** providers on every case, because `gemm` now has two registered
vendor implementations — so each shape reports an eager row under `torch` and a
compiled row under `torch_compile`, and the ratio between them is in one log.
Measured 2026-09-03 on one logical NeuronCore of a trn2.3xlarge, all 12 cases of
that file (fp8 peak 324.75 TF/core):

| M | K | N | eager us | compiled us | compiled TFLOPS | compiled MFU | speed-up |
|---|---|---|---|---|---|---|---|
| 1024 | 4096 | 4096 | 21,828.7 | 169.8 | 202.41 | 0.623 | 128.6x |
| 4096 | 4096 | 4096 | 68,351.6 | 532.4 | 258.15 | 0.795 | 128.4x |
| 8192 | 4096 | 4096 | 114,418.1 | 1,004.9 | 273.53 | 0.842 | 113.9x |
| 16384 | 4096 | 4096 | 210,020.0 | 1,949.7 | **281.97** | **0.868** | 107.7x |
| 1024 | 8192 | 1024 | 20,084.3 | 116.2 | 147.83 | 0.455 | 172.8x |
| 4096 | 8192 | 1024 | 55,351.1 | 328.8 | 209.00 | 0.644 | 168.4x |
| 8192 | 8192 | 1024 | 103,966.8 | 603.8 | 227.64 | 0.701 | 172.2x |
| 16384 | 8192 | 1024 | 199,293.2 | 1,167.0 | 235.54 | 0.725 | 170.8x |
| 1024 | 8192 | 8192 | 103,663.1 | 542.4 | 253.41 | 0.780 | 191.1x |
| 4096 | 8192 | 8192 | 186,857.2 | 2,349.6 | 233.98 | 0.721 | 79.5x |
| 8192 | 8192 | 8192 | 306,586.5 | 4,605.6 | 238.73 | 0.735 | 66.6x |
| 16384 | 8192 | 8192 | 488,241.8 | 9,093.1 | 241.84 | 0.745 | 53.7x |

The best point is **281.97 TFLOPS = 86.8% of the per-core fp8 peak** at
16384x4096x4096 — ten points above the 75.6% the probe reports, because the probe
only runs square shapes and 4096³ is not the best one. Read this table as the
single-core answer: the x4 to a per-chip figure is
[measured](#the-two-things-that-produce-12) for the 4096³ square case only, so the
982 TFLOPS/chip figure quoted above still rests on that case rather than on this
86.8% one, whose x4 is unmeasured.

Four details, each of which is a trap if you write your own version:

* **`XPU_PERF_NEURON_GEMM_COMPILE=1` is required**, and without it the provider does
  not register at all. That is deliberate: a registered provider is instantiated
  once per case, and `gemm.json`'s 848 non-fp8 cases would each print a rejection
  traceback. Opt-in keeps a default sweep byte-identical to the published one.
* **The workload has to set an explicit non-fp8 `dst_dtype`.** `nc_matmul` writes an
  fp32 `dst` on NeuronCore-v3, so something has to convert; the shipped file uses
  `bfloat16`, which is what an fp8 inference path emits. Leaving `dst_dtype` to
  default to `dtype` would make `write_bytes` claim 1 byte per element while the
  graph emits 2, so the provider rejects that case rather than reporting it.
* **Only `float8_e5m2` is accepted.** `float8_e4m3` maps to `torch.float8_e4m3fn`,
  which is the OCP encoding this chip has no matmul datapath for, so the provider
  rejects it with the reason rather than letting neuronx-cc fail mid-sweep.
* **`torch._dynamo.reset()` per case.** dynamo's default `cache_size_limit` is 8 and
  this file has 12 shapes, so without the reset the 9th shape onward would silently
  fall back to eager — reporting ~1.5 TFLOPS under a provider named
  `torch_compile`, which is the worst possible failure mode here.

Cost: the first of `perf()`'s two warm-up iterations pays a full neuronx-cc compile
for that shape, which is minutes per case rather than the ~3 s of the [eager
per-shape warm-up](TROUBLESHOOTING.md#a-sweep-is-taking-all-day-but-nothing-is-hung). Size any
watchdog accordingly.

##### `torch._scaled_mm` is not usable on this backend

`torch._scaled_mm` is the API that would express a real fp8 gemm — fp8 operands,
fp32 scales, bf16 accumulate — and it is what the GPU provider is *forced* onto,
since `torch.matmul` raises on fp8 under CUDA. So the two providers are not making
the same call, and it is worth knowing what happens if you try to make them match.

It dispatches, and it returns the right shape and dtype. It is also unusable: at
512³ a steady-state call takes **690 ms**, about 5,500x the bf16 matmul at the same
shape, and at 2048³ a single call did not return in three and a half minutes of
wall clock while burning 19 minutes of CPU at 555% — which is where an apparently
hung probe with 0% Neuron utilisation and ~50% host CPU comes from. `aten::mm`,
`aten::matmul` and `aten::_scaled_mm` are all absent from `_NEURON_OPS_REGISTRY`,
and `TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1` does not fire, so this is
not the documented host-fallback path either. Whatever it is doing, it is not a
route to the engines — use `torch.compile` on a plain `matmul` instead, which is
what the numbers above do.

**MXFP8 is not expressible in this harness at all.** `TORCH_DTYPE_MAPPING`
(`core/utils.py`) aliases `mxfloat8`, `mxfloat8_e4m3` and `mxfloat8_e5m2` to plain
`torch.float8_e4m3fn` / `float8_e5m2` with `dtype_size` 1, and no op def carries a
block-scale tensor anywhere. Microscaling is a block exponent plus an element
mantissa; with no E8M0 scale tensor there is nothing distinct to measure. So
`ops/torch/gemm.py` accepts the three plain fp8 spellings and deliberately leaves
the `mx` aliases rejected, rather than republishing these numbers under a label
that promises microscaling.

### Two index ops are pathologically slow, and one is an int64 index away from parity

| Op | mem_bw | Note |
|---|---|---|
| `embedding` / `index_select` | 631 GB/s | full memory bandwidth for one logical core |
| `index_add` | 98 GB/s | |
| `gather` | **1.34 GB/s** | flat across a 256x size range; wedges the compiler at `dim_size=65536` |
| `scatter` | **0.8 GB/s** | wedges the compiler at `dim_size=4096` |

`gather` and `index_select` select rows from the same tensor and differ only in
how the index is expressed: `IndexSelectOp` passes a 1-D index of
`dst_batch_size` int64s, which lowers to a whole-row DMA, while `GatherOp` passes
an index the same shape as the output (`[dst_batch_size, dim_size]`, built with
`.view(N, 1).expand(N, dim_size)`). That much is right, and it is **not a hardware
property** — but the reason is narrower and far more actionable than "a 2-D index
lowers to per-element access", which is what this section used to say.

**The cause is the index dtype.** The device has no int64. An int64 index is
converted on the way into the graph by a
`nki_kernels/stride2_gather.stride2_flat_gather_kernel` custom call, and that
conversion *materialises* the index — which destroys the stride-0 broadcast that
`.expand()` created and that the compiler needs in order to see whole rows. Pass an
int32 index instead and the view survives into the graph as a direct argument, with
these results (`xpu-perf-eager:latest`, one logical core, `src_batch_size = 1024`,
outputs verified bit-identical against the int64 path):

| Op | dtype | `dim_size` | int64 index | int32 index | speedup | int32 GB/s |
|---|---|---|---|---|---|---|
| `gather` | fp32 | 1024 | 6,363.6 us | 83.2 us | 76x | 100.8 |
| `gather` | fp32 | 8192 | 49,714.8 us | 170.9 us | **291x** | 392.8 |
| `gather` | fp32 | 32768 | 199,866.1 us | 460.8 us | **434x** | **582.6** |
| `gather` | bf16 | 8192 | 49,715.7 us | 136.5 us | 364x | 245.9 |
| `gather` | bf16 | 32768 | 199,789.4 us | 273.3 us | **731x** | 491.1 |
| `scatter_add_` | fp32 | 8192 | 1,194,040.8 us | 737.4 us | **1,619x** | 91.0 |
| `scatter_add_` | bf16 | 8192 | 1,191,633.4 us | 479.4 us | **2,486x** | 70.0 |

At `dim_size` 32768 an int32 `gather` reaches 582.6 GB/s against `index_select`'s
631 GB/s ceiling — **92%**. The 449x shortfall in the H100 comparison collapses to
1.08x. The speedup grows with `dim_size` because the two paths have different
units: the int64 path is per-element and holds ~5.9 ns/element flat, while the int32
path is per-byte and scales.

Three corollaries worth keeping straight:

- **Materialising the view is as bad as int64.** `int32` plus an explicit
  `.contiguous()` measures 6,191.6 us against the view's 72.3 us at
  `dim_size` 1024. The stride-0 layout, not the dtype, is what the lowering reads;
  int32 matters only because it is what lets the layout survive.
- **1-D-index ops are unaffected**, which is why they were never slow:
  `index_select` moves 77.8 → 69.9 us and `index_add_` 486.7 → 496.5 us. There is
  nothing to materialise.
- **`scatter` is not fixable this way and its 621x is real.** `ScatterOp` calls
  `dst.scatter_()`, i.e. `aten::scatter_.src`, which has no NKI implementation at
  all; both dtypes measure ~11 ms (0.75 vs 0.76 GB/s). Only `scatter_add` has a
  kernel.

The op defs are *not* changed to exploit this, because `torch.gather` requires an
int64 index on CPU and CUDA and accepts int32 only here — switching would make the
op def non-portable, and switching per backend would mean the two sides of the H100
comparison no longer run the same op def. The published rows stay as the honest
measurement of what the op def does today; this table is what the hardware can do,
and the fix belongs either upstream or in model code.

There is a second, independent reason the NKI path is unreachable, worth knowing
before anyone concludes a missing kernel is the problem. `nki_kernels/gather.py`
does ship a kernel, `GatherNKIImpl` guards it with exactly the condition our
tensors satisfy (`index.ndim == 2 and index.stride(1) == 0`, verified returning
True), and calling it directly measures 124.8 us against the shipped path's
6,345.2 us — 50.8x, matching output. It never runs because
`@neuron_op("aten::gather", priority=60)` **does not take effect**: the priority is
applied by a subclass factory in `python_ops/auto_registration.py` that the
registered instances bypass, so every implementation reports the class default 50
and the tie falls to import order — and `python_ops/gather.py` imports
`GatherMLIRImpl` at module top before its own decorator runs. All seven
multi-implementation ops are affected; four resolve to the wrong impl
(`aten::contiguous` picks `ContiguousBroadcastMLIRImpl` over
`ContiguousEmptyHloImpl`'s declared 200, and `gather` / `scatter_add` /
`scatter_add_` all pick MLIR over NKI). The dispatcher logs nothing, because
`python_ops/base.py` only logs when an implementation *fails*, not when a
lower-priority one wins — so silence here is not evidence that a gate rejected the
case.

Two things this rules out. It is **not** a CPU fallback: with
`TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1`, which turns a failed Neuron
implementation into a hard error instead of a silent host fallback, the run
completes unchanged (6,324.4 us against 6,323.5 us) and prints nothing. And it is
not the index construction: on CUDA the identical op def reaches 2,805 GB/s.

Both ops eventually hang the compiler outright, so give any run including them a
hard timeout or exclude them by name (`parse_tasks` honours comma-separated
`--task` lists). One caveat on the numbers: `GatherOp` inherits `prepare_args`
from `IndexSelectOp`, which declares a 1-D index, while `GatherOp.create_tensors`
builds a 2-D one — so the declared `io_bytes` does not describe the tensors the op
creates, and `gather`'s `mem_bw` should be read as approximate either way.

### Writes into a strided slice view are not in-place

Two op defs write into a *strided slice view* of a large pre-allocated tensor and
count only the slice in `write_bytes`:

```python
dst_k_cache = k_cache[kv_slot_id, :, cache_start:cache_end, :]       # store_kv_cache.py:276
dst_k_cache.copy_(src_k_data)                                        # store_kv_cache.py:280
packed_qkv[t0:t1, qk0:qk1, d0:d1].copy_(rotate(...))                 # rotary_embedding.py:173
```

`store_kv_cache` documents itself as "This operator is inplace." On eager Neuron
that is not what happens. The check needs no profiler: hold the update size fixed
and scan the buffer size. True in-place is a flat line; a functionalised
copy-the-whole-buffer is linear. Fixed 256 KB update, `k_cache` 8 → 32 → 128 MB
(16x), bf16, one core:

| Written how | 8 MB | 32 MB | 128 MB | Slope |
|---|---|---|---|---|
| `k_cache[0,:,cs:ce,:].copy_(src)` — the op def's pattern | 0.171 | 0.234 | 0.556 ms | **3.25x** |
| `cache.add_(1.0)` — O(buffer) by design, positive control | 0.321 | 0.656 | 1.774 ms | 5.53x |
| read the slice, write nothing | 0.083 | 0.085 | 0.108 ms | 1.30x (flat) |
| the same `copy_` but at offset 0 | 0.105 | 0.172 | 0.493 ms | 4.68x |
| **2-D contiguous dst**, `flat[cs:ce].copy_(src)` | 0.063 | **0.043** | **0.043 ms** | **0.69x (flat)** |

Three readings:

- **The cost tracks the buffer, not the update.** The 128 MB point spends 0.556 ms
  writing 256 KB. Scored as a whole-buffer copy that is 256 MB / 0.556 ms ≈ **460
  GB/s**, 63% of the per-core ceiling and a plausible real copy; scored as the
  slice it is 0.46 GB/s. The 8 MB and 32 MB points come out at 94 and 273 GB/s,
  which is the same curve plus ~0.1 ms of fixed overhead. That overhead is also why
  the slope is 3.25x over a 16x range rather than 16x, and why `add_` — which is
  genuinely O(buffer) — only reaches 5.53x.
- **The trigger is a non-contiguous destination, not slicing and not the offset.**
  A 2-D contiguous slice write is flat at 43 us — the true in-place path exists.
  Moving the strided write to offset 0 does not help.
- **`torch.compile(backend="neuron")` does not rescue it at cache sizes that
  matter.** It helps at 8 MB (0.139 vs 0.171 ms), then fails to compile at 32 MB
  and 128 MB with `BackendCompilerFailed: RuntimeError: Neuron backend NEFF
  execution setup failed`. Establishing input/output aliasing through dynamo is the
  usual advice for this; on this stack it is not available for a realistic KV cache.

`store_kv_cache` itself never reaches the device — see [Known
unsupported](#known-unsupported) — so no published number here is affected by it.
`rotary_embedding` does run, and this costs it **2.08x on its write step**, which
is 15% of that op's time. It is not the dominant term there; see
[`rotary_embedding` is not a Neuron result at
all](RESULTS.md#rotary_embedding-is-not-a-neuron-result-at-all).

One thing this is *not* evidence for: a general eager in-place problem. `add_` on a
whole tensor is O(buffer) because it is supposed to be, and the contiguous slice
write is flat. It is specifically the strided destination view.

Reproduce with `tools/probe_inplace_write.py`.

### Known unsupported

Two different things are easy to conflate, so they are kept apart: a case the
*base op definition* never implemented, which no vendor can fix from a
`vendor_ops` directory, and a case this backend genuinely cannot run.

| Cases | Blocker | Whose limit |
|---|---|---|
| `flash_attention` with any quantized dtype | `op_defs/llm_ops/flash_attention.py` accepts all-bfloat16 or bfloat16 + `int8` cache, and raises on everything else | base op def |
| **`store_kv_cache`, all 16 cases in `pre_fa_ops.json`** | every case sets `block_size: 512`, so `get_attn_info` classifies the cache as paged and `op_defs/llm_ops/store_kv_cache.py:257` raises `NotImplementedError("StoreKVCacheOp paged cache not implemented yet.")` | base op def |
| **`store_kv_cache` with `store_mode: "k"`**, independently of the above | `vendor_impl` creates `v_cache` only for `store_mode in ("both", "v")`, but `vendor_impl_run` reads `tensor_mapping["v_cache"]` unconditionally at `store_kv_cache.py:248` → `KeyError: 'v_cache'`. So removing `block_size` is not enough to make the file run | base op def |
| `quant_matmul` / `moe_quant_group_gemm` with `float8` / `mxfloat8` / `mxfloat4` / `int4` weights | both base impls accept only `int8/int8/int8 -> bfloat16` | base op def |
| `flash_attention` with a paged cache (`block_size`) | neither the NKI `flash_fwd` kernel nor native SDPA takes a block table | this backend |
| `flash_attention` chunked prefill (`cache_len > 0`, `q_len > 1`) **on the `torch` provider** | `is_causal` is top-left aligned; these need bottom-right | this backend (correctness) |
| ~~`flash_attention` multi-token decode~~ | now covered by the **`nkilib`** provider, which builds the active mask explicitly instead of relying on `is_causal` ([table](RESULTS.md#decode-the-nkilib-provider-attention_tkg)). Still unsupported on `torch` | — |
| `flash_attention` GQA / decode / `batch_size > 1` on the **XLA** runtime | the NKI `flash_fwd` kernel takes one contiguous K/V block for one head group; eager SDPA runs all three | this backend |
| `flash_attention` prefill at `q_len: 32768` | SDPA does not complete at that length — see below | this backend (capacity) |
| `p2p` | **the native runtime's `torch.distributed` backend has no point-to-point at all**: `RuntimeError: Backend neuron does not support send` on the sender and `... does not support recv` on the receiver, at every size. Not a placement or capacity limit -- it is measured on a trn2.48xlarge with the two ranks on one chip, and the same launch runs `broadcast` and `all_reduce_h2d` fine | this backend (missing feature) |
| collectives above ~1 GiB per rank | the 24 GiB per-core ceiling above | this backend (capacity) |

Three collectives were in a fourth category until now -- registered in
`op_defs/basic_ops/xccl_ops.py` and referenced by **no workload JSON anywhere in the
repo**, so untested on every backend rather than unsupported on any. They now have
files, and running them found one hard limit and one repo-wide bug:
[`broadcast`, `p2p` and `all_reduce_h2d`](RESULTS.md#three-collectives-that-had-never-run).

`dequant_kv_cache` appears in **no workload JSON in the repo**, on any backend. It
is untested, not unsupported. Twelve other ops were in that position until this
port added `single_test_ops/norm_ops.json`, `activation_ops.json`,
`moe_gating_ops.json` and `quant_ops.json` for them — `add_rms_norm`,
`head_rms_norm`, `qk_rms_norm`, `swiglu`, `moe_swiglu`, `moe_softmax_topk`,
`scale_dynamic_quant`, `quant_group_gemm_reduce_sum` and the `_dynamic_quant`
variants. Conversely `moe_scatter_dynamic_quant`,
`moe_quant_group_gemm_combine`, `quant_matmul` and `moe_quant_group_gemm` all do
run on the eager stack (int8 only for the last two) — an earlier version of the
table claimed Neuron could not provide int8 tensors, which was true of the XLA
path and is not true here.

**Long-context prefill has no measurable path on eager, and it fails by not
finishing rather than by erroring.** `q_len: 32768` at 80 q-heads was in
`fa_linear_ops.json` and was removed: it ran for **55 minutes** on an idle core,
83% CPU on one host thread, `walrus_driver` long since exited (so not a compile),
and never produced a result or an error. The same provider does `q_len: 10240` in
40.0 ms. Note what this is *not*: 32768 is a multiple of 512 and `B*H` is 80, so
this case clears every clause of the NKI flash gate described in
[Attention](RESULTS.md#attention) — it is failing **inside** a fused kernel, not for want of
one, which is why there is nothing to fall back to. The case was dropped rather
than left in the file to look like a hung machine. If you need long-context
attention numbers on Trainium, take them on the XLA runtime through the NKI
provider, at MHA.

Only three ops need vendor code at all; everything else runs its `op_defs` base
implementation through the `base` provider:

| Op | Provider | Runtime | Why |
|---|---|---|---|
| `gemm` | `torch` | both | rejects `int8` (not lowered through `torch.matmul`) and, since SDK 2.33, fp8 (eager fp8 matmul is numerically wrong — use `torch_compile`); runs `tfloat32` through a per-case `torch.compile` graph with neuronx-cc TF32 auto-cast (eager runtime only) — see [Re-validated on Neuron SDK 2.33](RESULTS.md#re-validated-on-neuron-sdk-233-official-dlami) |
| `all_gather` | `torch` | both | base uses `dist.all_gather_into_tensor`, unimplemented on the `xla` backend; the `neuron` backend implements it, so eager reproduces the base behaviour |
| `flash_attention` | `nki` / `torch` | `xla` / `eager` | no base implementation exists; NKI `flash_fwd` on XLA, `scaled_dot_product_attention` on eager |

## What would close these gaps

Nothing in this list is speculative about *where* the time goes — each item names a
measured gap above and the specific thing that would address it. In rough order of
value per unit of work:

1. **`rmsnorm_quant` from [nki-library](https://github.com/aws-neuron/nki-library),
   for the six 3.8-38x quantised ops.** They all share one unfused Python helper
   (`../../op_defs`'s `_dynamic_quant` path — see
   [the `*_dynamic_quant` family is 90-180x off](#the-_dynamic_quant-family-is-90-180x-off-and-it-is-one-shared-helper)),
   and a single fused norm+quant kernel replaces the whole chain. This is the
   largest number of affected ops for the least new code, and it does not need a new
   op def.
2. ~~**`attention_tkg`, for decode**~~ — **done, 2026-09-03.** Decode cannot reach
   the built-in NKI flash rewrite (see [Attention](RESULTS.md#attention)) and was leaving
   2.6-5.7x of this core's bandwidth unused. `nkilib`'s token-generation kernel is
   now wired as the `nkilib` provider and takes the best decode row to 42% of
   bandwidth, 4.09x SDPA at kv_len 16,384 — but 0.55x at 4,096, so both providers
   stay registered and the crossover is the result
   ([table](RESULTS.md#decode-the-nkilib-provider-attention_tkg)). What is *not* done: the
   remaining 58% of bandwidth, and an H100 run of the aligned workload to restate
   the per-chip decode ratio. **Prefill is a different case and was already done:**
   the SDPA rewrite lowers to `nkilib`'s `attention_cte`, calling that kernel by
   hand reproduces the same latency to 0.97x, and the 3.1x per-chip gap to the
   H100 is therefore that kernel against cuDNN/FlashAttention. Writing a NKI
   prefill provider would re-derive the number the table already has.
3. **An int32 index, if a *downstream* workload owns its own indices.** `gather` is
   291-731x faster with one
   ([details](#two-index-ops-are-pathologically-slow-and-one-is-an-int64-index-away-from-parity)).
   This benchmark's op defs deliberately keep int64 for portability, so the fix
   belongs in real model code, not here — but it is the single largest measured
   speedup available on this stack, and it is free.
4. **An fp8 `gemm` provider that compiles, for fp8 — and `float8_e5m2` cases to
   point it at.** The published 99x is the eager path in an `e4m3fn` this chip
   cannot multiply; compiled `e5m2` reaches 75.6% of the fp8 peak and closes the
   per-chip gap to 1.56x
   ([details](#fp8-the-sweep-measures-the-eager-path-in-a-dtype-this-chip-cannot-multiply)).
   The work is a provider that wraps the matmul in
   `torch.compile(..., dynamic=False)`, not a new kernel. `matmul_mxfp8` from
   `nkilib.experimental` is the further step and still needs the op def extended to
   carry a block-scale tensor, since microscaling has no meaning without one — the
   largest piece of work in the list, and the only one that changes shared code.

Two known defects in `torch_neuronx` itself are recorded rather than worked around,
because working around them in `vendor_ops` would hide them: the dropped
`@neuron_op(priority=)` override that keeps four ops on a slower implementation, and
`aten::scatter_.src` having no NKI kernel. Both are described in
[the index-ops section](#two-index-ops-are-pathologically-slow-and-one-is-an-int64-index-away-from-parity).

Three further gaps are in the **base op defs**, so they are not Neuron gaps and
closing them would move both backends. They are listed here because it is easy to
read them off the Neuron column as if they were:

- **`rotate()` in `core/utils.py:587` is five-plus materialising passes.** It costs
  Neuron and an H100 the same 94% of their respective peaks — see
  [`rotary_embedding` is not a Neuron result at
  all](RESULTS.md#rotary_embedding-is-not-a-neuron-result-at-all). One fused kernel would be
  ~17x on both.
- **Writes into a strided slice view are not in-place**, worth 2.08x on
  `rotary_embedding`'s write step. Sizing `packed_qkv` to the heads actually written
  makes the destination contiguous; see [Writes into a strided slice view are not
  in-place](#writes-into-a-strided-slice-view-are-not-in-place).
- **`store_kv_cache` has no runnable case on any backend**, for two independent
  reasons in the op def. Both are in [Known unsupported](#known-unsupported). Until
  they are fixed there is no `store_kv_cache` number to compare, on this backend or
  the GPU.

One gap is in the **Neuron `torch.distributed` backend** and cannot be closed from
this repo: it has no `send`/`recv`, so `p2p` — the only op here that measures a
single one-way transfer over a single link — raises `RuntimeError: Backend neuron
does not support send` at every size, on one chip or across the torus. Until a
Neuron release adds it, there is no point-to-point Trainium number to put against
NVLink, and the collectives are the closest available proxy. See
[Three collectives that had never run](RESULTS.md#three-collectives-that-had-never-run); the
`p2p_hop*` labels in `run_multicard_sweep.sh` exist to detect the day this changes.

Three things are **runnable and simply unrun**, which is a different kind of gap
from all of the above and the only kind that needs no code:

- **Multi-node collectives.** `launch.py` already takes `--node_world_size`,
  `--node_rank` and `--master_addr`; `core/backend.py:491-506` derives the global
  rank and world size from them and `core/engine.py:88-98` exports
  `MASTER_ADDR`/`RANK`/`WORLD_SIZE`. The path has never been exercised, on either
  backend. Two trn2.48xlarge boxes make `world_size` 128 reachable, and the
  interesting number is not the bandwidth but the *shape* of the drop from 64 to
  128 — 64 ranks are one torus, 128 are two torii and an EFA fabric, so this is
  where the `bus_bw` curve stops being a NeuronLink curve.
- **`host2device` at 8 ranks**, the missing control for `all_reduce_h2d`.
- **`gemm_fp8_compiled.json`** is in `vendor_ops/NEURON/workloads/` and in no sweep
  label, so the 245.50 TF / 75.6% MFU fp8 result above is documented but not
  reproducible by running a script. It needs an opt-in label, not a measurement.

And one whole directory is a Neuron gap of a different scale: `projects/infer_perf/`
has a `llm_perf/backends/GPU` and nothing else. Every number in this file is an
operator number; there is no end-to-end LLM serving comparison on Trainium in this
repo at all. `projects/train_perf` is a Megatron-LM checkout (CUDA), `trace_gen` is
documentation with a download script, and `xpu_oj` is an `llm_sim` — none of the
three has been looked at on this backend.

