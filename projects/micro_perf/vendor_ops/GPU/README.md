# GPU backend — reference numbers, and a Trainium2 comparison

Measured on a **p5.4xlarge** (1x H100 SXM5 80GB HBM3), 2026-09-02, against the
same workload files and the same provider code as the Trainium2 numbers in
[`../NEURON/README.md`](../NEURON/README.md), so the two are comparable case for
case.

## Summary

Normalised **per chip** — one H100 against one Trainium2, not against the quarter
of one that a logical NeuronCore is — the gap is not one number, it is three very
different ones. This first table is the **measured** set; it is not the whole
workload tree, and what is missing is listed with the reason
[below](#what-this-table-does-not-cover-yet):

| Workload | H100 / Trainium2, per chip | What that is |
|---|---|---|
| dense bf16 `gemm` | **1.35x** | silicon (nominal peak ratio is 1.48x) |
| elementwise / reductions | **~1.4x** | parity: 12 of 24 ops within 0.93-1.15x of the H100's %-of-own-peak |
| head/QK norms, `swiglu` | **0.35-1.26x** | Trainium2 *ahead* on 6 of 8 rows |
| attention prefill | **3.1x** | ~1.5x silicon, ~2.1x software — and the software is `nkilib`'s `attention_cte`, confirmed by identity, so there is no better kernel to reach for |
| `gelu`, `sin`/`cos`, `reduce_max` | **3.5-7.7x** | single-op lowering gaps, each with a fast sibling op |
| the 7 quantised ops | **3.8-38x** | on top of a shared unfused helper that costs the H100 3-14x too |
| `gemm` at fp8 | **~99x** as published, **~1.56x** measured | the published row is the eager path in an e4m3 encoding Trainium2 does not implement; compiled, in `e5m2`, it reaches 75.6% of its fp8 peak |
| `gather` / `scatter` | **449x / 621x** | the op def is exonerated; `gather` is one index dtype away from **1.08x**, `scatter` has no kernel |
| `topk`, `moe_softmax_topk` | **0.33x / 0.27x** | Trainium2 ~3x ahead per chip, and the x4 is now measured at these shapes |
| `gemm` at fp32 | **0.32x** | Trainium2 3.09x ahead; the nominal bar favours it 2.70x |
| `rotary_embedding` | **1.15x** | exactly the memory-bound bar — both backends get 5.9% of their own peak, so the missing 94% is the op def |

So the headline is that **Trainium2's silicon is fine and the gap is bimodal in
its software.** Nothing here lands between 1.4x and 3.1x, and nothing between
7.7x and 99x: ops are either at parity or off by one to three orders of
magnitude, which is the signature of missing kernels rather than slow hardware.
On bf16 gemm it delivers 90% of its own peak against the H100's 82% and lands
within 1.35x per chip — [and the four-core run confirms that x4 is
real](#how-to-compare-these-to-the-trainium2-numbers). On attention it delivers
32% against the H100's 69% — and that 32% is *already* a fused NKI flash kernel,
[measured by turning it off](#attention), so the remaining 2.1x is kernel quality
rather than an absent kernel. On fp8 the published row is not a hardware
number on either count: it is the eager path, in an `e4m3` encoding Trainium2 does
not implement. Compiled, in `e5m2`, one logical core does **245.50 TFLOPS at 75.6%
of its fp8 peak**, which puts the per-chip gap at **1.56x** against a 1.52x nominal
bar — [details](../NEURON/README.md#fp8-the-sweep-measures-the-eager-path-and-the-wrong-e4m3-encoding).
The 99x is real as a measurement of what the sweep currently runs, and it is not a
statement about the chip.

The `gather` row is the clearest case of the pattern, because it is now known
exactly what the two orders of magnitude are: the op def hands the device an
**int64** index, the device has no int64, and converting it materialises a
stride-0 broadcast that the fast lowering needs to see intact. The same call with
an int32 index is 291-731x faster and lands within 8% of `index_select` —
[details](../NEURON/README.md#two-index-ops-are-pathologically-slow-and-one-is-an-int64-index-away-from-parity).

Several results cut against the pattern and are worth knowing. Trainium2 uses
*more* of its own bandwidth than an H100 does on the head/QK norms and `swiglu`, it
is ~3x faster per chip at both selection ops, and it is 3.09x faster at fp32
`gemm` — so this is not a uniform deficit. And the H100
column repeatedly exonerates the benchmark: `gelu` matches `silu` there, `gather`
matches `index_select` there, and `head_rms_norm` is broken on *both* backends by
a hard-coded fp32 norm weight in the op def.

`rotary_embedding` is the cleanest of those exonerations, and the only case in this
comparison where the two backends agree to a significant figure: **5.89% of the
H100's own HBM peak against 5.90% of Trainium2's**, at 197.35 and 42.77 GB/s. Two
chips whose bandwidths differ by 4.62x cannot land on the same fraction of peak by
accident — the 94% that is missing is in the op def, which spends 78% of its time
in `rotate()` doing five-plus materialising elementwise passes where a fused kernel
would do one. Per chip that is 197.4 against 171.2, **1.15x**, exactly the
memory-bound bar. This row was previously blamed on Neuron's slow bf16 `sin`/`cos`;
that was wrong, since `cos` and `sin` are precomputed outside the timed region —
see [the NEURON README](../NEURON/README.md#rotary_embedding-is-not-a-neuron-result-at-all)
for the decomposition.

Memory-bound attention still tells the original story. During decode the H100
reads its KV cache at 80-86% of HBM peak, Trainium at 15-33% of its own — and
since the same Neuron core reaches 631 GB/s on a plain `index_select`, that is a
lowering result rather than a bandwidth wall.

Every number below carries the unit caveats in
[How to compare](#how-to-compare-these-to-the-trainium2-numbers). Read them before
quoting anything.

### What this table does not cover yet

Twelve workload files have a Trainium2 number and **no GPU column**, so no ratio
can be quoted for them: the seven in `xccl_ops/`, `gemm_ops.json`,
`moe_dispatch_ops.json`, `moe_combine_ops.json`, and two of the `vendor_test/`
files. `tools/run_comparison_sweep.sh` runs none of them. The remaining rows are
a different problem — a case no backend can run, or one only a single backend can —
and are marked as such. Most of this is a hardware limit, since a p5.4xlarge has
one GPU. **Exactly three files are runnable here today and simply unrun**:
`gemm_ops.json`, `moe_dispatch_ops.json` and `moe_combine_ops.json`. That is worth
being explicit about rather than leaving the table above looking complete.

| Workload | Ops | What Trainium2 got | Blocker | TODO |
|---|---|---|---|---|
| `xccl_ops/{all_reduce,all_gather,reduce_scatter,all_to_all}.json` | 4 collectives | best `bus_bw` at ws=4: `all_gather` 125.1, `all_reduce` 107.1, `reduce_scatter` 101.7, `all_to_all` 54.2 GB/s ([table](../NEURON/README.md#memory-bound-ops)) | **hardware** — `world_size` is 1 on one GPU | needs a p5.48xlarge (8x H100, NVLink). **The highest-value item in this list**: NVLink vs NeuronLink is the one remaining difference likely to matter at training scale, and nothing here constrains it |
| `single_test_ops/ccl_ops.json` | same 4 | **0 cases** — the file asks for `world_size: 8` and a trn2.3xlarge has 4 logical cores | **hardware**, both sides | needs 8 of either; on the Neuron side a trn2.48xlarge |
| `xccl_ops/device2device.json` | `device2device` | 648.7 GB/s (bf16) | **hardware** — needs two devices | same box. Note this op reads its operands exactly once, which is why it tops the Neuron memory-bound table |
| `xccl_ops/{device2host,host2device}.json` | 2 | 14.3 / 14.0 GB/s at ws=2, and the ws=4 column is not comparable (1 GiB cap, four ranks sharing one host DMA path) | **harness**, not hardware — both ops are registered on `XCCLEngine` (`op_defs/basic_ops/xccl_ops.py:569,627`), and `perf_engine.py:173` skips that engine outright when `len(device_ids) * node_world_size <= 1`. So although both files do list a `world_size: 1` case, a one-GPU launch dispatches to an engine that was never started and writes `note: engine_not_started` for every case (`perf_engine.py:352-364`) — exit 0, nothing measured. `XPU_PERF_ENGINES=XCCLEngine` does not help: the env filter is applied *before* that guard, not instead of it | needs any box with ≥2 GPUs — unlike the collectives it does not need NVLink or all eight. An earlier version of this row claimed these ran here as-is, on the strength of the `world_size: 1` entry in the JSON alone; they do not. Still worth having, as the one number in the comparison governed by neither chip's HBM but by a host link, so 14.3 GB/s currently has nothing to be read against |
| `single_test_ops/gemm_ops.json` | `moe_gating_gemm`, `quant_matmul`, `moe_quant_group_gemm` | `moe_gating_gemm` 21.89 TF / **48% MFU**; `quant_matmul` 15.78 "TOPS"; `moe_quant_group_gemm` **2.67-2.84 s at every shape** | **none** | run it. `moe_gating_gemm` is the interesting one — the only genuine compute-bound row outside plain `gemm`, at 48% of the fp32 peak against `gemm`'s 82% at the same dtype, with no GPU control to say which of those two numbers is the anomaly. The other two route through `fake_quant_gemm` and are [a bf16 simulation on every backend](../NEURON/README.md#the-quantized-ops-are-a-bf16-simulation-on-every-backend), so a GPU column there measures the same shared helper the `_dynamic_quant` rows already exposed |
| `single_test_ops/moe_dispatch_ops.json` | `moe_scatter_dynamic_quant` | 0.7 GB/s | **none** | run it. At 0.7 GB/s it is in `gather`/`scatter` territory, and the GPU column is what would say whether that is the op def or the lowering — exactly the question the H100 settled for `gather` |
| `single_test_ops/moe_combine_ops.json` | `moe_gather`, `moe_quant_group_gemm_combine` | `moe_gather` 9.2 GB/s | **none** | run it, same reason — `moe_gather` at 9.2 GB/s is 68x off `index_select` on the same chip |
| `single_test_ops/fa_ops.json` | paged `flash_attention` | **0 cases** — every case sets `block_size: 512` and no Neuron provider takes a block table | **the Neuron side**, not this one | the GPU can fill it in under `MODE=docker` (`flash_attn` does take a block table) and the sweep script already calls it, but there would be nothing to compare it against |
| `single_test_ops/pre_fa_ops.json` → `store_kv_cache` | 1 | **0 cases on either backend** | **base op def** — `store_kv_cache.py:257` raises on a paged cache and `:248` hits `KeyError: 'v_cache'` for `store_mode: "k"` | fix the op def first; this is shared code, so it is not a Trainium item |
| `vendor_test/quant_matmul.json`, `vendor_test/moe_quant_group_gemm.json` | 2 | 85 of 184 runnable, and **15 of 276** — ~16 min of wall clock per 2.7 s measurement | **none**, but the cost is prohibitive | not worth a GPU run until `moe_quant_group_gemm`'s per-expert host sync is fixed; `single_test_ops/gemm_ops.json` covers the same ops cheaply |
| `vendor_test{,_demo}/flash_attention.json` | 1 | **0 cases** (paged, plus `attn_mode=decode`) | same as `fa_ops.json` | subsumed by it |

One further gap belongs to neither backend: **`dequant_kv_cache` appears in no
workload JSON anywhere in the repo**, so it is untested rather than unsupported on
both. Twelve ops were in that position until this port added
`single_test_ops/{norm_ops,activation_ops,moe_gating_ops,quant_ops}.json`; this is
the thirteenth.

## Quick start

The DLAMI already ships a working CUDA PyTorch at `/opt/pytorch`, and building an
image on top of it buys nothing unless you need `flash_attn` specifically —
`vllm` 0.28.0 and `flashinfer` 0.6.16 are already installed there, so their extra
`rms_norm` providers register under `MODE=host` as-is. (That is also how the
[SDPA backend hazard](#attention) was found: `vllm` being importable is enough to
reconfigure attention.) Two deps *are* missing, and installing them into an
overlay avoids mutating an environment you did not create:

```bash
pip install --target ~/xpudeps jsonlines prettytable

cd projects/micro_perf
PYTHONPATH=$HOME/xpudeps:$(git rev-parse --show-toplevel)/src \
  /opt/pytorch/bin/python -u launch.py --backend GPU --device 0 \
    --report_dir /tmp/out --workload workloads/basic/tensor_gemm_ops/gemm.json
```

The full sweep, both modes:

```bash
cd projects/micro_perf
RESULTS=/tmp/gpu_results vendor_ops/GPU/tools/run_comparison_sweep.sh \
  2>&1 | tee /tmp/gpu_sweep.log

# or, for the flash_attn / flashinfer / vllm providers:
MODE=docker RESULTS=/tmp/gpu_results vendor_ops/GPU/tools/run_comparison_sweep.sh ...
```

`MODE=docker` needs `tools/Dockerfile.cuda` built first. It exists for one
specific reason: `op_defs/llm_ops/flash_attention.py` has **no base
implementation**, so without a fused-attention package installed every case in a
paged-attention workload is skipped and the launch **exits 0 having measured
nothing**. The Dockerfile therefore asserts `import flash_attn` at build time
rather than treating it as optional.

## Reproduce one row at a time

Checking one figure in the tables below does not need the sweep. Every row came
from exactly one `launch.py` call, and each of those calls has a label:

```bash
cd projects/micro_perf

LIST=1 vendor_ops/GPU/tools/run_comparison_sweep.sh    # print all 13; runs nothing
ONLY=gemm vendor_ops/GPU/tools/run_comparison_sweep.sh 2>&1 | tee /tmp/one.log
ONLY=single_norm_ops,single_quant_ops vendor_ops/GPU/tools/run_comparison_sweep.sh
```

`ONLY` takes commas or spaces, matches whole labels only (`ONLY=gemm` does not
also select `single_gemm_ops`), and changes nothing else: the log format and the
`$RESULTS/<label>/` layout are identical to a full run, so
`vendor_ops/NEURON/tools/analyze_sweep.py` reads a one-label log the same way.

**Every label here is single-GPU.** The elapsed times are what the published run
actually took on an idle p5.4xlarge, not the watchdog budgets in the script:

| Label | Workload | Elapsed | Covers |
|---|---|---|---|
| `gemm` | `basic/tensor_gemm_ops/gemm.json` | 403 s | [gemm](#gemm), including the 24 fp8 cases |
| `basic_vector_linear_ops` | `basic/vector_linear_ops` (all) | 175 s | [memory-bound](#memory-bound-ops) |
| `basic_vector_activation_ops` | `basic/vector_activation_ops` (all) | 64 s | " |
| `basic_vector_norm_ops` | `basic/vector_norm_ops` (all) | 51 s | " |
| `basic_vector_reduction_ops` | `basic/vector_reduction_ops` (all) | 84 s | " |
| `basic_vector_sfu_ops` | `basic/vector_sfu_ops` (all) | 289 s | " |
| `basic_vector_index_ops` | `basic/vector_index_ops` (all) | 62 s | `gather`/`scatter`, the [two ops Neuron is 100x off on](../NEURON/README.md#two-index-ops-are-pathologically-slow-and-one-is-an-int64-index-away-from-parity) |
| `single_norm_ops` | `llm/single_test_ops/norm_ops.json` | 84 s | [norm/activation/MoE](#norm-activation-and-moe-ops) |
| `single_activation_ops` | `llm/single_test_ops/activation_ops.json` | 52 s | " |
| `single_moe_gating_ops` | `llm/single_test_ops/moe_gating_ops.json` | **1,191 s** | " — the longest run here, and 9.7x the same file on one NeuronCore (123 s) |
| `single_quant_ops` | `llm/single_test_ops/quant_ops.json` | 17 s | " |
| `single_fa_linear_ops` | `llm/single_test_ops/fa_linear_ops.json` | 15 s | [attention](#attention) — all 10 prefill/decode/GQA cases |
| `single_fa_ops` | `llm/single_test_ops/fa_ops.json` | 9 s | paged attention. **Measures nothing under `MODE=host`** without `flash_attn`, and exits 0 |

That is 2,496 s of device time in total — **the whole comparison sweep is 42
minutes**, so reproducing all of it is usually cheaper than deciding which label
you need. The Trainium2 side is most of a day; see
[NEURON/README.md, Reproduce one row at a time](../NEURON/README.md#reproduce-one-row-at-a-time)
for its label list, and note that two labels are named differently there — `gemm`
is `basic_tensor_gemm_ops`, and `basic_vector_index_ops` is split into
`basic_index_ok` and `basic_index_slow` because `gather`/`scatter` have to be run
under their own watchdog on that side.

Three further labels exist but are **skipped in a default run** and reachable only
by naming them: `single_gemm_ops`, `single_moe_dispatch_ops` and
`single_moe_combine_ops`. They are the files with a Trainium2 number and no GPU
column that this box could close today — see
[What this table does not cover yet](#what-this-table-does-not-cover-yet). Running
them adds three rows to the comparison; they are gated so that a default run keeps
describing what the published numbers were actually taken with. `single_pre_fa_ops`
is in the same block for a different reason: the `rotary_embedding` row was measured
by hand rather than through the script.

## How to compare these to the Trainium2 numbers

This is the part that is easy to get wrong, and getting it wrong changes the
answer by 4x in either direction.

**One micro_perf "device" is not the same amount of hardware on the two
backends.** On this instance one device is one whole H100. On a trn2 one device
is one *logical NeuronCore* — a quarter of a Trainium2 chip at the default
`LNC=2`. So a raw per-device latency ratio flatters the GPU by roughly 4x for no
physical reason, and the following three columns are the ones that mean
something:

| Column | H100 SXM5 | One logical NeuronCore | One Trainium2 chip |
|---|---|---|---|
| bf16 / fp16 dense peak | 989.4 TFLOPS | 166.75 TFLOPS | 667 TFLOPS |
| fp8 dense peak | 1,978.9 TFLOPS | 324.75 TFLOPS | 1,299 TFLOPS |
| fp32 dense peak | 67.0 TFLOPS | 45.25 TFLOPS | 181 TFLOPS |
| HBM bandwidth | 3.35 TB/s | ~725 GB/s | ~2.9 TB/s |

Sparsity-doubled figures are excluded on both sides: no op in this suite feeds a
structured-sparse operand. The GPU table lives in
`src/xpu_perf/micro_perf/backends/GPU/backend_gpu.py` as `GPU_PEAK_TFLOPS`,
keyed on `torch.cuda.get_device_name()` longest-match-first — "NVIDIA H100 80GB
HBM3" and "NVIDIA H100 PCIe" both contain "H100" and differ by 30% on every
tensor-core row. A card that is not tabulated reports **no** MFU rather than a
wrong one.

**Compare MFU on compute-bound ops and `mem_bw` on memory-bound ones, and
compare them per chip.** At chip level the silicon ratio is only **1.48x** in
bf16 (989.4 vs 667) and **1.52x** in fp8. Anything larger than that in the tables
below is the software stack, not the hardware — which is the useful signal.

**The x4 is measured, not assumed — for gemm and for elementwise work.** Every
per-chip Trainium figure here is one logical core's number times four, so the
assumption is load-bearing and was checked directly: a `tensor_gemm_ops` run on
`--device 0,1,2,3` under `XPU_PERF_ENGINES=ComputeEngine` keeps four *different*
cases in flight, so each case's latency is a single-core latency measured while
the other three cores are competing for the same HBM.

| Measured on | 1 core | 4 cores, per case | Ratio |
|---|---|---|---|
| `gemm` bf16 peak | 149.42 TF (~90% MFU) | **149.59 TF (89.7% MFU)** | 1.001 |
| `gemm` fp32 peak | 37.67 TF (83.3% MFU) | **37.35 TF (82.5% MFU)** | 1.009 |
| `gemm`, 648 matched shapes, median by dtype | baseline | — | **1.004-1.011** |
| `add`/`mul`/`sub`/`cast`, 12 dtype combos | 593-617 GB/s | **594-616 GB/s** | 1.00-1.02 |
| `reduce_*` / `topk` / `moe`, median by op | baseline | — | **1.002-1.029** |
| `topk` at the published shape | 475.6 GB/s | **476.8 GB/s** | 0.998 (x4 = **4.01**) |
| `moe_softmax_topk` at the published shape | 74.4 GB/s | **74.4 GB/s** | 1.001 (x4 = **4.00**) |
| `gemm` fp8, 24 matched shapes | baseline | 1.02-2.16x slower | median **1.37** |

So for every workload class that matters to the headline, contention costs nothing
measurable: four cores each hold ~600 GB/s, i.e. ~2.4 TB/s of the chip's 2.9 TB/s
aggregate (83-85%), and bf16 gemm holds ~90% MFU per core with all four loaded.
**Multiplying by four is legitimate**, and 4 x 149.59 = 598.4 TFLOPS is a real
per-chip bf16 number rather than an extrapolation. The same now applies to the two
selection ops, which used to carry an explicit "extrapolated" flag.

Two classes of exception, both noted where they appear:

- **fp8, and only fp8, fails to scale at the median** (1.37-1.53x slower under
  contention). That is consistent with what it is actually doing — a software
  up-cast that moves bytes instead of using the matmul engine.
- **The smallest shapes in every family degrade 5-11x**, far worse than any median:
  `gemm` `M=2 K=1024 N=8192` **9.01x**, `topk` at `dim_size 128` **11.02x**,
  `moe_softmax_topk` at `num_tokens 1` **8.89x**. These are runs dominated by launch
  and synchronisation rather than by arithmetic, so they contend on something the
  large shapes never touch. Per-chip figures quoted from small shapes are not safe
  to multiply by four; the figures quoted in this document are all from the plateau.

**Attention** is still not measured this way — the workload has no multi-core
variant yet, so the 3.1x prefill gap keeps the "upper bound on Trainium, lower
bound on the gap" caveat.

## Attention

`workloads/llm/single_test_ops/fa_linear_ops.json`, all bf16, `head_dim` 128.
Nine of ten cases on both backends — the tenth is `q_len: 4` speculative decode,
which the provider rejects on purpose because `is_causal=True` aligns the mask to
the top-left and a multi-token decode step needs bottom-right.

Both sides ran the **same provider source**: `scaled_dot_product_attention`
against the same op def. That is deliberate — comparing Neuron to `flash_attn`
would compare two algorithms as well as two chips. On CUDA, SDPA is *not* a slow
fallback: it dispatches to a fused FlashAttention or cuDNN kernel and never
materialises the score matrix, which the 61-69% prefill MFU confirms.

Nor is it a fallback on Neuron for the **prefill** rows, and an earlier version of
this section said it was. `torch_neuronx` lowers SDPA to a NKI flash kernel inside
its dynamo backend (`_can_use_nki_flash_attention` in
`neuron_dynamo_backend/decompositions.py`, enabled by default through
`TORCH_NEURONX_ENABLE_NKI_SDPA`) for any call satisfying `L % 512 == 0 and
S % 512 == 0 and D <= 128 and B*H <= 512` with no attn_bias and no dropout.
Setting that variable to 0 takes the 80/8/128 `q_len` 4096 prefill from
**9,443.5 us to 60,499.9 us — 6.41x** — so the kernel is real and it is running.
The narrower statement this section used to rest on is still true and is all that
is true: `neuronxcc.nki.kernels.attention.flash_fwd` is HLO-traced and only loads
under `torch_xla`, so *that* kernel is unreachable on the eager stack.

The **decode** rows are a fallback, provably and permanently: `q_len == 1` can
never satisfy `L % 512 == 0`, and their `B*H` is 1280 and 5120 against a limit of
512, so the gate fails twice over. That is the asymmetry to keep in mind when
reading the two halves of the table below — prefill compares a fused kernel against
a fused kernel, decode compares a fused kernel against an unfused one.

> **Installing `vllm` used to halve every prefill number in this section, and the
> report gave no sign of it.** Worth reading even if you do not care about GPUs,
> because the shape of the trap is not CUDA-specific.
>
> Which fused backend SDPA picks is *process-global* state. `import vllm` calls
> `torch.backends.cuda.enable_cudnn_sdp(False)`, and the provider registry
> imports every vendor module — so having the vllm `rms_norm` provider installed
> reconfigured an unrelated attention op. cuDNN attention runs the 80/8 GQA
> prefill at `q_len` 4096 in **584 us**; PyTorch's FLASH backend, next in the
> priority order, takes **1,079 us** on the same shape. Measured through the op:
> all six prefill cases went 1.86-1.94x slower and the three decode cases moved
> only 1.09-1.13x, because a `q_len == 1` step cannot benefit from causality
> either way. A uniform 1.9x on one attention mode and nothing on the other is
> the signature to look for; it reads like noise if you only spot-check decode.
>
> `ops/torch/flash_attention.py` now re-enables all three fused backends in
> `vendor_impl`, so the numbers no longer depend on which unrelated packages are
> installed. With that in place a fresh run in the full-registry environment
> reproduced the table below to within 3% on all nine cases (prefill 60.6-68.8%
> MFU, decode 77.9-88.8% of HBM peak).
>
> The general lesson: **one provider's import can silently re-tune another
> provider's op.** `targets` records latency, not the configuration that produced
> it, so nothing in the report tree distinguishes the two runs. If a number moves
> and the workload did not, check what else got imported.

| Mode | q/kv | B | cache | q_len | H100 lat | H100 MFU | Trn2 lat | Trn2 MFU | lat ratio |
|---|---|---|---|---|---|---|---|---|---|
| prefill | 80/8 | 1 | 0 | 4,096 | 567.6 us | 61.2% | 9,507.6 us | 21.7% | 16.8x |
| prefill | 80/8 | 4 | 0 | 4,096 | 2,139.3 us | 65.0% | 34,422.1 us | 24.0% | 16.1x |
| prefill | 80/8 | 1 | 0 | 10,240 | 3,163.2 us | **68.6%** | 39,969.3 us | **32.2%** | 12.6x |
| prefill | 80/80 | 1 | 0 | 4,096 | 571.6 us | 60.8% | 9,520.2 us | 21.7% | 16.7x |
| prefill | 80/80 | 4 | 0 | 4,096 | 2,167.9 us | 64.1% | 34,757.8 us | 23.7% | 16.0x |
| prefill | 80/80 | 1 | 0 | 10,240 | 3,177.0 us | **68.3%** | 39,782.3 us | **32.4%** | 12.5x |

Decode is memory-bound — one query row against the whole cache does ~10 FLOPs per
byte — so MFU is the wrong column there and `mem_bw` is the right one:

| Mode | q/kv | B | cache | H100 lat | H100 GB/s | % of 3.35 TB/s | Trn2 lat | Trn2 GB/s | % of 725 GB/s |
|---|---|---|---|---|---|---|---|---|---|
| decode | 80/8 | 16 | 4,096 | 101.1 us | 2,662.8 | 79.5% | 1,976.7 us | 136.2 | 18.8% |
| decode | 80/8 | 64 | 4,096 | 372.9 us | 2,887.2 | **86.2%** | 4,511.3 us | 238.7 | **32.9%** |
| decode | 80/8 | 16 | 10,240 | 244.9 us | 2,742.8 | 81.9% | 6,066.2 us | 110.7 | 15.3% |

Four conclusions, in order of how much they matter:

- **Trainium2's attention gap is mostly software, not silicon — but not for the
  reason that looks obvious.** Per *chip*, the best prefill throughput is 678.97
  TFLOPS on the H100 against 4 x 53.99 = 216.0 TFLOPS on a Trainium2 — a **3.1x**
  gap where the peak-FLOPS ratio is only 1.48x. The other 2.1x is that the H100
  stack extracts 68% of its peak where the Neuron eager stack extracts 32%. It is
  tempting to attribute that to a missing fused kernel, and this bullet used to;
  the measurement above rules it out. Prefill already runs a fused NKI flash
  kernel — turning it off costs 6.41x, so 32% of peak *is* the fused kernel's
  score. The remaining 2.1x is kernel quality inside a fused implementation, which
  is a harder thing to close than a missing kernel but still software. **And it is
  not a case of the wrong kernel being picked either:** the kernel the rewrite
  lowers to is `nkilib`'s `attention_cte`, by object identity in
  `decompositions.py`, and calling `attention_cte` by hand with the launch
  arguments torch_neuronx uses reproduces the same latency to 0.97x
  ([details](../NEURON/README.md#attention)). So the 3.1x is `attention_cte`
  against cuDNN/FlashAttention at the same shape, with no better kernel sitting
  unused on the Neuron side.
- **Decode on Trainium is not bandwidth-limited, which means it is fixable.**
  The H100 reads its KV cache at 80-86% of HBM peak — that is a
  bandwidth-saturating kernel and there is nothing left to win. Trainium reads it
  at 15-33%, and for scale the best plain memory-bound op on the same core hits
  631 GB/s (`index_select`). So decode there is leaving 2.6-5.7x of *its own*
  available bandwidth on the floor. Both backends improve with batch and degrade
  with cache length, which is a per-call fixed cost being amortised rather than a
  ceiling.
- **GQA and MHA cost the same on both backends**, to within 1% at every shape, and
  on both the MHA row is marginally *faster* at `q_len 10240`. Prefill is
  compute-bound and GQA saves KV traffic rather than FLOPs, so this is the
  expected result on both. It also means neither backend's prefill numbers can
  demonstrate that `enable_gqa=True` avoids materialising the expanded cache —
  only decode could, and the workload covers GQA only there.
- **Batching to 4 helps the H100 and does nothing for Trainium.** H100 MFU goes
  61.2% -> 65.0%; Trainium goes 21.7% -> 24.0% with latency at 4 x 0.905 of the
  single-sequence case, i.e. essentially four sequential prefills. At `q_len 4096`
  one sequence already saturates the Neuron core.

## gemm

All 856 cases of `workloads/basic/tensor_gemm_ops/gemm.json`, in **403 seconds**.
Peak per dtype, with the Trainium2 figures for the same file alongside:

| Dtype | H100 peak | MFU | At shape | Trn2 per core | Trn2 MFU | Per chip: H100 / Trn2 |
|---|---|---|---|---|---|---|
| bfloat16 | **808.34** TF | 81.7% | 12288x8192x8192 | 149.42 TF | **~90%** | 808.3 / 597.7 = **1.35x** |
| float16 | 774.35 TF | 78.3% | 3072x8192x8192 | — | — | — |
| tfloat32 | 413.92 TF | **83.7%** | 3072x8192x8192 | rejected | — | — |
| float32 | 48.72 TF | 72.7% | 20480x8192x8192 | **37.67 TF** | **83.3%** | 48.7 / 150.7 = **0.32x** |
| float8_e4m3 | **1,527.85** TF | 77.2% | 4096x8192x8192 | 3.85 TF | 1.2% | 1,527.9 / 15.4 = **~99x** |
| float8_e5m2 | rejected | — | — | 4.77 TF | 1.5% | see below |
| fp8, best each side can actually do | 1,527.85 TF (`e4m3fn`) | 77.2% | 4096x8192x8192 | **245.50 TF** (`e5m2`, compiled) | **75.6%** | 1,527.9 / 982 = **1.56x** |

**On dense bf16 gemm, Trainium2 is competitive per chip — and extracts a *higher*
fraction of its own peak than the H100 does.** Both backends peak at the identical
shape, 12288x8192x8192: the H100 reaches 81.7% of 989.4, a logical NeuronCore
reaches ~90% of 166.75. Four cores is 597.7 TFLOPS against the H100's 808.3, a
**1.35x** gap where the nominal silicon ratio is 1.48x. This is the one place in
the comparison where Trainium2 needs no excuse — the matmul engine and its
lowering are both in good shape, and what remains is silicon.

**At fp32, Trainium2 wins outright — 3.09x per chip.** This cell used to hold a
stale 23.5 TF from an XLA run; the eager number for the same file is 37.67 TF per
logical core, 150.7 TF per chip, at **83.3%** of its own fp32 peak against the
H100's 72.7% of 67 TF. The nominal bar already favours Trainium2 here (2.70x on
peak fp32, against the H100's 1.48x advantage at bf16), and the stack collects
slightly more of it than CUDA does. fp32 gemm is not a shape most inference work
runs, so this is not a headline — but it is the counter-example to "the gap is
always in Trainium's software", and it is the reason the summary table's
`gemm` row is scoped to bf16.

**The x4 is measured, not assumed — with one caveat at tiny shapes.** Running
`gemm.json` on one core and on four cores concurrently and joining the 648 shapes
that both completed gives a median per-shape slowdown of **1.004-1.011x** by
dtype: four cores each do essentially the work one core alone does, so multiplying
a single-core figure by four is sound at the shapes that matter. The exceptions
are real but narrow. fp8 is the only dtype whose *median* does not scale
(1.46-1.53x) — unsurprising, since what it is timing is a software widening
competing for the same scalar engines. And the smallest shapes degrade badly under
concurrency: `M=2 K=1024 N=8192` takes **9.01x** longer per core with four cores
busy. Those are shapes where the run is dominated by launch and sync overhead
rather than by the matmul, so they contend on a shared resource that the large
shapes never touch. The same tail shows up in the reductions and in `moe`
([below](#memory-bound-ops)), which is what makes it a property of small-shape
dispatch rather than of any one op.

**fp8 is where the two stacks are least comparable, and the ~99x in the table is a
property of the sweep rather than of the chip.** On the H100 fp8 is a real
datapath: 1,527.9 TFLOPS at 77.2% MFU, **1.89x** faster than bf16, close to the 2x
the format promises. The Trainium2 cells next to it are 3.85 / 4.77 TFLOPS, 1.2-1.5%
MFU — and two independent things produce that, neither of which is "the hardware
has no fp8 gemm":

- **The encoding is wrong.** `float8_e4m3` maps to `torch.float8_e4m3fn`, the
  finite-only variant CUDA uses. Trainium1/2 implement the *other* e4m3, and the
  compiler says so by name: `[NCC_EVRF051] Data type F8E4M3FN is not supported on
  TRN1/TRN2.` The workaround flag that error recommends does not exist in
  neuronx-cc 2.27.2878.0.
- **The path is eager.** Eager has no fp8 gemm lowering for *either* encoding, so
  both fall onto the same software widening at ~1-2 TFLOPS.

Fix both — `torch.compile(backend="neuron", dynamic=False)`, in `e5m2` — and one
logical core does **245.50 TFLOPS at 75.6% of its 324.75 TF fp8 peak**, 1.82x its
own bf16 on a nominal 1.95x bar, 115.7x the eager number at the same shape. Unlike
eager fp8, that path scales across cores cleanly (1.018x worst case over four
concurrent runs), so **982 TFLOPS per chip and a real gap of 1.56x** where the
nominal fp8 peak ratio is 1.52x
([details](../NEURON/README.md#fp8-the-sweep-measures-the-eager-path-and-the-wrong-e4m3-encoding)).

Two things follow. The ~99x row stays in the table because it is what
`gemm.json` measures today and it is reproducible — closing it needs an fp8
provider that compiles and `float8_e5m2` cases to point at, not new silicon. And
**the fp8 row is not a like-for-like comparison and cannot be made into one within
this op def**: the H100's number is `e4m3fn` via `torch._scaled_mm`, Trainium2 has
no `e4m3fn` at all, and `torch._scaled_mm` rejects `e5m2 x e5m2` on CUDA. The two
chips share no fp8 format that both stacks will multiply, so the 1.56x above
compares each side's best fp8, in different encodings.

Two implementation notes, because the fp8 numbers depend on them:

- `torch.matmul` has **no** fp8 kernel on CUDA and raises, so unlike the Neuron
  provider it is not enough to opt the dtype in — `ops/torch/gemm.py` replaces the
  run function with `torch._scaled_mm`. That call requires a column-major second
  operand, so the provider declares `b` as `[N, K]` and passes `b.t()`: a
  transposed view of a row-major `[N, K]` *is* a column-major `[K, N]`, for free.
  Transposing inside the run function would put a K x N copy in the timed region,
  which is exactly the cost being measured around. Element count is unchanged, so
  the base def's `read_bytes` and `calc_flops` stay correct.
- **`e5m2 x e5m2` does not exist on CUDA.** `torch._scaled_mm` fails it with
  `ValueError: Multiplication of two Float8_e5m2 matrices is not supported` —
  e5m2 is a gradient format and is only ever one side of a mixed pair, which this
  op def cannot express since `a` and `b` share one `dtype`. The provider now
  reports it unsupported instead of raising mid-sweep. Worth contrasting with the
  Neuron result, where the asymmetry runs the other way: `e5m2` is the encoding
  Trainium2 *does* implement, and the only one that compiles. On the eager path it
  also comes out slightly faster than `e4m3` at all twelve shapes, consistent with
  both rows timing a software widening whose cost tracks the conversion rather than
  the multiply — though that is one reading among several, and the compiler's
  refusal of `e4m3fn` is the harder evidence.

Note also that `float32` and `tfloat32` are genuinely different hardware here and
the provider keeps them apart: it sets matmul precision `highest` for float32
(CUDA cores, no TF32 substitution) and `high` for tfloat32 (TF32 tensor cores).
The 60x TFLOPS spread between the two rows is that switch, not noise.

## Memory-bound ops

The H100 sits at **84-91% of HBM peak on 20 of 24** elementwise, reduction and
index ops — there is almost no variance to explain on this side. The Neuron core
spans **0.1% to 88%**. So this table is not really a hardware comparison; it is a
list of which Neuron lowerings are finished and which are not, with the GPU column
acting as the control that says what the op *should* cost.

Best `mem_bw` per op, each side at its own best dtype and shape, against its own
HBM peak. (Each side's best over the same workload files. Unlike the sections
below, this is peak-vs-peak rather than matched per shape — the 2026-09-01 Neuron
sweep's per-case jsonl for `basic/` is no longer on disk. The plateau numbers are
shape-insensitive enough that it makes little difference, but it is a weaker
comparison and is marked as such.)

| Op | H100 | % of 3.35 TB/s | Trn2 core | % of 725 GB/s | Neuron shortfall |
|---|---|---|---|---|---|
| `scatter` | 2,288.4 | 68.3% | 0.8 | 0.11% | **621x** |
| `gather` | 2,856.1 | 85.3% | 1.4 | 0.19% | **449x** |
| `gelu` | 2,985.2 | 89.1% | 83.7 | 11.5% | **7.7x** |
| `cos` | 2,982.4 | 89.0% | 110.6 | 15.3% | 5.8x |
| `sin` | 2,974.7 | 88.8% | 118.9 | 16.4% | 5.4x |
| `index_add` | 1,739.9 | 51.9% | 98.6 | 13.6% | 3.8x |
| `reduce_min` | 2,884.0 | 86.1% | 177.0 | 24.4% | 3.5x |
| `reduce_max` | 2,826.5 | 84.4% | 176.6 | 24.4% | 3.5x |
| `rms_norm` | 2,805.9 | 83.8% | 409.2 | 56.4% | 1.5x |
| `layer_norm` | 1,786.2 | 53.3% | 259.3 | 35.8% | 1.5x |
| `sqrt` | 2,987.3 | 89.2% | 564.5 | 77.9% | 1.15x |
| `log` | 2,985.0 | 89.1% | 564.6 | 77.9% | 1.14x |
| `div` | 3,034.8 | 90.6% | 579.9 | 80.0% | 1.13x |
| `exp` | 2,989.2 | 89.2% | 578.8 | 79.8% | 1.12x |
| `mul` | 3,034.8 | 90.6% | 599.9 | 82.7% | 1.10x |
| `add` | 3,024.8 | 90.3% | 601.9 | 83.0% | 1.09x |
| `silu` | 2,969.0 | 88.6% | 587.2 | 81.0% | 1.09x |
| `sub` | 3,034.2 | 90.6% | 608.6 | 84.0% | 1.08x |
| `cast` | 2,976.1 | 88.8% | 616.7 | 85.1% | 1.04x |
| `reduce_sum` | 2,985.0 | 89.1% | 639.0 | 88.1% | 1.01x |
| `index_select` | 2,861.7 | 85.4% | 631.6 | 87.1% | 0.98x |
| `embedding` | 2,843.2 | 84.9% | 631.5 | 87.1% | 0.97x |
| `softmax` | 2,134.7 | 63.7% | 495.8 | 68.4% | 0.93x |
| `topk` | 629.9 | 18.8% | 476.3 | 65.7% | **0.29x** |

"Neuron shortfall" is the ratio of the two %-of-own-peak columns, so 1.0 means
both backends extract the same fraction of the bandwidth they have. **Twelve of
the 24 ops are between 0.93x and 1.15x** — for most memory-bound work Trainium2
is simply not the problem, and the per-chip figure (4 x ~600 = ~2.4 TB/s,
[measured](#how-to-compare-these-to-the-trainium2-numbers)) lands within 1.4x of
one H100.

The GPU column settles four claims that the Neuron numbers alone could only
suggest:

- **`gelu`'s 7-14x deficit is lowering, not `erf`.** `NEURON/README.md` flags
  `gelu` at 83.7 GB/s against `silu` at 587.2 over identical shapes. On the H100
  `gelu` is 2,985.2 and `silu` 2,969.0 — `gelu` is marginally *faster*. The
  `erf`-based expansion therefore costs essentially nothing on hardware that
  compiles it well, and the entire Neuron gap is neuronx-cc.
- **`gather`/`scatter` is not the op def's *shape*, and for `gather` it is not the
  hardware either.** The base `GatherOp` builds an output-shaped index where
  `IndexSelectOp` passes a 1-D one, which was the natural suspect. Measured, that
  costs nothing on hardware that compiles it: on CUDA `gather` runs at 2,856 GB/s
  against `index_select`'s 2,862. What the shape *does* do on Neuron is expose a
  dtype problem — the index is built as int64 and expanded with `.expand()`, the
  device has no int64, and the int64→int32 conversion inserted on the way into the
  graph materialises the stride-0 broadcast that the fast lowering needs to see
  intact. Constructing the same index as int32 keeps it a view and runs **291-731x
  faster**, within 8% of `index_select`; forcing `.contiguous()` on the int32
  version makes it slow again, which is what identifies layout rather than dtype as
  the proximate cause. `scatter` is not fixable that way — `aten::scatter_.src` has
  no NKI implementation at all, so its 621x is real. The op def is deliberately
  *not* changed, since `torch.gather` requires int64 on CPU and CUDA; see
  [the NEURON README](../NEURON/README.md#two-index-ops-are-pathologically-slow-and-one-is-an-int64-index-away-from-parity)
  for the measured table and for the separate registration bug that keeps the NKI
  gather kernel unreachable even when the index is right.
- **The narrow-dtype and reduction anomalies are Neuron-only.** `sin`/`cos` are
  4x *slower* at bf16 than fp32 on Neuron; on the H100 `sin` is 2,884.9 bf16
  against 2,974.7 fp32, a 1.03x difference. `reduce_max` against `reduce_sum` is
  3.6x on Neuron and 1.06x on the H100. `div` at fp32 is 1.58x off `mul` on
  Neuron (379 vs 599) and *identical* on the H100 (both 3,034.8). Note that this
  does **not** carry over to `rotary_embedding`, which is the op those trig rows
  look like they should explain — it does not call `sin` or `cos` at all:

  | Case | Trn2 1 core | % of 725 | H100 | % of 3,350 | Per chip |
  |---|---|---|---|---|---|
  | prefill `q_len` 10240 | 42.77 | **5.90%** | 197.35 | **5.89%** | 1.15x |
  | prefill `cache_len` 5120 + `q_len` 5120 | 42.32 | 5.84% | 195.98 | 5.85% | 1.16x |
  | prefill `q_len` 32768 | 42.22 | 5.82% | 199.95 | 5.97% | 1.18x |
  | decode `batch_size` 16, `q_len` 1 | 0.99 | 0.14% | 3.80 | 0.11% | 0.96x |
  | decode `batch_size` 16, `q_len` 4 | 1.01 | 0.14% | 4.85 | 0.14% | 1.20x |

  The prefill rows are matched to a significant figure as a fraction of each chip's
  own peak, which is what identifies the op def rather than either stack:
  `rotate()` is 78% of the time and is five-plus materialising passes. The decode
  rows are not bandwidth on either side — `vendor_impl_run` loops over batches in
  Python, so `batch_size` 16 at `q_len` 1 is 16 dispatches of one token each.
  These are the only `pre_fa_ops.json` numbers in this comparison; the file's other
  op, `store_kv_cache`, does not run on **either** backend (two base-op-def
  blockers, [listed in the NEURON
  README](../NEURON/README.md#known-unsupported)).
- **Selection and sorting are where the H100 is the weak side**, and it is two ops
  rather than one, which is what makes it a pattern instead of an oddity:

  | Op | H100 | % of 3.35 TB/s | Trn2 core | % of 725 GB/s | Absolute | Per chip (measured) |
  |---|---|---|---|---|---|---|
  | `topk` (fp32, k=4) | 629.9 | 18.8% | 476.3 | 65.7% | H100 1.32x | 1,907.2 GB/s — **Trn2 3.03x** |
  | `moe_softmax_topk` (fp32) | 80.7 | **2.4%** | 74.4 | 10.3% | H100 1.08x | 297.5 GB/s — **Trn2 3.69x** |

  A whole H100 is only 8% faster than a *quarter of a Trainium2 chip* at
  `moe_softmax_topk`, on a 4.6x bandwidth advantage — and 2.4% of HBM peak is the
  worst number the H100 posts anywhere in this comparison. It is also slow in wall
  clock: `moe_gating_ops.json` is 56 cases and took the H100 **1,191 s**, three
  times the 403 s the entire 856-case `gemm.json` needed. Large-k selection over
  256 experts is a genuinely bad fit for a GPU, and the Neuron vector engine
  handles it comparatively well. The win depends entirely on the x4, since no single
  NeuronCore *beats* the H100 in absolute terms on either op — and the x4 is now
  measured at exactly these shapes rather than extrapolated: four concurrent cores
  give **4.01x** on `topk` and **4.00x** on `moe_softmax_topk`, so a Trainium2 chip
  really is ~3x an H100 at MoE routing. What the same run also found is that this
  holds only away from the smallest shapes: `topk` at `dim_size 128` degrades
  **11.02x** per core under four-core concurrency and `moe_softmax_topk` at
  `num_tokens 1` degrades **8.89x**, the same small-shape dispatch tail the
  [gemm section](#gemm) describes. Routing a single token per step will not see this
  advantage; routing a batch will.

## Norm, activation and MoE ops

These are matched **per shape**: both backends ran the same four workload files
this port added, so every row below is a median over cases with byte-identical
arguments rather than a peak-vs-peak comparison.

| Op | dtype | n | H100 best | % pk | Trn2 best | % pk | Shortfall, median (range) |
|---|---|---|---|---|---|---|---|
| `quant_group_gemm_reduce_sum` | fp8 | 2 | 231.4 | 6.9% | 1.2 | 0.2% | **38.2x** (28.4-48.0) |
| `quant_group_gemm_reduce_sum` | int8 | 2 | 255.6 | 7.6% | 1.5 | 0.2% | **31.9x** (23.4-40.4) |
| `scale_dynamic_quant` | bf16 | 14 | 135.3 | 4.0% | 1.2 | 0.2% | **28.4x** (1.7-42.4) |
| `swiglu_dynamic_quant` | bf16 | 14 | 178.7 | 5.3% | 1.4 | 0.2% | **24.4x** (1.6-32.8) |
| `add_rms_norm_dynamic_quant` | bf16 | 28 | 175.0 | 5.2% | 2.0 | 0.3% | **17.3x** (1.3-23.7) |
| `head_rms_norm_dynamic_quant` | bf16 | 14 | 77.7 | 2.3% | 1.0 | 0.1% | **10.3x** (1.5-23.9) |
| `moe_swiglu_dynamic_quant` | bf16 | 3 | 73.4 | 2.2% | 2.2 | 0.3% | **3.8x** (2.3-7.1) |
| `add_rms_norm` | bf16 | 14 | 2,044.9 | 61.0% | 269.4 | 37.2% | 1.26x (0.52-2.46) |
| `swiglu` | fp32 | 14 | 1,542.5 | 46.0% | 313.3 | 43.2% | 0.94x (0.40-3.12) |
| `swiglu` | bf16 | 14 | 1,043.1 | 31.1% | 291.9 | 40.3% | 0.74x (0.26-1.55) |
| `moe_swiglu` | bf16 | 3 | 988.6 | 29.5% | 306.2 | 42.2% | 0.70x (0.70-0.75) |
| `swiglu` | fp16 | 14 | 1,044.4 | 31.2% | 290.7 | 40.1% | 0.70x (0.43-1.55) |
| `moe_softmax_topk` | fp32 | 54 | 80.7 | **2.4%** | 74.4 | 10.3% | 0.65x (0.23-1.52) |
| `qk_rms_norm` | bf16 | 7 | 235.8 | 7.0% | 90.2 | 12.4% | 0.54x (0.41-0.66) |
| `head_rms_norm` | bf16 | 14 | 236.8 | 7.1% | 95.8 | 13.2% | 0.48x (0.32-0.67) |
| `qk_rms_norm` | fp32 | 7 | 814.1 | 24.3% | 374.5 | 51.7% | 0.42x (0.28-0.81) |
| `head_rms_norm` | fp32 | 14 | 816.1 | 24.4% | 431.0 | 59.5% | 0.35x (0.22-0.96) |

The table splits cleanly in two, with nothing between 1.26x and 3.8x.

**Below the split, Trainium2 is ahead on 9 of the 10 non-quantised rows** — it
extracts more of its own bandwidth than the H100 does, by up to 2.9x on
`head_rms_norm` at fp32. That is the mirror image of the basic-ops table, and it
cuts against the summary: the Neuron stack is not uniformly behind, it is
*bimodal*. Ops that reduce to a few large elementwise passes are fine or better;
ops needing a fused or well-chosen kernel are where it loses. The wide per-shape
ranges are the small-`num_tokens` end of each sweep, where Neuron's fixed
per-dispatch cost dominates; both backends reach their plateau by ~4096 tokens.

**Above the split are the seven quantised ops, and the H100 column is what makes
them interpretable.** `NEURON/README.md` reports these six `_dynamic_quant` ops at
87-183x slower than their unquantised siblings and traces it to one shared helper,
`smooth_per_token_dynamic_quant`, written as a chain of un-fused full-size
temporaries. The question that could not be answered from one backend was how much
of that is the helper and how much is Neuron. Now: the same helper reaches only
**2.2-5.3% of HBM peak on an H100**, 3-14x slower than the corresponding
unquantised op on hardware that does 84-91% on ordinary elementwise work. So the
helper is genuinely bad everywhere and is the dominant cause — but Neuron is a
further **3.8-38x** down on top of that. Roughly one order of magnitude is the
shared algorithm and one is Neuron-specific lowering. Fusing the helper is the
first fix for both backends, and it is not a Trainium workaround.

Two caveats on these rows specifically. Their absolute bandwidths are 1-2 GB/s on
Neuron, where run-to-run spread is about **±25%** — a re-run after the correctness
fix below moved individual peaks by up to 20% while the per-shape median stayed at
0.98-1.01x, so treat one significant figure as the precision. And the wide ranges
(1.3x at the low end) are again the small-`num_tokens` cases, where both backends
are launch-bound and the helper's cost has not yet appeared.

`smooth_per_token_dynamic_quant` also had a correctness bug, fixed in this port: it
multiplied `smooth_scale` by `per_token_scale` rather than `smoothed_input`, which
broadcasts to the same shape and so never failed, but made the output independent
of `hidden_states`. Re-measuring all eleven affected ops on Neuron after the fix
gave a median per-shape ratio of **0.98-1.01x** — the fix is performance-neutral
within noise, so every number published before it remains valid as a performance
figure. That is worth stating because the reverse would have invalidated a table.

**`head_rms_norm` and `qk_rms_norm` are broken on every backend, and the H100
proves it.** Both sit at 7.1% and 7.0% of HBM peak at bf16 on hardware that does
84-91% on ordinary elementwise work, and both are ~3.4x *slower* at bf16 than at
fp32 — the wrong direction for a memory-bound op with half the bytes. The cause
is in the op def, not either backend: `op_defs/llm_ops/head_rms_norm.py` declares
`token_data` as the case's dtype but hard-codes `norm_weight` as
`dtype=torch.float32` (`qk_rms_norm.py` does the same for both of its weights).
PyTorch says so out loud on the GPU, once per case:

```
UserWarning: Mismatch dtype between input and weight: input dtype = c10::BFloat16,
weight dtype = float, Cannot dispatch to fused implementation.
(Triggered internally at aten/src/ATen/native/layer_norm.cpp:346)
```

So the bf16 and fp16 rows of these two ops measure an unfused fallback on both
backends, which is why Neuron's 4.5x fp32-to-bf16 penalty looked Neuron-specific
and is not. Give `norm_weight` the same dtype as `token_data` to get numbers that
mean something; the tables above keep the as-shipped behaviour so they stay
comparable to the published Neuron run. (Separately, `vendor_impl_run` documents
itself as an in-place norm but rebinds `head_data` to `rms_norm`'s return value
instead of assigning into the slice, so `token_data` comes back unmodified. The
norm is still computed and the byte accounting still matches, so the timing
stands, but the op does not do what its comment says.)

## What is not measured here

Workload coverage — which files have a Trainium2 number and no GPU column, and
what blocks each — is a table in the Summary:
[What this table does not cover yet](#what-this-table-does-not-cover-yet).
Everything there is a TODO with a named file.

What is missing for reasons that are *not* about workload coverage:

- **Multi-chip anything.** One H100 against one Trainium2 chip is the widest
  comparison here. Nothing in this document constrains how either scales past one
  package, and the collectives rows that would are the hardware-blocked entries in
  the table above.
- **`torch.compile` on either side.** Both backends are measured eager, which is
  what `launch.py` runs. That is not a neutral choice: it is precisely what made
  the fp8 row read as 99x instead of 1.56x, and there is no reason to assume fp8
  is the only op where it matters. Any op whose Neuron number looks like an
  un-fused chain of elementwise passes — the `_dynamic_quant` family,
  `rotary_embedding` — is a candidate for the same correction, on both backends.
- **Numerics.** Every number here is latency or bandwidth. No case checks output
  against a reference, so a lowering that is fast because it is wrong would not be
  caught. Two bugs found in this port by reading rather than measuring
  (`smooth_per_token_dynamic_quant`'s scale multiply, `head_rms_norm`'s in-place
  claim) are the argument for adding that check.
