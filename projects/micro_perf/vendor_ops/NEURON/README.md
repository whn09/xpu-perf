# NEURON vendor ops (AWS Trainium / Inferentia)

micro_perf support for AWS Trainium and Inferentia through the
[Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/), on either of the
two PyTorch integrations Neuron offers: PyTorch/XLA (`xla`) or the newer
PyTorch-native eager stack (`eager`). The backend class lives at
`src/xpu_perf/micro_perf/backends/NEURON/`; this directory holds the vendor op
implementations, the default environment (`env.json`), and the sweep harness in
`tools/`.

**Prefer the eager runtime.** It is 2.7x faster on gemm, needs no compile cache,
and has far narrower run-to-run spread. The XLA path is supported and documented
because existing installations are on it.

Why the backend code looks the way it does — dispatch model, compilation
behaviour, collective plumbing, and the bugs each guard rail exists to prevent —
is in [IMPLEMENTATION.md](IMPLEMENTATION.md). This file is how to run it, what
the numbers are, and what to do when it breaks.

> **Validation status** — this is a port. The backend was written against the
> pre-refactor `micro_perf/` layout in
> [davidshtian/xpu-perf](https://github.com/davidshtian/xpu-perf), extended with
> trn2 support in [cszhz/xpu-perf](https://github.com/cszhz/xpu-perf) (that tree
> is preserved on the `legacy-neuron` branch), and rebased here onto current
> upstream. Verified on trn2.3xlarge on both runtimes, plus a full sweep of every
> workload file on eager — see [Reference numbers](#reference-numbers). Measured
> on one NeuronDevice only, so `world_size > 4` and cross-device collectives
> remain untested.

## Quick start

### 1. Pick a runtime

`detect_neuron_runtime()` chooses by looking for `torch_xla`; override with
`XPU_PERF_NEURON_RUNTIME=xla|eager|auto`. The choice is reported as
`neuron_runtime` in `get_backend_info()` — **check it before trusting a report.**
The backend is only offered when `/dev/neuron*` exists.

| | `xla` | `eager` |
|---|---|---|
| Requires | `torch_xla` + `libneuronxla` (PJRT plugin) | `torch_neuronx` only, currently ships as a container image |
| Dispatch | lazy, traced to HLO, compiled by `neuronx-cc` | eager, op by op |
| First-run cost | minutes to hours per shape | usually none, but see [the caveat](#a-run-that-looks-hung-is-usually-compiling) |

### 2. Install

**eager** — build the image (the native base ships no reporting stack, so
`launch.py` dies on `import prettytable` before it reaches the device), then run
with `--privileged`, without which `import torch_neuronx` fails with
`Failed to get Neuron instance information. Status: 1`:

```bash
cd projects/micro_perf

docker build -f vendor_ops/NEURON/tools/Dockerfile.eager \
    -t xpu-perf-eager:latest vendor_ops/NEURON/tools

docker run --rm --privileged \
    -v "$PWD/../..":/xpu-perf -w /xpu-perf/projects/micro_perf \
    -e PYTHONPATH=/xpu-perf/src xpu-perf-eager:latest \
    python launch.py --backend NEURON --device 0 \
    --workload workloads/neuron_smoke/gemm.json
```

The base image in `Dockerfile.eager` is the beta the reference numbers were
measured with, and it lives in an AWS-internal ECR repository. If you cannot pull
it, edit the `FROM` — nothing here depends on the tag beyond `import
torch_neuronx` working.

**xla** — Neuron SDK 2.x with `torch-neuronx` >= 2.1, `torch-xla`, `neuronx-cc`
and `aws-neuronx-runtime-lib`, all preinstalled on the
[Neuron DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/index.html),
plus `pip install -e .` from the repo root. If you are building a venv yourself,
`torch-neuronx` is not optional and a `vllm-neuron` inference venv is not a
substitute — see
[IMPLEMENTATION.md](IMPLEMENTATION.md#torch-neuronx-is-not-optional-xla-runtime),
because the failure mode is a report full of host-CPU numbers under a NEURON
label.

```bash
python3 -m venv ~/neuron_venv
~/neuron_venv/bin/pip install torch-neuronx neuronx-cc \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com
export PATH=$HOME/neuron_venv/bin:$PATH   # the bin dir, not just the interpreter
```

### 3. Run

```bash
cd projects/micro_perf

# smoke test: the backend end to end, no long compile
python launch.py --backend NEURON --device 0 --task_dir workloads/neuron_smoke

# one workload
python launch.py --backend NEURON --device 0 \
    --workload workloads/basic/tensor_gemm_ops/gemm.json

# name ops explicitly to skip the pathological ones (see below)
python launch.py --backend NEURON --device 0 \
    --task_dir workloads/basic/vector_index_ops \
    --task embedding,index_select,index_add
```

Collectives need two extra settings, both mandatory:

```bash
XPU_PERF_ENGINES=XCCLEngine XPU_PERF_XCCL_READY_TIMEOUT_S=2400 \
    python launch.py --backend NEURON --device 0,1,2,3 \
    --workload workloads/neuron_smoke/all_reduce.json
```

- **`XPU_PERF_ENGINES` must name exactly one engine.** Otherwise a
  `ComputeEngine` worker and an `XCCLEngine` worker land on each core and fight
  over it (`Logical Neuron Core(s) not available ... Available:0`). One
  NeuronCore is reservable by exactly one process, unlike a GPU — so collectives
  and non-collectives cannot share a launch. The trap: **an op whose engine is
  excluded is dropped with no diagnostic and exit code 0**, and engine membership
  does not follow the workload directory. `device2device` lives in
  `workloads/xccl_ops/` but registers under `ComputeEngine`
  (`op_defs/basic_ops/xccl_ops.py`), so an `XCCLEngine` launch measures none of
  its 76 cases and reports success. Always
  compare the case count in the report against what the launcher enumerated.
- **`XPU_PERF_XCCL_READY_TIMEOUT_S`** raises the upstream 60 s wait for rank 0 to
  finish its warmup `all_reduce`, which on a cold cache is tens of minutes of
  `neuronx-cc`. On expiry the parent exits and its children keep the cores, so
  the *next* launch fails too — see [Troubleshooting](#troubleshooting).

**Use all four cores.** `world_size=4` is the only size that runs every
collective — `all_to_all` rejects 2 outright (`supported sizes: 4, 8, 16, or
multiples of 32`, a collective-library rule, not an instance-type one) — and
every other collective is 1.1-2.8x faster at 4 than at 2. On the XLA runtime one
launch benches one world size, since sub-world groups do not complete there; on
eager they do.

For a whole-repo sweep do not use `--task all`; see
[Reproducing the full sweep](#reproducing-the-full-sweep) for why and what to use
instead.

## Reference numbers

All on trn2.3xlarge, one logical NeuronCore (a trn2 chip is 4 logical cores at
the default LNC=2). Read them with [Reading the numbers](#reading-the-numbers) —
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
[Reading the numbers](#reading-the-numbers);
`workloads/llm/single_test_ops/ccl_ops.json` is the only file with no runnable
case at all (it asks for `world_size: 8`).

Memory-bound ops. A quarter of the chip's 2.9 TB/s is ~725 GB/s, so that is the
ceiling to read these against:

| Op | Best mem_bw | Cases |
|---|---|---|
| `device2device` | 648.7 GB/s | 76 |
| `reduce_sum` | 639.0 GB/s | 33 |
| `index_select` | 631.6 GB/s | 44 |
| `embedding` | 631.5 GB/s | 44 |
| `softmax` | 495.8 GB/s | 33 |
| `topk` | 476.3 GB/s | 147 |
| `reduce_max` / `reduce_min` | 177.0 / 176.6 GB/s | 33 each |
| `index_add` | 98.4 GB/s | 66 |
| `gather` | 1.34 GB/s | 16 of 44 (compiler wedge) |
| `scatter` | 0.8 GB/s | 5 of 44 (compiler wedge) |

`reduce_max`/`reduce_min` at 177 GB/s against `reduce_sum` at 639 is a 3.6x gap
between three reductions over the same shapes, so the max/min lowering is leaving
bandwidth on the table. `gather` and `scatter` are a different phenomenon — see
[Two index ops are pathologically slow](#two-index-ops-are-pathologically-slow).

Compute, best per op: `moe_gating_gemm` 21.9 TFLOPS at **MFU 48.4%**;
`quant_matmul` plateaus at 12-16 "TOPS" but see
[the quantized ops are a bf16 simulation](#the-quantized-ops-are-a-bf16-simulation-on-every-backend);
`rms_norm` MFU 0.34%, which is correct for a memory-bound op.

Collectives, best `bus_bw`:

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
see [Troubleshooting](#troubleshooting). Cap `batch_size` at 262,144 (1 GiB at
fp32, far past the bandwidth plateau) to avoid it entirely;
`tools/cap_xccl_workloads.py` does this.

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

### Most `workloads/llm/` cases do not run

| Workload | Cases | Measured | Note |
|---|---|---|---|
| `vendor_test/flash_attention.json` | 588 | **0** | 420 rejected by the base op def, 168 by this backend (paged cache) |
| `vendor_test_demo/flash_attention.json` | 9 | **0** | same, plus 2 `attn_mode=decode` |
| `single_test_ops/fa_ops.json` | 11 | **0** | every case sets `block_size: 512` and GQA `[80, 8, 128]` |
| `vendor_test/quant_matmul.json` | 736 | 85, of 184 runnable | only the int8 quarter is runnable; the other 552 are `float8`/`mxfloat8`/`mxfloat4` and the base op def rejects them. A 5 h timeout cut the rest |
| `vendor_test/moe_quant_group_gemm.json` | 1380 | **15**, of 276 runnable | same dtype gate, but far worse per case: 15 in 4 h, ~16 min of wall clock each |
| `single_test_ops/ccl_ops.json` | — | **0** | asks for `world_size: 8`; a trn2.3xlarge has 4 logical cores |

So the flash_attention figures above come from
`workloads/neuron_smoke/flash_attention.json`, the only flash_attention workload
in the repo with a runnable case here: no `block_size`, MHA `[8, 8, 128]`,
all-bfloat16. `flash_fwd` takes one contiguous K/V block, so only prefill is
expressible, and it also needs all-bfloat16 with a linear cache, MHA
(`q_head_num == kv_head_num`), `batch_size == 1` and `cache_len == 0`. The eager
`torch` provider accepts the same envelope so both
runtimes report the same cases, even though SDPA is more general — GQA is
excluded deliberately, because expanding kv heads needs a `repeat_interleave`,
a real copy that would land inside the timed region.

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

### Two index ops are pathologically slow

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
`.view(N, 1).expand(N, dim_size)`), which lowers to per-element indexed access.
That is the whole 470x difference — **not a hardware property.**

Both eventually hang the compiler outright, so give any run including them a hard
timeout or exclude them by name (`parse_tasks` honours comma-separated
`--task` lists). One caveat on the numbers: `GatherOp` inherits `prepare_args`
from `IndexSelectOp`, which declares a 1-D index, while `GatherOp.create_tensors`
builds a 2-D one — so the declared `io_bytes` does not describe the tensors the op
creates, and `gather`'s `mem_bw` should be read as approximate either way.

### Known unsupported

Two different things are easy to conflate, so they are kept apart: a case the
*base op definition* never implemented, which no vendor can fix from a
`vendor_ops` directory, and a case this backend genuinely cannot run.

| Cases | Blocker | Whose limit |
|---|---|---|
| `flash_attention` with any quantized dtype | `op_defs/llm_ops/flash_attention.py` accepts all-bfloat16 or bfloat16 + `int8` cache, and raises on everything else | base op def |
| `quant_matmul` / `moe_quant_group_gemm` with `float8` / `mxfloat8` / `mxfloat4` / `int4` weights | both base impls accept only `int8/int8/int8 -> bfloat16` | base op def |
| `flash_attention` with a paged cache, `attn_mode=decode`, or GQA | neither the NKI `flash_fwd` kernel nor native SDPA covers those | this backend |
| `p2p` | needs send/recv across multiple NeuronDevices | this backend |
| collectives above ~1 GiB per rank | the 24 GiB per-core ceiling above | this backend (capacity) |

`scale_dynamic_quant`, `add_rms_norm_dynamic_quant`,
`head_rms_norm_dynamic_quant`, `swiglu_dynamic_quant` and `dequant_kv_cache`
appear in **no workload JSON in the repo**, on any backend. They are untested,
not unsupported. Conversely `moe_scatter_dynamic_quant`,
`moe_quant_group_gemm_combine`, `quant_matmul` and `moe_quant_group_gemm` all do
run on the eager stack (int8 only for the last two) — an earlier version of the
table claimed Neuron could not provide int8 tensors, which was true of the XLA
path and is not true here.

Only three ops need vendor code at all; everything else runs its `op_defs` base
implementation through the `base` provider:

| Op | Provider | Runtime | Why |
|---|---|---|---|
| `gemm` | `torch` | both | rejects `tfloat32` (an NVIDIA format) and `int8` (not lowered through `torch.matmul`) |
| `all_gather` | `torch` | both | base uses `dist.all_gather_into_tensor`, unimplemented on the `xla` backend; the `neuron` backend implements it, so eager reproduces the base behaviour |
| `flash_attention` | `nki` / `torch` | `xla` / `eager` | no base implementation exists; NKI `flash_fwd` on XLA, `scaled_dot_product_attention` on eager |

## Reproducing the full sweep

`tools/` holds what produced the eager full-sweep numbers. `launch.py --task all`
does not survive this workload set; the comment block at the top of
`run_full_sweep.sh` lists the four separate reasons.

```bash
cd projects/micro_perf

# 1. Build the image (as in Quick start).
docker build -f vendor_ops/NEURON/tools/Dockerfile.eager \
    -t xpu-perf-eager:latest vendor_ops/NEURON/tools

# 2. Sweep, on an idle trn2.3xlarge. Budget a day.
IMAGE=xpu-perf-eager:latest setsid nohup vendor_ops/NEURON/tools/run_full_sweep.sh &

# 3. Per-run accounting: cases tried, measured, and grouped rejection reasons.
python3 vendor_ops/NEURON/tools/analyze_sweep.py /tmp/neuron_sweep.log

# 4. One op's full scaling curve instead of a summary line.
python3 vendor_ops/NEURON/tools/analyze_sweep.py /tmp/neuron_sweep.log all_to_all
```

`IMAGE`, `REPO`, `LOG`, `RESULTS` and `DOCKER` all come from the environment, so
a different image or a non-`sudo` docker needs no edit. Two things to know:

- **A killed run writes no report, but its results are recoverable.** micro_perf
  writes CSV/jsonl only when a launch finishes, so a watchdog kill loses
  everything on disk — but every case is printed to stdout as it completes.
  `recover_from_log.py <log> <outdir>` rebuilds the CSVs from that. Several of
  the numbers above came through this path.
- **Measured counts are not enumerated counts.** Compare `analyze_sweep.py`'s
  `measured` against the table in
  [Most `workloads/llm/` cases do not run](#most-workloadsllm-cases-do-not-run),
  not against what the launcher printed at startup.

The script deliberately skips `workloads/llm/single_test_ops/ccl_ops.json` and
caps `workloads/xccl_ops/` sizes. Both exclusions are load-bearing, not tidying:
the second is what keeps a rank from OOMing, which would hang the launch forever.

## Troubleshooting

### A run that looks hung is usually compiling

On the XLA runtime this is normal: every distinct shape is compiled by
`neuronx-cc` on first use, 5-15 minutes per op, hours for a sweep. **On eager it
also happens** — "eager" is not a promise that `neuronx-cc` never runs. Most ops
dispatch to a prebuilt kernel, but one the runtime has no kernel for falls back
to the full XLA-era compile: `gather` at bf16 / `dim_size=8192` sat in
`neuronx-cc compile module.mlir --framework XLA --target trn2 --lnc 2 -O1` for
over two hours at 199% CPU before it was killed. So check for a `neuronx-cc` or
`walrus_driver` process before assuming a deadlock, and budget a per-shape
timeout on both runtimes.

`find /var/tmp/neuron-compile-cache -name "*.neff" | wc -l` shows cache growth.
Run under tmux or screen so an SSH drop does not kill a compile.

### `timeout docker run` does not bound anything

It signals the docker *client*; the container is a child of the daemon, so the
launch is orphaned and keeps holding the core. Use a watchdog that runs
`docker kill <name>` against the container — `tools/run_full_sweep.sh` does.

### A collective run hangs forever after a rank dies

`XCCLEngine` has no liveness check and no timeout on the results it waits for, so
when a worker OOMs during compilation the launcher blocks indefinitely: `ps`
shows the workers in state `Z` and the launcher in `sleep`, with nothing on
stdout after the last completed case. It sat 35 minutes before being killed from
outside. Bound collective runs externally, and cap sizes so the OOM never
happens (see the 24 GiB ceiling above).

### An op produced no output, no error, and exit code 0

Its engine is not in `XPU_PERF_ENGINES`. See the note in
[Quick start](#3-run) — `device2device` is the one that catches people.

### A killed run still holds the cores

```bash
pkill -9 -f multiprocessing && pkill -9 -f neuronx-cc
find /var/tmp/neuron-compile-cache -name "*.lock" -delete   # killed compilers leave locks
neuron-ls   # confirm the cores are free
```

A timed-out collective launch is the common case: the parent exits but its
non-daemon children do not. Note that `pkill -f launch.py` matches its own
command line if the invoking shell's arguments contain that text, so bracket the
pattern:

```bash
ps -o pid=,cmd= -u "$USER" | grep -E "[l]aunch\.py|[s]pawn_main"
pkill -9 -f "[l]aunch\.py"; pkill -9 -f "[s]pawn_main"
```

### `NRT_FAILURE` / `Logical Neuron Core(s) not available ... (cores busy, ret=-16)`

Another process owns the cores — or, on a multi-device launch, one of the
single-process-per-core constraints (see
[IMPLEMENTATION.md](IMPLEMENTATION.md#one-process-per-neuroncore)). `neuron-ls`
attributes cores to a PID but can miss a holder in another container, so check
directly:

```bash
sudo bash -c 'for p in /proc/[0-9]*; do for fd in "$p"/fd/*; do
  case "$(readlink "$fd" 2>/dev/null)" in /dev/neuron*)
    echo "$(basename $p) $(tr -d "\0" < $p/cmdline)"; break;; esac; done; done'
sudo docker ps        # a second container is an easy holder to miss
```

Match `/dev/neuron*`, not any path containing `neuron` — the looser pattern also
matches a process whose *own executable* lives under `/opt/aws/neuron`, which
says nothing about the device. And `neuron-top` / `neuron-monitor` appear in this
list: they open `/dev/neuron0` to read counters without reserving a core, so
watching the device looks identical to using it. Filter them out before treating
a non-empty list as busy, or waiting for an idle machine will wait forever.

### Suspiciously fast results

A gemm reporting ~1,900 TFLOPS on the XLA runtime is a pruned graph, not a fast
gemm — see
[IMPLEMENTATION.md](IMPLEMENTATION.md#xla-compilation-dominates-first-run-time).
`torch_neuronx.get_fallback_ops()` lists ops that silently ran on CPU (across a
basic sweep only `aten::normal_` falls back), and `get_backend_info()` reports
which runtime was selected.
