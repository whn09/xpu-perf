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
from [MFU](#mfu) — 166.75 for bf16/fp16, 45.25 for fp32, 324.75 for fp8.

**The two fp8 rows are in the table for completeness and should not be read as
fp8 hardware numbers** — they are the eager path, in an e4m3 encoding this chip
does not implement. Compiled, in `e5m2`, the same core does **245.50 TFLOPS at
75.6% of its fp8 peak**. See
[fp8: the sweep measures the eager path, and the wrong e4m3 encoding](#fp8-the-sweep-measures-the-eager-path-and-the-wrong-e4m3-encoding).

**90% of the dense bf16 peak on one logical core is the headline number for this
chip**, and it is 20 points above the 1024x4096x4096 row in the table above: that
shape is too small to hide the ~60 us dispatch floor, and it is the shape a quick
smoke test uses. `quant_matmul` is not an int8 hardware number on any backend —
see [the quantized ops are a bf16 simulation](#the-quantized-ops-are-a-bf16-simulation-on-every-backend).

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

**Two exceptions.** The first is dtype-specific: fp8 `gemm` (medians 1.464 and
1.525) and fp32 `reduce_sum` (median 1.369) are the only groups whose *median* does
not scale, where every other dtype of every other op sits between 1.000 and 1.025.
For fp8 that is consistent with it timing a software up-cast rather than the matmul
engine — and the compiled path, which *does* reach the engines, scales cleanly
instead: 1.018x worst case over four cores
([details](#fp8-the-sweep-measures-the-eager-path-and-the-wrong-e4m3-encoding)). fp32 `reduce_sum`
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
[small-op latencies](#small-op-latencies-measure-the-runtime-not-the-chip) — so
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
| `device2device` | — | — | **648.7** |
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
parity](#two-index-ops-are-pathologically-slow-and-one-is-an-int64-index-away-from-parity).

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
problem](#writes-into-a-strided-slice-view-are-not-in-place), and it is real but
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
  [the `*_dynamic_quant` family](#the-_dynamic_quant-family-is-90-180x-off-and-it-is-one-shared-helper).
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
[Known unsupported](#known-unsupported)). All bf16, `head_dim` 128.

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
That is what the 15-33% of bandwidth reflects. The named fix is
`nkilib`'s `attention_tkg` (token-generation attention), which is installed in the
beta images and would be wired as a second provider rather than a change to this
one — see [What would close these gaps](#what-would-close-these-gaps).

For the same nine cases measured on an H100 with the same provider code, see
[`../GPU/README.md`](../GPU/README.md).

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

### fp8: the sweep measures the eager path, and the wrong e4m3 encoding

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

**1. `torch.float8_e4m3fn` is not a format Trainium2 has.** There are two e4m3
encodings. `f8e4m3` has infinities and a conventional NaN set; `f8e4m3fn` is
finite-only — no infinities, one NaN pattern, one extra exponent value of range.
They are not interchangeable and hardware implements one or the other. Trainium1
and Trainium2 implement `f8e4m3`. PyTorch's `torch.float8_e4m3fn` — which is what
CUDA uses, and what `TORCH_DTYPE_MAPPING` gives the workload's `float8_e4m3` — is
the other one. Ask the compiler for it and it says so by name:

```
[NCC_EVRF051] Data type F8E4M3FN is not supported on TRN1/TRN2. Target TRN3 or
later hardware, or use the --experimental-unsafe-fp8e4m3fn-as-fp8e4m3 flag to cast
F8E4M3FN to F8E4M3.
```

That flag does not exist in the `neuronx-cc` these numbers were taken with
(2.27.2878.0 — `compile --help` has no fp8 options at all), so on this stack
`float8_e4m3` has **no route to the tensor engines by any means**. `float8_e5m2`
has no such split and is supported directly.

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

**What is *not* claimed here.** The workload files are unchanged and the published
rows are still eager, so the table above stands as what `gemm.json` measures.
Getting the better number into the sweep needs `float8_e5m2` cases run through a
compiled provider, which is a different provider rather than a flag. And it is a
genuine cross-backend asymmetry that the H100's fp8 figure is an e4m3fn figure,
Trainium2 has no e4m3fn at all, and Trainium2's figure has to be quoted in e5m2 —
which `torch._scaled_mm` in turn rejects on CUDA
([details](../GPU/README.md#gemm)). The two chips have no fp8 format in common
that both stacks will multiply, which is worth knowing before planning a mixed
fleet.

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
all](#rotary_embedding-is-not-a-neuron-result-at-all).

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
| `flash_attention` chunked prefill (`cache_len > 0`, `q_len > 1`) or multi-token decode | `is_causal` is top-left aligned; these need bottom-right | this backend (correctness) |
| `flash_attention` GQA / decode / `batch_size > 1` on the **XLA** runtime | the NKI `flash_fwd` kernel takes one contiguous K/V block for one head group; eager SDPA runs all three | this backend |
| `flash_attention` prefill at `q_len: 32768` | SDPA does not complete at that length — see below | this backend (capacity) |
| `p2p` | needs send/recv across multiple NeuronDevices | this backend |
| collectives above ~1 GiB per rank | the 24 GiB per-core ceiling above | this backend (capacity) |

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
[Attention](#attention) — it is failing **inside** a fused kernel, not for want of
one, which is why there is nothing to fall back to. The case was dropped rather
than left in the file to look like a hung machine. If you need long-context
attention numbers on Trainium, take them on the XLA runtime through the NKI
provider, at MHA.

Only three ops need vendor code at all; everything else runs its `op_defs` base
implementation through the `base` provider:

| Op | Provider | Runtime | Why |
|---|---|---|---|
| `gemm` | `torch` | both | rejects `tfloat32` (an NVIDIA format) and `int8` (not lowered through `torch.matmul`); accepts `float8_e4m3` / `float8_e5m2`, which the base op def gates out — see [fp8 is measured on the eager path, in an encoding this chip does not implement](#fp8-the-sweep-measures-the-eager-path-and-the-wrong-e4m3-encoding) |
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
2. **`attention_tkg`, for decode — and nothing for prefill.** Decode cannot reach
   the built-in NKI flash rewrite (see [Attention](#attention)) and is leaving
   2.6-5.7x of this core's bandwidth unused. `nkilib` ships a token-generation
   attention kernel for exactly this shape. It goes in as a third
   `flash_attention` provider, so it neither perturbs the existing eager numbers
   nor needs the XLA runtime. **Prefill is a different case and is already done:**
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
   point it at.** The published 99x is the eager path in an encoding this chip does
   not implement; compiled `e5m2` reaches 75.6% of the fp8 peak and closes the
   per-chip gap to 1.56x
   ([details](#fp8-the-sweep-measures-the-eager-path-and-the-wrong-e4m3-encoding)).
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
  all](#rotary_embedding-is-not-a-neuron-result-at-all). One fused kernel would be
  ~17x on both.
- **Writes into a strided slice view are not in-place**, worth 2.08x on
  `rotary_embedding`'s write step. Sizing `packed_qkv` to the heads actually written
  makes the destination contiguous; see [Writes into a strided slice view are not
  in-place](#writes-into-a-strided-slice-view-are-not-in-place).
- **`store_kv_cache` has no runnable case on any backend**, for two independent
  reasons in the op def. Both are in [Known unsupported](#known-unsupported). Until
  they are fixed there is no `store_kv_cache` number to compare, on this backend or
  the GPU.

## Reproduce one row at a time

The full sweep is most of a day, and checking one figure in the tables above does
not need it. Every row came from exactly one `launch.py` call, and each of those
calls has a label in one of the two scripts:

```bash
cd projects/micro_perf

# What is there. Prints label, device list, budget and launch arguments; touches
# no device, so it is safe on a machine someone else is using.
LIST=1 vendor_ops/NEURON/tools/run_full_sweep.sh
LIST=1 vendor_ops/NEURON/tools/run_new_workloads.sh

# One row.
ONLY=single_gemm_ops IMAGE=xpu-perf-eager:latest \
    vendor_ops/NEURON/tools/run_full_sweep.sh
tail -f /tmp/neuron_sweep.log        # both scripts log to a file, not the terminal

# Several. Commas or spaces, either works in either script.
ONLY=basic_index_ok,basic_index_slow IMAGE=xpu-perf-eager:latest \
    vendor_ops/NEURON/tools/run_full_sweep.sh
```

`ONLY` matches whole labels only, so `ONLY=gemm` selects nothing rather than
quietly running `single_gemm_ops`. Everything else is unchanged: the log format and
the `$RESULTS/<label>/` layout are identical to a full run, so
`analyze_sweep.py` reads a one-label log the same way. The
[waiting-for-an-idle-chip behaviour](#reproducing-the-full-sweep) still applies to
each run — that is the point of going through the script rather than calling
`launch.py` by hand.

**24 of the 28 labels in `run_full_sweep.sh` are single-core** (`--device 0`, one
logical NeuronCore, a quarter of a chip at LNC=2). Elapsed times are from the logs
the published numbers came from, on an otherwise idle trn2.3xlarge:

| Label | Workload | Dev | Elapsed | Covers |
|---|---|---|---|---|
| `basic_tensor_gemm_ops` | `basic/tensor_gemm_ops` (all) | 0 | 4,939 s | [gemm](#eager-runtime-full-sweep) — 856 cases, the longest single-core run |
| `basic_vector_linear_ops` | `basic/vector_linear_ops` (all) | 0 | not recorded | [memory-bound](#memory-bound-ops) |
| `basic_vector_activation_ops` | `basic/vector_activation_ops` (all) | 0 | not recorded | " |
| `basic_vector_sfu_ops` | `basic/vector_sfu_ops` (all) | 0 | not recorded | " |
| `basic_vector_norm_ops` | `basic/vector_norm_ops` (all) | 0 | 413 s | " |
| `basic_vector_reduction_ops` | `basic/vector_reduction_ops` (all) | 0 | 591 s | " |
| `basic_index_ok` | `basic/vector_index_ops`, `embedding,index_select,index_add` | 0 | not recorded | the three index ops that are fine |
| `basic_index_slow` | same dir, `gather,scatter` | 0 | **killed at 1,240 s** | [the two that are not](#two-index-ops-are-pathologically-slow-and-one-is-an-int64-index-away-from-parity). Expect a kill; results come back with `recover_from_log.py` |
| `single_gemm_ops` | `llm/single_test_ops/gemm_ops.json` | 0 | 182 s | `moe_gating_gemm` (48% MFU), `quant_matmul`, `moe_quant_group_gemm` |
| `single_fa_linear_ops` | `llm/single_test_ops/fa_linear_ops.json` | 0 | **killed at 784 s** | [attention](#eager-runtime) — prefill, decode, GQA. `run_new_workloads.sh` gives it 21,600 s instead of 5,400 for this reason |
| `single_fa_ops` | `llm/single_test_ops/fa_ops.json` | 0 | 15 s | **0 cases** — paged, and no Neuron provider takes a block table |
| `single_pre_fa_ops` | `llm/single_test_ops/pre_fa_ops.json` | 0 | 177 s | `rotary_embedding`; `store_kv_cache` runs 0 cases |
| `single_moe_dispatch_ops` | `llm/single_test_ops/moe_dispatch_ops.json` | 0 | 32 s | `moe_scatter_dynamic_quant` |
| `single_moe_combine_ops` | `llm/single_test_ops/moe_combine_ops.json` | 0 | 61 s | `moe_gather` |
| `single_norm_ops` | `llm/single_test_ops/norm_ops.json` | 0 | 873 s | [norm/activation/MoE](#the-_dynamic_quant-family-is-90-180x-off-and-it-is-one-shared-helper) |
| `single_activation_ops` | `llm/single_test_ops/activation_ops.json` | 0 | 312 s | " |
| `single_moe_gating_ops` | `llm/single_test_ops/moe_gating_ops.json` | 0 | 123 s | " — and 9.7x faster than the same file on an H100 (1,191 s) |
| `single_quant_ops` | `llm/single_test_ops/quant_ops.json` | 0 | 172 s | " |
| `quant_matmul` | `llm/vendor_test/quant_matmul.json` | 0 | **killed at 1,988 s** | 85 of 184 cases; recover from the log |
| `moe_quant_group_gemm` | `llm/vendor_test/moe_quant_group_gemm.json` | 0 | **killed at 565 s** | 15 of 276 — [~16 min of wall clock per 2.7 s measurement](#most-workloadsllm-cases-do-not-run) |
| `fa_vendor_test` | `llm/vendor_test/flash_attention.json` | 0 | 16 s | **0 cases** |
| `demo_quant_matmul` | `llm/vendor_test_demo/quant_matmul.json` | 0 | 41 s | the small versions of the three above |
| `demo_moe_quant_group_gemm` | `llm/vendor_test_demo/moe_quant_group_gemm.json` | 0 | 38 s | " |
| `demo_flash_attention` | `llm/vendor_test_demo/flash_attention.json` | 0 | 17 s | " (0 cases) |
| `xccl4` | capped copies of `xccl_ops/` | 0,1,2,3 | 762 s | the 4 collectives plus `device2host`/`host2device`. Needs `XPU_PERF_ENGINES=XCCLEngine`, which the script sets |
| `d2d` | capped `xccl_ops/device2device.json` | 0,1 | 142 s | `device2device`, 648.7 GB/s |
| `chip4_gemm` | `basic/tensor_gemm_ops` (all) | 0,1,2,3 | 1,480 s | whether "x4" is real |
| `chip4_mem` | `basic/vector_linear_ops` (all) | 0,1,2,3 | 187 s | " |

"not recorded" means that label ran under an earlier version of the script that did
not print an elapsed line, not that it failed. **A kill is not a failure either**:
micro_perf writes its CSVs only when a launch finishes, but every case is printed as
it completes, so `recover_from_log.py <log> <outdir>` rebuilds the reports — several
of the published numbers came through that path. Whether a 137 was the watchdog or a
hand kill to free the chip is not recoverable from the log, so treat these elapsed
figures as "how long it ran", not "how long it needs".

`run_new_workloads.sh` has 12 labels of its own, 9 of them single-core, and takes
the same `LIST`/`ONLY`. The two overlap deliberately: `single_norm_ops`,
`single_activation_ops`, `single_moe_gating_ops`, `single_quant_ops` and
`single_fa_linear_ops` appear in both, with a larger budget in the newer script.
`core1_gemm` there is `basic_tensor_gemm_ops` under a different name, paired with
`chip4_gemm` so `analyze_scaling.py` can join them.

The GPU side is label-for-label the same idea and much cheaper — 13 labels, all
single-GPU, 42 minutes end to end. See
[GPU/README.md, Reproduce one row at a time](../GPU/README.md#reproduce-one-row-at-a-time);
two labels are spelled differently there (`gemm` for `basic_tensor_gemm_ops`, and
one `basic_vector_index_ops` instead of the `basic_index_ok`/`basic_index_slow`
split, which exists here only because `gather`/`scatter` need their own watchdog).

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

# 2b. Or just the workloads added after the first sweep, plus the 4-core runs.
#     Same machinery, about an hour. Log: /tmp/neuron_new.log.
IMAGE=xpu-perf-eager:latest setsid nohup vendor_ops/NEURON/tools/run_new_workloads.sh &

# 3. Per-run accounting: cases tried, measured, and grouped rejection reasons.
python3 vendor_ops/NEURON/tools/analyze_sweep.py /tmp/neuron_sweep.log

# 4. One op's full scaling curve instead of a summary line.
python3 vendor_ops/NEURON/tools/analyze_sweep.py /tmp/neuron_sweep.log all_to_all

# 5. Is "x4" real? Join a 1-core report against a 4-core one, shape by shape.
#    Only the matched pairs from step 2b are comparable; the labels are
#    core1_gemm/chip4_gemm and core1_reduction/chip4_reduction.
ONLY="core1_gemm chip4_gemm core1_reduction chip4_reduction chip4_moe" \
    IMAGE=xpu-perf-eager:latest vendor_ops/NEURON/tools/run_new_workloads.sh
python3 vendor_ops/NEURON/tools/analyze_scaling.py \
    /tmp/new_results/core1_gemm /tmp/new_results/chip4_gemm

# 6. The gather int64-vs-int32 finding, on device, in one file (~2 min).
sudo docker run --rm --privileged -v "$PWD":/w -w /w xpu-perf-eager:latest \
    python3 vendor_ops/NEURON/tools/probe_index_dtype.py

# 7. The in-place-write finding: the slope test, plus rotary_embedding
#    decomposed at the shape the table publishes (~4 min, needs a few GB HBM).
sudo docker run --rm --privileged -v "$PWD":/w -w /w xpu-perf-eager:latest \
    python3 vendor_ops/NEURON/tools/probe_inplace_write.py
```

`IMAGE`, `REPO`, `LOG`, `RESULTS`, `DOCKER` and `WAIT_BUDGET_S` all come from the
environment, so a different image or a non-`sudo` docker needs no edit. Three
things to know:

- **Both scripts refuse to start a run while anything else holds a NeuronCore**,
  and wait `WAIT_BUDGET_S` (default 1800) for it to clear before skipping that
  run. Raise it when you are queueing behind someone else's sweep. What they wait
  on is a process with `/dev/neuron*` open, plus any container with a `python` in
  it — an idle `docker run -it` shell left open after a sweep does not count.

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

`flash_attention` on the eager runtime is the case most likely to be mistaken for
a hang, because it is slow exactly once. A cold `80/8/128` GQA prefill at
`q_len: 4096` spent **over 13 minutes** in `walrus_driver --optlevel 2` and had
produced no result when it was killed; the same case measured **9,512.6 us** on
the next run, because `NKI_ENABLE_TRACE_CACHE=1` (the default) persists the kernel
cache across processes and the first run had populated it. A bare
`scaled_dot_product_attention` at that shape is ~33 ms, so if the first case of
`fa_linear_ops.json` has been silent for ten minutes, that is a compile and not a
broken vendor op — the check is a live `walrus_driver`, not elapsed time. Two
practical consequences: give an FA sweep a per-case budget in hours rather than
minutes, and warm each distinct shape once before timing anything you intend to
publish.

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

### `❌ 导入失败 ._<something>：attempted relative import beyond top-level package`

macOS AppleDouble sidecars in the synced tree, not a code problem. `rsync`/`scp`
from a Mac leaves a 220-byte `._foo.py` next to every `foo.py` it copied, and
`parse_vendor_ops` (`core/common_utils.py`) imports every `*.py` it finds, so it
tries `..foo` and gets two leading dots. The ops themselves still register —
check the `Provider: torch` table for the real answer. `ls` hides these files
because the name starts with a dot:

```bash
find ~/xpu-perf -name "._*" -delete
rsync -a --exclude="._*" ...        # or pass --iconv=. / use COPYFILE_DISABLE=1
```

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
