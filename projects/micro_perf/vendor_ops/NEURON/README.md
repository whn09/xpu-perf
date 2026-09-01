# NEURON vendor ops (AWS Trainium / Inferentia)

micro_perf support for AWS Trainium and Inferentia through the
[Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/), on either of the
two PyTorch integrations Neuron offers — PyTorch/XLA or the newer PyTorch-native
eager stack. See [Two runtimes](#two-runtimes).

The backend class lives at `src/xpu_perf/micro_perf/backends/NEURON/`; this
directory holds the vendor op implementations and the default environment.

> **Validation status** — this is a port. The Neuron backend was originally
> written against the pre-refactor `micro_perf/` layout in
> [davidshtian/xpu-perf](https://github.com/davidshtian/xpu-perf) and extended
> with trn2 support and XCCL fixes in
> [cszhz/xpu-perf](https://github.com/cszhz/xpu-perf); that tree is preserved on
> the `legacy-neuron` branch of this fork along with its measured results. The
> code here is rebased onto current upstream (`src/xpu_perf/` package +
> `ProviderRegistry` vendor mixins).
>
> Verified on a trn2.3xlarge (2026-08-31): device discovery,
> `get_backend_info()`, op/provider registration, `env.json` handling, report
> writing, the deferred-`torch_xla` import contract, single-core measurements
> for `gemm`, `add`, `softmax` and the NKI `flash_attention` kernel, and
> two-core `all_reduce` / `all_gather`. `gemm` lands within a few percent of the
> `legacy-neuron` baseline measured in March, which is the check that the
> numbers are real — see
> [Reference numbers](#reference-numbers-measured-on-trn23xlarge).
>
> Verified on a second trn2.3xlarge (2026-09-01) on the **PyTorch-native eager
> stack** (Beta 4 image, torch 2.12.1 / torch-neuronx 2.12.3 / neuronx-cc
> 2.27.2878 / nki 0.6.0, host driver 2.30.2.0, no `torch_xla`): the same six
> cases, with `flash_attention` running through native SDPA instead of NKI. Run
> twice, on hosts whose driver did and did not match the image's expected
> version, agreeing to within run-to-run noise. See
> [Two runtimes](#two-runtimes).
>
> Measured on one NeuronDevice only (4 logical cores at LNC=2), so
> `world_size` > 4, cross-device collectives, and the ops listed under
> [Known unsupported](#known-unsupported) remain untested here.

## Supported hardware

- AWS Inferentia2 — inf2 instances
- AWS Trainium — trn1 / trn1n instances
- AWS Trainium2 — trn2 instances

## Two runtimes

Neuron ships two mutually exclusive PyTorch integrations, and this backend
supports both:

| | `xla` | `eager` |
|---|---|---|
| Dispatch | lazy, traced into HLO, compiled by `neuronx-cc` | eager, op by op |
| Requires | `torch_xla` + `libneuronxla` (PJRT plugin) | `torch_neuronx` only |
| Device string | `xla` | `neuron` (a privateuse1 backend) |
| Sync | `xm.mark_step()` + `xm.wait_device_ops()` | `torch_neuronx.synchronize()` |
| PG backend | `xla` | `neuron` |
| First-run cost | minutes to hours of compilation per shape | usually none, but see below |

"Eager" is not a promise that `neuronx-cc` never runs. Most ops dispatch straight
to a prebuilt kernel, but an op the runtime has no kernel for still falls back to
building one, and that fallback is the full XLA-era compile. `gather` at
`bfloat16` / `dim_size=8192` sat in `neuronx-cc compile module.mlir --framework
XLA --target trn2 --lnc 2 -O1` for over two hours at 199% CPU before it was
killed. So budget a per-shape timeout on the eager path too, and be aware that a
run which appears hung is more likely compiling than deadlocked — check for a
`neuronx-cc` or `walrus_driver` process before assuming the worst.

`detect_neuron_runtime()` picks one by looking for `torch_xla` with
`importlib.util.find_spec` — availability only, so the check itself never claims
a NeuronCore or drags `torch_xla` into a process that is about to fork workers.
Override with `XPU_PERF_NEURON_RUNTIME=xla|eager|auto` (default `auto`). Where
`torch_xla` exists the XLA path is chosen, so nothing about an existing
installation changes. The selected runtime is reported as `neuron_runtime` in
`get_backend_info()` — check it before trusting a report.

Most of this file describes the XLA path, because that is where nearly all of
the sharp edges are. The eager path is simpler; what follows is everything that
differs.

### Running the eager stack

The PyTorch-native stack currently ships as a container image rather than
through pip. It needs `--privileged`, otherwise `import torch_neuronx` dies with
`Failed to get Neuron instance information. Status: 1`:

```bash
docker run --rm --privileged \
    -v "$PWD":/xpu-perf -w /xpu-perf/projects/micro_perf \
    -e PYTHONPATH=/xpu-perf/src <native-image> \
    python launch.py --backend NEURON --device 0 \
    --workload workloads/neuron_smoke/gemm.json
```

The image carries no reporting dependencies, so add `prettytable jsonlines
flask` to it (or to a derived image) first.

### Timing: never use `torch_neuronx.Event`

It looks like the CUDA-event equivalent and it is not one. `elapsed_time()`
returns 25-30 us regardless of the work submitted: a 1024x4096x4096 bf16 gemm
and an 8192x4096x4096 one both "take" ~24 us, which would be 1,154 and 11,273
TFLOPS. It is not measuring device execution. `_core_perf_eager()` therefore
times with `time.perf_counter_ns()` around a single
`torch_neuronx.synchronize()`, which scales linearly with FLOPs. No keepalive
reference is needed the way the XLA path needs one — eager dispatch has already
executed the op by the time `core_run()` returns, so there is no graph for dead
code elimination to prune.

### Use the device string `neuron`, never `neuron:0`

Under `init_process_group` the native backend sets each rank's local device
start index to its local rank, so on rank 1 the only valid index is 1.
`torch.empty`/`torch.randn` with `neuron:0` raise there — and `torch.full`
*silently returns a `neuron:1` tensor*, which is worse than raising. Bare
`neuron` always resolves to the current device, so
`get_torch_device_name()` returns it unindexed on this runtime.

### The process group comes before the device

This inverts the XLA ordering. `torch_neuronx.distributed.backend`'s
`_neuron_runtime_setup` asserts the runtime is *not* yet initialised, because it
wants to assign cores from `LOCAL_RANK` itself, publish `NEURON_RT_ROOT_COMM_ID`
through the store and run an nrt barrier. So on this runtime nothing may touch
the device until `init_process_group` has returned, and the check that the
process really is on a NeuronCore has to come *after* it rather than before.
`LOCAL_WORLD_SIZE` must be set too, or the backend infers local rank by
rendezvousing on IP addresses; `set_device()` defaults it from `WORLD_SIZE`.

### Sub-world process groups work here

Unlike the XLA path, a group narrower than the world does complete (verified
with a `ranks=[0,1]` group in a `world_size=4` job), so the "bench one
world_size per launch" restriction does not apply — `perf()` gates that skip on
the XLA runtime only. `dist.all_gather_into_tensor` is also implemented, so the
`xm.all_gather` override is unnecessary and the `all_gather` vendor op
reproduces the base behaviour instead.

### No NKI attention kernel

`neuronxcc.nki.kernels.attention.flash_fwd` is traced into HLO and fails on this
runtime deep inside the kernel with `No module named 'torch_neuronx.pyhlo'`. The
native entry point is `torch_neuronx.wrap_nki` (note that
`torch_neuronx.nki_kernel` is a module, not a callable), but it expects kernels
written against the standalone `nki` package, which as of 0.6.0 ships no kernel
library to point it at. So `flash_attention` is measured through native
`scaled_dot_product_attention` instead — see
[Op coverage](#op-coverage). Both providers are registered conditionally on the
detected runtime, so exactly one is available.

### Eager dispatch costs ~55-65 us per op, and that is the floor

This is the one number to internalise about this runtime. Measured on
trn2.3xlarge with a chain of `silu(x + b)` on 1024x1024 bf16 tensors, varying
only how many ops sit in one region:

| Ops in region | eager | `torch.compile(backend="neuron")` | speedup |
|---|---|---|---|
| 1 | 139.9 us | 102.7 us | 1.36x |
| 4 | 512.2 us | 97.2 us | 5.27x |
| 16 | 2,035.0 us | 205.9 us | 9.89x |
| 64 | 8,236.4 us | 706.5 us | 11.66x |

Eager scales linearly at about 64 us per op while the compiled region barely
moves, which puts the cost squarely in dispatch, not in the arithmetic. So every
small-op figure in [Reference numbers](#reference-numbers-measured-on-trn23xlarge)
— `add` at 48.8 us, `softmax` at 53.8 us — is essentially *all* dispatch
overhead. **Those numbers measure the runtime, not the chip**, and no smaller
number is reachable on this stack. `gemm` at 1024x4096x4096 is the smallest
shape in the table where the arithmetic clearly dominates.

### torch.compile does not help micro_perf

It is the obvious thing to reach for, and the measurements say no. On Neuron
`torch.compile(backend="neuron")` is not eager-plus-fusion: it lowers through
`torch_mlir` to StableHLO and compiles a NEFF with neuronx-cc, so it is the
graph-compiled path again, entered through dynamo instead of LazyTensor. One
NEFF launch costs ~95-100 us against ~55 us for one eager dispatch, and
micro_perf times exactly one op per region — there is nothing to amortise the
launch over:

| Case | eager | compiled | speedup |
|---|---|---|---|
| gemm bf16 1024x4096x4096 | 282.9 us | 289.9 us | 0.98x |
| gemm bf16 2048x4096x4096 | 530.9 us | 539.9 us | 0.98x |
| add bf16 1024x1024 | 52.5 us | 75.5 us | 0.69x |
| softmax bf16 1024x1024 | 55.8 us | 75.3 us | 0.74x |
| add bf16 2048x1024 | 58.0 us | 73.0 us | 0.80x |
| softmax bf16 2048x1024 | 68.6 us | 79.7 us | 0.86x |
| sdpa bf16 causal 2048x8x128 | 346.5 us | 332.4 us | 1.04x |
| sdpa bf16 causal 4096x8x128 | 780.9 us | 781.8 us | 1.00x |

Large ops are unchanged (same tensor-engine kernel either way), small ops get
20-30% *worse*, and `flash_attention` gains nothing — so compiling does not
recover a fused attention kernel that eager SDPA was missing. A
`torch.compile` option is therefore not offered: it would never win, and it
would reintroduce the dead-code-elimination hazard that eager is immune to (see
[Timing](#timing-never-use-torch_neuronxevent)) for nothing.

Where it does win is fusion across many ops — the 11.6x above, and 1.27x on a
llama-shaped `gemm -> silu -> gemm`. That is a model-level concern, and
measuring single ops is what micro_perf is for.

Two notes if you do use it here anyway:

- **`dynamic=False` is mandatory.** Benchmarking the same function at two shapes
  makes dynamo mark the varying dimension dynamic, and neuronx-cc rejects that:
  `[NCC_EMOD025] Dynamic shape is not supported: instruction 'parameter' has
  shape 'bf16[?,4096]'`. The error can surface at an unrelated later device
  call, so also call `torch._dynamo.reset()` between cases.
- Compilation is 1.4-5.1 s per graph for 8-323 nodes — minutes-to-hours faster
  than the XLA path, but not free. `torch_neuronx.get_dynamo_metrics()` reports
  node count and lowering/compile time per graph, which is also the cheapest way
  to confirm a graph really was compiled and run.

### Checking a run really was on-device

`torch_neuronx.get_fallback_ops()` lists ops that silently ran on CPU. Across a
basic op sweep only `aten::normal_` (tensor initialisation) falls back.

## Requirements

For the XLA runtime:

- Neuron SDK 2.x, `torch-neuronx` >= 2.1, `torch-xla`, `neuronx-cc`,
  `aws-neuronx-runtime-lib` — all preinstalled on the
  [Neuron DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/index.html).
- `pip install -e .` from the repo root, so `xpu_perf` is importable.

The backend is only offered when `/dev/neuron*` exists; on a machine without the
driver `--backend NEURON` simply is not listed.

### `torch-neuronx` is not optional (XLA runtime)

`torch-neuronx` is what pulls in `libneuronxla`, which ships the Neuron PJRT
plugin (`libneuronpjrt.so`). Without it `torch_xla` finds no plugin and falls
back to **CPU**, announcing it only as `WARNING:root:Defaulting to
PJRT_DEVICE=CPU` — every op then "runs" and reports plausible-looking numbers
measured on the host CPU. `set_device()` therefore pins `PJRT_DEVICE=NEURON` and
asserts the resolved device is not CPU, so this fails loudly instead of
producing a mislabelled report.

Note that a Neuron stack is not automatically a *training/XLA* stack. An
inference venv built around `vllm-neuron` can carry `neuronx-cc`, `torch-xla`
and `libtorch-neuronx-lite` while having no `torch-neuronx` and no PJRT plugin
at all; NKI kernels also import `torch_neuronx` at call time and raise
`ModuleNotFoundError` there. Build a separate venv for benchmarking:

```bash
python3 -m venv ~/neuron_venv
~/neuron_venv/bin/pip install torch-neuronx neuronx-cc \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com
```

Run with that venv's `bin` on `PATH`, not just its interpreter: `torch_neuronx`
shells out to the `libneuronpjrt-path` executable during import and dies with
`FileNotFoundError` if it is not found.

```bash
export PATH=$HOME/neuron_venv/bin:$PATH
```

## Quick start

```bash
cd projects/micro_perf

# one workload, one NeuronCore
python launch.py --backend NEURON --device 0 \
    --workload workloads/basic/tensor_gemm_ops/gemm.json

# a small shape set, for checking the backend end to end without a long compile
python launch.py --backend NEURON --device 0 --task_dir workloads/neuron_smoke

# every basic op (expect hours of neuronx-cc compilation on the first run)
python launch.py --backend NEURON --device 0 --task all

# collectives: one process per NeuronCore, world_size matching the device count
python launch.py --backend NEURON --device 0,1 \
    --workload workloads/neuron_smoke/all_reduce.json

# long sweeps are easier to drive over the server/client split
python server.py --backend NEURON --device 0        # terminal 1
python client.py --task all                          # terminal 2
```

## Op coverage

Only three ops need vendor code; every other op in `op_defs` runs its base
implementation unchanged, via the `base` provider.

| Op | Provider | Runtime | Why it needs an override |
|---|---|---|---|
| `gemm` | `torch` | both | Rejects `tfloat32` (an NVIDIA format) and `int8` (not lowered through `torch.matmul`) |
| `all_gather` | `torch` | both | Base uses `dist.all_gather_into_tensor`, unimplemented on the `xla` backend, so the XLA path uses `xm.all_gather`; the `neuron` backend implements it, so the eager path reproduces the base behaviour |
| `flash_attention` | `nki` | `xla` only | Has no base implementation at all; uses the `flash_fwd` NKI kernel |
| `flash_attention` | `torch` | `eager` only | No NKI attention kernel exists for the native stack; uses `scaled_dot_product_attention` |

### flash_attention constraints

`flash_fwd` takes one contiguous K/V block, so only prefill is expressible —
decode with a paged KV cache is not. It also requires all-bfloat16 with a linear
cache, MHA (`q_head_num == kv_head_num`, so no GQA), `batch_size == 1`, and
`cache_len == 0`.

The eager `torch` provider accepts exactly the same envelope so that the two
runtimes report the same cases, even though SDPA itself is more general. GQA is
excluded deliberately: expanding the kv heads needs a `repeat_interleave`, a real
copy that would land inside the timed region.

### Known unsupported

Two different things get conflated here easily, so they are kept apart: a case
the *base op definition* never implemented, which no vendor can fix from a
`vendor_ops` directory, and a case this backend genuinely cannot run. An earlier
version of this table blamed Neuron for several rows that are actually the first
kind; it was rewritten after sweeping every shipped workload on the eager stack.

| Cases | Blocker | Whose limit |
|---|---|---|
| `flash_attention` with any quantized dtype (`int8`/`float8`/`mxfloat8` for q, k, or the compute dtypes) | `op_defs/llm_ops/flash_attention.py` accepts all-bfloat16 or bfloat16 + `int8` cache, and raises on everything else | base op def |
| `quant_matmul` and `moe_quant_group_gemm` with `float8` / `mxfloat8` / `mxfloat4` / `int4` weights | both base impls accept only `int8/int8/int8 -> bfloat16` | base op def |
| `flash_attention` with a paged cache (`block_size` set in the workload), `attn_mode=decode`, or GQA | neither the NKI `flash_fwd` kernel nor native SDPA covers those; see [flash_attention constraints](#flash_attention-constraints) | this backend |
| `p2p` | Needs send/recv across multiple NeuronDevices | this backend |
| `all_gather` / `all_reduce` / `all_to_all` above ~1 GiB per rank | One logical NeuronCore gets 24 GiB (96 GiB / 4 at LNC=2) and the compiler counts I/O plus an equal scratchpad, so an 8 GiB buffer asks for 32 GiB and fails with `NCC_EOOM001` | this backend (capacity) |

Three corrections worth calling out, because the old table got them wrong:

- **`all_to_all` runs fine on a single trn2.3xlarge.** The constraint is not the
  instance type; it is the world size. Neuron rejects `world_size=2` outright
  (`unsupported world size 2, supported sizes: 4, 8, 16, or multiples of 32`),
  and a trn2.3xlarge has exactly 4 logical cores at LNC=2. Run it with
  `--device 0,1,2,3` and all 76 cases pass.
- **`moe_scatter_dynamic_quant`, `moe_quant_group_gemm_combine`, `quant_matmul`
  and `moe_quant_group_gemm` all run on the eager stack** (int8 only for the last
  two). The old row claimed Neuron cannot provide int8 tensors; that was true of
  the XLA path and is not true here.
- **`scale_dynamic_quant`, `add_rms_norm_dynamic_quant`,
  `head_rms_norm_dynamic_quant`, `swiglu_dynamic_quant` and `dequant_kv_cache`
  appear in no workload JSON in the repo.** They have never been exercised on any
  backend, so they are untested, not unsupported. Listing them as blocked implied
  someone had tried.

### What the shipped LLM workloads actually measure

Worth knowing before quoting a number from `workloads/llm/`: most of those cases
do not run, and the ones that do often measure a reference simulation rather than
a kernel.

| Workload | Cases | Measured | Note |
|---|---|---|---|
| `vendor_test/flash_attention.json` | 588 | **0** | 420 rejected by the base op def, 168 by this backend (paged cache) |
| `vendor_test_demo/flash_attention.json` | 9 | **0** | same, plus 2 `attn_mode=decode` |
| `single_test_ops/fa_ops.json` | 11 | **0** | every case sets `block_size: 512` and GQA `[80, 8, 128]` |
| `vendor_test/quant_matmul.json` | 736 | int8 only (184) | the other 552 are `float8`/`mxfloat8`/`mxfloat4`, rejected by the base op def |
| `vendor_test/moe_quant_group_gemm.json` | 1380 | int8 only (276) | ditto |
| `single_test_ops/ccl_ops.json` | — | **0** | asks for `world_size: 8`; a trn2.3xlarge has 4 logical cores |

So the flash_attention figures in [Reference numbers](#reference-numbers-measured-on-trn23xlarge)
come from `workloads/neuron_smoke/flash_attention.json`, which was written for
this backend and is the only flash_attention workload in the repo with a runnable
case: no `block_size`, MHA `[8, 8, 128]`, all-bfloat16.

**The quantized ops are a bf16 simulation, on every backend.** `quant_matmul`,
`moe_quant_group_gemm`, `moe_quant_group_gemm_combine` and
`quant_group_gemm_reduce_sum` all route through `fake_quant_gemm`
(`core/utils.py`), which casts the int8 operands **to bfloat16**, matmuls, then
scales in fp32. No int8 arithmetic happens anywhere. `grep -rl` across
`vendor_ops/` finds no vendor implementation of any of them, for NEURON or GPU,
so this is what every backend reports. Two consequences:

- Their TOPS figures describe `fake_quant_gemm`, not a quantized datapath. Measured
  here: `quant_matmul` plateaus around 12-16 "TOPS", which is ~7-10% of the bf16
  peak — about what a bf16 matmul with an int8 upcast on both operands and an fp32
  scaling epilogue should cost.
- `moe_quant_group_gemm` is worse than a simulation of the wrong dtype: it is a
  Python `for` loop over experts whose slice bounds are read out of device
  tensors, so it syncs to the host once per expert and recompiles for each
  data-dependent shape. Its latency is **2.7 s and completely flat** from 1 token
  to 640 tokens — a 640x range with no change, because none of the time is
  arithmetic. Do not quote it as a MoE number for any accelerator.

### Ops that run but are pathologically slow

Two of the index ops are worth a warning, because they will dominate any sweep
they are part of and their numbers say nothing about the hardware.

| Op | mem_bw | Note |
|---|---|---|
| `embedding` | 631 GB/s | full memory bandwidth for one logical core |
| `index_select` | 631 GB/s | same |
| `index_add` | 98 GB/s | |
| `gather` | **1.34 GB/s** | flat across a 256x size range; wedges the compiler at `dim_size=65536` |
| `scatter` | **0.8 GB/s** | wedges the compiler at `dim_size=4096` |

`gather` and `index_select` select rows from the same tensor and differ only in
how the index is expressed: `IndexSelectOp` passes a 1-D index of
`dst_batch_size` int64s, which lowers to a whole-row DMA, while `GatherOp`
passes an index the same shape as the output (`[dst_batch_size, dim_size]`, built
with `.view(N, 1).expand(N, dim_size)`), which lowers to per-element indexed
access. That is the whole 470x difference — not a hardware property.

Both also eventually hang the compiler outright: `neuronx-cc compile module.mlir
--framework XLA --target trn2 --lnc 2 -O1` plus `walrus_driver` sit at ~200% CPU
indefinitely, `gather` at `float32 dim_size=65536` and `scatter` at `float32
dim_size=4096`. A `gather` case ran for over two hours before being killed. Give
any run that includes these two a hard timeout, or exclude them:
`--task embedding,index_select,scatter` style task lists are honoured by
`parse_tasks`, so naming ops explicitly is the easy way out.

One caveat on the numbers above: `GatherOp` inherits `prepare_args` from
`IndexSelectOp`, which declares the index tensor as 1-D `[dst_batch_size]`, but
`GatherOp.create_tensors` builds a 2-D `[dst_batch_size, dim_size]` one. The
declared `io_bytes` therefore does not describe the tensors the op actually
creates. Whether that costs real traffic depends on whether the backend
materialises the stride-0 `expand`, which was not verified here — but the
reported `mem_bw` for `gather` should be read as approximate either way.

## MFU and where the denominator comes from

`micro_perf` reports `mfu` = `calc_flops_power / peak_tflops`, and this backend
supplies `peak_tflops` from `NEURON_CHIP_PEAK_TFLOPS` in `backend_neuron.py`.
The figures are AWS's published dense peaks per *chip*, from
`general/arch/neuron-hardware/trainium2.html`:

| | trn1 (NeuronCore-v2) | trn2 (NeuronCore-v3) |
|---|---|---|
| FP32 | 48 TFLOPS | 181 TFLOPS |
| BF16 / FP16 / TF32 | 191 TFLOPS | 667 TFLOPS |
| FP8 | not published | 1,299 TFLOPS |

Four things to keep in mind before quoting an MFU from here.

- **The denominator is per logical NeuronCore, not per chip.** micro_perf treats
  one logical core as one device, so the per-chip peak is divided by the number
  of logical cores the chip reports. A trn2 chip is 8 physical NeuronCore-v3 and
  splits into 4 logical cores at the default LNC=2, so the bf16 denominator is
  667 / 4 = **166.75 TFLOPS** — but 83.375 at LNC=1. The split is read from
  `neuron-ls` rather than assumed, so both are handled; check
  `logical_neuroncore_config` in the run's `backend` block if a number surprises
  you.
- **Sparse peaks are excluded on purpose.** AWS also quotes 2,563 TFLOPS sparse
  for trn2, but no micro_perf op feeds a sparse operand, so scoring against it
  would understate every result by ~4x.
- **int8 and fp4 have no published peak, so those ops report no MFU at all.**
  That covers the `int8` and `mxfloat4` variants of `quant_matmul` and the
  `moe_*_gemm` family. `mxfloat8` *is* scored, against the fp8 peak: the repo's
  own dtype table maps `mxfloat8` onto `torch.float8_e4m3fn`, so MX there is a
  block-scaling scheme over fp8 operands and the multiplies go through the fp8
  datapath. If you need an int8 number, quote TOPS and leave MFU blank rather
  than borrowing the fp8 denominator.
- **A memory-bound op's MFU is correctly near zero, and that is not a finding.**
  `add` at 1024x1024 counts one FLOP per element, so it reports ~0.01% MFU — the
  arithmetic intensity is 0.5 FLOP/byte and no accelerator can do better on it.
  Read `mem_bw(GB/s)` for those ops. Only ops reporting no arithmetic at all get
  no MFU field. Note also that HBM bandwidth (2.9 TB/s on trn2) is a *per-chip*
  resource shared by all four logical cores, so unlike TFLOPS there is no clean
  per-device bandwidth denominator to divide by — one core running alone can
  exceed a naive 1/4 share.

As a cross-check on the denominator: a standalone probe (not through
micro_perf) of a 8192x4096x4096 bf16 gemm on one logical core reached 143
TFLOPS, i.e. 86% of 166.75 — the right shape of number for a large gemm, which
is the main reason to believe 166.75 is the correct per-core figure rather than
667 or 83.375.

## Reference numbers (measured on trn2.3xlarge)

### XLA runtime

Measured 2026-08-31 on trn2.3xlarge, one NeuronCore, by this backend: torch
2.9.1, torch-xla 2.9.0, torch-neuronx 2.9.0.2.15.32035, neuronx-cc 2.23.6484.
The `legacy` column is the pre-refactor branch's own run from 2026-03-09
(neuronx-cc 2.23.6484, torch-neuronx 2.9.0.2.12.22436), kept as a cross-check.

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
| all_reduce | bf16 | 1024x1024, world_size=2 | 659.5 us | 3.18 GB/s bus | n/a | trn2.48xlarge only |
| all_gather | bf16 | 1024x1024, world_size=2 | 1,108.3 us | 0.95 GB/s bus | n/a | trn2.48xlarge only |

`gemm` is the number that matters for trusting a run: it agrees with the March
baseline to within 2-9%, and it is the only op here where a wrong device or a
pruned graph is unmistakable.

**These are single-run figures, and run-to-run spread is wide.** Repeating the
same shapes against a warm compile cache gave gemm fp32 1,258 us, fp16 628 us,
bf16 570 us and all_reduce 1,040 us — 15-25% from the table above, in both
directions. Compare orders of magnitude, not digits: a real fp32 gemm here is
"about a millisecond", and anything reporting tens of microseconds is a pruned
graph or the wrong device.

At 1024x1024 both `add` and `softmax` cost 610-710 us, which is graph-launch
time rather than memory bandwidth — the GB/s figures for them are not
meaningful. Their agreeing with *each other* is the useful signal.

Three cautions when comparing:

- **`softmax` cannot detect a CPU fallback.** On this host the CPU produced
  18.0 us. Check `gemm` — CPU and Neuron differ by orders of magnitude there.
- **The legacy `softmax` baseline of 16.8 us is not a real measurement.** Two
  launch-dominated ops on identically sized tensors cannot differ 60x, yet that
  run reported `add` at 1,026 us against `softmax` at 16.8 us. The softmax
  graph was pruned; see the dead-code note under
  [XLA compilation dominates first-run time](#xla-compilation-dominates-first-run-time).
  Run today, `legacy-neuron` prunes `gemm` too, reporting 17-19 us and ~1,900
  TFLOPS for all three dtypes. Treat that branch as a record of what was run,
  not as a trustworthy baseline.
- Collective baselines on `legacy-neuron` were taken on **trn2.48xlarge**, not
  trn2.3xlarge, so they are not directly comparable.

### Eager runtime

Measured 2026-09-01 on a trn2.3xlarge, one logical NeuronCore, in the
PyTorch-native image: torch 2.12.1, torch-neuronx 2.12.3.0.1636, neuronx-cc
2.27.2878.0, nki 0.6.0, host driver 2.30.2.0. The `xla` column is the table
above, taken on a second trn2.3xlarge.

| Op | Dtype | Shape | Latency | Metric | MFU | `xla` latency |
|---|---|---|---|---|---|---|
| gemm | fp32 | 1024x4096x4096 | 990.9 us | 34.7 TFLOPS | 77% | 1,462.2 us |
| gemm | fp16 | 1024x4096x4096 | 305.2 us | 112.6 TFLOPS | 68% | 727.9 us |
| gemm | bf16 | 1024x4096x4096 | 276.8 us | 124.1 TFLOPS | 74% | 758.5 us |
| add | bf16 | 1024x1024 | 45.4 us | 138.7 GB/s | ~0% | 711.2 us |
| softmax | bf16 | 1024x1024 | 51.5 us | 81.4 GB/s | ~0% | 612.2 us |
| flash_attention | bf16 | prefill q_len=2048, 8 heads, dim 128 | 539.3 us | 15.9 TFLOPS | 10% | 1,658.6 us (NKI) |
| all_reduce | bf16 | 1024x1024, world_size=2 | 105.3 us | 19.9 GB/s bus | n/a | 659.5 us |
| all_gather | bf16 | 1024x1024, world_size=2 | 91.1 us | 11.5 GB/s bus | n/a | 1,108.3 us |

**The host driver version turned out not to matter here.** The same sweep on a
host whose driver was 2.x.8955.0 against the image's expected 2.30.2.0 — which
makes the runtime log `nrta_tensor_read/write` warnings and fall back to
synchronous tensor IO — agreed with this table to within run-to-run noise
(gemm bf16 279.4 us, add 48.8 us, all_reduce 105.0 us). Worth knowing, since
that warning looks alarming and is easy to mistake for the cause of a slow
result.

The eager path is faster across the board here, but read the gap carefully
rather than as a hardware result:

- **The small ops are not a hardware comparison at all.** `add` and `softmax` at
  1024x1024 are launch-bound on both runtimes, so 49 us vs 711 us is the cost of
  cutting and dispatching an HLO graph versus dispatching one op. It says
  nothing about memory bandwidth.
- **gemm is the honest comparison**, and bf16 at 2.7x is large enough to be
  real. The two stacks have different compilers (neuronx-cc 2.27 vs 2.23), so
  part of the gap is the compiler version rather than the dispatch model.
- **flash_attention is not the same computation path** — native SDPA against the
  NKI `flash_fwd` kernel — so the 3x is a comparison of two implementations,
  not of two runtimes.
- Unlike the XLA table, these needed no warm compile cache, and run-to-run
  spread is far narrower: there is no compilation to hit or miss.

### Eager runtime, full sweep

Measured 2026-09-01 on the same host and image, sweeping every workload file in
the repo. 3,000-odd cases; the per-workload accounting of what ran and what was
rejected is under
[What the shipped LLM workloads actually measure](#what-the-shipped-llm-workloads-actually-measure).
`workloads/llm/single_test_ops/ccl_ops.json` is the only file with no runnable
case here at all — it asks for `world_size: 8`.

Memory-bound ops, one logical NeuronCore. A quarter of the chip's 2.9 TB/s is
~725 GB/s, so that is the ceiling to read these against:

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

`reduce_max` and `reduce_min` at 177 GB/s against `reduce_sum` at 639 is a 3.6x
gap between three reductions over the same shapes, so the max/min lowering is
leaving bandwidth on the table. `gather` and `scatter` are a different phenomenon
entirely — see
[Ops that run but are pathologically slow](#ops-that-run-but-are-pathologically-slow).

Collectives, best `bus_bw` per op. The world_size=2 column comes from
`--device 0,1` and the world_size=4 column from `--device 0,1,2,3`:

| Op | ws=2 | ws=4 |
|---|---|---|
| `all_gather` | 64.5 GB/s | 125.1 GB/s |
| `all_reduce` | 37.8 GB/s | 107.1 GB/s |
| `reduce_scatter` | 91.8 GB/s | 101.7 GB/s |
| `all_to_all` | not supported at ws=2 | 54.2 GB/s |
| `device2host` | 14.3 GB/s | 4.2 GB/s |
| `host2device` | 14.0 GB/s | 4.1 GB/s |

Three things to know before reusing these:

- **Use all four cores.** Every collective is substantially faster at world_size 4,
  and `all_to_all` does not run at 2 at all. A world_size=2 run on a single chip
  barely exercises the interconnect.
- **The host-transfer rows are not comparable across the two columns.** The ws=4
  run capped transfers at 1 GiB to stay inside the per-core HBM budget (see
  below), while the ws=2 run reached 8 GiB, and `device2host` needs the larger
  sizes to saturate. Four ranks also contend for the same host DMA path:
  4.2 x 4 is in the same range as 14.3 x 2.
- `all_to_all` has a ~360 us fixed floor that dominates everything below 8 MB,
  then climbs to a ~40 GB/s plateau (fp32) at 256-512 MB.

**The largest sizes in `workloads/xccl_ops/` do not fit.** Each file sweeps
`batch_size` to 2,097,152 x `dim_size` 1024, i.e. 8 GiB at fp32. One logical
NeuronCore has 24 GiB (96 GiB / 4 at LNC=2, confirmed by the runtime reporting
`total_hbm=25769803776`), and `neuronx-cc` budgets I/O plus an equal scratchpad,
so an 8 GiB buffer asks for 32 GiB and fails:

```
[ERROR] [NCC_EOOM001] Maximum peak HBM usage of 32.00GB exceeds HBM limit of
24.00GB for Trn2. This consists of 16.00GB I/O tensors, 0B intermediate tensors,
and 16.00GB internal (scratchpad) allocations
```

**And an OOM in a rank hangs the run permanently.** When `all_gather` hit the
above, both worker processes died and became zombies while the launcher stayed in
`sleep`, waiting for results that could never arrive — `XCCLEngine` neither
notices a dead child nor times out. It sat there 35 minutes before being killed
from outside. This is why the sweep script wraps every launch in a
`docker kill`-based watchdog rather than trusting the launcher to finish. Cap
`batch_size` at 262,144 (1 GiB at fp32, far past the bandwidth plateau) to avoid
the situation entirely.

## How this backend differs from GPU

### XLA compilation dominates first-run time (XLA runtime)

Every distinct tensor shape is compiled by `neuronx-cc` on first use — 5-15
minutes per op on inf2, hours for a full sweep. Later runs reuse
`/var/tmp/neuron-compile-cache/`. Run under tmux or screen so an SSH drop does
not kill a compile. Check cache growth with:

```bash
find /var/tmp/neuron-compile-cache -name "*.neff" | wc -l
```

Three backend behaviours follow from this, all in `backend_neuron.py`:

- `perf()` caps the tensor-copy count at 4. The GPU path allocates up to 256
  copies to defeat CPU cache reuse, but each copy is a `clone()` in the same
  graph; 256 of them exceed 10 MB of HLO and take neuronx-cc over five minutes.
  NeuronCores have no CPU cache to defeat, so 4 is enough.
- `perf()` calls `mark_step()` / `wait_device_ops()` right after
  `create_tensors()`. Without it the first copy carries
  `empty -> clone -> op` while later copies carry `empty -> op`; those are
  distinct graphs, so warmup keeps re-compiling and neuronx-cc intermittently
  fails with `type must be number, but is null`.
- `core_perf()` warms up at least 4 iterations and issues one `mark_step()` per
  iteration, so the timed loop reuses the graph compiled during warmup instead
  of fusing into a new one.
- `core_perf()` holds the last `core_run()` result in a local across
  `mark_step()`. A lazy tensor dropped before the graph is cut is dead code and
  XLA prunes it, so an op whose output goes nowhere compiles to nothing and
  "runs" in the time it takes to launch an empty graph — a gemm reporting
  1896 TFLOPS is this bug, not a fast gemm. Warmup retains it too, so both
  loops compile the same graph.

### Timing and profiling

There is no usable CUDA-event equivalent on either runtime, so timing is
`time.perf_counter_ns()` around an explicit `mark_step()` + `wait_device_ops()`
on XLA, or around `torch_neuronx.synchronize()` on eager. The eager stack does
expose a `torch_neuronx.Event` with an `elapsed_time()`, but it does not measure
anything — see
[Timing: never use `torch_neuronx.Event`](#timing-never-use-torch_neuronxevent).
There is no kernel-level profiler either: the `kernels` field in results is
always empty. For kernel detail use
[Neuron Profile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html)
separately.

### Device model

One NeuronCore is one micro_perf device. `set_device()` pins the worker with
`NEURON_RT_VISIBLE_CORES`. Note that trn2 defaults to LNC=2, pairing two
physical cores into each logical core.

| Instance | NeuronCores |
|---|---|
| inf2.xlarge / inf2.8xlarge | 2 |
| inf2.24xlarge | 12 |
| inf2.48xlarge | 24 |
| trn1.2xlarge | 2 |
| trn1.32xlarge | 32 |
| trn2.48xlarge | 64 |

### Collectives

Everything in this subsection and the two that follow it describes the XLA
runtime; for the eager runtime see
[The process group comes before the device](#the-process-group-comes-before-the-device)
and [Sub-world process groups work here](#sub-world-process-groups-work-here).
The one-process-per-NeuronCore rules below apply to both.

Python object exchange (`all_gather_object`) hangs on the `xla` process group
backend, so it runs over gloo instead. Upstream's `xccl_infer_loop` already
creates a gloo group for its own exchanges; `initialize_ccl()` additionally
pre-creates one gloo group per possible `group_size` for use inside `perf()`.
They are built eagerly, while all ranks are still in lockstep, because
`new_group()` has to be called by every rank in the default group — including
non-members — and inside `perf()` only the active ranks are running.

XLA collectives on a sub-group narrower than the world do not complete on
Neuron. `perf()` therefore skips any case whose `world_size` differs from the
launched world size, so **bench one world_size per launch** — `--device 0,1` for
`world_size=2`, and so on.

Not every world size is legal. `all_to_all` is rejected at 2:

```
unsupported world size 2, supported sizes: 4, 8, 16, or multiples of 32
```

This is a collective-library rule, not an instance-type one — a trn2.3xlarge has
exactly 4 logical cores, so `--device 0,1,2,3` satisfies it. Prefer world_size 4
here in general: it is the only size that runs every op, and every other
collective is 1.1-2.8x faster at 4 than at 2. See
[Eager runtime, full sweep](#eager-runtime-full-sweep) for both columns, and for
the 24 GiB per-core limit that the largest `workloads/xccl_ops/` sizes exceed.

**A rank that dies takes the run with it, silently and forever.** `XCCLEngine`
has no liveness check and no timeout on the results it waits for, so when a
worker OOMs during compilation the launcher blocks indefinitely — `ps` shows the
workers in state `Z` and the launcher in `sleep`, with nothing on stdout after
the last completed case. Bound collective runs from outside (a watchdog that
`docker kill`s the container, or `timeout` on the launcher process itself)
instead of relying on the launch to return.

### torch_xla has to be told the world exists

Each worker narrows `NEURON_RT_VISIBLE_CORES` to one core, so by default the
Neuron PJRT plugin sees a one-device, one-process job. `ProcessGroupXla` takes
its size from `xr.world_size()`, which then reports 1 no matter what `world_size`
`init_process_group` was given, and `new_group()` fails with:

```
ValueError: the new group's world size should be less or equal to the world
size set by init_process_group
```

`set_device()` fixes this in two steps, both before anything queries the world
size (`xr.world_size()` caches its first answer):

1. `NEURON_PJRT_PROCESS_INDEX`, `NEURON_PJRT_WORLD_SIZE` and
   `NEURON_PJRT_PROCESSES_NUM_DEVICES` are exported before `torch_xla` is
   imported, which gets the plugin to `process_count() == 2` and
   `global_device_count() == 2`.
2. `xm.set_replication(device, [device])` is called after the import. Without
   it `_xla_get_replication_devices_count()` is 0 and `xr.world_size()`
   short-circuits to 1 even though the device count is right.

### One process per NeuronCore

Three further constraints come from a NeuronCore being reservable by exactly one
process, unlike a GPU. All three show up as the same misleading error, so they
are worth recognising:

```
NRT:nrt_allocate_neuron_cores  Logical Neuron Core(s) not available -
  Requested:lnc0-lnc0 Available:0 Logical Core size:2 (cores busy, ret=-16)
```

- **One engine at a time.** `perf_engine` starts every engine in
  `ENGINE_TYPE_MAPPING`, so a multi-device launch puts a `ComputeEngine` worker
  *and* an `XCCLEngine` worker on each device simultaneously. Set
  `XPU_PERF_ENGINES=XCCLEngine` to bench collectives, and leave it unset (or
  `ComputeEngine`) for everything else. Collectives and non-collectives
  therefore cannot share one launch on Neuron.

  The cost of that setting is that **an op whose engine is excluded is dropped
  with no diagnostic at all** — it appears in the enumerated case list, then
  produces no results, no warning, and exit code 0. This is easy to walk into
  because engine membership does not follow the workload directory:
  `device2device` lives in `workloads/xccl_ops/` and takes a device list like a
  collective, but is registered `@ProviderRegistry.register_base_impl(
  "device2device", "ComputeEngine")` (`op_defs/basic_ops/xccl_ops.py`), so under
  `XPU_PERF_ENGINES=XCCLEngine` it silently measures nothing. Run it with
  `ComputeEngine` and check the case count in the report against the case count
  the launcher printed.
- **`nrt_init()` must not run concurrently** — for independent workers. When two
  of them reserve cores on the same NeuronDevice within milliseconds *both*
  fail, so `set_device()` serialises them with a file lock
  (`XPU_PERF_NEURON_INIT_LOCK`, default `/tmp/xpu_perf_neuron_init.lock`; set it
  to `off` to disable). Ranks of a collective launch are exempt: once the
  topology above is set the plugin assigns their cores itself and brings them up
  together, so holding a lock through that would deadlock — rank 0 would wait
  for rank 1, which would be waiting for the lock.
- **The ready timeout has to fit a cold compile.** `XCCLEngine.start()` waits
  for rank 0 to finish the warmup `all_reduce` in `xccl_infer_loop`, which on a
  cold cache is tens of minutes of `neuronx-cc`. The upstream default is 60 s;
  raise it with `XPU_PERF_XCCL_READY_TIMEOUT_S`. When it does expire the parent
  calls `sys.exit(-1)` while its non-daemon children keep running and keep the
  cores, so the *next* launch fails on busy cores too — clean up before retrying
  (see [Troubleshooting](#troubleshooting)).

```bash
XPU_PERF_ENGINES=XCCLEngine XPU_PERF_XCCL_READY_TIMEOUT_S=2400 \
    python launch.py --backend NEURON --device 0,1 \
    --workload workloads/neuron_smoke/all_reduce.json
```

### torch_xla import ordering

Importing `torch_xla` initialises PJRT and claims the NeuronCores visible to the
process. If that happens in the parent, the spawned workers get nothing. So
`torch_xla` is never imported at module level; it is imported inside
`set_device()`, after `NEURON_RT_VISIBLE_CORES` has been narrowed.

That ordering conflicts with upstream's `xccl_infer_loop`, which calls
`initialize_ccl()` *before* `set_device()` — but the `xla` process group backend
needs `torch_xla` already imported. `initialize_ccl()` therefore records the
request and `set_device()` performs the actual `init_process_group`. Both
methods run on every rank in the same order, so the group is still formed
collectively.

### pin_memory

Neuron hosts have no NVIDIA driver, so `pin_memory()` on a CPU tensor raises.
`BackendNEURON.__init__` patches it to return the tensor unchanged.

## Troubleshooting

**A killed run still holds the cores.** A worker killed mid-compile (Ctrl-C, SSH
drop, timeout) can leave a zombie holding a NeuronCore:

```bash
pkill -9 -f multiprocessing && pkill -9 -f neuronx-cc
neuron-ls   # confirm the cores are free
```

**Compilation hangs after a killed run.** Killed compilers leave lock files:

```bash
find /var/tmp/neuron-compile-cache -name "*.lock" -delete
```

**A timed-out collective launch leaves workers holding the cores.** The parent
exits but its non-daemon children do not, so the next launch fails on busy cores.
Check for strays before retrying — and note that `pkill -f launch.py` will match
its own command line if you run it from a shell whose arguments contain that
text, so bracket the pattern:

```bash
ps -o pid=,cmd= -u "$USER" | grep -E "[l]aunch\.py|[s]pawn_main"
pkill -9 -f "[l]aunch\.py"; pkill -9 -f "[s]pawn_main"
```

**`NRT_FAILURE` / `Logical Neuron Core(s) not available ... (cores busy,
ret=-16)`.** Another process owns the cores — or, on a multi-device launch, one
of the three single-process-per-core constraints in [Collectives](#collectives).
`neuron-ls` attributes cores to a PID, but it can miss a holder in another
container, so also check directly:

```bash
sudo bash -c 'for p in /proc/[0-9]*; do for fd in "$p"/fd/*; do
  case "$(readlink "$fd" 2>/dev/null)" in /dev/neuron*)
    echo "$(basename $p) $(tr -d "\0" < $p/cmdline)"; break;; esac; done; done'
sudo docker ps        # a second container is an easy holder to miss
```

Match `/dev/neuron*`, not any path containing `neuron`: the looser pattern also
matches a process whose *own executable* lives under `/opt/aws/neuron`, which
says nothing about the device. And note that `neuron-top` and `neuron-monitor`
appear in this list — they open `/dev/neuron0` to read counters without
reserving a core, so simply watching the device looks identical to using it.
Filter them out before treating a non-empty list as "busy", or waiting for an
idle machine will wait forever.
