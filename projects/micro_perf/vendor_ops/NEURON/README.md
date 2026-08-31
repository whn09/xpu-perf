# NEURON vendor ops (AWS Trainium / Inferentia)

micro_perf support for AWS Trainium and Inferentia through the
[Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/), using PyTorch/XLA.

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
> Measured on one NeuronDevice only (4 logical cores at LNC=2), so
> `world_size` > 4, cross-device collectives, and the ops listed under
> [Known unsupported](#known-unsupported) remain untested here.

## Supported hardware

- AWS Inferentia2 — inf2 instances
- AWS Trainium — trn1 / trn1n instances
- AWS Trainium2 — trn2 instances

## Requirements

- Neuron SDK 2.x, `torch-neuronx` >= 2.1, `torch-xla`, `neuronx-cc`,
  `aws-neuronx-runtime-lib` — all preinstalled on the
  [Neuron DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/index.html).
- `pip install -e .` from the repo root, so `xpu_perf` is importable.

The backend is only offered when `/dev/neuron*` exists; on a machine without the
driver `--backend NEURON` simply is not listed.

### `torch-neuronx` is not optional

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
implementation on XLA unchanged, via the `base` provider.

| Op | Provider | Why it needs an override |
|---|---|---|
| `gemm` | `torch` | Rejects `tfloat32` (an NVIDIA format) and `int8` (not lowered through `torch.matmul`) |
| `all_gather` | `torch` | Base uses `dist.all_gather_into_tensor`, unimplemented on the `xla` backend; uses `xm.all_gather` |
| `flash_attention` | `nki` | Has no base implementation at all; uses the `flash_fwd` NKI kernel |

### flash_attention constraints

`flash_fwd` takes one contiguous K/V block, so only prefill is expressible —
decode with a paged KV cache is not. It also requires all-bfloat16 with a linear
cache, MHA (`q_head_num == kv_head_num`, so no GQA), `batch_size == 1`, and
`cache_len == 0`.

### Known unsupported

| Ops | Blocker |
|---|---|
| 8 quantization ops (`scale_dynamic_quant`, `add_rms_norm_dynamic_quant`, `head_rms_norm_dynamic_quant`, `swiglu_dynamic_quant`, `moe_scatter_dynamic_quant`, `quant_matmul`, `moe_quant_group_gemm`, `dequant_kv_cache`) | Need int8/fp8 tensors, which Neuron XLA does not provide |
| `all_to_all` | Needs the Mesh algorithm, unavailable when every rank sits on one NeuronDevice; requires inf2.24xlarge / trn1.32xlarge or larger |
| `p2p` | Needs send/recv across multiple NeuronDevices |

## Reference numbers (measured on trn2.3xlarge)

Measured 2026-08-31 on trn2.3xlarge, one NeuronCore, by this backend: torch
2.9.1, torch-xla 2.9.0, torch-neuronx 2.9.0.2.15.32035, neuronx-cc 2.23.6484.
The `legacy` column is the pre-refactor branch's own run from 2026-03-09
(neuronx-cc 2.23.6484, torch-neuronx 2.9.0.2.12.22436), kept as a cross-check.

| Op | Dtype | Shape | Latency | Metric | legacy (2026-03-09) |
|---|---|---|---|---|---|
| gemm | fp32 | 1024x4096x4096 | 1,462.2 us | 23.5 TFLOPS | 1,432.3 us |
| gemm | fp16 | 1024x4096x4096 | 727.9 us | 47.2 TFLOPS | 698.7 us |
| gemm | bf16 | 1024x4096x4096 | 758.5 us | 45.3 TFLOPS | 697.4 us |
| add | fp32 | 1024x1024 | 681.6 us | 18.5 GB/s | 1,090.4 us |
| add | bf16 | 1024x1024 | 711.2 us | 8.8 GB/s | 1,026.7 us |
| softmax | fp32 | 1024x1024 | 655.9 us | 12.8 GB/s | 16.8 us (see below) |
| softmax | bf16 | 1024x1024 | 612.2 us | 6.9 GB/s | 16.8 us (see below) |
| flash_attention | bf16 | prefill q_len=2048, 8 heads, dim 128 | 1,658.6 us | 5.2 TFLOPS | not measured |
| all_reduce | bf16 | 1024x1024, world_size=2 | 659.5 us | 3.18 GB/s bus | trn2.48xlarge only |
| all_gather | bf16 | 1024x1024, world_size=2 | 1,108.3 us | 0.95 GB/s bus | trn2.48xlarge only |

`gemm` is the number that matters for trusting a run: it agrees with the March
baseline to within 2-9%, and it is the only op here where a wrong device or a
pruned graph is unmistakable.

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

## How this backend differs from GPU

### XLA compilation dominates first-run time

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

There is no CUDA-event equivalent, so timing is `time.perf_counter_ns()` around
an explicit `mark_step()` + `wait_device_ops()`. There is no kernel-level
profiler either: the `kernels` field in results is always empty. For kernel
detail use
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
sudo bash -c 'for p in /proc/[0-9]*; do for fd in $p/fd/*; do
  case "$(readlink "$fd" 2>/dev/null)" in *neuron*)
    echo "$(basename $p) $(tr -d "\0" < $p/cmdline)"; break;; esac; done; done'
sudo docker ps        # a second container is an easy holder to miss
```
