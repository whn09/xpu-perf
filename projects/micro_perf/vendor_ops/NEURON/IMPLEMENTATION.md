# NEURON backend implementation notes

Why the Neuron backend code looks the way it does. Every section here exists
because something failed, silently or loudly, and the guard rail is what remains.
For how to run it and what the numbers are, see [README.md](README.md).

The backend class is `src/xpu_perf/micro_perf/backends/NEURON/backend_neuron.py`;
the vendor ops and environment are in this directory.

- [Two runtimes](#two-runtimes)
- [The eager runtime](#the-eager-runtime)
- [The XLA runtime](#the-xla-runtime)
- [Collectives](#collectives)
- [Device model](#device-model)
- [Timing and profiling](#timing-and-profiling)

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
| First-run cost | minutes to hours of compilation per shape | usually none |

`detect_neuron_runtime()` picks one by looking for `torch_xla` with
`importlib.util.find_spec` — availability only, so the check itself never claims
a NeuronCore or drags `torch_xla` into a process that is about to fork workers.
`XPU_PERF_NEURON_RUNTIME=xla|eager|auto` overrides it (default `auto`). Where
`torch_xla` exists the XLA path is chosen, so nothing about an existing
installation changes. The selection is reported as `neuron_runtime` in
`get_backend_info()`.

The eager path is much simpler. Nearly all of the sharp edges below are XLA's.

## The eager runtime

### Never use `torch_neuronx.Event`

It looks like the CUDA-event equivalent and it is not one. `elapsed_time()`
returns 25-30 us regardless of the work submitted: a 1024x4096x4096 bf16 gemm and
an 8192x4096x4096 one both "take" ~24 us, which would be 1,154 and 11,273 TFLOPS.
It is not measuring device execution. `_core_perf_eager()` therefore times with
`time.perf_counter_ns()` around a single `torch_neuronx.synchronize()`, which
scales linearly with FLOPs.

No keepalive reference is needed the way the XLA path needs one: eager dispatch
has already executed the op by the time `core_run()` returns, so there is no graph
for dead code elimination to prune.

### Use the device string `neuron`, never `neuron:0`

Under `init_process_group` the native backend sets each rank's local device start
index to its local rank, so on rank 1 the only valid index is 1.
`torch.empty`/`torch.randn` with `neuron:0` raise there — and `torch.full`
*silently returns a `neuron:1` tensor*, which is worse than raising. Bare `neuron`
always resolves to the current device, so `get_torch_device_name()` returns it
unindexed on this runtime.

### The process group comes before the device

This inverts the XLA ordering. `torch_neuronx.distributed.backend`'s
`_neuron_runtime_setup` asserts the runtime is *not* yet initialised, because it
wants to assign cores from `LOCAL_RANK` itself, publish `NEURON_RT_ROOT_COMM_ID`
through the store and run an nrt barrier. So nothing may touch the device until
`init_process_group` has returned, and the check that the process really is on a
NeuronCore has to come *after* it rather than before. `LOCAL_WORLD_SIZE` must be
set too, or the backend infers local rank by rendezvousing on IP addresses;
`set_device()` defaults it from `WORLD_SIZE`.

### Sub-world process groups work here

Unlike the XLA path, a group narrower than the world does complete (verified with
a `ranks=[0,1]` group in a `world_size=4` job), so the "bench one world_size per
launch" restriction does not apply — `perf()` gates that skip on the XLA runtime
only. `dist.all_gather_into_tensor` is also implemented, so the `xm.all_gather`
override is unnecessary and the `all_gather` vendor op reproduces the base
behaviour instead.

### No NKI attention kernel

`neuronxcc.nki.kernels.attention.flash_fwd` is traced into HLO and fails on this
runtime deep inside the kernel with `No module named 'torch_neuronx.pyhlo'`. The
native entry point is `torch_neuronx.wrap_nki` (note that
`torch_neuronx.nki_kernel` is a module, not a callable), but it expects kernels
written against the standalone `nki` package, which as of 0.6.0 ships no kernel
library to point it at. So `flash_attention` is measured through native
`scaled_dot_product_attention` instead. Both providers are registered
conditionally on the detected runtime, so exactly one is available.

### Eager dispatch costs ~55-65 us per op, and that is the floor

This is the one number to internalise about this runtime. Measured on
trn2.3xlarge with a chain of `silu(x + b)` on 1024x1024 bf16 tensors, varying only
how many ops sit in one region:

| Ops in region | eager | `torch.compile(backend="neuron")` | speedup |
|---|---|---|---|
| 1 | 139.9 us | 102.7 us | 1.36x |
| 4 | 512.2 us | 97.2 us | 5.27x |
| 16 | 2,035.0 us | 205.9 us | 9.89x |
| 64 | 8,236.4 us | 706.5 us | 11.66x |

Eager scales linearly at about 64 us per op while the compiled region barely
moves, which puts the cost squarely in dispatch, not in the arithmetic. So every
small-op figure in the reference tables is essentially *all* dispatch overhead,
and no smaller number is reachable on this stack.

### torch.compile does not help micro_perf

It is the obvious thing to reach for, and the measurements say no. On Neuron
`torch.compile(backend="neuron")` is not eager-plus-fusion: it lowers through
`torch_mlir` to StableHLO and compiles a NEFF with neuronx-cc, so it is the
graph-compiled path again, entered through dynamo instead of LazyTensor. One NEFF
launch costs ~95-100 us against ~55 us for one eager dispatch, and micro_perf
times exactly one op per region — there is nothing to amortise the launch over:

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
recover a fused attention kernel that eager SDPA was missing. A `torch.compile`
option is therefore not offered: it would never win, and it would reintroduce the
dead-code-elimination hazard that eager is immune to for nothing.

Where it does win is fusion across many ops — the 11.6x above, and 1.27x on a
llama-shaped `gemm -> silu -> gemm`. That is a model-level concern, and measuring
single ops is what micro_perf is for.

Two notes if you use it here anyway:

- **`dynamic=False` is mandatory.** Benchmarking the same function at two shapes
  makes dynamo mark the varying dimension dynamic, and neuronx-cc rejects that:
  `[NCC_EMOD025] Dynamic shape is not supported: instruction 'parameter' has
  shape 'bf16[?,4096]'`. The error can surface at an unrelated later device call,
  so also call `torch._dynamo.reset()` between cases.
- Compilation is 1.4-5.1 s per graph for 8-323 nodes — minutes-to-hours faster
  than the XLA path, but not free. `torch_neuronx.get_dynamo_metrics()` reports
  node count and lowering/compile time per graph, which is also the cheapest way
  to confirm a graph really was compiled and run.

## The XLA runtime

### `torch-neuronx` is not optional (XLA runtime)

`torch-neuronx` is what pulls in `libneuronxla`, which ships the Neuron PJRT
plugin (`libneuronpjrt.so`). Without it `torch_xla` finds no plugin and falls back
to **CPU**, announcing it only as `WARNING:root:Defaulting to PJRT_DEVICE=CPU` —
every op then "runs" and reports plausible-looking numbers measured on the host
CPU. `set_device()` therefore pins `PJRT_DEVICE=NEURON` and asserts the resolved
device is not CPU, so this fails loudly instead of producing a mislabelled report.

A Neuron stack is not automatically a *training/XLA* stack. An inference venv
built around `vllm-neuron` can carry `neuronx-cc`, `torch-xla` and
`libtorch-neuronx-lite` while having no `torch-neuronx` and no PJRT plugin at all;
NKI kernels also import `torch_neuronx` at call time and raise
`ModuleNotFoundError` there. Build a separate venv for benchmarking, and put its
`bin` on `PATH` rather than just using its interpreter — `torch_neuronx` shells
out to the `libneuronpjrt-path` executable during import and dies with
`FileNotFoundError` if it is not found.

### XLA compilation dominates first-run time

Every distinct tensor shape is compiled by `neuronx-cc` on first use — 5-15
minutes per op on inf2, hours for a full sweep. Later runs reuse
`/var/tmp/neuron-compile-cache/`. Four backend behaviours follow, all in
`backend_neuron.py`:

- `perf()` caps the tensor-copy count at 4. The GPU path allocates up to 256
  copies to defeat CPU cache reuse, but each copy is a `clone()` in the same
  graph; 256 of them exceed 10 MB of HLO and take neuronx-cc over five minutes.
  NeuronCores have no CPU cache to defeat, so 4 is enough.
- `perf()` calls `mark_step()` / `wait_device_ops()` right after
  `create_tensors()`. Without it the first copy carries `empty -> clone -> op`
  while later copies carry `empty -> op`; those are distinct graphs, so warmup
  keeps re-compiling and neuronx-cc intermittently fails with `type must be
  number, but is null`.
- `core_perf()` warms up at least 4 iterations and issues one `mark_step()` per
  iteration, so the timed loop reuses the graph compiled during warmup instead of
  fusing into a new one.
- `core_perf()` holds the last `core_run()` result in a local across
  `mark_step()`. **A lazy tensor dropped before the graph is cut is dead code and
  XLA prunes it**, so an op whose output goes nowhere compiles to nothing and
  "runs" in the time it takes to launch an empty graph — a gemm reporting
  1896 TFLOPS is this bug, not a fast gemm. Warmup retains it too, so both loops
  compile the same graph.

### torch_xla has to be told the world exists

Each worker narrows `NEURON_RT_VISIBLE_CORES` to one core, so by default the
Neuron PJRT plugin sees a one-device, one-process job. `ProcessGroupXla` takes its
size from `xr.world_size()`, which then reports 1 no matter what `world_size`
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
2. `xm.set_replication(device, [device])` is called after the import. Without it
   `_xla_get_replication_devices_count()` is 0 and `xr.world_size()`
   short-circuits to 1 even though the device count is right.

### torch_xla import ordering

Importing `torch_xla` initialises PJRT and claims the NeuronCores visible to the
process. If that happens in the parent, the spawned workers get nothing. So
`torch_xla` is never imported at module level; it is imported inside
`set_device()`, after `NEURON_RT_VISIBLE_CORES` has been narrowed.

That ordering conflicts with upstream's `xccl_infer_loop`, which calls
`initialize_ccl()` *before* `set_device()` — but the `xla` process group backend
needs `torch_xla` already imported. `initialize_ccl()` therefore records the
request and `set_device()` performs the actual `init_process_group`. Both methods
run on every rank in the same order, so the group is still formed collectively.

### pin_memory

Neuron hosts have no NVIDIA driver, so `pin_memory()` on a CPU tensor raises.
`BackendNEURON.__init__` patches it to return the tensor unchanged.

## Collectives

The gloo and sub-world notes here are XLA-specific; the
one-process-per-NeuronCore rules apply to both runtimes.

Python object exchange (`all_gather_object`) hangs on the `xla` process group
backend, so it runs over gloo instead. Upstream's `xccl_infer_loop` already
creates a gloo group for its own exchanges; `initialize_ccl()` additionally
pre-creates one gloo group per possible `group_size` for use inside `perf()`.
They are built eagerly, while all ranks are still in lockstep, because
`new_group()` has to be called by every rank in the default group — including
non-members — and inside `perf()` only the active ranks are running.

XLA collectives on a sub-group narrower than the world do not complete on Neuron.
`perf()` therefore skips any case whose `world_size` differs from the launched
world size, so on that runtime **bench one world_size per launch**.

### One process per NeuronCore

Three constraints come from a NeuronCore being reservable by exactly one process,
unlike a GPU. All three surface as the same misleading error:

```
NRT:nrt_allocate_neuron_cores  Logical Neuron Core(s) not available -
  Requested:lnc0-lnc0 Available:0 Logical Core size:2 (cores busy, ret=-16)
```

- **One engine at a time.** `perf_engine` starts every engine in
  `ENGINE_TYPE_MAPPING`, so a multi-device launch puts a `ComputeEngine` worker
  *and* an `XCCLEngine` worker on each device simultaneously.
  `XPU_PERF_ENGINES` (added in `core/perf_engine.py`) restricts that, at the cost
  of silently dropping any op whose engine is excluded — see the note in
  [README.md](README.md#3-run).
- **`nrt_init()` must not run concurrently** for independent workers. When two of
  them reserve cores on the same NeuronDevice within milliseconds *both* fail, so
  `set_device()` serialises them with a file lock (`XPU_PERF_NEURON_INIT_LOCK`,
  default `/tmp/xpu_perf_neuron_init.lock`; set it to `off` to disable). Ranks of
  a collective launch are exempt: once the PJRT topology is set the plugin assigns
  their cores itself and brings them up together, so holding a lock through that
  would deadlock — rank 0 would wait for rank 1, which would be waiting for the
  lock.
- **The ready timeout has to fit a cold compile.** `XCCLEngine.start()` waits for
  rank 0 to finish the warmup `all_reduce` in `xccl_infer_loop`, which on a cold
  cache is tens of minutes of `neuronx-cc`. The upstream default is 60 s;
  `XPU_PERF_XCCL_READY_TIMEOUT_S` (added in `core/engine.py`) overrides it. When
  it does expire the parent calls `sys.exit(-1)` while its non-daemon children
  keep running and keep the cores, so the *next* launch fails on busy cores too.

Both `XPU_PERF_ENGINES` and `XPU_PERF_XCCL_READY_TIMEOUT_S` are the two changes
this port makes outside `vendor_ops`, and both default to the previous behaviour.

## Device model

One NeuronCore is one micro_perf device. `set_device()` pins the worker with
`NEURON_RT_VISIBLE_CORES`. trn2 defaults to LNC=2, pairing two physical cores into
each logical core.

| Instance | NeuronCores |
|---|---|
| inf2.xlarge / inf2.8xlarge | 2 |
| inf2.24xlarge | 12 |
| inf2.48xlarge | 24 |
| trn1.2xlarge | 2 |
| trn1.32xlarge | 32 |
| trn2.48xlarge | 64 |

Supported hardware: Inferentia2 (inf2), Trainium (trn1 / trn1n), Trainium2
(trn2).

## Timing and profiling

There is no usable CUDA-event equivalent on either runtime, so timing is
`time.perf_counter_ns()` around an explicit `mark_step()` + `wait_device_ops()` on
XLA, or around `torch_neuronx.synchronize()` on eager. The eager stack does expose
a `torch_neuronx.Event` with an `elapsed_time()`, but it does not measure anything
— see [Never use `torch_neuronx.Event`](#never-use-torch_neuronxevent).

There is no kernel-level profiler either: the `kernels` field in results is always
empty. For kernel detail use
[Neuron Profile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html)
separately. `torch_neuronx.get_fallback_ops()` lists ops that silently ran on CPU.
