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
> `ProviderRegistry` vendor mixins) and **has not yet produced a verified
> measurement on hardware**. Numbers in the section below come from the legacy
> branch and are carried over as expectations to check the port against, not as
> results produced by it.
>
> Validated so far on a trn2.3xlarge: device discovery, `get_backend_info()`,
> op/provider registration, `env.json` handling, report writing, and the
> deferred-`torch_xla` import contract. Still unverified: every latency number,
> the NKI `flash_attention` path, and all collectives.

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

## Reference numbers (from `legacy-neuron`, trn2.3xlarge)

Measured 2026-03-09 on trn2.3xlarge, one NeuronCore: torch-xla 2.9.0,
torch-neuronx 2.9.0.2.12.22436, neuronx-cc 2.23.6484. These are the targets to
check a run against; the full set is under
`micro_perf/benchmark/basic/**/neuron/` on that branch.

| Op | Dtype | Shape | Latency | Metric |
|---|---|---|---|---|
| gemm | fp32 | 1024x4096x4096 | 1,432.3 us | 24.0 TFLOPS |
| gemm | fp16 | 1024x4096x4096 | 698.7 us | 49.2 TFLOPS |
| gemm | bf16 | 1024x4096x4096 | 697.4 us | 49.3 TFLOPS |
| add | bf16 | 1024x1024 | 1,026.7 us | 6.1 GB/s |
| add | fp32 | 1024x1024 | 1,090.4 us | 11.5 GB/s |
| softmax | bf16 | 1024x1024 | 16.8 us | 249.3 GB/s |
| softmax | fp32 | 1024x1024 | 16.8 us | 499.4 GB/s |

`add` really is ~60x slower than `softmax` here: at these sizes the graph launch
dominates, and it is the shape of the number, not its size, that tells you the
measurement is real.

Two cautions when comparing:

- **`softmax` cannot detect a CPU fallback.** On this host the CPU produced
  18.0 us against Neuron's 16.8 us. Check `gemm` — CPU and Neuron differ by
  orders of magnitude there.
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

**`NRT_FAILURE` / `Logical Neuron Core(s) not available ... (cores busy,
ret=-16)`.** Another process owns the cores. `neuron-ls` attributes them to a
PID, but it can miss a holder in another container, so also check directly:

```bash
sudo bash -c 'for p in /proc/[0-9]*; do for fd in $p/fd/*; do
  case "$(readlink "$fd" 2>/dev/null)" in *neuron*)
    echo "$(basename $p) $(tr -d "\0" < $p/cmdline)"; break;; esac; done; done'
sudo docker ps        # a second container is an easy holder to miss
```
