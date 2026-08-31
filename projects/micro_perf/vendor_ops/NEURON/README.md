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
> `ProviderRegistry` vendor mixins) and **has not yet been re-run on hardware**.
> Numbers in the section below come from the legacy branch and are carried over
> as expectations to check the port against, not as results produced by it.

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

## Reference numbers (from `legacy-neuron`, inf2.8xlarge)

Neuron SDK 2.x, torch-neuronx 2.9.0, neuronx-cc 2.22. trn2.3xlarge results for
the basic ops are under `micro_perf/benchmark/basic/**/neuron/` on that branch.

| Op | Dtype | Shape | Latency | Metric |
|---|---|---|---|---|
| gemm | fp32 | 1024x4096x4096 | 10,740 us | 3.2 TFLOPS |
| gemm | fp16 | 1024x4096x4096 | 6,260 us | 5.5 TFLOPS |
| gemm | bf16 | 1024x4096x4096 | 7,082 us | 4.9 TFLOPS |
| flash_attention | bf16 | 2048 seq, 8h MHA prefill | 994 us | 8.6 TFLOPS |
| softmax | bf16 | 1024x1024 | 24 us | 233 GB/s |
| add_rms_norm | bf16 | 128x4096 | 80 us | 52.7 GB/s |
| moe_gating_gemm | bf16→fp32 | 128x4096x8 | 69 us | 0.12 TFLOPS |
| all_reduce | bf16 | 1024x1024, 2 cores | 41 us | 51.3 GB/s |
| all_gather | bf16 | 1024x1024, 2 cores | 17 us | 124.7 GB/s |

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
