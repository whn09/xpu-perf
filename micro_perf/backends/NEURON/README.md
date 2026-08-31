# NEURON Backend for ByteMLPerf Micro-Benchmark

Micro-benchmark backend for **AWS Trainium and Inferentia** accelerators using the [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/).

> **Branch note** — `legacy-neuron` is a known-working baseline preserved on the
> pre-refactor `micro_perf/` layout. It is the tree from
> [cszhz/xpu-perf](https://github.com/cszhz/xpu-perf) (which itself builds on the Neuron
> backend from [davidshtian/xpu-perf](https://github.com/davidshtian/xpu-perf)), and it
> lags `bytedance/xpu-perf` by 9 commits — notably the refactor that moved the framework
> into `src/xpu_perf/` and vendor ops into `projects/micro_perf/vendor_ops/`.
> Use this branch to see what was originally run; use `main` for the port onto current
> upstream, which is where the maintained code lives.
>
> **The numbers below no longer reproduce, and some were never valid.** Re-run on a
> trn2.3xlarge on 2026-08-31 (torch-xla 2.9.0, neuronx-cc 2.23.6484 — the same compiler
> as the original run), this branch reports gemm at 17-19 us and ~1,900 TFLOPS for fp32,
> fp16 and bf16 alike, which is physically impossible. `core_perf()` discards the
> `core_run()` result before `mark_step()`, so XLA prunes the op being measured as dead
> code and what gets timed is an empty graph launch.
>
> The same bug is visible inside the original results: `add` at 1,026 us against
> `softmax` at 16.8 us, on identically sized 1024x1024 tensors, where both ops are
> graph-launch bound and must cost about the same. The softmax figure was a pruned graph
> even then. `gemm` was measured correctly in March and matches `main` today to within
> 2-9%, so the gemm numbers are the only ones worth trusting here.
>
> `main` fixes this by keeping the result reachable across `mark_step()`. Treat this
> branch as a record of what was run, not as a baseline.

## Supported Hardware

- AWS Inferentia2 (inf2 instances)
- AWS Trainium (trn1/trn1n instances)
- AWS Trainium2 (trn2 instances)

## Requirements

- AWS Neuron SDK 2.x
- `torch-neuronx` >= 2.1
- `torch-xla` (matching PyTorch version)
- `neuronx-cc` (Neuron compiler)
- `aws-neuronx-runtime-lib` (Neuron runtime)

All dependencies come pre-installed on [Neuron DLAMIs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/index.html).

## Quick Start

```bash
cd micro_perf

# Single op benchmark
python launch.py --backend NEURON --device 0 --workload workloads/basic/tensor_gemm_ops/gemm.json

# All basic ops
python launch.py --backend NEURON --task all --device 0

# Specific ops
python launch.py --backend NEURON --task add,gemm,softmax --device 0

# Multiple NeuronCores (for XCCL collective ops)
python launch.py --backend NEURON --task all_reduce
```

## Op Coverage

**51 ops implemented** — full parity with the GPU backend.

| Category | Ops | Provider |
|---|---|---|
| Vector Linear (4) | add, sub, mul, cast | torch |
| Vector SFU (6) | div, sin, cos, exp, log, sqrt | torch |
| Vector Reduction (4) | reduce_max, reduce_min, reduce_sum, topk | torch |
| Vector Norm (3) | layer_norm, rms_norm, softmax | torch |
| Vector Activation (2) | gelu, silu | torch |
| Vector Index (5) | embedding, gather, index_select, scatter, index_add | torch |
| Tensor GEMM (1) | gemm (float32, float16, bfloat16) | torch |
| LLM Basic (3) | scale_dynamic_quant, add_rms_norm_dynamic_quant, add_rms_norm | torch |
| LLM MOE (8) | moe_gating_gemm, moe_softmax_topk, moe_scatter_dynamic_quant, quant_matmul, moe_quant_group_gemm, moe_swiglu_dynamic_quant, swiglu_dynamic_quant, moe_gather | torch |
| LLM Attention (6) | head_rms_norm, head_rms_norm_dynamic_quant, rotary_embedding, store_kv_cache, dequant_kv_cache, flash_attention | torch / nki |
| XCCL (9) | all_reduce, reduce_scatter, all_gather, all_to_all, broadcast, p2p, host2device, device2host, device2device | torch |

### Flash Attention

Flash attention uses the **NKI (Neuron Kernel Interface)** `flash_fwd` kernel from `neuronxcc.nki.kernels.attention`. This is a hardware-optimized kernel that runs natively on NeuronCores.

- Provider: `nki`
- Mode: prefill only (batch_size=1, contiguous Q/K/V, causal mask)
- Dtype: bfloat16
- Head config: MHA only (q_head_num must equal kv_head_num); GQA not supported by NKI kernel
- Decode mode with paged KV cache is not supported by the NKI kernel

### Unsupported dtypes

- `tfloat32` — NVIDIA-specific, rejected by GEMM op
- `int8` GEMM — not supported via `torch.matmul` on Neuron

## Test Status

Tested on **inf2.8xlarge** with Neuron SDK 2.x, torch-neuronx 2.9.0, neuronx-cc 2.22.

### Verified passing (43 ops)

All basic compute ops, GEMM with 3 dtypes, index ops, transfers, and LLM ops:

| Op | Dtype | Shape | Latency | Metric |
|---|---|---|---|---|
| gemm | fp32 | 1024x4096x4096 | 10,740 us | 3.2 TFLOPS |
| gemm | fp16 | 1024x4096x4096 | 6,260 us | 5.5 TFLOPS |
| gemm | bf16 | 1024x4096x4096 | 7,082 us | 4.9 TFLOPS |
| softmax | bf16 | 1024x1024 | 24 us | 233 GB/s |
| reduce_max | fp32 | 1024x1024 | 18 us | - |
| reduce_min | fp32 | 1024x1024 | 19 us | - |
| reduce_sum | fp32 | 1024x1024 | 20 us | - |
| topk | fp32 | 1024x1024 k=10 | 32 us | - |
| gelu | bf16 | 1024x1024 | 46 us | - |
| silu | bf16 | 1024x1024 | 45 us | - |
| layer_norm | bf16 | 1024x1024 | 69 us | - |
| rms_norm | bf16 | 1024x1024 | 72 us | - |
| sqrt | fp32 | 1024x1024 | 1,150 us | - |
| sub | bf16 | 1024x1024 | 1,179 us | - |
| exp | fp32 | 1024x1024 | 1,216 us | - |
| log | fp32 | 1024x1024 | 1,325 us | - |
| mul | bf16 | 1024x1024 | 1,324 us | - |
| div | fp32 | 1024x1024 | 1,349 us | - |
| cos | fp32 | 1024x1024 | 1,683 us | - |
| sin | fp32 | 1024x1024 | 1,701 us | - |
| add | bf16 | 1024x1024 | 2,695 us | - |
| embedding | bf16 | 1024x1024 | - | - |
| index_select | bf16 | 1024x1024 | - | - |
| gather | bf16 | 1024x1024 | - | - |
| scatter | bf16 | 1024x1024 | - | - |
| index_add | bf16 | 1024x1024 | - | - |
| device2device | bf16 | 1024x1024 | - | - |
| host2device | bf16 | 1024x1024 | - | - |
| device2host | bf16 | 1024x1024 | - | - |
| add_rms_norm | bf16 | 128x4096 | 80 us | 52.7 GB/s |
| head_rms_norm | bf16 | 128x32x128 | 117 us | 17.9 GB/s |
| moe_softmax_topk | fp32 | 128x8 k=2 | - | - |
| cast | bf16 | 1024x1024 | - | - |
| moe_gating_gemm | bf16→fp32 | 128x4096x8 | 69 us | 0.12 TFLOPS |
| rotary_embedding | bf16 | 128x48x128 prefill | 799 us | 3.4 GB/s |
| store_kv_cache | bf16 | 128x8x128 prefill | 629 us | 1.7 GB/s |
| moe_gather | bf16 | 128x4096 8e top2 | 662 us | 11.1 GB/s |
| flash_attention | bf16 | 2048 seq, 8h MHA prefill (nki) | 994 us | 8.6 TFLOPS |
| all_reduce | bf16 | 1024x1024 2-core | 41 us | 51.3 GB/s |
| reduce_scatter | bf16 | 1024x1024 2-core | - | - |
| broadcast | bf16 | 1024x1024 2-core | - | - |
| all_gather | bf16 | 1024x1024 2-core | 17 us | 124.7 GB/s |

### Blocked ops (10 ops)

These ops load successfully but cannot be benchmarked due to Neuron platform limitations.

| Category | Ops | Blocker |
|---|---|---|
| LLM quant ops (8) | scale_dynamic_quant, add_rms_norm_dynamic_quant, head_rms_norm_dynamic_quant, swiglu_dynamic_quant, moe_scatter_dynamic_quant, quant_matmul, moe_quant_group_gemm, dequant_kv_cache | Require int8/fp8 dtype, unsupported on Neuron XLA (see #7) |
| XCCL (1) | all_to_all | Requires Mesh algorithm, needs multi-NeuronDevice instance (see #6) |
| XCCL (1) | p2p | Requires multi-NeuronDevice instance for send/recv |

### How to resume testing

```bash
cd micro_perf

# Step 1: Run all basic ops (will compile new shapes, ~2-3 hours first time)
# Recommend running in tmux/screen so SSH disconnect won't kill it
python launch.py --backend NEURON --task all --device 0

# Step 2: Run LLM ops (needs LLM workload format)
python launch.py --backend NEURON --device 0 --workload workloads/llm/test_ops/gemm_ops.json
python launch.py --backend NEURON --device 0 --workload workloads/llm/test_ops/fa_prefill.json

# Step 3: Run XCCL ops (needs 2 NeuronCores)
python launch.py --backend NEURON --device 0,1 --workload workloads/llm/test_ops/ccl_ops.json

# Tip: Check compilation cache hit rate
find /var/tmp/neuron-compile-cache -name "*.neff" | wc -l
```

### Known issues

1. **Stale NeuronCore lock**: If a benchmark process is killed during XLA compilation
   (e.g., Ctrl+C, SSH disconnect, timeout), the zombie subprocess may hold the NeuronCore.
   Fix: `pkill -9 -f multiprocessing && pkill -9 -f neuronx-cc` then verify with `neuron-ls`.

2. **Stale compilation locks**: Killed compilers leave `.lock` files that block future compilations.
   Fix: `find /var/tmp/neuron-compile-cache -name "*.lock" -delete`

3. **pin_memory()**: Patched in BackendNEURON to return self (no NVIDIA driver on Neuron machines).

4. **torch_xla import ordering**: `import torch_xla` must NOT happen at module level in the
   parent process — it grabs NeuronCores via PJRT init. Deferred to `set_device()` which
   runs only in child subprocesses.

5. **XCCL ops framework deadlock (FIXED)**: The XCCLEngine used `dist.all_gather_object()`
   which hangs on XLA backend. Fixed by overriding `xccl_infer_loop` and `perf` in
   BackendNEURON: replaced `all_gather_object` with `_broadcast_object` (XLA broadcast +
   pickle) for task distribution and `_gather_objects` (XLA all_gather + pickle) for result
   collection. Also overrode `op_group_barrier` with `xm.mark_step()`/`xm.wait_device_ops()`.

6. **all_to_all on single NeuronDevice**: `all_to_all` requires Mesh algorithm which is
   not supported when all ranks are on the same NeuronDevice (e.g., inf2.8xlarge has 2
   NeuronCores on 1 device). Needs multi-device instances (inf2.24xlarge+, trn1.32xlarge).

7. **Quantization ops unsupported**: 8 LLM ops require int8/fp8 dtype tensors which Neuron
   XLA does not support: scale_dynamic_quant, add_rms_norm_dynamic_quant,
   head_rms_norm_dynamic_quant, swiglu_dynamic_quant, moe_scatter_dynamic_quant,
   quant_matmul, moe_quant_group_gemm, dequant_kv_cache.

## Architecture Notes

### XLA Compilation

The Neuron backend uses PyTorch/XLA. The **first run of each unique tensor shape triggers
XLA compilation** through `neuronx-cc`, which can take 5-15 minutes per op on inf2.
Subsequent runs with the same shapes use cached compiled graphs from
`/var/tmp/neuron-compile-cache/`.

This means:
- First benchmark run for a new workload will be very slow (hours for full suite)
- Second run will be fast (seconds per op)
- Run in tmux/screen to survive SSH disconnects
- The benchmark framework uses extra warmup iterations to absorb compilation overhead

### Device Management

Each NeuronCore is treated as a separate device. The backend uses `NEURON_RT_VISIBLE_CORES`
environment variable to assign one NeuronCore per benchmark subprocess.

```
inf2.xlarge    → 2 NeuronCores
inf2.8xlarge   → 2 NeuronCores
inf2.24xlarge  → 12 NeuronCores
inf2.48xlarge  → 24 NeuronCores
trn1.2xlarge   → 2 NeuronCores
trn1.32xlarge  → 32 NeuronCores
trn2.48xlarge  → 64 NeuronCores
```

### Timing

No CUDA events are available. Timing uses `time.perf_counter_ns()` after explicit XLA
synchronization (`xm.mark_step()` + `xm.wait_device_ops()`).

### Profiling

Kernel-level profiling (like torch.profiler with CUDA) is not currently supported.
The `kernels` field in results will be empty. For Neuron-specific profiling, use
[Neuron Profile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html) externally.

## File Structure

```
backends/NEURON/
├── backend_neuron.py      # BackendNEURON class
├── env_neuron.py           # Default environment variables
├── provider_neuron.py      # NKI provider detection
├── README.md               # This file
└── ops/                    # 51 op mapping files
    ├── add.py              # Re-export of core AddOp
    ├── flash_attention.py  # NKI flash_fwd kernel (prefill only)
    ├── gemm.py             # Subclass rejecting tfloat32/int8
    └── ...                 # 48 more op files
```
