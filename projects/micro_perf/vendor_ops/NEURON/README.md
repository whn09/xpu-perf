# NEURON vendor ops (AWS Trainium / Inferentia)

micro_perf support for AWS Trainium and Inferentia through the
[Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/), on either of the two
PyTorch integrations Neuron offers: the PyTorch-native eager stack (`eager`,
`torch-neuronx` 2.x) or PyTorch/XLA (`xla`). The backend class lives at
`src/xpu_perf/micro_perf/backends/NEURON/`; this directory holds the vendor op
implementations (`ops/`), the default environment (`env.json`), extra workloads
(`workloads/`) and tools (`tools/`).

**Prefer the eager runtime.** It is 2.7x faster on gemm, needs no compile cache, and
has far narrower run-to-run spread. The XLA path is supported because existing
installations are on it.

**Status.** Every provider was re-checked for numerical correctness on the official
Neuron SDK 2.33 DLAMI on a trn2.48xlarge (2026-11-03,
[details](RESULTS.md#re-validated-on-neuron-sdk-233-official-dlami)). Trainium3 is in
the MFU peak table and has an MX provider, but **nothing here has run on Trn3
hardware yet** — start a Trn3 session with [step 2](#2-check-the-install).

| File | What is in it |
|---|---|
| this README | how to install, check and run, per dtype and per op family |
| [RESULTS.md](RESULTS.md) | every measured number: SDK 2.33 re-validation, single-chip sweep, multi-chip, XLA |
| [READING_THE_NUMBERS.md](READING_THE_NUMBERS.md) | what the numbers do and do not mean (MFU, launch-bound ops, fp8, quantized-op simulation, known-unsupported list, open gaps) |
| [REPRODUCE.md](REPRODUCE.md) | re-measuring one published row, and the full sweep |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | hangs, held cores, silent drops, suspiciously fast results |
| [IMPLEMENTATION.md](IMPLEMENTATION.md) | why the backend code looks the way it does |

## Quick start

### 1. Environment

**Official Neuron DLAMI (recommended).** The multi-framework DLAMI ships the eager
stack as a venv; nothing to build:

```bash
source /opt/aws_neuronx_venv_torch_neuronx_2_14_0/bin/activate   # SDK 2.33
git clone https://github.com/whn09/xpu-perf.git && cd xpu-perf
pip install -e .
cd projects/micro_perf && pip install -r requirements.txt
```

Every command below runs from `projects/micro_perf`.

`pip install -e .` must not touch numpy: on Python 3.12 the Neuron compiler needs
numpy >= 2, and an older pin here once downgraded it and broke every compile with
`module 'numpy' has no attribute 'long'`. If you see that error,
`pip install "numpy>=2"`.

`detect_neuron_runtime()` chooses `xla` if `torch_xla` is importable and `eager`
otherwise; override with `XPU_PERF_NEURON_RUNTIME=xla|eager|auto`. The choice is
reported as `neuron_runtime` in the report — **check it before trusting one.**

| | `eager` | `xla` |
|---|---|---|
| Requires | `torch_neuronx` 2.x (DLAMI venv above) | `torch_xla` + `libneuronxla` + `torch-neuronx` |
| Dispatch | eager, op by op | lazy, traced to HLO, compiled by `neuronx-cc` |
| First-run cost | seconds per shape, but see [the caveat](TROUBLESHOOTING.md#a-run-that-looks-hung-is-usually-compiling) | minutes to hours per shape |

<details><summary>Other environments: the beta container image, a hand-built XLA venv</summary>

**eager in a container** — the native base image ships no reporting stack, so build
`tools/Dockerfile.eager` and run with `--privileged` (without it `import
torch_neuronx` fails with `Failed to get Neuron instance information. Status: 1`):

```bash
docker build -f vendor_ops/NEURON/tools/Dockerfile.eager \
    -t xpu-perf-eager:latest vendor_ops/NEURON/tools
docker run --rm --privileged \
    -v "$PWD/../..":/xpu-perf -w /xpu-perf/projects/micro_perf \
    -e PYTHONPATH=/xpu-perf/src xpu-perf-eager:latest \
    python launch.py --backend NEURON --device 0 \
    --workload workloads/neuron_smoke/gemm.json
```

Its `FROM` is the AWS-internal beta image the [beta-image tables](RESULTS.md#beta-image-measurements)
were measured with; edit it if you cannot pull that.

**xla** — `torch-neuronx` is not optional, and a `vllm-neuron` inference venv is not
a substitute ([why](IMPLEMENTATION.md#torch-neuronx-is-not-optional-xla-runtime): the
failure mode is a report full of host-CPU numbers under a NEURON label).

```bash
python3 -m venv ~/neuron_venv
~/neuron_venv/bin/pip install torch-neuronx neuronx-cc \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com
export PATH=$HOME/neuron_venv/bin:$PATH   # the bin dir, not just the interpreter
```
</details>

### 2. Check the install

```bash
python vendor_ops/NEURON/tools/verify_providers.py --device 0 --mx-sim
```

Two things `launch.py` never does. It lists which providers actually registered —
every provider file swallows its own import errors, so after an SDK change a broken
one simply disappears from the sweep — and it runs each provider once and compares
the output with an fp32 host reference on the same inputs, passing a case only
within 3x of the host's own error in that dtype. `--mx-sim` adds the MX kernel in
the NKI CPU simulator at target trn3 (works on any host). About 10 minutes on Trn2; exit
code 0 means everything passed. **Run it first on any new SDK or instance type**,
and on Trn3 before the first sweep: it is the only thing here that would notice a
provider returning wrong numbers at a plausible latency.

### 3. Run

```bash
# smoke test: the backend end to end, no long compile
python launch.py --backend NEURON --device 0 --task_dir workloads/neuron_smoke

# one workload
python launch.py --backend NEURON --device 0 \
    --workload workloads/basic/tensor_gemm_ops/gemm.json

# name ops explicitly to skip the pathological ones
python launch.py --backend NEURON --device 0 \
    --task_dir workloads/basic/vector_index_ops \
    --task embedding,index_select,index_add
```

`--device N` is one *logical* NeuronCore (4 per chip at the default LNC=2; a
trn2.48xlarge or trn3.48xlarge has 64). One core can be held by one process only,
so give concurrent runs different cores. For a whole-repo sweep do not use
`--task all` — see [REPRODUCE.md](REPRODUCE.md#reproducing-the-full-sweep).

### 4. Which dtype runs where

| dtype | op | how | notes |
|---|---|---|---|
| fp32 / fp16 / bf16 | everything | default | |
| tfloat32 | `gemm` | default (`torch` provider) | neuronx-cc TF32 auto-cast, scoped to one compiled graph per case: 106.8 TFLOPS at 4096³ vs 35.4 fp32 on Trn2 |
| fp8 e5m2 | `gemm` | `XPU_PERF_NEURON_GEMM_COMPILE=1` + `vendor_ops/NEURON/workloads/gemm_fp8_compiled.json` | `torch_compile` provider; set an explicit `dst_dtype` |
| fp8 e4m3 (OCP e4m3fn) | `gemm` | same flag + `vendor_ops/NEURON/workloads/gemm_fp8_e4m3_compiled.json` (or `ONLY=trn3_fp8_e4m3_compiled`) | **Trn3 only** — Trn2 has no e4m3fn datapath (`NCC_EVRF051`) |
| fp8, eager | `gemm` | — | rejected: eager fp8 matmul returns wrong products on SDK 2.33 ([details](RESULTS.md#re-validated-on-neuron-sdk-233-official-dlami)) |
| MXFP8 / MXFP4 | `quant_matmul` | `XPU_PERF_NEURON_MX=1`, see [step 5](#5-mxfp8--mxfp4-on-trainium3) | **Trn3 only** |
| int8 | `quant_matmul` and family | default | a bf16 simulation on every backend ([why](READING_THE_NUMBERS.md#the-quantized-ops-are-a-bf16-simulation-on-every-backend)) |

### 5. MXFP8 / MXFP4 on Trainium3

`ops/nkilib/quant_matmul.py` runs `quant_matmul` on the NeuronCore-v4 MX datapath
(`nisa.nc_matmul_mx`, 2517 TFLOPS per chip), for the compute dtypes
`op_defs/llm_ops/quant_matmul.md` defines: `mxfp8_e4m3fn`, `mxfp8_e5m2`,
`mxfp4_e2m1fn`, any activation x weight combination.

```bash
# on a trn3 instance
python vendor_ops/NEURON/tools/verify_providers.py --device 0 --mx-sim --only quant_matmul
XPU_PERF_NEURON_MX=1 python launch.py --backend NEURON --device 0 \
    --workload vendor_ops/NEURON/workloads/quant_matmul_mx.json

# or both Trn3-only labels (MX + OCP-e4m3 compiled fp8 gemm) through the sweep script
RUNNER=venv ONLY=trn3 vendor_ops/NEURON/tools/run_full_sweep.sh
```

- **What is measured.** Operands arrive pre-quantized — x4-packed along K, one E8M0
  scale per 32 values — exactly as the int8 op receives int8 activations with their
  scales, so the timed region is the MX GEMM. Activation quantization is a separate
  op in this suite. The workload sweeps four format pairs (e4m3·e4m3, e5m2·e5m2,
  e4m3 activations · fp4 weights, fp4·fp4) over four `hidden_size x new_hidden_size`
  shapes, at prefill `num_tokens` 128-8192 and decode `num_tokens` 1-64 (144 cases).
  Constraints: `hidden_size % 512 == 0`, `new_hidden_size % 1024 == 0`, `dst_dtype`
  bf16.
- **Decode shapes.** `nc_matmul_mx` takes an even token count of at most 128 per
  tile, so the provider pads `num_tokens` to the next even number (1 → 2) up to 128
  and to the next multiple of 128 above it. The padding tokens are computed but not
  counted in `calc_flops`; at decode sizes the op is weight-read bound, so read
  `mem_bw` rather than MFU there. Verified in the trn3 simulator at T = 1, 2, 8, 63,
  64, 128, 200 (LNC 1 and 2) and compiled for trn3.
- **MFU** is scored against 2517 TFLOPS per chip (629.25 per logical core at LNC=2),
  the published MXFP8/MXFP4 peak.
- **Why opt-in.** Registering a vendor provider for `quant_matmul` replaces the base
  implementation, which would drop the int8 rows. With the flag set,
  `ops/torch/quant_matmul.py` re-registers the base as `torch`, so both run.
- **Validated without Trn3 hardware.** In the NKI simulator at target trn3, all nine
  format pairs at LNC 1 and 2 match an exact host dequant-matmul to bf16 output
  rounding (rel 1.66e-3), and negative controls (doubled scales, swapped fp4
  nibbles, permuted weights) are caught; neuronx-cc compiles it for trn3 cleanly. On
  Trn3, `verify_providers.py` additionally checks the device output against the host
  reference.
- **Kernel structure.** A weight slab (512-2048 columns, as wide as SBUF allows)
  stays resident across token tiles, and each 128-token tile brings its whole K
  extent and scales in 5 DMAs through K-major HBM views. The first version issued 5
  DMAs per 128x512 K tile; compiled for trn3, the restructure cuts DMA instructions
  per core 9-27x (e.g. 10,416 -> 756 at 512x8192x28672, 42,112 -> 1,556 at 8192³
  with fp4 weights) and total instructions 4-12x. That is an instruction count, not
  a timing — the compiler gives no latency estimate for NKI kernels and Trn2 cannot
  run them — so the first Trn3 MFU is still the real test; nkilib's tuned
  `experimental/matmul_mxfp8` is the next step if it is low.
- **Refused off Trn3, by instance type.** neuronx-cc will compile the kernel for
  trn3 on a Trn2 host if the target is overridden, and the 2.35 runtime then
  *executes* the trn3 NEFF on the Trn2 core and faults it
  (`SEQUENCER_ERROR_SUBTYPE_BAD_OPCODE`) until the process exits. Do not override
  `NEURON_PLATFORM_TARGET_OVERRIDE` in a process that also touches the device.

### 6. Collectives

```bash
XPU_PERF_ENGINES=XCCLEngine XPU_PERF_XCCL_READY_TIMEOUT_S=2400 \
    python launch.py --backend NEURON --device 0,1,2,3 \
    --workload workloads/neuron_smoke/all_reduce.json
```

- **`XPU_PERF_ENGINES` must name exactly one engine.** Otherwise a `ComputeEngine`
  worker and an `XCCLEngine` worker land on each core and fight over it (`Logical
  Neuron Core(s) not available ... Available:0`), so collectives and
  non-collectives cannot share a launch. The trap: **an op whose engine is excluded
  is dropped with no diagnostic and exit code 0**, and engine membership does not
  follow the workload directory — `device2device` lives in `workloads/xccl_ops/` but
  registers under `ComputeEngine`, so an `XCCLEngine` launch measures none of its 76
  cases and reports success. Compare the case count in the report against what the
  launcher enumerated.
- **`XPU_PERF_XCCL_READY_TIMEOUT_S`** raises the upstream 60 s wait for rank 0's
  warmup `all_reduce`, which on a cold cache is tens of minutes of `neuronx-cc`. On
  expiry the children keep the cores and the *next* launch fails too — see
  [TROUBLESHOOTING.md](TROUBLESHOOTING.md#a-collective-run-hangs-forever-after-a-rank-dies).
- **World size and placement decide what runs.** `all_to_all` rejects 2 ranks
  (`supported sizes: 4, 8, 16, or multiples of 32`), and on Trn2 also rejects 4
  ranks spread one per chip; a group whose chips are not torus neighbours falls back
  to Mesh, where ring collectives above 40 MiB fail outright. Use contiguous cores
  (`--device 0,1,2,3`, `0-7`, ...). Measured scaling to 64 ranks is in
  [RESULTS.md](RESULTS.md#multi-chip-trn248xlarge). On the XLA runtime one launch
  benches one world size; on eager sub-world groups work.

### 7. Trainium3 and UltraServer notes

- `trn3.48xlarge` is recognised by instance type: MFU uses 671 (bf16/fp16/tf32),
  183 (fp32) and 2517 (fp8/MX) TFLOPS per chip, from the Neuron Trainium3
  architecture page. Before this, a Trn3 report silently had no `mfu` field.
- Choices that are *conservative* on Trn3, not wrong: the nkilib `rms_norm` SBUF
  budget is the smallest generation's (Trn3 has 32 MiB per core vs Trn2's 24), and
  `torch_tiled` attention's size threshold was tuned on Trn2.
- **Multi-node (one UltraServer, 4 x trn3.48xlarge).** UltraServer mode is per
  process, not machine state (`neuron-ls` showing `ultraserver-mode: unset` is
  normal). Every rank needs `NEURON_RT_ULTRASERVER_MODE=1`,
  `NEURON_RT_ULTRASERVER_SERVER_ID` (= `/sys/class/neuron_device/reservation_id`)
  and `NEURON_RT_ULTRASERVER_NODE_ID` (= `/sys/class/neuron_device/node_id`), with
  node ranks assigned in (reservation_id, node_id, FQDN) order — what AWS's
  UltraServer init script does. If those are persisted in `/etc/environment`, unset
  them for single-node runs. **xpu-perf's multi-node launch
  (`--node_world_size/--node_rank/--master_addr`) has never been run on Neuron.**
- The SDK 2.33 component versions are runtime-lib 2.35.30 and collectives 2.35.43;
  a "2.35 vs SDK 2.33" mismatch is not a mismatch.

## Results at a glance

One logical NeuronCore unless stated. Full tables, conditions and caveats in
[RESULTS.md](RESULTS.md).

| What | Trn2 result |
|---|---|
| gemm bf16 1024x4096x4096 | 276.8-285 us, 120-124 TFLOPS, MFU 0.72-0.74 |
| gemm tf32 32768x1024x8192 (SDK 2.33) | 131.6 TFLOPS, 79% of peak |
| gemm fp8 e5m2 4096³, compiled | 245.5 TFLOPS, 75.6% of the fp8 peak |
| `all_gather` / `all_reduce` / `all_to_all`, 64 ranks, bf16 bus_bw | 166.4 / 146.5 / 15.7 GB/s |
| Trn3 | not measured yet |
