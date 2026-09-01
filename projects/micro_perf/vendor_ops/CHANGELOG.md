# Changelog

All notable changes to `vendor_ops` are documented in this file.

This changelog follows a Keep a Changelog style and semantic versioning.

## [Unreleased]

### Added

- Added `NEURON` vendor implementations for AWS Trainium / Inferentia:
  a `torch` provider overriding `gemm` (rejects tfloat32 and int8) and
  `all_gather` (uses `xm.all_gather`), plus an `nki` provider for
  `flash_attention` built on the `flash_fwd` kernel shipped with neuronx-cc.
  Every other op runs its `op_defs` base implementation unchanged.
- Added `NEURON/env.json` and `NEURON/README.md`.
- `NEURON`: support the PyTorch-native ("eager") Neuron stack alongside
  PyTorch/XLA. The two integrations are mutually exclusive — the native image
  ships neither `torch_xla` nor `libneuronxla`, so nothing `xm.*`-shaped exists
  there. `detect_neuron_runtime()` chooses between them with
  `importlib.util.find_spec` (availability only, so the check never claims a
  NeuronCore or imports `torch_xla` into a process about to fork workers), and
  `XPU_PERF_NEURON_RUNTIME=xla|eager|auto` overrides it. XLA is chosen wherever
  `torch_xla` is installed, so existing installations are unaffected. The
  selected runtime is reported as `neuron_runtime` in `get_backend_info()`.
- `NEURON`: added a `torch` provider for `flash_attention` on the eager runtime,
  using `scaled_dot_product_attention`. It accepts the same restricted envelope
  as the `nki` provider (bf16 MHA prefill, linear cache, `batch_size == 1`) so
  both runtimes report the same cases. Exactly one of the two providers
  registers, decided by the detected runtime: the bundled
  `neuronxcc.nki.kernels.attention.flash_fwd` is traced into HLO and fails on
  the native stack with `No module named 'torch_neuronx.pyhlo'`, while nki 0.6.0
  ships no kernel for `torch_neuronx.wrap_nki` to wrap.
- `NEURON`: `all_gather` now reproduces the base `all_gather_into_tensor`
  behaviour on the eager runtime, which implements it; the `xm.all_gather`
  override applies to the XLA runtime only.

### Fixed

- `NEURON`: report provider versions independently, so a missing
  `torch-neuronx` no longer discards the known `torch-xla` version.
- `NEURON`: pin `PJRT_DEVICE=NEURON` and assert the resolved device is not CPU.
  Without the Neuron PJRT plugin `torch_xla` silently fell back to CPU, and the
  benchmark reported host-CPU latencies under a NEURON label.
- `NEURON`: hold the `core_run()` result across `mark_step()` in `core_perf()`.
  Discarded lazy tensors are dead code and XLA pruned the op being measured,
  producing impossible results such as a gemm at 1896 TFLOPS.
- `NEURON`: serialise `nrt_init()` across independent workers with a file lock in
  `set_device()`. Workers are spawned together, and two of them reserving cores
  on one NeuronDevice within milliseconds made *both* fail with
  `cores busy, ret=-16`. Collective ranks are exempt, since the plugin brings
  them up together and the lock would deadlock them.
- `NEURON`: time the eager runtime with `time.perf_counter_ns()` around
  `torch_neuronx.synchronize()`, not with `torch_neuronx.Event`. Its
  `elapsed_time()` returns 25-30 us regardless of the work submitted — a
  1024x4096x4096 bf16 gemm and an 8192x4096x4096 one both "took" ~24 us, which
  would be 1,154 and 11,273 TFLOPS.
- `NEURON`: return the device string `neuron` unindexed on the eager runtime.
  The native distributed backend sets each rank's local device start index to
  its local rank, so `neuron:0` raised on rank 1 for `torch.empty`/`torch.randn`
  — and `torch.full` silently returned a `neuron:1` tensor instead.
- `NEURON`: on the eager runtime, form the process group before touching the
  device, inverting the XLA ordering. `_neuron_runtime_setup` asserts the
  runtime is not yet initialised, so the on-device verification now runs after
  `init_process_group` rather than before it, and `LOCAL_WORLD_SIZE` is defaulted
  from `WORLD_SIZE` so the backend does not infer local rank from IP addresses.
- `NEURON`: only skip cases whose `world_size` differs from the launched world
  size on the XLA runtime. Sub-world process groups do complete on the eager
  runtime, so that restriction no longer applies there.
- `NEURON`: describe the process topology to the Neuron PJRT plugin and call
  `xm.set_replication()` in `set_device()`. Each worker sees one core, so
  `xr.world_size()` reported 1 and `ProcessGroupXla` rejected every group wider
  than one rank — no collective could run at all.

Two supporting changes land outside `vendor_ops`, in `micro_perf/core`, because
a NeuronCore can only be reserved by one process where a GPU can host several.
Both default to the previous behaviour:

- `XPU_PERF_ENGINES` (`core/perf_engine.py`) restricts which engines start.
  Without it a multi-device launch puts a `ComputeEngine` and an `XCCLEngine`
  worker on every device at once, and the second set cannot get cores.
- `XPU_PERF_XCCL_READY_TIMEOUT_S` (`core/engine.py`) overrides the hardcoded
  60 s wait for rank 0 to report ready, which does not fit the cold-cache
  `neuronx-cc` compile of the warmup collective.

## [0.1.0] - 2026-04-14

### Added

- Established standalone maintenance baseline for `vendor_ops`.
- Added `VERSION` file for sub-module semantic versioning.
- Added `CHANGELOG.md` for release-level change tracking.
- Added maintenance `README.md` with scope, version rules, and release checklist.
