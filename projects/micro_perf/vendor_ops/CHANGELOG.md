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

### Fixed

- `NEURON`: report provider versions independently, so a missing
  `torch-neuronx` no longer discards the known `torch-xla` version.
- `NEURON`: pin `PJRT_DEVICE=NEURON` and assert the resolved device is not CPU.
  Without the Neuron PJRT plugin `torch_xla` silently fell back to CPU, and the
  benchmark reported host-CPU latencies under a NEURON label.
- `NEURON`: hold the `core_run()` result across `mark_step()` in `core_perf()`.
  Discarded lazy tensors are dead code and XLA pruned the op being measured,
  producing impossible results such as a gemm at 1896 TFLOPS.
- `NEURON`: serialise `nrt_init()` across workers with a file lock in
  `set_device()`. Workers are spawned together, and two of them reserving cores
  on one NeuronDevice within milliseconds made *both* fail with
  `cores busy, ret=-16`.

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
