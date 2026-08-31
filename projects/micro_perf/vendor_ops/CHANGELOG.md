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

## [0.1.0] - 2026-04-14

### Added

- Established standalone maintenance baseline for `vendor_ops`.
- Added `VERSION` file for sub-module semantic versioning.
- Added `CHANGELOG.md` for release-level change tracking.
- Added maintenance `README.md` with scope, version rules, and release checklist.
