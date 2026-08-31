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

## [0.1.0] - 2026-04-14

### Added

- Established standalone maintenance baseline for `vendor_ops`.
- Added `VERSION` file for sub-module semantic versioning.
- Added `CHANGELOG.md` for release-level change tracking.
- Added maintenance `README.md` with scope, version rules, and release checklist.
