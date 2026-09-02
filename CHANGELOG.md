# Changelog

All notable changes to `xpu-perf` are documented in this file.

This changelog follows semantic versioning and Keep a Changelog style.

## [Unreleased]

### Added

- `projects/micro_perf/workloads/llm/single_test_ops/`: workload files for the op
  families that had an `op_defs` implementation but no workload JSON anywhere in
  the repo, so they were untested rather than unsupported on every backend —
  `norm_ops.json` (`add_rms_norm`, `head_rms_norm`, `qk_rms_norm` and the two
  `_dynamic_quant` variants), `activation_ops.json` (`swiglu`, `moe_swiglu` and
  their `_dynamic_quant` variants), `moe_gating_ops.json` (`moe_softmax_topk`,
  both `compute_mode`s) and `quant_ops.json` (`scale_dynamic_quant`,
  `quant_group_gemm_reduce_sum`). Each sweeps `num_tokens` from 1 to 32768 so the
  launch-bound region and the bandwidth plateau are both visible, and each has
  the companion `.md` the directory's other files have.
- `projects/micro_perf/workloads/llm/single_test_ops/fa_linear_ops.json`: the
  `fa_ops.json` head configurations over a **linear** kv cache. Every case in
  `fa_ops.json` sets `block_size: 512`, i.e. a paged cache, so a vendor that
  implements only a contiguous cache has no runnable case in that file at all and
  no GQA or decode number can be obtained from it. This file covers GQA and MHA
  prefill, a batched prefill, and decode at two batch sizes.
- `projects/micro_perf/workloads/basic/tensor_gemm_ops/gemm.json`: `float8_e4m3`
  and `float8_e5m2` cases. The base `GemmOp` gates `dtype` to the four float
  formats, so these are unsupported until a vendor overrides `vendor_parser` —
  the same arrangement `tfloat32` already has, where the case exists in the
  workload and each backend answers for itself. The M list is short on purpose:
  fp8 costs 60-85 ms per case where bf16 costs 1 ms, so a 52-point sweep would
  not finish.

### Fixed

- `src/xpu_perf/micro_perf/core/utils.py`: `smooth_per_token_dynamic_quant`
  quantised the wrong tensor. It computed `torch.mul(smooth_scale,
  per_token_scale)` — a `[1, hidden_size]` smoothing vector times a
  `[num_tokens, 1]` scale — where it meant `torch.mul(smoothed_input,
  per_token_scale)`. Both broadcast to `[num_tokens, hidden_size]`, so nothing
  failed and every shape check passed, but the returned `quant_tokens` did not
  depend on `hidden_states` at all. This is the quantisation body of
  `scale_dynamic_quant`, `add_rms_norm_dynamic_quant`,
  `head_rms_norm_dynamic_quant`, `swiglu_dynamic_quant`,
  `moe_swiglu_dynamic_quant` and `moe_scatter_dynamic_quant`, on every backend.
  The correct version also reads a `[num_tokens, hidden_size]` operand instead of
  a `[1, hidden_size]` one, so it is marginally *more* work, not less.

## [0.1.0] - 2026-04-14

### Added

- Initial public project baseline in `pyproject.toml`.
- Established standalone release management for:
  - `projects/micro_perf/op_defs`
  - `projects/micro_perf/vendor_ops`
