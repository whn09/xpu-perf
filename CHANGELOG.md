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
- `projects/micro_perf/vendor_ops/NEURON/tools/analyze_scaling.py`: joins a
  one-core report against a four-core report case by case, so the `x4` in every
  published per-chip Trainium2 figure is checked rather than assumed. Reports the
  median and the tail separately per op and per `(op, dtype)`, the worst shape per
  op, and the ratio *at the peak shape* — which is the cell a README actually
  publishes and which the all-shape median is the wrong check for.
- `projects/micro_perf/vendor_ops/NEURON/tools/probe_index_dtype.py`: the `gather`
  investigation as one runnable file — the int64-vs-int32 index sweep, the
  `.contiguous()` control that identifies layout rather than dtype as the
  proximate cause, the 1-D `index_select` control, `scatter_add_`, and a dump of
  `_NEURON_OPS_REGISTRY` showing the declared-vs-effective implementation
  priorities.

### Changed

- `projects/micro_perf/vendor_ops/NEURON/README.md`,
  `projects/micro_perf/vendor_ops/GPU/README.md`: **every per-chip Trainium2
  figure is now measured instead of extrapolated.** Running the compute-bound,
  reduction, selection and MoE workloads on one logical core and again on four
  gives a median 4c/1c latency ratio of 1.002–1.029 across 648+ matched shapes, so
  `x4` was close to right — but with two exceptions the multiplication hid. fp8
  `gemm` medians are 1.464 (e4m3) and 1.525 (e5m2), i.e. an effective x2.6–2.7,
  and *every* family has a small-shape contention tail of 5–11x (`gemm` bf16
  M=2/K=1024/N=8192 9.01x, `topk` 1024x128 k=8 11.02x, `moe_softmax_topk`
  num_tokens=1 8.89x, `reduce_max` fp32 1024x2048 8.33x). The rule now stated in
  both files: do not multiply a small-shape single-core number by four.
- `projects/micro_perf/vendor_ops/NEURON/README.md`,
  `projects/micro_perf/vendor_ops/GPU/README.md`: the `gather` row is reframed
  around its cause. The 449x is an **int64-index artefact**, not an op-level
  deficit: the device has no int64, so the index is converted on the way into the
  graph by a NKI custom call, and that conversion materialises the stride-0
  broadcast `GatherOp.create_tensors` builds with `.expand()` — which the fast
  lowering needs to see intact. The identical call with an int32 index is 8–704x
  faster and lands within 8% of `index_select`; int32 plus `.contiguous()` is
  92–575x slower again, which is what identifies layout rather than dtype.
  `scatter_add_` moves 547–2453x on the same change. Plain `scatter` does not
  move, because `aten::scatter_.src` has no implementation at all — so its 621x is
  real. The op defs are deliberately left on int64, since `torch.gather` requires
  it on CPU and CUDA and the published rows are the honest cross-backend number.
- `projects/micro_perf/vendor_ops/GPU/README.md`: the fp32 `gemm` cell now holds a
  native-eager 37.67 TF/core (83.3% MFU, 150.7 TF/chip) in place of a stale 23.5
  TF from an XLA run, which reverses the comparison — Trainium2 is **3.09x ahead**
  of the H100's 48.72 TF on a 2.70x nominal bar.

### Fixed

- `projects/micro_perf/vendor_ops/NEURON/ops/torch/flash_attention.py`,
  `projects/micro_perf/vendor_ops/GPU/ops/torch/flash_attention.py`,
  `projects/micro_perf/vendor_ops/NEURON/README.md`,
  `projects/micro_perf/vendor_ops/GPU/README.md`: all four claimed the Neuron
  eager runtime has no fused flash kernel, and attributed the attention gap to
  that. **It is false, and the A/B falsifies it.** `torch_neuronx` lowers SDPA to a
  NKI flash kernel in its own dynamo backend (`_can_use_nki_flash_attention`,
  enabled by default via `TORCH_NEURONX_ENABLE_NKI_SDPA`) whenever
  `L % 512 == 0 and S % 512 == 0 and D <= 128 and B*H <= 512` with no attn_bias
  and no dropout. Setting the variable to 0 takes the 80/8/128 GQA prefill at
  `q_len` 4096 from 9,443 us to 60,500 us — 6.41x — so the prefill rows *are* a
  fused-kernel score and the remaining gap is kernel quality, not a missing
  kernel. Every prefill case clears the gate and no decode case can reach it
  (`q_len == 1` never satisfies `L % 512 == 0`, and `B*H` is 1280/5120 there),
  which is what the decode rows reflect. The narrower true statement the old claim
  rested on is kept: `neuronxcc.nki.kernels.attention.flash_fwd` is HLO-traced and
  loads only under torch_xla.
- `projects/micro_perf/vendor_ops/NEURON/README.md`: records a second
  `torch_neuronx` defect found while chasing the `gather` row — the
  `@neuron_op(priority=)` override is dropped. Every registered instance reports
  the class default of 50 regardless of what it declared, so implementation
  selection falls to import order. Verified for `aten::gather`
  (`GatherMLIRImpl` declared 40, `GatherNKIImpl` declared 60, both effective 50),
  `aten::scatter_add`, `aten::scatter_add_`, `aten::contiguous` (five impls
  declaring 95/90/100/200/50), `aten::copy_` and `aten::_to_copy`.

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
