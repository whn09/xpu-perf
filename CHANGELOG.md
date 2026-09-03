# Changelog

All notable changes to `xpu-perf` are documented in this file.

This changelog follows semantic versioning and Keep a Changelog style.

## [Unreleased]

### Added

- `projects/micro_perf/vendor_ops/NEURON/ops/nkilib/`: a second NEURON
  `flash_attention` provider, `nkilib`, calling `nkilib.core.attention.attention_tkg`
  (token-generation attention) for the **decode** rows. The existing `torch`/SDPA
  provider cannot reach a fused kernel there — torch_neuronx's rewrite gate needs
  `L % 512 == 0` and `B*H <= 512`, and a decode step offers `L == 1` and `B*H` of
  1280-5120 — so decode was measuring an unfused score-matrix path whose bandwidth
  *falls* with cache length. `attention_tkg`'s rises: measured on Trn2 at 80/8/128
  GQA bf16, the two cross between kv_len 4,096 (0.55x) and 8,192 (1.56x), reaching
  **4.09x at 16,384** and 42% of the core's memory bandwidth. Both providers stay
  registered because neither wins everywhere. It also covers `q_len 4` speculative
  decode, which the SDPA provider rejects because PyTorch's `is_causal` aligns to
  the top-left of a non-square score matrix.
  `attention_tkg` is not directly launchable — it takes a `BufferManager` and a
  caller-allocated `out` — so the `@nki.jit` entry point is part of the provider.
- `projects/micro_perf/vendor_ops/NEURON/workloads/fa_decode_tkg.json`: the decode
  rows of `fa_linear_ops.json` restated with `kv_len` on a multiple of 128, which the
  kernel asserts (`cache_len 4095` + `q_len 1` rather than 4096 + 1 — a 0.02%
  shorter context), plus kv_len 8,192 and 16,384 so the crossover between the two
  providers is visible. Both providers run this file, so one launch produces the
  comparison.
- `projects/micro_perf/vendor_ops/NEURON/tools/probe_attention_tkg.py`: the
  configuration sweep behind that provider — flat vs paged prior (paged is 4-8x
  faster and the only one that ever beats SDPA), the `block_len` sweep showing the
  kernel overrides the caller's choice, the QK-swap eligibility check showing 80/8
  GQA can never reach the kernel's fast MM1 path, and the correctness check against
  SDPA (rel_err 0.0033-0.0065 everywhere).
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
- `projects/micro_perf/vendor_ops/NEURON/tools/probe_inplace_write.py`: tests
  whether the two op defs that write into a strided slice view of a large
  pre-allocated tensor — `store_kv_cache` and `rotary_embedding` — actually get an
  in-place write. Part 1 is the slope test (hold the update size fixed, scan the
  buffer size; flat means in-place, linear means a functionalised full-buffer copy)
  with `add_` as the positive control and a contiguous-destination control. Part 2
  decomposes `rotary_embedding` at the shape the README publishes, which the slope
  test cannot cover because its prefill cases have `q_len == num_tokens`.
- `projects/micro_perf/vendor_ops/NEURON/tools/probe_attention_kernel.py`: settles
  which kernel backs the prefill attention rows, by object identity rather than by
  benchmark — `torch_neuronx`'s `decompositions.py` imports `attention_cte` from
  `nkilib` at line 24 and wraps *that object* at line 71, so
  `dc.attention_cte is attention_cte` is the whole proof and it is `True`. Then
  times the same shape three ways (SDPA; `attention_cte` launched the way
  `torch_neuronx` launches it; `attention_cte` launched the obvious way) and checks
  every variant's output against the SDPA reference, because two of the three are
  traps: the `lnc` subscript takes a bare int, and omitting it costs 1.85x by
  running on one half of the LNC2 pair.
- `projects/micro_perf/vendor_ops/NEURON/tools/probe_fp8_datapath.py`: separates the
  two independent causes of the 1.2% fp8 MFU — the `e4m3fn`-vs-`f8e4m3` encoding
  mismatch and the absence of an eager fp8 gemm lowering — by sweeping
  {bf16, e5m2, e4m3fn} x {eager, compiled} at two square shapes. Documents two
  reproduce gotchas in code: `torch.compile(..., dynamic=False)` is mandatory when
  compiling one function body at two shapes, and the `e4m3fn` compile must run last
  because its failure latches an error that the *next* device op inherits.

### Changed

- `projects/micro_perf/vendor_ops/GPU/README.md`: **the Summary table now has an
  `attention decode` row, and the bimodality claim above it is corrected.** Decode
  was the one workload with a measured Trainium2 number and no line in the table
  everybody reads first; it is **1.9-3.2x per chip** with the better of the two
  providers at each shape, 1.6-2.7x of that software against a 1.16x bandwidth bar.
  Two claims did not survive it. The table said "nothing here lands between 1.4x and
  3.1x", offered as evidence that the gap is bimodal — decode lands in the middle of
  that band, and the band was an artifact of decode never having been measured on an
  aligned workload, so the split is now stated as a gradient with the surviving
  claim (nothing in the far group is explained by peak FLOPS or peak bandwidth)
  separated from the retracted one. And decode's software residual was described as
  "much smaller than the prefill row's 2.1x"; it straddles it (ahead at kv_len 4,096,
  behind at 8,192), so the honest distinction is the *sign of the curve* rather than
  the midpoint — prefill's residual grows with the problem size and decode's stops
  once the right provider is used. The cross-chip decode block is also now a
  `###` subsection so the summary row can link to it, and it carries the SDPA-only
  per-chip series (1.86x to 11.13x) that the "provider-dependent" range refers to.

- `projects/micro_perf/vendor_ops/GPU/README.md`,
  `projects/micro_perf/vendor_ops/GPU/tools/run_comparison_sweep.sh`,
  `projects/micro_perf/vendor_ops/GPU/ops/torch/flash_attention.py`,
  `projects/micro_perf/vendor_ops/NEURON/README.md`: **corrected the documented
  reason `single_fa_ops` is empty.** Three places said, or implied, that
  `fa_ops.json`'s paged cases were the `fa2` provider's and that the GPU side
  could fill that row once `flash_attn` was installed. It cannot: installing
  `flash_attn` unlocks **0 of the 11 cases**. All 11 set `block_size: 512`, which
  is the only thing that makes `cache_type` `"paged"` (`core/utils.py:427`), and
  `FA2Op.vendor_parser` (inherited unchanged by `fa3`) rejects the file from both
  ends — its 9 **prefill** cases because that path demands
  `cache_type == "linear"` (`flash_attn_func` takes no block table; only
  `flash_attn_with_kvcache` does), and its 2 **decode** cases, which are paged as
  that path wants, because the same parser demands an all-bfloat16 dtype set and
  both carry `cache_dtype: int8`. The file needs an **int8-KV paged** kernel and
  no provider in the tree has one — `ops/flashinfer/` and `ops/vllm/` hold only
  `rms_norm.py`, and flashinfer is already installed on the H100 box. So this is
  a missing provider, not a missing dependency, and it is a gap on both backends
  rather than a Neuron-only one.

- `projects/micro_perf/vendor_ops/NEURON/tools/run_full_sweep.sh`,
  `projects/micro_perf/vendor_ops/NEURON/tools/run_new_workloads.sh` (**removed**),
  `projects/micro_perf/vendor_ops/NEURON/README.md`: **two sweep scripts had
  drifted into disagreeing about how to measure the same thing.**
  `run_new_workloads.sh` existed so a re-run of "just what changed" took an hour
  instead of a day, but `ONLY=<labels>` already does that, and maintaining two
  label lists cost results three separate ways: `single_fa_linear_ops` had a
  21,600 s budget in one script and a fatal 5,400 s in the other; `gemm`'s 24 fp8
  cases were reachable only from the newer script, because they are *last* in
  `gemm.json` at 20-570 s each and the watchdog fires inside them at the 5,400 s
  the float cases need — which, since micro_perf writes its CSVs only when a launch
  finishes, also loses the 832 float cases; and `single_fa_decode_tkg` had been
  added only to the older one. The second script is deleted and its labels folded
  in: `chip4_reduction` and `chip4_moe` are now section 7, and its
  `core1_gemm`/`core1_reduction` were `basic_tensor_gemm_ops` and
  `basic_vector_reduction_ops` under different names. `basic_tensor_gemm_ops` now
  has 28,800 s, which makes one launch serve three purposes (float cases, fp8 tail,
  and the one-core report `analyze_scaling.py` joins against `chip4_gemm`);
  `single_fa_linear_ops` and `chip4_gemm` get the larger budgets. 31 labels, one
  script.
- `projects/micro_perf/vendor_ops/GPU/README.md`: **the cross-chip decode ratio is
  measured rather than deferred.** The H100 has now run
  `../NEURON/workloads/fa_decode_tkg.json` — the same six comparable cases as the
  Trainium side — so the row that said "no new cross-chip decode ratio is quoted"
  now carries a table: **1.9-3.2x per chip on a 1.16x nominal bandwidth bar**, i.e.
  1.6-2.7x software, the narrowest residual in the comparison outside the parity
  rows. The H100's own column is flat at 69-91% of HBM peak across all six cases, so
  the entire cross-chip movement in this row comes from the Trainium side and from
  which provider wins: SDPA at kv_len 4,096, `attention_tkg` from 8,192 up. The
  `q_len 4` case is rejected on **both** backends (PyTorch's `is_causal` aligns to
  the top-left of a non-square score matrix), so `nkilib`'s speculative-decode
  coverage still has no GPU counterpart.
- `projects/micro_perf/vendor_ops/GPU/README.md`,
  `projects/micro_perf/vendor_ops/GPU/tools/run_comparison_sweep.sh`,
  `projects/micro_perf/vendor_ops/NEURON/README.md`: **the GPU Summary table read
  as an inventory of the workload tree and was only the measured subset of it.**
  Twelve workload files have a Trainium2 number and no GPU column, so no ratio can
  be quoted for them, and the table gave no sign they existed. A new
  `What this table does not cover yet` subsection tabulates all of them with the
  op names, what the Trainium2 side got, the blocker, and what closing the row
  would settle. The split that matters is that only three of the entries are
  hardware-blocked — the four collectives and `device2device` need more than the
  one GPU a p5.4xlarge has — while five files
  (`xccl_ops/device2host.json`, `xccl_ops/host2device.json`,
  `single_test_ops/gemm_ops.json`, `moe_dispatch_ops.json`,
  `moe_combine_ops.json`) are runnable on the existing instance and are missing
  only because nobody has run them. The same list is now a TODO block in the sweep
  script, next to the existing "deliberately NOT here" note, so the two cannot
  drift. `What is not measured here` at the foot of the README stops duplicating
  the coverage list and instead records the gaps that are not about coverage:
  multi-chip scaling, `torch.compile` on either side (the fp8 correction is the
  demonstration that this is not a neutral omission), and the absence of any
  numerical check, which is what let two op-def bugs in this port be found by
  reading rather than by measuring. The NEURON README's own
  `Most workloads/llm/ cases do not run` table now cross-links to it, since the
  two cover mirror-image gaps.
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

- `projects/micro_perf/vendor_ops/NEURON/README.md`: a section on in-place
  semantics. `store_kv_cache` documents itself as "This operator is inplace"; on
  eager Neuron a write into a strided slice view is not. Fixed 256 KB update over an
  8 → 128 MB `k_cache`, the write scales 3.25x where a contiguous 2-D slice write is
  flat at 43 us, and the 128 MB point's 0.556 ms is 460 GB/s scored as a whole-buffer
  copy against 0.46 GB/s scored as the slice. The trigger is the non-contiguous
  destination, not slicing and not the offset. `torch.compile(backend="neuron")` —
  the usual advice, since dynamo establishes input/output aliasing — helps at 8 MB
  and then fails to compile at 32 MB and 128 MB with
  `RuntimeError: Neuron backend NEFF execution setup failed`, so it is not a remedy
  at realistic KV cache sizes.

### Fixed

- `projects/micro_perf/vendor_ops/NEURON/README.md`,
  `projects/micro_perf/vendor_ops/GPU/README.md`: **both READMEs concluded that
  Trainium2 reaches no fp8 datapath at all, and told the reader not to plan capacity
  around fp8 on this stack. That was too broad.** The 1.2% MFU measurement was
  right; what it measures is the *eager* path in an encoding the chip does not
  implement, not the hardware. Two independent causes, neither of which alone
  explains the number. (1) **Encoding:** `float8_e4m3` maps to
  `torch.float8_e4m3fn`, the finite-only variant CUDA uses; Trainium1/2 implement
  the other e4m3, and the compiler says so by name — `[NCC_EVRF051] Data type
  F8E4M3FN is not supported on TRN1/TRN2`. The workaround flag that error recommends
  does not exist in neuronx-cc 2.27.2878.0 (`compile --help` has no fp8 options at
  all). (2) **Path:** eager has no fp8 gemm lowering for *either* encoding, so both
  land on the same software widening at ~1-2 TFLOPS. Fix both — `e5m2` under
  `torch.compile(backend="neuron", dynamic=False)` — and one logical core does
  **245.50 TFLOPS at 75.6% of its 324.75 TF fp8 peak** at 4096^3, 1.82x its own
  bf16 on a nominal 1.95x bar and 115.7x the eager figure at the same shape. That
  path also scales across cores where eager fp8 does not (1.018x worst case over
  four concurrent runs against 1.46-1.53x eager), so **982 TFLOPS per chip and a
  1.56x gap to the H100** where the nominal fp8 peak ratio is 1.52x — not 99x. The
  99x row stays in the tables because it is what `gemm.json` measures today;
  closing it needs an fp8 provider that compiles and `float8_e5m2` cases to point
  at. Also newly documented: the fp8 row **cannot** be made like-for-like inside
  this op def, since the H100's number is `e4m3fn` via `torch._scaled_mm`,
  Trainium2 has no `e4m3fn`, and `torch._scaled_mm` rejects `e5m2 x e5m2` on CUDA —
  the two chips share no fp8 format both stacks will multiply. And
  `torch._scaled_mm` is not a usable route on Neuron: 690 ms at 512^3 (~5,500x the
  bf16 matmul), and at 2048^3 a single call did not return in 3.5 minutes while
  burning 555% host CPU, which is the "50% CPU, 0% Neuron" symptom it produces.
- `projects/micro_perf/vendor_ops/NEURON/README.md`,
  `projects/micro_perf/vendor_ops/GPU/README.md`: the attention sections said the
  prefill rows are a fused NKI kernel but left open which kernel, and listed writing
  a NKI prefill provider as work that would close the gap. **It would not: the
  kernel the SDPA rewrite lowers to already *is* `nkilib`'s `attention_cte`**, by
  object identity in `torch_neuronx`'s `decompositions.py` (line 24 imports it, line
  71 wraps it, line 935 launches it with `tp_q=True, tp_k=True` and KV left at its
  own head count so the kernel does the GQA). Calling `attention_cte` by hand with
  those arguments reproduces the SDPA latency to **0.97x** (7,407.4 vs 7,636.7 us at
  B=1/HQ=80/HKV=8/L=4096/D=128) with bit-identical output — one kernel, reached two
  ways. So the residual ~2.1x software component of the 3.1x prefill gap is this
  kernel against cuDNN/FA, and there is no unused kernel in `nkilib` to reach for.
  The `What would close these gaps` list now scopes its NKI-attention item to
  decode (`attention_tkg`). Recorded as a trap because it nearly produced a false
  finding: launched without the `lnc` subscript, `attention_cte` runs on one half of
  the LNC2 pair and measures 1.85x slower, which reads as "the nkilib kernel is
  worse than what SDPA gets".
- `projects/micro_perf/vendor_ops/NEURON/README.md`,
  `projects/micro_perf/vendor_ops/GPU/README.md`: **`rotary_embedding`'s 42.8 GB/s
  was attributed to Neuron's slow bf16 `sin`/`cos`. That is impossible** — `cos` and
  `sin` are precomputed by `precompute_freqs_cis` when the tensors are created, and
  `rotate()` is only `cat`/`mul`/`add`, so no trig runs inside the timed region. The
  op was run on an H100 to settle it (it had no GPU number, since `pre_fa_ops.json`
  was never run there) and the two backends agree to a significant figure as a
  fraction of their own peak: **5.89% for the H100 at 197.35 GB/s against 5.90% for
  one Trainium2 core at 42.77**. Two chips 4.62x apart in bandwidth do not land on
  the same fraction of peak by accident — the cost is in the op def. Decomposing the
  body confirms it: `rotate()` is 78.1% of the 10,995.6 us, five-plus materialising
  elementwise passes where a fused kernel would be one pass (318 us at peak). Per
  chip the op is 1.15x, exactly the memory-bound bar. The decode rows are a third
  thing again and are not bandwidth on either side: `vendor_impl_run` loops over
  batches in Python, so `batch_size` 16 at `q_len` 1 is 16 dispatches of one token.
- `projects/micro_perf/vendor_ops/NEURON/README.md`: two `store_kv_cache` blockers
  added to the "Known unsupported" table, both **base op def** and so affecting the
  GPU backend identically. All 16 cases in `pre_fa_ops.json` set `block_size: 512`,
  which makes the cache paged and hits
  `raise NotImplementedError("StoreKVCacheOp paged cache not implemented yet.")` at
  `store_kv_cache.py:257`. Independently, `store_mode: "k"` cases fail with
  `KeyError: 'v_cache'` at `store_kv_cache.py:248`, because `vendor_impl` creates
  `v_cache` only for `store_mode in ("both", "v")` while `vendor_impl_run` reads it
  unconditionally — so removing `block_size` is not enough to make the file run.
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
