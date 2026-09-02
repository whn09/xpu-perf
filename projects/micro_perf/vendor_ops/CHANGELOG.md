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
- `NEURON`: report `peak_tflops` and `mfu`. `llm_ops.md` has always specified MFU
  against "the corresponding nominal computing power" for the op's
  `compute_dtype`, but no backend shipped a nominal figure, so MFU was
  uncomputable. `Backend.get_peak_tflops(dtype)` is the new hook (returning
  `None` by default, so backends without a spec table are unaffected), and this
  backend fills it from AWS's published dense per-chip peaks divided by the
  logical-core count `neuron-ls` reports — 667 / 4 = 166.75 bf16 TFLOPS per
  device on a trn2 at LNC=2. Sparse peaks are excluded (no op feeds a sparse
  operand), and int8 / fp4 report no MFU because AWS publishes no peak for them.
- `NEURON/README.md`: added reference numbers from a full sweep of every workload
  file in the repo on the eager runtime (trn2.3xlarge, 2026-09-01), covering the
  memory-bound ops, both world sizes for the collectives, and a per-workload
  accounting of how many cases each file actually measures. Three findings worth
  reading before planning a run: `ccl_ops.json` asks for `world_size: 8` and so
  has no runnable case on a 4-core instance; the largest `xccl_ops/` sizes (8 GiB
  at fp32) exceed the 24 GiB a logical NeuronCore gets, and an OOM there hangs
  the launch permanently rather than failing it; and world_size 4 is the only
  size that runs every collective, since `all_to_all` rejects 2.
- `NEURON/IMPLEMENTATION.md`: split out of the README, which had grown to 933
  lines with Quick start buried at line 264 behind the runtime internals. The
  README now runs Quick start -> reference numbers -> how to read them ->
  reproducing -> troubleshooting, and this file holds why the backend code is
  shaped the way it is (dispatch model, compilation behaviour, collective
  plumbing, and the bug each guard rail prevents). Nothing was dropped in the
  split; the README also gained a symptom-indexed troubleshooting section, since
  several failure modes were previously only findable inside prose.
- `NEURON/tools/`: the harness that produced those numbers, so the sweep is
  reproducible rather than just described. `Dockerfile.eager` (the native image
  plus the reporting deps `launch.py` needs), `run_full_sweep.sh` (per-workload
  launches under a `docker kill` watchdog — `timeout docker run` cannot bound a
  launch, since it signals the client while the container is a child of the
  daemon), `cap_xccl_workloads.py` (size-capped `xccl_ops` copies that fit one
  core), `analyze_sweep.py` (per-run accounting: tried / measured / grouped
  rejection reason, plus a per-case curve mode) and `recover_from_log.py`
  (rebuilds CSVs from stdout, since micro_perf writes reports only on a clean
  finish and a killed run otherwise leaves nothing).
- `NEURON`: the eager `flash_attention` provider now covers GQA and single-token
  decode. GQA goes through `enable_gqa=True`, which lets SDPA broadcast the kv
  heads internally instead of `repeat_interleave` materialising a
  `q_head_num`-sized cache inside the timed region; decode is the same view plus
  slice as prefill, because `arg_type: llm` gives every sequence the same
  `q_len`/`cache_len`. The envelope no longer mirrors the NKI provider's, which
  stays MHA-prefill-only. Two exclusions are deliberate and are now stated as
  such: chunked prefill (`cache_len > 0` with `q_len > 1`) and speculative decode
  (`q_len > 1`) both need the causal mask aligned to the bottom-right of a
  non-square score matrix, and PyTorch's `is_causal=True` aligns to the top-left
  — passing it anyway does not fail, it silently attends to the wrong keys.
- `NEURON/README.md`: published the numbers the 2026-09-01 sweep measured but
  never reported. Per-dtype gemm peaks (bf16 149.42 TFLOPS at 12288x8192x8192,
  **90% of the per-core dense peak** — 20 points above the 1024x4096x4096 row a
  smoke test uses, which is too small to hide the dispatch floor) and best
  `mem_bw` for 27 memory-bound ops at each of fp32/fp16/bf16. Five entries in
  that table are lowering quality rather than hardware and are called out: `gelu`
  is 7-14x slower than `silu` over identical shapes, `sin`/`cos` are 4x *slower*
  at bf16 than at fp32 (which is what puts `rotary_embedding` at 42.8 GB/s),
  `reduce_max`/`reduce_min` are 3.6x off `reduce_sum`, fp32 `div` is 1.6x off
  fp32 `mul`, and `rms_norm`/`layer_norm` report byte-identical fp16 and bf16
  latencies, so their bf16 rows are not measuring bf16 arithmetic.
- `NEURON/tools/run_new_workloads.sh`: runs just the workload files added after
  the 2026-09-01 sweep plus the whole-chip aggregate launches, in about an hour
  rather than most of a day.
- `NEURON/tools/run_full_sweep.sh`: a section 7 that re-runs `tensor_gemm_ops` and
  `vector_linear_ops` with all four logical cores busy. What this does and does
  not measure needs stating, because the label invites the wrong reading: a
  `ComputeEngine` spawns one worker per device and they all pull from one shared
  queue, so four *different* cases are in flight and every latency reported is
  still a single-core latency — just one measured while three other cores compete
  for the same HBM. The diff against the single-core runs is the measurement.
- `NEURON`: `gemm` accepts `float8_e4m3` and `float8_e5m2`. The base `GemmOp` gates
  `dtype` to the four float formats, so fp8 is unmeasurable unless a vendor opts
  in. It runs — `torch.matmul` on `torch.float8_e4m3fn` dispatches to the device
  and returns fp8 — but the README states plainly that this is an fp8 *storage*
  number, not an fp8 datapath number, and gives the control that proves it:
  casting both operands to bfloat16 by hand costs 49.8 ms at 4096³ against the fp8
  matmul's 84.6 ms and a native bf16 matmul's 1.04 ms, so the up-cast is
  essentially the entire cost and no fp8 engine is being reached.
  `torch._scaled_mm`, which would express a real fp8 gemm, returns at 64³ but
  wedges `neuronx-cc` past 40 minutes at 4096³. The `mxfloat8*` spellings stay
  rejected on purpose: `TORCH_DTYPE_MAPPING` aliases them to the same plain fp8
  types and no op def carries a block-scale tensor, so accepting them would
  republish these numbers under a label that promises microscaling.
- `NEURON/tools/`: both sweep scripts wait on a `WAIT_BUDGET_S` (default 1800)
  before skipping a run whose cores are still held, so a sweep can be queued
  behind someone else's. The idle check no longer counts containers that merely
  exist: an interactive `docker run -it` shell left open after a sweep holds no
  NeuronCore, and counting it made the wait unsatisfiable, so what counts now is a
  process with `/dev/neuron*` open plus any container with a live `python` in it.
- `NEURON/README.md`: the full fp8 sweep (24 cases, 2 dtypes x 12 shapes) and the
  three properties of it that prove the cast rather than the matmul is being timed:
  MFU stays inside 0.2-1.5%, `e5m2` beats `e4m3` at every single shape by 20-35%
  (which no tensor engine would do, but an emulated widening to bf16 does, since
  e5m2 maps onto bf16 with a shift and a mask while e4m3 needs its exponent
  rebiased), and latency tracks element count rather than FLOPs — 16x the
  arithmetic for 4.1x the time. Also notes that `gemm.json` now enumerates 856
  cases with the fp8 ones last, so a watchdog sized for the float cases cuts the
  run before reaching them.
- `NEURON/README.md`: a troubleshooting entry for the one case most likely to be
  read as a hang. A cold `80/8/128` GQA prefill at `q_len: 4096` spent over 13
  minutes in `walrus_driver` and produced nothing; the identical case then measured
  9,512.6 us, because `NKI_ENABLE_TRACE_CACHE=1` persists the kernel cache across
  processes and the killed run had populated it. The distinguishing check is a live
  `walrus_driver`, not elapsed time.
- `NEURON/README.md`: two notes on reading a single gemm result. The reported MFU
  on the eager runtime is end-to-end and carries the ~60 us dispatch floor inside
  it, so a 1024x4096x4096 bf16 gemm at 0.73 is a tensor engine at ~93%; and a
  `❌ 导入失败 ._<op>` line is macOS AppleDouble sidecars in an rsync'd tree, not a
  broken vendor op — `parse_vendor_ops` imports every `*.py` it finds, and `._x.py`
  becomes the relative import `..x`.
- `NEURON/README.md`: an attention reference table — nine of the ten cases in
  `single_test_ops/fa_linear_ops.json` through the eager SDPA provider (the tenth
  is `q_len: 4` speculative decode, rejected on purpose). Prefill MFU runs
  21.7-32.4%, improving with sequence length and barely moving with batch, since
  at `q_len 4096` one sequence already saturates the core — `batch_size: 4` costs
  4 x 0.905 of a single prefill, i.e. essentially four sequential ones. GQA and MHA
  come out within 1% at every shape, which is the expected result rather than a
  suspicious one (prefill is compute-bound and GQA saves KV traffic, not FLOPs) but
  which also means these numbers *cannot* demonstrate that `enable_gqa=True` avoids
  materialising the expanded cache. The decode rows are memory-bound at ~10 FLOPs
  per byte, so `mem_bw` and not MFU is their column: 110.7-238.7 GB/s against a
  ~725 GB/s core, i.e. 15-33%, where the best plain memory-bound op on the same
  core reaches 631 GB/s. Decode is therefore not bandwidth-limited here, it is
  leaving 2.6-5.7x of available bandwidth unused.
- `NEURON/tools/run_new_workloads.sh`: an `ONLY="<labels>"` filter, so re-running
  one file after its op changed does not mean sitting through gemm's 856 cases
  again.
- `NEURON/README.md`: documented the ops that run but are not worth reporting as
  hardware results — `gather` at 1.34 GB/s and `scatter` at 0.8 GB/s against
  `index_select`'s 631 GB/s, a 470x gap. This was first attributed to the base op
  defs building an output-shaped index (per-element access) where `index_select`
  passes a 1-D one (whole-row DMA); the H100 run has since falsified that, since
  on CUDA `gather` reaches 2,805 GB/s against `index_select`'s 2,862. The index
  construction costs nothing, and the gap is Neuron lowering.

- `GPU`: report `peak_tflops` and `mfu`, so a GPU run and a Neuron run can be put
  side by side. `Backend.get_peak_tflops` defaults to `None` and only the NEURON
  backend overrode it, which left MFU blank on the very backend the comparison
  needs it for. `GPU_PEAK_TFLOPS` tabulates published *dense* per-GPU peaks for
  H100 SXM / H100 PCIe / H200 / A100, matched against
  `torch.cuda.get_device_name()` longest-key-first — "NVIDIA H100 80GB HBM3" and
  "NVIDIA H100 PCIe" both contain "H100" and differ by 30% on every tensor-core
  row, so a shortest-match lookup would quietly misreport one of them. A card that
  matches nothing reports no MFU rather than a wrong one, and A100 carries no fp8
  entry because Ampere has no fp8. Unlike the Neuron hook there is nothing to
  divide by: one CUDA device is one whole accelerator, where one Neuron device is a
  quarter of a chip.
- `GPU`: `gemm` accepts `float8_e4m3` / `float8_e5m2`, via `torch._scaled_mm`.
  Opting in is not sufficient on CUDA the way it is on Neuron — `torch.matmul` has
  no fp8 kernel and raises — so the provider also replaces the run function. Two
  details keep the measurement honest: `b` is declared as `[N, K]` instead of
  `[K, N]` so that `b.t()` is the column-major operand `_scaled_mm` requires
  *as a free view*, rather than a K*N transpose copy inside the timed region
  (element count is unchanged, so the base def's `read_bytes` / `calc_flops` stay
  correct); and the scales are preallocated fp32 ones, because deriving them from
  the data would time a reduction over both operands instead of the gemm.
  `float8_e5m2` is now rejected rather than attempted: cuBLAS has no e5m2 x e5m2
  kernel (`ValueError: Multiplication of two Float8_e5m2 matrices is not
  supported`) because e5m2 is a gradient format and only ever one side of a mixed
  pair, which this op def cannot express since `a` and `b` share one `dtype`.
  Measured on an H100 SXM at 4096³: **1,352.7 TFLOPS / 68.4% MFU**, 1.75x faster
  than bf16 on the same shape — against 3.85 TFLOPS / 1.2% for the same op def on
  a logical NeuronCore, which is the clearest available evidence that Trainium2's
  fp8 here is storage and the H100's is a datapath.
- `GPU`: added a `torch` provider for `flash_attention` using
  `scaled_dot_product_attention`, mirroring the NEURON eager provider's envelope
  case for case. Two reasons it is not redundant with `fa2`. Coverage: `fa2`
  accepts prefill only at `batch_size == 1` with a linear cache and decode only
  with a *paged* cache, so it measures 4 of the 10 cases in
  `single_test_ops/fa_linear_ops.json` and turns away the other 6. And
  comparability: the NEURON eager runtime has no fused flash kernel at all, so its
  attention numbers come from this same source against the same op def — putting
  them next to `flash_attn` would compare two algorithms as well as two chips.
  This is not a slow path on CUDA (SDPA dispatches to a fused FlashAttention or
  cuDNN kernel; prefill measures 61-69% MFU), but note that `targets["kernels"]`
  cannot tell you which backend ran: `core_perf` drops any kernel whose launch
  count differs from `prefer_iterations`, and a fused SDPA call launches more than
  one per iteration, so the list comes back empty.
- `GPU/README.md`: p5.4xlarge (1x H100 SXM5) reference numbers next to the
  Trainium2 ones for the same nine attention cases and the same gemm dtypes, plus
  the unit-comparison rules the whole exercise depends on — one micro_perf device
  is a whole H100 but only a *quarter* of a Trainium2 chip, so a per-device
  latency ratio flatters the GPU ~4x for no physical reason. Normalised per chip
  the bf16 silicon ratio is 1.48x (989.4 vs 667 TFLOPS) while delivered attention
  throughput differs by 3.1x, so most of the attention gap is the software stack;
  and Trainium's decode reads its KV cache at 15-33% of that core's bandwidth
  where the H100 reaches 80-86% of its own, which makes it a fixable lowering
  problem rather than a bandwidth wall. States what is *not* covered: collectives
  (world_size 1 on this instance), paged attention (no Neuron provider takes a
  block table), multi-chip, and `torch.compile`.
- `GPU/tools/run_comparison_sweep.sh`: a `MODE=host` (now the default) that runs
  against an existing interpreter instead of a container. A DLAMI already has a
  working CUDA PyTorch, and the published numbers were taken this way — `PYTHON`
  points at it and `EXTRA_PYTHONPATH` at a `pip install --target` overlay carrying
  the two reporting deps, so the user's environment is never mutated. `MODE=docker`
  is still there and is what you want for the `flash_attn` / `flashinfer` / `vllm`
  providers. Under host mode `timeout` genuinely bounds the run, since the process
  is a direct child rather than a child of the docker daemon.
- `GPU/tools/`: `Dockerfile.cuda` and `run_comparison_sweep.sh`. The Dockerfile
  exists mainly for one non-obvious reason: `flash_attention` has no base
  implementation (`op_defs/llm_ops/flash_attention.py` raises), so without
  `flash_attn` installed every attention case is skipped and the sweep reports
  success having measured none — the build therefore asserts `import flash_attn`
  rather than leaving it optional, while `flashinfer` and `vllm` (extra `rms_norm`
  providers) are allowed to fail. The sweep script runs the same workload files the
  NEURON numbers came from so the two report trees line up, omits `xccl_ops` and
  `ccl_ops.json` (world_size 1 on a single-GPU box), and states in its header that
  MFU rather than latency is the comparable column.
- `GPU/README.md`: two new matched cross-backend sections, so the comparison is no
  longer only attention and gemm. `Memory-bound ops` ranks 24 ops by "Neuron
  shortfall" — the ratio of the two %-of-own-HBM-peak columns, which is the only
  fair statistic when one side has 4.6x the bandwidth — and `Norm, activation and
  MoE ops` does the same per shape for 17 norm / activation / MoE / quant ops.
  The result is that the gap is *bimodal*, not uniform: Trainium2 is at parity on
  12 of the 24 memory-bound ops (0.93-1.15x) and ahead on 9 of the 10
  non-quantised norm/activation rows, while the quantised ops sit at 3.8-38x, fp8
  gemm at ~99x, and `gather`/`scatter` at 449x/621x. Nothing lands between 1.4x
  and 3.1x, and nothing between 7.7x and 99x — which is the shape of a software
  gap in specific lowerings, not of a slower chip.
- `GPU/README.md`: recorded that the per-chip x4 extrapolation is now *measured*
  rather than assumed, for gemm and for elementwise work, and that it has a
  counter-example. With all four logical cores loaded, bf16 `gemm` still delivers
  149.59 TFLOPS per case (89.7% MFU) against 149.42 single-core, and
  `add`/`mul`/`sub`/`cast` over 12 dtype combinations give 594-616 GB/s against
  593-617 — ratios of 1.001 and 1.00-1.02, so 4 x 149.59 = 598.4 TFLOPS per chip
  is real. The fp8 gemm path degrades 1.02-2.16x (median 1.37x) under the same
  contention, which is consistent with it being a software up-cast that moves
  bytes rather than a use of the matmul engine.
- `NEURON/tools/run_new_workloads.sh`: a `core1_gemm` run of the same `task_dir`
  as `chip4_gemm`. The 2026-09-01 sweep's per-case gemm jsonl is gone from disk,
  and comparing `chip4`'s peak against a *published* peak can only show that the
  best case is unchanged — it cannot show whether some particular shape degrades
  under contention, which is the whole question.
- `NEURON/tools/run_new_workloads.sh`: `core1_reduction`, `chip4_reduction` and
  `chip4_moe`, to measure the x4 for selection and sorting. `topk` and
  `moe_softmax_topk` are the two ops where a single logical NeuronCore comes
  within 8-32% of a whole H100, so the per-chip claim there rests entirely on
  scaling — and unlike the elementwise ops already measured, these are the ones
  most likely to serialise on something shared. Cheap to run: the 56-case
  `moe_gating_ops.json` takes 123 s on Neuron, against 1,191 s on the H100.

### Fixed

- `GPU`: pin the fused SDPA backend set in `flash_attention`'s `vendor_impl`.
  Which backend `scaled_dot_product_attention` picks is process-global state, and
  any *other* provider's import can change it: `import vllm` calls
  `torch.backends.cuda.enable_cudnn_sdp(False)`, and the provider registry imports
  every vendor module, so merely having vllm installed silently reconfigured an
  unrelated op. It cost **1.9x on every prefill case** and left no trace in the
  report — cuDNN attention runs an 80/8 GQA prefill at `q_len` 4096 in 584 us,
  where PyTorch's FLASH backend, next in the priority order, takes 1,079 us for
  the same shape. Decode moved only 1.09-1.13x, since a `q_len == 1` step cannot
  exploit causality either way, which is what made the symptom easy to misread as
  noise. Re-enabling all three backends restores PyTorch's own default priority
  whatever else got imported, so a number here no longer depends on which
  unrelated packages happen to be installed; with the fix the full-registry
  environment that had broken it reproduces the published table within 3%
  (prefill 60.6-68.8% MFU, decode 77.9-88.8% of HBM peak). Done as a one-time
  global set rather than an `sdpa_kernel` context, which would sit inside the
  timed region. The general lesson is worth more than the fix: `targets` records
  latency but not the configuration that produced it, so one provider's import can
  re-tune another provider's op with nothing in the report to show it.
- `NEURON/README.md`: withdrew the claim that the six dynamic-quant ops "would be
  similarly bad on a GPU". The same code has since run on an H100 SXM5, and
  "similarly" was too strong in both directions: the shared helper is indeed bad
  there — 73.4-178.7 GB/s, i.e. 2.2-5.3% of a 3.35 TB/s peak, and 3-14x slower
  than the corresponding unquantised op — but a logical NeuronCore gets
  0.15-0.33% of *its* peak on the same shapes, a further 4-23x down. So roughly
  one order of magnitude is the shared per-token-quantisation algorithm and one is
  Neuron-specific lowering. The section now says plainly not to quote these six as
  quantisation performance for any accelerator, and points at the full table in
  `GPU/README.md`.
- `NEURON/README.md`: replaced the guess that the dynamic-quant correctness fix
  makes the op "marginally *more* work" with the measurement — per-shape median
  0.98-1.01x across eleven ops, i.e. performance-neutral within noise, so every
  previously published figure remains valid and nothing needed re-baselining.
  Added that precision on these rows is one significant figure (±25% at 1-2 GB/s),
  which is the bound the pre/post comparison itself establishes.
- `NEURON/README.md`: `head_rms_norm` and `qk_rms_norm` hard-code `norm_weight` as
  `dtype=torch.float32` while `token_data` follows the case dtype, so
  `F.rms_norm` cannot take its fused path on *any* backend. This is a base-op-def
  problem, not a Neuron one, and CUDA says so out loud: `UserWarning: Mismatch
  dtype between input and weight: input dtype = c10::BFloat16, weight dtype =
  float, Cannot dispatch to fused implementation.` Documented rather than changed,
  since altering an op def would invalidate both backends' existing numbers.
- `NEURON/README.md`: corrected three claims that the sweep disproved.
  (1) The eager runtime's first-run cost was given as "none"; an op the runtime
  has no prebuilt kernel for still falls back to a full `neuronx-cc` compile, and
  `gather` at bf16 / `dim_size=8192` sat in one for over two hours. (2) Eight
  quantized ops were listed as unsupported by Neuron; they are not implemented by
  *any* vendor, the base defs gate them to `int8/int8/int8 -> bfloat16`, and that
  path is `fake_quant_gemm` — a bf16 matmul with scale multiplies, on every
  backend — so the numbers are not int8 hardware numbers anywhere. Four of them
  do run on eager. (3) `all_to_all` was said to be unsupported on this instance
  type; it is gated on world size (4, 8, 16, or multiples of 32) and runs at
  `--device 0,1,2,3`. The "Known unsupported" table now separates a base-op-def
  limit from a backend limit, and distinguishes untested-because-no-workload-uses-it
  from unsupported.
- `NEURON/README.md`: documented that an op whose engine is excluded by
  `XPU_PERF_ENGINES` is dropped with no warning and exit code 0. Engine
  membership does not follow the workload directory — `device2device` sits in
  `workloads/xccl_ops/` but registers under `ComputeEngine`, so a collectives
  launch measures none of it and reports success.
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
