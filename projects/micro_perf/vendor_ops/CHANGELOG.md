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
- `NEURON/README.md`: two notes on reading a single gemm result. The reported MFU
  on the eager runtime is end-to-end and carries the ~60 us dispatch floor inside
  it, so a 1024x4096x4096 bf16 gemm at 0.73 is a tensor engine at ~93%; and a
  `❌ 导入失败 ._<op>` line is macOS AppleDouble sidecars in an rsync'd tree, not a
  broken vendor op — `parse_vendor_ops` imports every `*.py` it finds, and `._x.py`
  becomes the relative import `..x`.
- `NEURON/README.md`: documented the ops that run but are not worth reporting as
  hardware results — `gather` at 1.34 GB/s and `scatter` at 0.8 GB/s against
  `index_select`'s 631 GB/s, which is a 470x gap caused entirely by the base op
  defs building an output-shaped index (per-element access) where `index_select`
  passes a 1-D one (whole-row DMA), not by anything Neuron-specific.

### Fixed

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
