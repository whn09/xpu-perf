# Changelog

All notable changes to `xpu-perf` are documented in this file.

This changelog follows semantic versioning and Keep a Changelog style.

## [Unreleased]

### Added

- `projects/micro_perf/workloads/models/qwen3_5_27b/`: eight workload files (392
  cases) shaped from `Qwen/Qwen3.5-27B`'s real `config.json` rather than from powers
  of two, at TP=1 and TP=4, plus a README documenting the provenance of every shape.
  The model's nine distinct GEMM `K.N` pairs and `basic/tensor_gemm_ops/gemm.json`'s
  four have an **empty intersection**, which is why per-op numbers could not previously
  be attached to a per-model claim without interpolating. Covers `head_dim 256`
  attention, `intermediate_size 17408`, the 248320 vocabulary, partial rotary
  (`rope_dim 64` against a `rope_dim 256` control), the gated-delta-net path
  decomposed into registered ops, and the TP=4 all_reduce message shape. Reachable
  from both sweep scripts as gated `qwen3_5_27b_*` labels; excluded from the default
  runs so the published comparison table keeps meaning one thing.
  The README also lists the nine things the model executes that **no op def can
  express** — the delta-rule scan itself, `conv1d`, `cumsum`, triangular inverse,
  `softplus`/`tanh`, `bmm`, `where`, `index_copy_`/`pad`/`cat`, and a paged
  `store_kv_cache`. Three of those are exactly where the working vllm-neuron port had
  to hand-roll a workaround.

  Measured on one H100 80GB HBM3: 348 of 384 cases in 4 min 45 s. **`head_dim 256`
  costs the H100 nothing** — 701.9 TFLOPS / 70.9% MFU on prefill at `q_len` 10240, and
  decode is bandwidth-bound at 2698 GB/s (92% of achievable). GEMM peaks at 81.9% MFU
  and the non-power-of-two 17408 costs nothing, but the tiny-N GDN projections (`N` 96
  and 24) run at 1-21% of peak and the 128×128 delta-rule scan tile reaches only 13.5%
  MFU — the concrete form of the claim that the GDN path is utilisation-bound, not
  FLOP-bound. Partial rotary is worth 2.8-3.6x in prefill. `swiglu` peaks at 1041 GB/s
  against `silu`'s 2952 on the same card, and `topk` over the vocabulary costs
  90-300 us almost independently of `k`.

  Measured on one Trn2 chip (`trn2.3xlarge`, PyTorch-native Beta 4, eager), all 384
  overlapping cases plus 6 Trainium-only `all_reduce`, in 41 min. Ratios are per
  **logical core**, of which a chip has four (the harness's own `peak_tflops` is
  166.75, one quarter of 667), so ~4-6x raw is per-chip parity against the two chips'
  989.4/667 = 1.48x bf16 peak ratio. **The parity claim holds for the arithmetic and
  fails for this model**, for reasons that are all kernel coverage rather than silicon:

  - **Dense GEMM confirms parity, in Trainium2's favour**: 152.7 TFLOPS on one logical
    core is **91.6% MFU**, better than the H100's 81.9%. Median 4.42x raw ≈ 1.1x per
    chip. The fp32 128×128 scan tile is ~1.9x *faster* per chip, which is the dtype
    `mamba_ssm_dtype: float32` puts 48 of 64 layers' scan in. Exception: `lm_head`
    (5120→248320) degrades as tokens rise, 147 → 94 → 71 TFLOPS at 1024 → 4096 →
    10240, where the H100 holds ~750 — 2.56x per chip at the top. Tiny-N is a
    non-issue: Trn2 is absolutely *faster* at ≤1024 tokens.
  - **`head_dim 256` reaches no fused attention path at all**: `nkilib`'s
    `attention_tkg` rejects decode (`P_MAX = 128` partitions) *and* prefill
    (token-generation only), and torch_neuronx's SDPA rewrite gate also needs
    `D <= 128`, so the union is empty and the fallback materialises the score matrix.
    Prefill 24/4/256 is **37.6x at `q_len` 4096 and 153.9x at 10240**, with throughput
    *falling* 15.0 → 4.6 TFLOPS as the H100's rises 566 → 702 — superquadratic, so the
    gap grows with context and cannot be extrapolated from a short measurement. The
    vision tower's head_dim 72 sits in the normal 4.0-9.6x band precisely because it is
    under the limit.
  - **`gelu` is 12-15x below its own chip's roofline**: flat at **31-42 GB/s at every
    size**, 52x median / 61.7x max. The control is `silu`, same chip, same dtype, same
    `base` provider, at **475-489 GB/s** — ~1.9 TB/s per chip against the H100's 2.9,
    exactly the expected 1.45x. Confined to the vision tower's `gelu_pytorch_tanh`.
  - **`topk` has a cliff above `k = 8`**: 557 us at k=1 and k=8, **5312 us at k=50**,
    at every vocabulary size and batch, while the H100 is flat in `k`. At k ≤ 8
    Trainium2 is at or ahead of parity per chip; k=50 is what production sampling uses.
  - **`rms_norm`'s 13.8-14.4x at 10240 tokens is mostly a provider mismatch** — the
    H100 runs `flashinfer`/`vllm` fused kernels, Neuron runs `base` — but its 190.9
    GB/s is still 2.5x under `silu` on the same chip. `add_rms_norm`, `qk_rms_norm` and
    `head_rms_norm` run `base` on both sides and land at 1.6-3.7x raw, i.e. Trainium2
    ahead per chip, which is the cleaner comparison.
  - **Partial rotary buys Trainium2 1.2-1.4x where it buys the H100 3.6x**, so
    `partial_rotary_factor: 0.25` is close to being ignored.
  - **`all_reduce` at TP=4 is 4.1x over the decode budget**: the 160 KiB message the
    step sends 128 times measures 118.1 us against the 28.5 us needed to stay under 10%
    of a 36.5 ms step — 15.1 ms, or 41% of the step, in collectives. Latency is flat at
    105-118 us from 10 KiB to 640 KiB, so this is fixed overhead, not bandwidth
    (which reaches a respectable 106.6 GB/s at 100 MB).

  `store_kv_cache` and `rotary_embedding` **decode** rows are excluded from all of the
  above: they measure the op defs' per-sequence Python loop on both backends, and their
  224 ms / 2.3 ms ratio is launch overhead, not a chip result.

- **Four of the gaps above are now closed**, each by a different mechanism, and each
  verified end-to-end through the harness on both backends rather than in a standalone
  script:

  | gap | before | after | mechanism |
  | --- | --- | --- | --- |
  | `gelu` | 15.41x per chip | **4.35x** | the benchmark was measuring the wrong function |
  | `topk` | 14.2x per chip worst | **2.91x** | wire up a kernel nkilib already ships |
  | attention `head_dim 256` | 153.6x raw worst | **38.4x** | tile the query axis; no kernel at all |
  | `rms_norm` | 3.45x per chip bf16, 3.61x fp32 | **1.50x / 1.26x** | write the kernel nkilib does *not* ship in this layout |

- `projects/micro_perf/op_defs/basic_ops/vector_activation_ops.py`: `GeluOp` now takes
  an `approximate` argument, torch's own, defaulting to `"none"` so every pre-existing
  `gelu` row keeps its meaning. `models/qwen3_5_27b/activation_ops.json` asks for both
  modes (42 → 48 cases; the directory total is 392 → 398). This matters because
  Qwen3.5's config specifies `gelu_pytorch_tanh` while
  `torch.nn.functional.gelu` defaults to the **erf** form, so the published `gelu`
  numbers were of an activation this model does not use — and the two are 3.75x apart on
  Trainium2. Same chip, same shapes, one launch: 16384 × 4304 bf16 goes from 6698.7 us
  / 42.1 GB/s to **1749.0 us / 161.3 GB/s**, i.e. 61.65x raw against the H100's
  `approximate="tanh"` becomes 17.40x, **15.41x → 4.35x per chip**.

  The cause is `erf` alone and it is isolated by measurement: at 16384 × 4304 bf16 on
  one logical core `erf` costs **6387.1 us at 44.2 GB/s** — 6387 of erf-gelu's 6699 —
  while `tanh` (499.5 us, 564.7 GB/s), `sigmoid` (493.5, 571.6) and `exp` (508.6,
  554.6) all sit on the SFU activation-table roofline. One missing lowering, not a
  bandwidth property. Two results from `vendor_ops/NEURON/tools/probe_gelu_lowering.py`
  worth keeping: **do not hand-write the polynomial** (6496.6 us against 1795.8 for the
  fused call — 3.6x *worse*, because each elementwise step is its own device round trip
  on an eager backend), and the residue is real (161.3 GB/s is still ~2.9x under `silu`
  on the same chip, so a fused NKI gelu is worth that much again).

- `projects/micro_perf/vendor_ops/NEURON/ops/nkilib/topk.py`: a second `topk` provider
  calling `nkilib.core.topk.rotational_topk`, which is **flat in `k`** — 607.6 us at
  k=50 against 605.5 at k=8 over the 248320 vocabulary, where `torch.topk` steps from
  559.8 to 5310.2. `sorted` makes no measurable difference on either side, so the cliff
  is torch's algorithm switch and not the sort; values match `torch.topk` exactly.
  Neither provider wins everywhere and both now run every case: nkilib takes all 12
  `k=50` rows (2.43-8.81x) and loses all 24 `k ≤ 8` rows (0.27-0.99x), and its cost
  grows with `BxS` where torch's is flat in batch. Dispatching to the better of the two
  moves the op from a worst row of 56.9x raw / 14.2x per chip to **11.6x / 2.91x**, and
  the geomean over 36 rows from 1.28x to **0.71x per chip** — from behind the H100 to
  ahead of it. Two notes for use outside the benchmark: the indices carry nkilib's own
  `index_dtype`, not `torch.int64`, so a sampler must cast; and the config takes an
  `nl` dtype, never a numpy one — `np.float32` fails at *lowering* time with `error:
  numpy dtypes are not supported as arguments`, which reads like a shape limitation
  rather than the one-word type bug it is.

- `projects/micro_perf/vendor_ops/NEURON/ops/nkilib/rms_norm.py`: a second `rms_norm`
  provider, and a kernel written here rather than wired up — nkilib's `rmsnorm_tkg`
  produces `[128, BxS, H//128]`, the layout its own sharded-matmul caller wants, and the
  harness declares `dst` as `[T, H]`, so using it would move the cost into a transpose
  inside the timed region.

  Unlike `gelu` there was nothing to fix in the op def: it already calls the single fused
  aten op. The finding is *where* the time goes. At 10240 × 5120 bf16 on one logical
  core, `(x*x).mean(-1)` **alone** costs 934.7 us of the whole op's 1127.3 at 112.2 GB/s,
  while `x*x` alone is 533.3 (393.2 GB/s), `silu` 464.2 (451.8) and a bare `clone()`
  362.1 (579.2) — so 83% of the op is the row reduction, as its own pass over HBM, and
  the multiply is another one after it. No torch-level spelling helps: written out by
  hand it is 1.9x *worse* natively and 3.5x worse with an fp32 reduction, because on an
  eager backend every intermediate is a whole tensor through HBM.

  The kernel loads a 128-row tile into SBUF once and does the square, the row sum, the
  rsqrt and both multiplies on-chip. Two `nisa` instructions carry it:
  `nisa.activation` squares **and** free-axis-reduces in the same pass
  (`reduce_op`/`reduce_res`), and `nisa.scalar_tensor_tensor` applies
  `(x * inv_rms) * gamma` in one pass with the `[p, 1]` broadcast free. Rows go on the
  128 partitions, which keeps the reduction on the cheap axis and both DMAs contiguous.
  Both providers run every case (harness-measured, `ONLY=qwen3_5_27b_norm`):

  | dtype | tokens | `torch` | `nkilib` | GB/s | best H100 | was | now |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | bf16 | 1024 | 92.2 us | 92.2 us | 227.5 | 41.7 us | 2.21x | 2.21x |
  | bf16 | 4096 | 282.5 us | **203.2 us** | 412.8 | 41.2 us | 6.86x | **4.93x** |
  | bf16 | 10240 | 1095.3 us | **475.8 us** | 440.7 | 79.5 us | 13.78x | **5.98x** |
  | fp32 | 1024 | 119.4 us | **103.7 us** | 404.5 | 36.5 us | 3.27x | **2.84x** |
  | fp32 | 4096 | 501.2 us | **308.9 us** | 543.2 | 71.3 us | 7.03x | **4.33x** |
  | fp32 | 10240 | 2178.5 us | **763.0 us** | 549.8 | 150.9 us | 14.44x | **5.06x** |

  440.7 GB/s is 98% of `silu`'s 451.8 on the same core and dtype, i.e. the streaming
  roofline, and 1.50x/1.26x per chip against the 1.16x an HBM-bound op should show. Below
  1024 tokens the kernel loses by a flat 20-30 us of fixed cost; that crossover is left
  visible in the results rather than hidden behind a `vendor_parser` rejection, as
  `topk`'s is.
  Values match aten to rounding (bf16 0.01563 against its 0.01562, fp32 0.00001 against
  0.00000).

  Four things about nki 0.6.0 that each cost a run to find, all recorded in the file:
  **`nl.rms_norm` is unusable** — it hands its own `[p, 1]` rsqrt to
  `nisa.tensor_tensor`, which rejects it (`'dst' free total elements 5120 != 'rhs' free
  total elements 1`), at *lowering* time. **`nl.tile_size` properties cannot be read
  outside a kernel trace**: they resolve the NeuronCore generation through an nki backend
  that only exists while tracing, so reading one in `vendor_parser` fails every case of
  the op with "No backend set. Call `_activate_backend()`…", naming none of this; the
  SBUF bound is now a documented constant. **nki traces the source of every function a
  kernel calls**, so it refuses a `raise` in a helper ("NKI does not support 'raise'
  statements") and refuses a call to a nested `def` ("inner functions can only be used
  as fori_loop/while_loop body arguments") — the tile arithmetic is therefore raise-free
  and the loop body inline. And **the grid is not optional**: `grid=1` gives 815.7 us
  where `grid=2` gives 502.8, 1.62x for a launch subscript, because a logical core is two
  physical halves at `logical-neuroncore-config: 2`.

- `projects/micro_perf/vendor_ops/NEURON/ops/torch_tiled/`: a third `flash_attention`
  provider for the `head_dim > 128` prefill shapes that reach **no** fused path — same
  SDPA, called once per query tile. The motivation is the superquadratic scaling
  recorded above: the score matrix at the worst shape is `24 × 10240 × 10240 × 2 B` =
  **5.03 GB** against 24 GB of usable HBM per core, and tiling the query axis caps that
  at `tile / q_len` of it. Measured: 24/4/256 at `q_len` 10240 goes 282.0 ms → **70.4
  ms** (4.00x), 6/1/256 → **19.0 ms** (3.72x), so the section's worst ratio drops from
  **153.6x to 38.4x** and stops growing with context. Output is numerically identical.

  Gated off below a 1 GiB score matrix, which the measurements bracket tightly: 201 MB
  **regresses 31%**, 805 MB is a wash at 1.00x, 1.17 GB wins 3.72x. `torch` stays the
  implementation for those rows. Splitting `head_dim` instead cannot work at all —
  softmax sits between the two matmuls, so two fused `D=128` calls cannot compose into
  a `D=256` result — and the ceiling is unchanged: the `head_dim 128` control reaches
  61.3 TFLOPS where this reaches 18.3, so a real 256-partition NKI kernel is still
  worth ~3x more. One trap, recorded because it fails silently: `is_causal=True` cannot
  be reused per tile, since PyTorch aligns the implied mask to the **top-left** of a
  non-square score matrix while a query tile against its prefix needs bottom-right.

- `projects/micro_perf/vendor_ops/NEURON/tools/probe_gelu_lowering.py`,
  `probe_attention_head_dim_256.py`, `probe_topk_rotational.py`,
  `probe_rms_norm_lowering.py`: the four measurements
  the fixes above were decided from, kept because each answers a question that recurs.
  Respectively: which formulation of an activation the backend actually lowers well;
  whether a `head_dim` over the partition limit can be rescued by tiling rather than by
  a kernel; and whether a nkilib kernel clears a `torch` cliff. Note the
  attention probe's numbers are 8-10% pessimistic against the provider's, because it
  expands GQA with `repeat_interleave` where the provider passes `enable_gqa`.

### Fixed

- `projects/micro_perf/vendor_ops/NEURON/ops/torch/topk.py`: registering a vendor
  provider for an op **silently removes the `base` one**, and the `topk` provider above
  hit it. `core/op.py:153-155` inserts `base` only for ops with no vendor provider at
  all, so `OP_MAPPING["topk"]` existing at all was enough to stop `torch.topk` being
  measured — the run produced `topk/nkilib/` and no `topk/base/` at all, with no error
  message, quietly dropping the baseline the `k` cliff was diagnosed from. Reading
  `engine.py:128` (which does loop every provider) and `common_utils.py:455-480` (which
  does write one file per provider) suggests the opposite, so this is worth stating
  plainly: **a vendor provider replaces the base implementation, it does not join it.**
  Fixed by registering the inherited implementation under its own name, `torch`, which
  is the convention `flash_attention` already followed with its
  `ops/torch/flash_attention.py` beside `ops/nkilib/flash_attention.py`. Both providers
  now run all 36 cases. Note the path rename: these rows land in `topk/torch/` where
  they used to land in `topk/base/`.

- `projects/micro_perf/vendor_ops/NEURON/ops/torch/rms_norm.py`: the same base-suppression
  trap, for the `rms_norm` provider above, and worth a second entry because it is now
  confirmed to be a property of the registry rather than something specific to `topk`:
  adding `ops/nkilib/rms_norm.py` alone would have stopped `F.rms_norm` being measured at
  all, taking with it the baseline every ratio in that entry is quoted against. Both
  providers now run all 12 cases, and it is also what covers the shapes the kernel
  rejects (a token count that does not split into equal 128-row tiles, a hidden size too
  wide for one partition's SBUF, `add_residual`, `float16`). Same path rename:
  `rms_norm/torch/` where these rows used to be `rms_norm/base/`.

- `projects/micro_perf/vendor_ops/NEURON/ops/nkilib/flash_attention.py`: the prefill
  rejection message told the reader that prefill "is measured by the `torch` provider,
  which reaches a fused NKI kernel there (`attention_cte`)" without qualification. That
  is only true for `head_dim <= 128` — torch_neuronx's SDPA rewrite gate has the same
  128 limit `attention_tkg` does — so at `head_dim 256` the message was promising
  coverage that does not exist, which is exactly the case where someone reads it.
  Now states the head_dim condition and cites the measured consequence (prefill falls
  to 4.6 TFLOPS and gets *worse* with sequence length).

- `projects/micro_perf/op_defs/llm_ops/store_kv_cache.py`: the linear-cache branch
  could never run. It copied a `[q_len, kv_head_num * head_dim]` slice onto a
  `[kv_head_num, q_len, head_dim]` cache slice — `transpose(0, 1)` without first
  splitting the head dimension out — which `copy_` cannot broadcast at **any** shape,
  so every case failed with a size-mismatch `RuntimeError`. The quantised path was
  separately wrong: `static_quant`'s contract is `[num_tokens, hidden_size]` against a
  `[1, hidden_size]` scale, and it was being handed an already-transposed tensor.
  Fixed by quantising while the data is still token-major and splitting the head
  dimension before the transpose. This is the real reason
  `llm/single_test_ops/pre_fa_ops.json` had no runnable `store_kv_cache` case on
  either backend — the int8 cache dtype was only the second reason — and the eight
  rows in `models/qwen3_5_27b/pre_attention_ops.json` are the first that execute.
  Note that this op's *decode* rows still measure the op def's per-sequence Python
  loop (`store_kv_cache.py:260`, and `rotary_embedding.py:152` has the same shape):
  at batch 64 that is 2310 us and a reported 0.2 GB/s, so a cross-backend ratio taken
  from those rows would mostly compare kernel-launch overhead.

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

- Both sweep scripts (`vendor_ops/NEURON/tools/run_full_sweep.sh`,
  `vendor_ops/GPU/tools/run_comparison_sweep.sh`): **`ONLY` now also accepts a prefix
  group** — a token matches every label beginning with it followed by an underscore. So
  the model-shaped set is `ONLY=qwen3_5_27b` rather than eight comma-separated labels
  nobody retypes correctly, and with `RESULTS=` it lands in one tree per backend:

  ```bash
  RESULTS=/tmp/qwen3_5_27b_neuron LOG=/tmp/qwen3_5_27b_neuron.log \
      ONLY=qwen3_5_27b vendor_ops/NEURON/tools/run_full_sweep.sh
  RESULTS=/tmp/qwen3_5_27b_gpu ONLY=qwen3_5_27b \
      vendor_ops/GPU/tools/run_comparison_sweep.sh 2>&1 | tee /tmp/qwen3_5_27b_gpu.log
  ```

  `ONLY=chip4` gets the four four-core labels the same way. Whole-label matching is tried
  first, so a group name cannot shadow a label, and the documented guarantee that
  `ONLY=gemm` does **not** select `single_gemm_ops` still holds — the prefix has to end at
  an underscore. Both scripts keep identical `want()` semantics on purpose: the two sides
  of a comparison have to be selectable the same way. This is deliberately not a new
  script; a second file with its own list of labels is what
  `run_new_workloads.sh` was, and it drifted (see the note at the top of
  `run_full_sweep.sh`).

- `projects/micro_perf/vendor_ops/GPU/README.md`: the `Cross-chip decode, on the
  aligned workload` subsection **moves out of the Summary and into `## Attention`**,
  next to the decode discussion it belongs to. It is a six-case eleven-column table
  about one provider on one op, and sitting third from the top of the file it read as
  if it were a headline. The Summary keeps the decode row and its link, so the anchor
  still resolves from both places. Two claims in the section it moved next to are
  corrected while touching it: the decode bullet said the 1.6-2.7x residual was "the
  narrowest anywhere in this comparison outside the parity rows", which is wrong twice
  over — it straddles the prefill bullet's 2.1x, and `rms_norm`/`layer_norm` sit at
  1.5x — so it now says what actually distinguishes decode, that its residual stops
  growing with problem size.
- `projects/micro_perf/vendor_ops/GPU/README.md`: **the `gather`/`scatter` row now
  says how much its 510x/715x costs an LLM, which is much less than the magnitude
  suggests.** Nothing in `op_defs/llm_ops/` calls `torch.gather` or
  `Tensor.scatter_`: the MoE combine path uses `index_add_` (`moe_gather.py:154`,
  `moe_quant_group_gemm_combine.py:239`, and `llm_ops.md:768` states it outright),
  embedding lookup uses `index_select`/`embedding`, and paged-KV gathering happens
  inside an attention kernel. Those measure 3.8x, 0.98x and 0.97x respectively. The
  row is therefore evidence about how neuronx-cc lowers a broadcast int64 index — a
  class of bug worth knowing — not a cost an inference server pays today; the
  LLM-shaped version of the question is `moe_scatter_dynamic_quant` (0.7 GB/s) and
  `moe_gather` (9.2 GB/s), which still have no GPU column.
- `projects/micro_perf/vendor_ops/GPU/README.md`: **the Summary table is rebuilt —
  grouped by op family, with the two normalisations separated into their own
  columns.** It had accumulated rows in no order (`gemm` at bf16, fp8 and fp32 were
  1st, 8th and 11th), three naming conventions for the same op (`dense bf16 gemm`
  vs `gemm at fp8`), and cells that named nothing at all ("the 7 quantised ops").
  Rows now read `gemm` bf16/fp8/fp32, then both `flash_attention` modes, then the
  24 basic ops split by outcome, then the layer ops, then the quantised ones, each
  naming the ops it covers.

  The reordering surfaced a worse problem: **the ratio column mixed two different
  normalisations under a header that said "per chip".** Five of the twelve rows
  quoted the "Neuron shortfall" figure from the memory-bound and norm tables, which
  is already divided by the 1.16x bandwidth bar (`shortfall = per-chip / 1.1552`
  exactly, since 3.35 TB/s / (725 GB/s x 4) = 1.1552). So `gather`'s headline 449x
  was the software residual and its true per-chip ratio is 510x; `scatter` 621x is
  really 715x; the `gelu` group's 3.5-7.7x is 4.0-8.9x; the quantised ops' 3.8-38x
  is 4.4-44x; the norm/`swiglu` range 0.35-1.26x is 0.40-1.46x. Both columns are now
  present and defined under the table, with the bar named per family (1.48x
  compute-bound bf16, 1.16x bandwidth-bound, dtype-specific for the other `gemm`
  rows), and the residual column is the one to read as "how much is software".
  The published shortfall numbers are kept verbatim in that column so they still
  match their source tables.

  Also: **an `attention decode` row, and the bimodality claim above it corrected.**
  Decode was the one workload with a measured Trainium2 number and no line in the
  table everybody reads first; it is **1.9-3.2x per chip** with the better of the two
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
