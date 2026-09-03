# Qwen3.5-27B shaped workloads

Every other workload directory in this tree sweeps powers of two. This one sweeps the
shapes one real model actually executes, so that a per-op number can be attached to a
per-model claim without an interpolation step in between.

All shapes below are derived from `Qwen/Qwen3.5-27B`'s `config.json`. Nothing here is
a new op: every key is an op already registered in `op_defs/`, so these files run
today on both backends. What the model needs and the harness *cannot* express yet is
listed at the end, and that list is the more interesting half of this directory.

Run them with the sweep scripts, which have a gated label per file:

```bash
# GPU
LIST=1 vendor_ops/GPU/tools/run_comparison_sweep.sh | grep qwen
ONLY=qwen3_5_27b_gemm vendor_ops/GPU/tools/run_comparison_sweep.sh

# Neuron
ONLY=qwen3_5_27b_gemm,qwen3_5_27b_attention vendor_ops/NEURON/tools/run_full_sweep.sh
```

or one file at a time through `launch.py --workload workloads/models/qwen3_5_27b/<f>.json`.

## Why this model is not a dense transformer

The parity question these files exist to answer ("a sub-40B bf16 dense model should
run about the same on one Trn2 chip as on one H100") assumes a stack of identical
attention+MLP blocks. Qwen3.5-27B is not that:

| config.json | value | consequence for these files |
| --- | --- | --- |
| `num_hidden_layers` | 64 | but only 16 of them are attention |
| `layer_types` / `full_attention_interval: 4` | 16 `full_attention`, 48 `linear_attention` | three quarters of the layers are gated-delta-net, not attention — `deltanet_ops.json` |
| `hidden_size` | 5120 | every `hidden_size` and every GEMM K below |
| `intermediate_size` | 17408 | not a power of two; `gemm.json`'s `K.N` pairs never touch it |
| `num_attention_heads` / `num_key_value_heads` | 24 / 4 | GQA 6:1; TP=4 is the maximum clean TP |
| `head_dim` | **256** | above the Neuron flash-attention kernels' `MAX_HEAD_DIM` of 128; the H100's SDPA path takes it (measured below) |
| `attn_output_gate` | true | q projection is 2x wide: 24×256×2 = 12288, not 6144 |
| `partial_rotary_factor` | 0.25 | rotary covers 64 of the 256 dims — `rope_dim: 64` |
| `vocab_size`, `tie_word_embeddings` | 248320, false | a 2.54 GB embedding *and* a separate 2.54 GB lm_head |
| `linear_num_key_heads` … | 16×128 K, 48×128 V, conv kernel 4 | the GDN projection widths below |
| `mamba_ssm_dtype` | float32 | the bf16→fp32 boundary in `deltanet_ops.json` |
| `mtp_num_hidden_layers` | 1 | one speculative draft token → `q_len: 4` decode case |
| vision tower | 27 layers, hidden 1152, inter 4304, 16 heads, patch 16 | head_dim 72 and `gelu` at 4304 |

For scale: MLP 17.11B params, linear-attention 5.56B, full-attention 1.68B,
embed+head 2.54B, vision 0.42B — 27.31B total, 54.6 GB of bf16 weights.

## TP=1 and TP=4 conventions

Each file carries both. TP=1 is the shape as written in the config; TP=4 is the same
shape after a 4-way tensor-parallel split, which is the configuration a single Trn2
chip actually runs (4 logical cores, and `num_key_value_heads: 4` makes 4 the largest
TP that needs no KV-head replication). 24 / 4 / 16 / 48 / 17408 / 248320 all divide by
4 cleanly, so the TP=4 row is a plain division with no padding:

| | TP=1 | TP=4 |
| --- | --- | --- |
| q heads / kv heads | 24 / 4 | 6 / 1 |
| GDN value heads | 48 | 12 |
| MLP intermediate | 17408 | 4352 |
| vocab (lm_head N) | 248320 | 62080 |

Reading a TP=4 row as "one core's share" and a TP=1 row as "the whole chip doing it
alone" is the intended comparison; neither is more real than the other.

## What is in each file

### `gemm_ops.json` — 114 cases

Nine distinct `K.N` pairs at TP=1 and the same nine at TP=4, each across
`num_tokens` 1 / 16 / 64 / 1024 / 4096 / 10240 (batch-1 decode, small-batch decode,
large-batch decode, and three prefill lengths). The mapping:

| `hidden_size.new_hidden_size` (TP=1) | layer | note |
| --- | --- | --- |
| 5120 → 17408 | MLP gate_proj and up_proj | ×2 per layer, ×64 layers — the single biggest FLOP consumer |
| 17408 → 5120 | MLP down_proj | |
| 5120 → 12288 | attention q_proj | 2x wide because `attn_output_gate` |
| 5120 → 1024 | attention k_proj, v_proj | 4 kv heads × 256 |
| 6144 → 5120 | attention o_proj *and* GDN out_proj | same shape, two different layer types |
| 5120 → 10240 | GDN in_proj_qkv | 16×128 K + 48×128 V + q |
| 5120 → 6144 | GDN in_proj_z | the gate branch |
| 5120 → 96 | GDN in_proj_ba | tiny-N; see the runtime hazard below |
| 5120 → 248320 | lm_head | the largest single GEMM in the model |

Plus a `[128, 128]` tile at `num_tokens` 16384 / 65536 / 262144 in bf16 and fp32, as
a proxy for the delta-rule chunk matmuls: the GDN scan is a serial chain of 128×128
products, and its cost is set by how well a backend does at that tile size when the
chain cannot be batched away.

Note that `gemm.json`'s four `K.N` pairs and this model's nine have an **empty
intersection**. That is the concrete reason this directory exists.

### `attention_ops.json` — 32 cases

`flash_attention` at `[q_head_num, kv_head_num, head_dim]` of `[24, 4, 256]` (TP=1)
and `[6, 1, 256]` (TP=4):

* prefill at `q_len` 4096 and 10240;
* decode at batch 16 / cache 4096, batch 16 / cache 10240, batch 64 / cache 4096;
* decode at `q_len: 4` — the MTP speculative-decode step, one draft token verified
  alongside the real one. Expect the SDPA-based providers to reject this shape on
  both backends; that rejection is itself the result.

Vision attention at `[16, 16, 72]` / `[4, 4, 72]`, `q_len` 1024 and 4096.

Also 16 `softmax` cases in the op's `llm` arg_type, which is the 4-D attention softmax
`[batch, head_num, q_seq_len, kv_seq_len]` rather than anything hidden-state shaped.
These exist as an unfused-attention proxy: on a backend where `head_dim 256` misses the
fused kernel, the softmax is the part of the fallback that a fused kernel would have
kept in on-chip memory, so its standalone cost is a floor on what the fallback pays.

Two caveats to read the output with:

* **`head_dim: 256` is the point of these cases, and whether it falls off the fast path
  is backend-specific.** On the H100 it does not — 82% of the peak a power-of-two
  head_dim reaches (see the measured section). On Trainium2 the kernels cap head_dim at
  128 (`MAX_HEAD_DIM`) and torch_neuronx's NKI SDPA rewrite gate also requires
  `D <= 128`, so the 16 attention layers are expected to fall back to torch there. If
  that holds, the published attention comparison (prefill 3.1x, decode 1.9-3.2x) does
  not describe these layers and the real gap is wider, not narrower.
* **`is_causal=True` is hardcoded** in `op_defs/llm_ops/flash_attention.py`, and
  vision attention is not causal. The vision rows are therefore an upper bound on a
  cheaper op, not a faithful measurement. Fixing that needs an op-def change.

### `norm_ops.json` — 90 cases

`rms_norm` and `add_rms_norm` at `hidden_size: 5120` (bf16 and fp32 for the former,
because the residual stream is bf16 but the reduction may not be); `qk_rms_norm` at
the two head sets; `head_rms_norm` at `[48, 128, 0, 48]` / `[12, 128, 0, 12]` for
GDN's gated output norm and `[16, 128, 0, 16]` / `[4, 128, 0, 4]` for its q/k
normalisation. `num_tokens` follows the same six-point ladder as the GEMMs.

### `activation_ops.json` — 42 cases

`swiglu` at `hidden_size` 17408 / 4352 — note the op def's `hidden_size` is the
*output* width, so the input tensor is `[num_tokens, 2 × hidden_size]`, which is
already how the fused gate+up projection lands. `silu` at the GDN gate and conv
widths (6144 / 1536 / 10240 / 2560), and `gelu` at the vision intermediate
(4304 / 1076) — the config says `gelu_pytorch_tanh`, and the op def's plain `gelu`
is the closest registered form.

### `pre_attention_ops.json` — 24 cases

`rotary_embedding` with `rope_dim: 64` — the real partial rotary — and `rope_dim: 256`
as a full-rotation control, so the file measures whether a backend actually saves
anything on the partial case or pads it back to full width. Both head sets, prefill
and decode.

`store_kv_cache` at `block_size: 0`, the linear cache. The published `pre_fa_ops.json`
has *no* runnable `store_kv_cache` case on any backend; fixing the layout bug that
caused that (see the measured section) makes these the first rows that execute. The
paged form is not here because the op def raises `NotImplementedError` for it. This op
is in-place, so read its bandwidth number with that in mind — and read only its
prefill rows, since the decode rows measure the op def's per-sequence Python loop.

### `sampling_ops.json` — 54 cases

`softmax` and `topk` over the 248320-entry vocabulary (and 62080 at TP=4), at
`k` 1 / 8 / 50 for greedy, small beam and typical top-k sampling; `embedding` from
the 248320-row table into the six token counts. The vocabulary is 2.5x the 98304 of
the models the existing files assume, and softmax/topk over it is a fixed per-step
cost that does not shrink with batch. `softmax` here is the `default` arg_type — a
2-D `[num_tokens, vocab]` reduction; the `llm` arg_type of the same op is the 4-D
attention softmax and lives in `attention_ops.json`.

### `deltanet_ops.json` — 30 cases

The gated-delta-net path decomposed into the registered ops it is built from:
`cast` bf16→fp32 at the 6144 / 1536 gate widths (the `mamba_ssm_dtype: float32`
boundary), `exp` on the per-head decay at 48 / 12 heads, `mul` for the gating,
`reduce_sum` at dim 128 for the L2 normalisation, and `index_select` at
`dim_size: 786432` for reading a sequence's recurrent state page (155 MB per sequence
in fp32, constant in context length — that number is why the state read is worth a
row of its own).

This is a decomposition, not the op. See the omissions below.

### `ccl_ops.json` — 6 cases

`all_reduce` at `world_size: 4`, `hidden_size: 5120`, the same six token counts.
This is the exact message the TP=4 decode step sends: 128 all-reduces per step
(2 per layer × 64), 160 KiB each at batch 16. For that traffic to cost under 10% of a
36.5 ms step, each one has to finish in under 28.5 us — which is the number to check
the output against. Needs 4 devices; it does nothing on a one-GPU box.

## Measured: H100 80GB HBM3, 2026-09-03

348 of 384 cases ran, in 4 min 45 s for all seven single-device files. The 36 that did
not are accounted for at the end of this section. Trainium2 numbers are not here yet —
the machine was busy — so nothing below is a comparison, only the GPU column.

**`head_dim: 256` is not a problem on the H100.** This was the open question, and the
answer is that the torch SDPA provider takes it and reaches 82% of the peak a
power-of-two head_dim reaches:

| shape | latency | TFLOPS | MFU |
| --- | --- | --- | --- |
| prefill, 24/4/256, q_len 10240 | 1836 us | 701.9 | 0.709 |
| prefill, 24/4/256, q_len 4096 | 364 us | 566.2 | 0.572 |
| prefill, 6/1/256 (TP=4), q_len 10240 | 564 us | 571.7 | 0.578 |
| decode, 24/4/256, B=64, cache 4096 | 399 us | 16.2 | 0.016 — 2698 GB/s |
| decode, 24/4/256, B=16, cache 10240 | 267 us | 15.1 | 0.015 — 2518 GB/s |

Decode's low MFU is the expected memory-bound result: 2698 GB/s is 92% of the
~2948 GB/s this card actually achieves. So on this side of the comparison the model's
unusual head_dim costs nothing, which sharpens rather than softens the question for
Trainium2, where `MAX_HEAD_DIM` is 128 and these 16 layers are expected to fall back
to torch.

**GEMM peaks at 81.9% MFU** (810.6 TFLOPS, `17408→5120` at 10240 tokens), and the MLP
and lm_head shapes all sit at 726-810 TFLOPS once there are ≥1024 tokens — the
non-power-of-two `17408` costs nothing. Two shapes do not:

| K → N | 4096 tokens | 10240 tokens | what it is |
| --- | --- | --- | --- |
| 5120 → 17408 | 798.5 | 781.0 | MLP gate/up |
| 17408 → 5120 | 805.6 | 810.6 | MLP down |
| 5120 → 248320 | 755.9 | 726.7 | lm_head |
| 5120 → 1024 | 695.2 | 731.5 | k/v_proj |
| **5120 → 96** | **66.9** | **207.7** | GDN in_proj_ba |
| **5120 → 24** | **11.7** | **46.4** | same, TP=4 |
| **128 → 128** | — | 133.1 @ 262144 | delta-rule scan tile |

The tiny-N projections run at 1-21% of peak, and the 128×128 scan tile reaches only
13.5% MFU even at 262144 rows. Both are real model shapes, and they are the concrete
form of the point that the GDN path is utilisation-bound rather than FLOP-bound: its
1.7% share of prefill FLOPs is executed at roughly a sixth of the efficiency the MLP
gets. At batch 1 the weight GEMMs are cleanly bandwidth-bound as expected
(`5120→17408` 67.9 us at 2626 GB/s, lm_head 833 us at 3053 GB/s).

**Partial rotary is worth 2.8-3.6x in prefill.** `rope_dim: 64` against the
`rope_dim: 256` control: 426 us vs 1522 us at 24 heads / 10240 tokens. So the backend
really does skip the untouched 192 dims rather than padding back to full width.

**Two op defs measure their own Python loop, not the chip.** `store_kv_cache` and
`rotary_embedding` both iterate over the batch in Python
(`store_kv_cache.py:260`, `rotary_embedding.py:152`), so their decode rows are
launch-bound: `store_kv_cache` at B=64 takes 2310 us and reports 0.2 GB/s, and
`rotary_embedding` decode lands at 1.6-6.9 ms, where partial rotary appears *slower*
than full because the loop dominates the signal entirely. Their prefill rows (B=1, one
iteration) are meaningful — `store_kv_cache` 160 us at 525 GB/s — but do not read a
cross-backend ratio off the decode rows of these two ops; it would mostly compare
kernel-launch overhead.

**Other rows worth keeping:**

* `swiglu` peaks at 1041 GB/s while `silu` reaches 2952 on the same card — the fused
  activation is 2.8x off the elementwise roofline, which is the opposite of what the
  name suggests.
* `topk` over the 248320 vocabulary costs 90-300 us and is **almost independent of
  `k`** (1, 8 and 50 differ by less than the run-to-run spread). At batch 1 that is
  129 us moving 7.7 GB/s — a fixed per-step cost comparable to a whole decode step's
  attention, and it does not amortise until the batch is large.
* Vocabulary `softmax` in bf16 is *slower* than in fp32 (1589 vs 2842 GB/s at the
  attention-shaped case), consistent with an internal upcast.
* `qk_rms_norm` and `head_rms_norm` peak at 1052 and 1188 GB/s — roughly 40% of what
  `rms_norm` gets through the vllm/flashinfer providers (2780 GB/s). These are
  head-sliced partial-width norms with no vendor provider, so `base` is what runs.
* The GDN state page read (`index_select`, `dim_size` 786432) is 275 GB/s for one
  sequence and 2726 GB/s for 64 — it needs a batch to reach the roofline.

**The 36 cases that did not run:**

* 16 `store_kv_cache` cases failed before the fix in this same change. The linear path
  copied `[q_len, kv_head_num * head_dim]` onto a `[kv_head_num, q_len, head_dim]`
  cache slice, which cannot broadcast at *any* shape, and the quantised path applied a
  `[1, kv*head_dim]` scale to an already-transposed tensor. Both are fixed in
  `op_defs/llm_ops/store_kv_cache.py`, and the 8 remaining cases here are the first
  runnable `store_kv_cache` rows on either backend. This is also the real reason the
  published `pre_fa_ops.json` had no runnable case — not only its int8 cache dtype.
* 8 paged (`block_size: 512`) cases: `StoreKVCacheOp paged cache not implemented yet`
  is an explicit `raise` in the op def. Removed from the file and listed as an
  omission below rather than left to error on every run.
* 12 `softmax` cases: this file originally passed `arg_type: "llm"` with
  `hidden_size`, but `softmax`'s `llm` form is the *attention* softmax
  (`[batch, head_num, q_seq_len, kv_seq_len]`), not a hidden-state form. The
  vocabulary softmax is `arg_type: "default"`; both are now present, the attention
  form in `attention_ops.json` where it belongs, as an unfused-attention proxy for the
  backend on which `head_dim 256` misses the kernel.
* 2 `flash_attention` MTP cases (`q_len: 4`) rejected with "SDPA attention decode only
  supports q_len == 1; a multi-token decode step needs a bottom-right aligned causal
  mask." That is the predicted result and the file keeps them deliberately.

## Runtime hazard: `max_data_cnt` on the GPU backend

`core/backend.py` sizes its buffer rotation as `1 GiB / tensor_size` with **no upper
bound** on the GPU backend (`backend_neuron.py` caps it at 4). A small tensor
therefore allocates and shuffles tens of thousands of buffers to run 2-10 iterations.
26 of the cases here are small enough to land above 400 buffers; the worst are
`swiglu` at `num_tokens: 1` (~62k) and `gemm` at `N: 24` (~4k). These are real model
shapes — batch-1 decode and the tiny `in_proj_ba` — so they are kept rather than
trimmed.

In the event this did **not** bite: all seven single-device files finished in 4 min
45 s on the H100, so the worry was misplaced at this scale — the pathology that made
one case take 474 s needed a tensor two orders of magnitude smaller again. Recorded
here anyway because it is the first thing to suspect if a case appears to hang, and
because the Neuron side allocates differently.

## What is missing, and why it is not in these files

The model executes these, and no op def exists, so none of them can be added as
JSON. Listed in rough order of how much they matter:

1. **A gated-delta-net / delta-rule op.** The largest gap. `deltanet_ops.json`
   measures the pieces; it cannot measure the chunked scan itself, whose cost is
   dominated by a serial dependency and a matrix inverse rather than by any of the
   elementwise ops around it. At ~0.83 GFLOP/token it is only 1.7% of prefill FLOPs
   but it is latency-bound, so the FLOP share understates its time share, and 48 of
   64 layers are made of it. Adding an op def needs two decisions: whether the
   `(I - A)^-1` solve lives inside the op, and whether the chunk size is a parameter.
2. **`conv1d`.** Twice over: the depthwise causal convolution with
   `linear_conv_kernel_dim: 4` in every GDN layer, and the vision patch embedding's
   strided conv. The vllm-neuron port keeps `F.conv1d` behind a default-off flag and
   ships four hand-rolled shifted slices plus a multiply-add instead, which is
   direct evidence that this shape's cost is worth measuring.
3. **`cumsum`.** The same port replaced `torch.cumsum` with `g @ (cols >= rows)` —
   a matmul standing in for a scan. Whether that trade is right is exactly the kind
   of question a micro-benchmark answers.
4. **Triangular inverse / `eye` / triangular solve.** The port hand-rolls
   `_strictly_lower_inverse` as blocked elimination rather than calling any standard
   solve, for the same reason.
5. **`softplus`, `tanh`.** The GDN decay path. `sigmoid` and `rsqrt` are also absent
   as standalone ops but are at least exercised inside `silu` and `rms_norm`.
6. **Batched matmul (`bmm`).** Every attention-shaped product with a batch dimension
   that is not going through the fused kernel — which, given `head_dim: 256`, is all
   16 attention layers.
7. **`where`.** Masking non-finite KV entries; on Neuron `v * mask` is not a
   substitute, since `inf * 0` is NaN.
8. **`index_copy_`, `pad`, `cat` / `stack` / `chunk`.** Cheap individually, but the
   GDN state update and the conv-state ring buffer are made of them.
9. **A paged `store_kv_cache`.** Not a missing op but a missing path:
   `StoreKVCacheOp` raises `NotImplementedError` for `cache_type == "paged"`, so only
   the linear cache can be measured, and a real server pages. The linear branch it
   would be modelled on is now correct (see above), which is the prerequisite.

Separately, the seven `moe_*` ops are **not** applicable to this model: it is dense,
`mlp_only_layers` is empty, and there are no experts. They matter for Qwen3.5-35B-A3B,
not here.

Three of the eight items above (conv1d, cumsum, triangular inverse) are exactly the
three places the working vllm-neuron port had to write a workaround. That coincidence
is the strongest available argument that these gaps are real rather than cosmetic.
