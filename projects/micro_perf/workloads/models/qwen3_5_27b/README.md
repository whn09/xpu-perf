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
| `head_dim` | **256** | above every Neuron flash kernel's 128-partition limit, so no fused attention path exists there (measured: up to 154x); the H100's SDPA takes it for free |
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

* **`head_dim: 256` is the point of these cases, and it is where the two backends
  diverge most.** On the H100 it costs nothing — 82% of the peak a power-of-two head_dim
  reaches. On Trainium2 it misses *every* fused path (`nkilib`'s `P_MAX = 128` and
  torch_neuronx's SDPA rewrite gate both cap at 128), and the measured result is 37.6x
  at `q_len` 4096 rising to 153.9x at 10240. The published attention comparison
  (prefill 3.1x, decode 1.9-3.2x) therefore does **not** describe these 16 layers. See
  the Trainium2 section.
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
not are accounted for at the end of this section. The Trainium2 column, and the
comparison, are in the section after this one.

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

## Measured: Trainium2 vs H100, 2026-09-03

`trn2.3xlarge` (one Trn2 chip, PyTorch-native Beta 4, eager runtime) against the H100
above. 384 cases overlap; the 6 `all_reduce` cases are Trainium-only, since the GPU box
has one card. All eight files finished in 41 min (attention 393 s, gemm 967 s,
pre_attention 456 s, norm 294 s, activation 164 s, deltanet 106 s, ccl 41 s).

### How to read a ratio here

The harness runs one **logical core**, and a Trn2 chip has four. Its own
`peak_tflops` field says so: 166.75 TFLOPS per logical core, one quarter of the chip's
667. The same division applies to bandwidth. So a raw Trn2/H100 latency ratio has to be
divided by 4 before it can be compared against the 989.4/667 = **1.48x** ratio of the
two chips' bf16 peaks:

| raw per-core ratio | per-chip meaning |
| --- | --- |
| ~4-6x | parity — this is the expected band, not a finding |
| ~1.5-4x | Trainium2 is **ahead** per chip |
| >8x | a real gap; something is off the fast path |

Every number below is per-core-raw with the per-chip reading given alongside. The band
is what makes the outliers legible: most of this model is in it, and the three things
that are not are the whole result.

### Verdict per op

| op | raw ratio (median) | per chip | reading |
| --- | --- | --- | --- |
| `gemm`, dense bf16 | 4.42x | ~1.1x | **parity**, Trn2 slightly ahead |
| `head_rms_norm` | 1.63x | 0.41x | Trn2 ahead |
| `rms_norm` (small) / `qk_rms_norm` | 2.2x | 0.56x | Trn2 ahead — but see the caveat |
| `reduce_sum`, `embedding`, `cast` | 2.4-3.2x | 0.6-0.8x | Trn2 ahead |
| `swiglu`, `add_rms_norm` | 3.3-3.7x | ~0.9x | parity |
| `silu`, `mul`, `exp` | 4.7-6.1x | 1.2-1.5x | parity, at the bandwidth ratio |
| `topk`, `k <= 8` | 3.1-6.2x | 0.8-1.6x | parity |
| `softmax` (vocab) | 6.7x | 1.7x | slightly behind |
| `flash_attention`, `head_dim 256` | **9.7x median, 154x max** | up to **38x** | **broken** |
| `gelu` | **52x median, 61.7x max** | **13-15x** | **broken** |
| `topk`, `k = 50` | **28-57x** | **7-14x** | **cliff above k=8** |

`store_kv_cache` and `rotary_embedding` decode rows are excluded from that table; they
measure the op defs' per-sequence Python loop on both backends (`store_kv_cache` at
B=64 is 224 ms on Trn2 against 2.3 ms on the H100, which is a launch-overhead ratio,
not a chip result). Their prefill rows are kept and discussed below.

### GEMM is not the problem — Trainium2 wins it on MFU

Trn2 reaches **152.7 TFLOPS on one logical core, 91.6% MFU** against its 166.75 peak.
The H100's best on the same file is 810.6 TFLOPS, **81.9% MFU**. On the arithmetic the
model is actually built out of, this chip is the more efficient of the two, and the
non-power-of-two `17408` costs it nothing. Two shapes fall off:

| K → N | tokens | Trn2 TFLOPS | H100 TFLOPS | raw | per chip |
| --- | --- | --- | --- | --- | --- |
| 5120 → 17408 (MLP gate/up) | 10240 | 148.9 | 781.0 | 5.25x | 1.31x |
| 17408 → 5120 (MLP down) | 10240 | 152.7 | 810.6 | 5.31x | 1.33x |
| **5120 → 248320 (lm_head)** | 1024 | 147.0 | 664.5 | 4.52x | 1.13x |
| **5120 → 248320 (lm_head)** | 4096 | 94.2 | 755.9 | 8.03x | 2.01x |
| **5120 → 248320 (lm_head)** | 10240 | 71.0 | 726.7 | 10.3x | **2.56x** |
| 128 → 128 bf16 (scan tile) | 262144 | 31.8 | 133.1 | 4.19x | 1.05x |
| 128 → 128 **fp32** (scan tile) | 262144 | 15.8 | 33.2 | 2.10x | **0.53x** |

The lm_head **degrades as tokens rise** — 147 → 94 → 71 TFLOPS where the H100 holds
~750 — so a large-batch prefill pays 2.6x per chip on the single biggest GEMM in the
model. N=248320 at TP=1 is a 2.54 GB weight against 24 GB of usable HBM per core; the
shape is legal but clearly not tiled for. At TP=4 (N=62080) it behaves.

The fp32 128×128 scan tile is the good news the other way: **Trainium2 is ~1.9x faster
per chip**, and `mamba_ssm_dtype: float32` means that is the dtype 48 of the 64 layers
actually run their scan in.

Tiny-N is a non-issue: at ≤1024 tokens `5120 → 96` and `5120 → 24` are 0.75-0.8x raw,
i.e. Trn2 is **absolutely faster**, and only reach 4-5x at 10240 tokens.

### `head_dim: 256` has no fused attention path on Neuron at all

This is the finding that overturns the parity expectation for this specific model.

| shape | Trn2 | H100 | raw ratio |
| --- | --- | --- | --- |
| prefill 24/4/256, q_len 4096 | 13.7 ms, 15.0 TFLOPS | 364 us, 566.2 | **37.6x** |
| prefill 24/4/256, q_len 10240 | 282.6 ms, 4.6 TFLOPS | 1836 us, 701.9 | **153.9x** |
| prefill 6/1/256 (TP=4), q_len 4096 | — | — | 21.9x |
| prefill 6/1/256 (TP=4), q_len 10240 | — | — | 125.3x |
| decode 24/4/256, various | — | — | 4.6-10.4x |
| vision prefill 16/16/72 | — | — | 4.0-9.6x |

Two things make this qualitatively different from a slow kernel. Neuron's throughput
**falls** with sequence length, 15.0 → 4.6 TFLOPS, while the H100's rises 566 → 702:
cost is growing faster than the O(n²) the algorithm implies, which is the signature of
a materialised score matrix rather than a tiled one. And the ratio therefore *grows*
with context — 37.6x at 4096, 153.9x at 10240 — so it cannot be extrapolated from a
short-sequence measurement.

The provider log says why, from three directions at once:

* `nkilib`'s `attention_tkg` rejects every decode case — "head_dim 256 exceeds the 128
  partitions the kernel puts it on" (`P_MAX = 128`);
* `attention_tkg` is token-generation only, so it rejects all prefill by design;
* `torch_neuronx`'s NKI SDPA rewrite gate (`_can_use_nki_flash_attention`) also
  requires `D <= 128`, so the `torch` provider's fallback is an unfused score matrix.

The union of those three conditions is empty for this model. `attention_tkg` — the
kernel that *wins* the published decode row by 4.09x — is unreachable at head_dim 256,
and the vision tower's head_dim 72 lands in the normal 4-9.6x band precisely because it
is under the limit. The H100's SDPA takes 256 with no penalty (82% of the peak a
power-of-two head_dim reaches), so the entire gap is on the Neuron side, and it is a
kernel-coverage gap rather than a silicon one.

Consequence for the model: these are 16 of 64 layers and ~1.68B of 27.31B params, but
at 10240-token prefill they would dominate total time outright.

### `gelu` is pinned at ~40 GB/s — a software fault, not bandwidth

| shape | Trn2 | H100 | raw |
| --- | --- | --- | --- |
| 16384 × 4304 | 6698.9 us, **42.1 GB/s** | 108.5 us, 2599.6 GB/s | 61.7x |
| 4096 × 4304 | 1660.8 us, **42.5 GB/s** | 32.0 us, 2206.3 GB/s | 52.0x |
| 16384 × 1076 | 1826.9 us, **38.6 GB/s** | 31.9 us, 2208.9 GB/s | 57.2x |

Trn2's `gelu` is flat at **31-42 GB/s across every size** — it does not scale at all.
The control is `silu` on the same chip, same dtype, comparable shapes, also the `base`
provider: **475-489 GB/s**, which is ~1.9 TB/s per chip against the H100's 2.9 and
lands exactly on the expected 1.45x chip ratio. So the elementwise roofline is healthy
and `gelu` specifically is **12-15x below it**. Nothing about the hardware explains a
12x spread between two activations that read and write the same bytes; this is a
missing or scalar-decomposed `erf`/`tanh` path.

It is confined to the 27-layer vision tower (`gelu_pytorch_tanh` at intermediate 4304),
so it does not touch text-only decode — but it makes the vision encoder the second
thing to fix, and it is likely the cheapest of the three to fix.

### `topk` falls off a cliff above `k = 8`

| dim | B | k=1 | k=8 | k=50 |
| --- | --- | --- | --- | --- |
| 248320 | 1 | 557.0 us | 558.3 us | **5312.1 us** |
| 248320 | 64 | 621.3 us | 593.9 us | **5367.9 us** |
| 62080 | 1 | 164.9 us | 165.1 us | **1368.2 us** |
| 62080 | 64 | 191.3 us | 169.7 us | **1361.0 us** |

A 9.5x step between k=8 and k=50, at every vocabulary size and batch, while the H100 is
**flat in k** (90-193 us throughout, as the GPU section already noted). At k ≤ 8 Trn2
is 1.4-6.2x raw, i.e. at or ahead of parity per chip; at k=50 it is 28-57x. There is
evidently a small-k path and a general sort above it. Typical top-k sampling uses
k=50, so production sampling hits the slow side: 5.3 ms per decode step at TP=1, still
1.4 ms at TP=4, against a ~36.5 ms step budget.

### `rms_norm`'s 14x is mostly a provider mismatch, but not entirely

At 10240 tokens `rms_norm` is 13.8x (bf16) and 14.4x (fp32) raw — far outside the band.
Provider labels explain most of it: the H100 runs **`flashinfer`/`vllm`** fused kernels
at 2638-2780 GB/s, Trn2 runs **`base`** torch. That is not an apples-to-apples
comparison of silicon, and the honest statement is that Neuron has no fused RMSNorm
provider registered in this harness.

The residue is still real, though: Trn2's `base` rms_norm gets 190.9 GB/s at 10240
tokens and 297-331 GB/s at 4096 — **declining with size, and 2.5x under `silu`'s
475 GB/s on the same chip**. So even against its own elementwise roofline there is
about 2.5x available here. `add_rms_norm`, `qk_rms_norm` and `head_rms_norm` run `base`
on *both* sides and sit at 1.6-3.7x raw, i.e. Trainium2 ahead per chip — which is the
cleaner comparison and the reason to read the 14x as software, not hardware.

### Partial rotary saves the H100 3.6x and Trainium2 almost nothing

`rope_dim: 64` against the `rope_dim: 256` control, prefill only (B=1, so no loop
artifact):

| shape | Trn2 rope64 | Trn2 rope256 | Trn2 saving | H100 saving |
| --- | --- | --- | --- | --- |
| 24 heads, 10240 tokens | 4876.0 us | 5905.6 us | **1.21x** | **3.57x** |
| 24 heads, 4096 tokens | 2177.4 us | 2554.7 us | 1.17x | ~2.8x |
| 6 heads, 10240 tokens | 1315.7 us | 1810.4 us | 1.38x | — |

`partial_rotary_factor: 0.25` should make this nearly 4x cheaper. The H100 collects
most of that; Neuron collects 1.2-1.4x, so it is doing close to the full-width rotation
either way. Worth ~3.5 ms per 10240-token prefill at TP=1 — small next to attention,
but it is the same class of problem and cheap to check.

### `all_reduce` at TP=4 is 4.1x over the decode budget

Trainium-only, `world_size: 4`, `hidden_size: 5120`:

| tokens | bytes | latency | bus_bw |
| --- | --- | --- | --- |
| 1 | 10 KiB | 105.1 us | — |
| 16 | 160 KiB | **118.1 us** | — |
| 64 | 640 KiB | 113.7 us | — |
| 1024 | 10 MB | 199.6 us | — |
| 4096 | 40 MB | 654.8 us | — |
| 10240 | 100 MB | 1476.1 us | 106.6 GB/s |

The interesting row is the one the `ccl_ops.json` section names in advance: the TP=4
decode step sends **160 KiB, 128 times per step** (2 per layer × 64), and for that to
stay under 10% of a 36.5 ms step each has to finish in **28.5 us**. Measured: 118.1 us,
**4.1x over**. 128 × 118.1 us = 15.1 ms, i.e. **41% of the step spent in collectives**.

Latency is flat at 105-118 us from 10 KiB to 640 KiB, so this is fixed overhead, not
bandwidth — bandwidth only becomes the limit past ~10 MB, where 106.6 GB/s is
respectable. The implication is that TP=4 within one chip is not free for this model,
and the alternative (TP=1 per core, 54.6 GB of weights against 24 GB per core) does not
fit. That tension is worth its own measurement.

### Summary against the parity question

The claim this directory was built to test — "a sub-40B bf16 dense model should run
about the same on one Trn2 chip as on one H100" — holds for the *arithmetic* and fails
for this *model*:

* **Dense GEMM: confirmed, emphatically.** 91.6% MFU per logical core beats the H100's
  81.9%. The MLP, which is 17.11B of 27.31B params, is at parity or slightly ahead per
  chip. The fp32 scan tile is ~1.9x ahead.
* **Elementwise and norm: confirmed.** `silu` lands on the 1.45x chip ratio; several
  norms are ahead per chip.
* **But three specific software gaps dominate the model's real runtime**, none of them
  a silicon property: no fused attention at `head_dim 256` (up to 154x, growing with
  context), `gelu` 12-15x under its own chip's roofline, and `topk` above k=8.
  Collectives at TP=4 add a fourth.

So parity is a statement about the chip, not a prediction about this checkpoint. All
four items are fixable in kernels; none requires different hardware. The head_dim-256
limit is the one that would need real work — a 256-partition flash kernel, or a
two-pass split over the partition axis.

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
