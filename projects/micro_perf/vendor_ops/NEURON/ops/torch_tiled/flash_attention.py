"""Query-axis tiled prefill, for the `head_dim > 128` shapes that have no fused kernel.

Both of Trainium2's fused attention paths are gated on `head_dim <= 128`: nkilib puts
head_dim on the 128 partitions of `P_MAX`, and torch_neuronx's SDPA rewrite requires
`D <= 128` (`_can_use_nki_flash_attention`, see ../torch/flash_attention.py). A model
with `head_dim 256` -- Qwen3.5-27B -- misses both, and SDPA falls back to its math
decomposition. Measured cost of that on one Trn2 logical core, bf16, against an H100:

    24/4/256 prefill      q_len 4096     13.7 ms     H100  364 us
    24/4/256 prefill      q_len 10240   282.6 ms     H100 1836 us

Splitting head_dim in two does not help, because softmax sits between the two matmuls:
`QK^T` can be accumulated as `Q1@K1^T + Q2@K2^T`, but the softmax needs the whole score
row before `P@V` can start, so two fused `D=128` calls cannot compose into a `D=256`
result.

What does help is tiling the query axis, which needs no kernel. The clue is that the
cost grows 20.6x for a 2.5x length increase (2.5^3.4) where O(n^2) traffic predicts
6.25x -- something is scaling worse than the algorithm, and the candidate is the score
matrix itself: `24 x 10240 x 10240 x 2 B` = **5.03 GB** against 24 GB of usable HBM per
core, plus a causal mask of the same shape. Tiling caps both at `tile / q_len` of that.

Measured through the harness on `workloads/models/qwen3_5_27b/attention_ops.json`, this
provider against the `torch` one, and the resulting ratio to an H100:

    shape       q_len    torch      tiled    gain   TFLOPS    H100      was      now
    24/4/256    10240  282043 us   70443 us  4.00x    18.3  1836 us  153.6x    38.4x
     6/1/256    10240   70514 us   18975 us  3.72x    17.0   564 us  125.1x    33.7x
    24/4/256     4096   13689 us   gated off   --      --    364 us   37.6x       --
     6/1/256     4096    3529 us   gated off   --      --    161 us   21.9x       --

Run-to-run spread on the two tiled rows is about 2% (a second launch gave 4.03x and
3.81x).

The 4096 rows are why the gate below is a second constant rather than the same one.
Measured with the gate forced open: a score matrix of 201 MB (6/1/256 at 4096)
**regresses 31%** when tiled, and one of 805 MB (24/4/256 at 4096) is a **wash at
1.00x** -- 13636 us against 13689 -- while 1.17 GB and 4.8 GB win 3.7x and 4.0x. So the
crossover is above 805 MB, and `MIN_TOTAL_SCORE_BYTES` is set at 1 GiB, between the wash
and the smallest win. The tile *size* is a different question -- how large a chunk still
streams efficiently -- and `TILE_SCORE_BYTES` answers it at 512 MiB, which picks 1024 for
the 24-head row and 2048 for the 6-head row, both the best or within 3% of the best of
the 512/1024/2048 columns `tools/probe_attention_head_dim_256.py` swept.

Note that probe's numbers are 8-10% pessimistic against the table above because it
expands GQA with `repeat_interleave` where this provider passes `enable_gqa`. That is
also the whole story of the 24/4/256 4096 row: `enable_gqa` alone takes the untiled call
from 14986 to 13699 us, which is most of the 1.20x the probe credited to tiling.

Output is numerically identical to SDPA -- 0.01562 max abs error against an
explicitly-masked fp32 reference at 6/1/256 q_len 2048, the same figure SDPA itself gets,
i.e. bf16 rounding and nothing else.

How far this gets: the `head_dim 128` control on the same shapes reaches 42.6 and 61.3
TFLOPS, where tiled `head_dim 256` reaches 17.0 and 18.3. So tiling recovers ~4x of a
~22x gap and a genuine 256-partition NKI kernel is worth roughly 3x more on top. This is
the part that can be had without writing one.

Two deliberate conservatisms, both of which make the reported number a lower bound:

* the per-tile mask is built inside the timed region, once per call, exactly as the
  probe did. A model would build it once and reuse it across all 64 layers.
* `enable_gqa` is kept rather than `repeat_interleave`, so k/v are not materialised at
  q_head_num.
"""
from xpu_perf.micro_perf.core.op import ProviderRegistry

from xpu_perf.micro_perf.backends.NEURON.backend_neuron import (
    RUNTIME_EAGER,
    detect_neuron_runtime,
)

# How much score matrix one tile may hold. Set by measurement, not by a hardware
# quantity: at this budget the tile rule reproduces the best column of the swept
# 512/1024/2048 on every shape measured.
TILE_SCORE_BYTES = 512 << 20

# Below this, do not tile at all. Separate from the above because it answers a
# different question -- whether tiling pays, not how big a chunk to use. 201 MB
# regresses 31%, 805 MB is a wash at 1.00x, 1.17 GB wins 3.81x, so the crossover
# is above 805 MB and this sits between the wash and the smallest win.
MIN_TOTAL_SCORE_BYTES = 1 << 30

# Below 512 a tile stops amortising its own launch and mask; above 2048 the score
# matrix is large enough again that the measured columns flatten out.
MIN_TILE = 512
MAX_TILE = 2048

try:
    if detect_neuron_runtime() != RUNTIME_EAGER:
        raise ImportError("tiled SDPA attention requires the eager runtime")

    import torch

    def _pick_tile(score_bytes_per_row: int, q_len: int) -> int:
        """Largest power-of-two tile whose scores fit the budget, clamped."""
        tile = TILE_SCORE_BYTES // max(score_bytes_per_row, 1)
        tile = max(MIN_TILE, min(MAX_TILE, tile))
        tile = 1 << (tile.bit_length() - 1)
        return min(tile, q_len)

    @ProviderRegistry.register_vendor_impl("flash_attention", "torch_tiled")
    class NeuronTiledSDPAFlashAttentionOp:
        """SDPA once per query tile, so the score matrix never lands whole in HBM.

        Rejections, all reported as unsupported rather than failing later. Each one
        is a case where this provider would either be wrong or would only reproduce
        the `torch` provider's number under a second name:

        * `head_dim <= 128` -- the fused NKI rewrite is reachable, and tiling it
          would break it up for nothing.
        * decode, or any `q_len == 1` -- there is no query axis to tile.
        * `is_causal` false -- without a causal prefix every tile needs the whole
          of k/v, so the score matrix shrinks but the work does not, and the
          correctness argument below does not apply either.
        * a score matrix under `MIN_TOTAL_SCORE_BYTES` -- measured to be a wash at
          best and to regress by up to 31%.
        * a paged cache and a non-bfloat16 dtype set, for the same reasons
          ../torch/flash_attention.py rejects them: both put a copy inside the
          timed region.

        The correctness point that is easy to get wrong: `is_causal=True` cannot be
        reused per tile. PyTorch aligns the implied mask to the **top-left** of a
        non-square score matrix, so a query tile against its whole prefix -- which
        needs bottom-right alignment -- would silently attend to the wrong keys.
        The mask is therefore built explicitly, as `ki <= qi` with `qi` offset by
        the tile start.
        """

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            if self.cache_type != "linear":
                raise ValueError(
                    "tiled SDPA attention only supports a linear cache, not "
                    f"cache_type={self.cache_type}."
                )

            if not (
                self.dtype == "bfloat16"
                and self.dst_dtype == "bfloat16"
                and self.cache_dtype == "bfloat16"
                and self.qk_compute_dtype == "bfloat16"
                and self.pv_compute_dtype == "bfloat16"
            ):
                raise ValueError(
                    "tiled SDPA attention only supports an all-bfloat16 dtype set."
                )

            if self.head_dim <= 128:
                raise ValueError(
                    f"head_dim {self.head_dim} <= 128 reaches the fused NKI rewrite; "
                    "use the `torch` provider, which is 2.5-3.6x faster there than "
                    "anything this tiling can do."
                )

            if self.attn_mode != "prefill":
                raise ValueError(
                    f"tiled SDPA attention is prefill-only, not attn_mode="
                    f"{self.attn_mode}; a decode step has no query axis to tile."
                )

            if self.cache_lens[0] != 0:
                raise ValueError(
                    "tiled SDPA attention prefill only supports cache_len == 0, got "
                    f"{self.cache_lens[0]}."
                )

            if not self.is_causal:
                raise ValueError(
                    "tiled SDPA attention needs is_causal; without a causal prefix "
                    "every tile reads the whole of k/v and the work is unchanged."
                )

            q_len = self.q_lens[0]
            if q_len <= MIN_TILE:
                raise ValueError(
                    f"q_len {q_len} is not longer than one tile ({MIN_TILE})."
                )

            # bf16 scores, whole matrix, all heads and batches at once.
            self.score_bytes_per_row = self.batch_size * self.q_head_num * \
                self.kv_lens[0] * 2
            total = self.score_bytes_per_row * q_len
            if total < MIN_TOTAL_SCORE_BYTES:
                raise ValueError(
                    f"the whole score matrix is {total / (1 << 20):.0f} MiB, under the "
                    f"{MIN_TOTAL_SCORE_BYTES >> 20} MiB above which tiling has been "
                    "measured to pay -- below it tiling is a wash at best (1.00x at "
                    "805 MiB) and 31% slower at worst (201 MiB). The `torch` provider "
                    "is the implementation for this shape."
                )

        def vendor_impl(self):
            super().vendor_impl()

            self.q_len = self.q_lens[0]
            self.kv_len = self.kv_lens[0]
            self.tile = _pick_tile(self.score_bytes_per_row, self.q_len)

            self.sdpa_kwargs = {"scale": self.softmax_scale}
            if self.q_head_num != self.kv_head_num:
                self.sdpa_kwargs["enable_gqa"] = True

            self._run_func = self.tiled_sdpa_run

        def tiled_sdpa_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            # (num_tokens, q_head_num, head_dim) -> (bs, heads, seq_q, head_dim)
            q_sdpa = q.view(
                self.batch_size, self.q_len, self.q_head_num, self.head_dim
            ).transpose(1, 2)

            outs = []
            for t0 in range(0, self.q_len, self.tile):
                t1 = min(t0 + self.tile, self.q_len)

                # The causal prefix for this tile ends at t1, so the caches are
                # sliced there -- that slice, not the mask, is what keeps the score
                # matrix small on the early tiles.
                k_sdpa = k_cache[:, :, :t1, :]
                v_sdpa = v_cache[:, :, :t1, :]

                # Explicit, bottom-right aligned: everything before t0 is
                # unconditionally visible, and the last (t1 - t0) columns are causal.
                qi = torch.arange(t0, t1, device=q.device).view(-1, 1)
                ki = torch.arange(t1, device=q.device).view(1, -1)

                outs.append(
                    torch.nn.functional.scaled_dot_product_attention(
                        q_sdpa[:, :, t0:t1, :],
                        k_sdpa,
                        v_sdpa,
                        attn_mask=ki <= qi,
                        **self.sdpa_kwargs,
                    )
                )

            out = torch.cat(outs, dim=2)

            # Back to the packed layout op_defs declares for "out".
            return out.transpose(1, 2).reshape(
                self.num_tokens, self.q_head_num, self.head_dim
            )

except Exception:
    pass
