"""Is nkilib's `attention_tkg` worth a provider for the decode rows?

`ops/torch/flash_attention.py` measures decode through
`scaled_dot_product_attention`, and SDPA cannot reach torch_neuronx's fused NKI
path there: `_can_use_nki_flash_attention` wants `L % 512 == 0 and S % 512 == 0 and
D <= 128 and B*H <= 512`, and a decode step has `L == 1` and `B*H` in the
thousands. nkilib ships `attention_tkg` for that shape. This script answers, on
device, whether calling it is actually faster -- and the answer turns out to depend
entirely on context length.

Measured on Trn2 (one chip, LNC 2), 80/8/128 GQA, bf16, `kv_bytes / wall time`
where `kv_bytes = 2 * B * kv_h * kv_len * d * 2`:

    B    kv_len  q_len  block_len   SDPA            attention_tkg      ratio
    16     4096      1  flat         950.9 us  282     7693.0 us   35   0.12x
    16     4096      1  128          953.8 us  281     1876.8 us  143   0.51x
    16     4096      1  32           938.0 us  286     1836.5 us  146   0.51x
    16     8192      1  128         3886.4 us  138     2311.2 us  232   1.68x
    16    10240      1  flat        6938.5 us   97    14490.3 us   46   0.48x
    16    10240      1  128         6934.6 us   97     2337.8 us  287   2.97x
    16    10240      1  64          6936.8 us   97     2447.5 us  274   2.83x
    16    10240      1  32          6900.1 us   97     2413.4 us  278   2.86x
    16    10240      1  16          6892.0 us   97     2414.9 us  278   2.85x
    16    16384      1  128        15847.3 us   68     3372.2 us  318   4.70x
    16    10240      4  128        5532.1 us  121     2920.0 us  230   1.89x
    64     4096      1  128        3343.8 us  321     5236.1 us  205   0.64x
    64    10240      1  128       11606.0 us  231     8814.1 us  305   1.32x
    16     4096      1  128 (64 q heads, QK-swap eligible)
                                   833.3 us  322     1701.2 us  158   0.49x

Four things to read out of that:

1. **The two cross between 4096 and 8192.** SDPA's throughput *falls* with context
   length (282 -> 68 GB/s) because the unfused path materialises a score matrix per
   step; `attention_tkg`'s *rises* (143 -> 318 GB/s) as its fixed per-batch cost
   amortises. Neither provider dominates, so both are worth keeping: at kv_len 4096
   SDPA is 2x the faster one.
2. **Block KV is not optional.** A flat `k_prior`/`v_prior` is 4-8x slower than the
   same call with pages, and it is the one configuration that is slower than SDPA
   everywhere.
3. **`block_len` is advisory.** `resize_cache_block_len_for_attention_tkg_kernel`
   overrides it -- at kv_len 10240 all of 128/64/32/16 are reduced to 16 and land
   within 2% of each other. It prints "reducing block length by 8x" when it does.
4. **QK-swap is unreachable at the published head count.** `is_qk_swapped` needs
   `s_active * q_per_group` to satisfy `% 32 == 0 or 32 % it == 0` and
   `128 % it == 0`; 80/8 GQA gives `q_per_group = 10`, so decode gets 10 and fails
   both. Dropping to 64 query heads makes it eligible and changes nothing
   measurable, so the missing fast MM1 path is not what costs the 4096 row.

rel_err against SDPA is 0.0033-0.0065 across every configuration above, i.e. this
is the same attention, not a cheaper one.

`attention_tkg` is not launchable as it stands: it takes a `BufferManager` and a
caller-allocated `out`, so the `@nki.jit` entries here are hand-written. The
constraints they encode were all found by trial:

  * `qk_in_sb=True` is required whenever `fuse_rope=False`, hence the two
    `alloc_stack` + `dma_copy` stages;
  * every `alloc_stack` must be inside `sbm.open_scope()`, the caller's included;
  * block KV needs `strided_mm1=False`, `tp_k_prior=True` and an
    `active_blocks_table` with `table.shape[1] * block_len == curr_sprior`;
  * `curr_sprior % 128 == 0`, and `curr_sprior` counts the active tokens, so it is
    `kv_len` -- which is why the provider ships its own workload file;
  * `q` must be pre-scaled by `1/sqrt(d)`, because `fuse_rope=False` leaves the
    kernel nowhere to apply the softmax scale;
  * GQA folds kv heads into the batch: `bs = B * kv_h`, `q_head = q_h // kv_h`;
  * `active_mask[k, b, h, s] == 1` iff query `s` attends to active key `k`, so
    causal within the active window is `k <= s`;
  * `k_active` must be the *tail* of the cache. The kernel copies it over the last
    `s_active` prior slots in SBUF, so `curr_sprior = kv_len` covers the whole
    cache exactly once.

Run it (Trn2, PyTorch-native image, one Neuron device is enough):

    scp projects/micro_perf/vendor_ops/NEURON/tools/probe_attention_tkg.py Trn2:/tmp/
    ssh Trn2 'sudo docker run --rm --device /dev/neuron0 -v /tmp:/t -w /t \
        xpu-perf-beta4:latest python3 /t/probe_attention_tkg.py'

`ONLY=1,6` picks individual rows of CASES. Each shape pays a one-off neuronx-cc
compile of 7-90 s, printed separately from the timed loop.
"""
import math
import os
import time

import torch
import torch_neuronx  # noqa: F401

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.attention.attention_tkg import attention_tkg
from nkilib.core.attention.attention_tkg_utils import (
    AttnTKGConfig,
    is_batch_sharded,
    is_qk_swapped,
    is_s_prior_sharded,
)
from nkilib.core.utils.allocator import create_auto_alloc_manager

DEV = "neuron"
P_MAX = 128
LNC = 2


def log(*a):
    print(*a, flush=True)


def bench(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def _cfg(bs, q_head, s_active, curr_sprior, d_head, block_len):
    return AttnTKGConfig(
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        curr_sprior=curr_sprior,
        full_sprior=curr_sprior,
        d_head=d_head,
        block_len=block_len,
        tp_k_prior=True,
        # A flat (unpaged) prior needs the strided MM1 layout; block KV forbids it.
        strided_mm1=(block_len == 0),
        use_pos_id=True,
        fuse_rope=False,
        use_gpsimd_sb2sb=True,
        qk_in_sb=True,
        k_out_in_sb=False,
        out_in_sb=False,
        enable_fa_s_prior_tiling=True,
    )


def _stage_qk(sbm, q, k_active, bs, q_head, s_active, d_head):
    """qk_in_sb=True: the kernel wants q and the active K already in SBUF."""
    q_sb = sbm.alloc_stack((d_head, bs * q_head * s_active), dtype=q.dtype,
                           buffer=nl.sbuf, name="q_sb")
    nisa.dma_copy(q_sb, q)
    k_sb = sbm.alloc_stack((d_head, bs * s_active), dtype=k_active.dtype,
                           buffer=nl.sbuf, name="k_active_sb")
    nisa.dma_copy(k_sb, k_active)
    return q_sb, k_sb


@nki.jit
def tkg_flat(q, k_active, v_active, k_prior, v_prior, active_mask, pos_ids,
             bs, q_head, s_active, curr_sprior, d_head):
    """k_prior/v_prior as one contiguous [bs, 1, s_prior, d] run per sequence."""
    out = nl.ndarray((bs, q_head, d_head, s_active), dtype=q.dtype,
                     buffer=nl.shared_hbm)
    sbm = create_auto_alloc_manager()
    sbm.open_scope(name="tkg-entry")
    q_sb, k_sb = _stage_qk(sbm, q, k_active, bs, q_head, s_active, d_head)
    attention_tkg(
        q=q_sb, k_active=k_sb, v_active=v_active,
        k_prior=k_prior, v_prior=v_prior, mask=active_mask, out=out,
        cfg=_cfg(bs, q_head, s_active, curr_sprior, d_head, 0),
        sbm=sbm, rope_pos_ids=pos_ids,
    )
    sbm.close_scope()
    return out


@nki.jit
def tkg_block(q, k_active, v_active, k_prior, v_prior, active_mask, pos_ids,
              blocks_table, bs, q_head, s_active, curr_sprior, d_head,
              block_len):
    """k_prior/v_prior as [bs * n_blocks, block_len, d] pages plus a table."""
    out = nl.ndarray((bs, q_head, d_head, s_active), dtype=q.dtype,
                     buffer=nl.shared_hbm)
    sbm = create_auto_alloc_manager()
    sbm.open_scope(name="tkg-entry")
    q_sb, k_sb = _stage_qk(sbm, q, k_active, bs, q_head, s_active, d_head)
    attention_tkg(
        q=q_sb, k_active=k_sb, v_active=v_active,
        k_prior=k_prior, v_prior=v_prior, mask=active_mask, out=out,
        cfg=_cfg(bs, q_head, s_active, curr_sprior, d_head, block_len),
        sbm=sbm, rope_pos_ids=pos_ids, active_blocks_table=blocks_table,
    )
    sbm.close_scope()
    return out


def run_case(B, HQ, HKV, KV_LEN, D, block_len, S_ACTIVE=1):
    GRP = HQ // HKV
    BF = B * HKV               # the kernel's batch, kv heads folded in
    CACHE = KV_LEN - S_ACTIVE
    SCALE = 1.0 / math.sqrt(D)
    tag = f"block_len={block_len}" if block_len else "flat"
    log(f"\n=== B={B} {HQ}/{HKV}/{D} kv_len={KV_LEN} q_len={S_ACTIVE} -> "
        f"bs_folded={BF} q_head={GRP} | {tag} ===")

    swapped = is_qk_swapped(
        bs=B, q_head=HQ, d_head=D, s_active=S_ACTIVE, curr_sprior=KV_LEN,
        lnc=LNC, p_max=P_MAX, is_block_kv=block_len > 0, is_2byte_kv=True,
        fp8_packed=False, fuse_rope=False, kv_heads=HKV,
    )
    log(f"  batch_sharded={is_batch_sharded(BF, GRP, S_ACTIVE, KV_LEN, P_MAX)} "
        f"sprior_sharded={is_s_prior_sharded(BF, GRP, S_ACTIVE, KV_LEN, P_MAX)} "
        f"qk_swapped={swapped}")

    torch.manual_seed(0)
    q = torch.randn(B * S_ACTIVE, HQ, D, dtype=torch.bfloat16).to(DEV)
    k_cache = torch.randn(B, HKV, KV_LEN, D, dtype=torch.bfloat16).to(DEV)
    v_cache = torch.randn(B, HKV, KV_LEN, D, dtype=torch.bfloat16).to(DEV)

    # Causal over the whole cache: query s sees absolute positions [0, CACHE+s].
    # `is_causal=True` would align to the top-left of the non-square score matrix
    # and attend to the wrong keys, which is why the SDPA provider refuses q_len>1.
    if S_ACTIVE > 1:
        j = torch.arange(KV_LEN).view(1, -1)
        i = torch.arange(S_ACTIVE).view(-1, 1)
        sdpa_mask = (j <= (CACHE + i)).view(1, 1, S_ACTIVE, KV_LEN).to(DEV)
    else:
        sdpa_mask = None

    def sdpa():
        """Exactly what ops/torch/flash_attention.py runs."""
        qs = q.view(B, S_ACTIVE, HQ, D).transpose(1, 2)
        return torch.nn.functional.scaled_dot_product_attention(
            qs, k_cache, v_cache, attn_mask=sdpa_mask, scale=SCALE,
            is_causal=False, enable_gqa=True,
        ).transpose(1, 2).reshape(B * S_ACTIVE, HQ, D)

    t_sdpa = bench(sdpa)
    ref = sdpa().float().cpu()
    kv_bytes = 2 * B * HKV * KV_LEN * D * 2
    log(f"  SDPA               {t_sdpa:10.1f} us  {kv_bytes / t_sdpa / 1e3:7.1f} GB/s")

    # q: [d, bs*q_head*s_active], free axis b*q_head*s_active + h*s_active + s.
    q_tkg = ((q * SCALE).view(B, S_ACTIVE, HKV, GRP, D).permute(4, 0, 2, 3, 1)
             .reshape(D, BF * GRP * S_ACTIVE).contiguous())
    k_active = (k_cache[:, :, CACHE:, :].reshape(BF * S_ACTIVE, D).t().contiguous())
    v_active = (v_cache[:, :, CACHE:, :].reshape(BF, 1, S_ACTIVE, D).contiguous())

    ka = torch.arange(S_ACTIVE).view(-1, 1)
    qa = torch.arange(S_ACTIVE).view(1, -1)
    am = (ka <= qa).to(torch.uint8).view(S_ACTIVE, 1, 1, S_ACTIVE)
    active_mask = am.expand(S_ACTIVE, BF, GRP, S_ACTIVE).contiguous().to(DEV)
    # The prior mask is generated in-kernel as `iota < pos_ids`, so this is the
    # cache length rather than an absolute position.
    pos_ids = torch.full((BF, S_ACTIVE), CACHE, dtype=torch.float32).to(DEV)

    if block_len:
        nblk = KV_LEN // block_len
        k_prior = k_cache.reshape(BF * nblk, block_len, D)
        v_prior = v_cache.reshape(BF * nblk, block_len, D)
        table = torch.arange(BF * nblk, dtype=torch.int32).reshape(BF, nblk).to(DEV)
        kern = tkg_block[LNC]

        def tkg():
            return kern(q_tkg, k_active, v_active, k_prior, v_prior,
                        active_mask, pos_ids, table, BF, GRP, S_ACTIVE,
                        KV_LEN, D, block_len)
    else:
        k_prior = k_cache.view(BF, 1, KV_LEN, D)
        v_prior = v_cache.view(BF, 1, KV_LEN, D)
        kern = tkg_flat[LNC]

        def tkg():
            return kern(q_tkg, k_active, v_active, k_prior, v_prior,
                        active_mask, pos_ids, BF, GRP, S_ACTIVE, KV_LEN, D)

    t0 = time.perf_counter()
    out = tkg()
    torch.neuron.synchronize()
    log(f"  compile            {(time.perf_counter() - t0) * 1e3:10.1f} ms")

    got = (out.float().cpu().reshape(B, HKV, GRP, D, S_ACTIVE)
           .permute(0, 4, 1, 2, 3).reshape(B * S_ACTIVE, HQ, D))
    rel = (got - ref).abs().max().item() / ref.abs().max().item()

    t_tkg = bench(tkg)
    log(f"  attention_tkg[{LNC}]   {t_tkg:10.1f} us  "
        f"{kv_bytes / t_tkg / 1e3:7.1f} GB/s"
        f"   {t_sdpa / t_tkg:5.2f}x vs SDPA   rel_err {rel:.3g}")


CASES = [
    # (B, HQ, HKV, KV_LEN, D, block_len[, q_len]); block_len 0 == flat prior
    (16, 80, 8, 4096, 128, 0),
    (16, 80, 8, 4096, 128, 128),
    (16, 80, 8, 4096, 128, 32),
    (16, 64, 8, 4096, 128, 128),    # q_per_group = 8 -> QK-swap eligible
    (16, 64, 8, 4096, 128, 0),
    (64, 80, 8, 4096, 128, 128),
    (16, 80, 8, 10240, 128, 128),
    (16, 80, 8, 10240, 128, 0),
    (16, 80, 8, 8192, 128, 128),
    (16, 80, 8, 16384, 128, 128),
    (16, 80, 8, 10240, 128, 128, 4),   # speculative decode; SDPA refuses this row
    (64, 80, 8, 10240, 128, 128),
    (16, 80, 8, 10240, 128, 64),       # block_len sweep: the kernel overrides all
    (16, 80, 8, 10240, 128, 32),       # of these down to 16
    (16, 80, 8, 10240, 128, 16),
]

if __name__ == "__main__":
    only = os.environ.get("ONLY")
    cases = [CASES[int(i)] for i in only.split(",")] if only else CASES
    for c in cases:
        try:
            run_case(*c)
        except Exception as e:
            log(f"  FAILED: {type(e).__name__}: {str(e)[:600]}")
