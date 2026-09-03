"""Decode attention through nkilib's token-generation kernel, `attention_tkg`.

Why a second provider rather than a change to `ops/torch/flash_attention.py`:

`torch_neuronx` does fuse `scaled_dot_product_attention` into a NKI flash kernel,
but only when `L % 512 == 0 and S % 512 == 0 and D <= 128 and B*H <= 512`
(`_can_use_nki_flash_attention`). A decode step has `L == q_len == 1` and
`B*H` in the thousands, so it fails two of those conditions and can never reach
the fused path -- SDPA falls back to an unfused score-matrix decode. nkilib ships
`attention_tkg` for exactly this shape, and on the long-context rows it is worth a
lot:

    B=16 80/8/128 GQA, bf16, one logical core pair, kv bytes / wall time

      kv_len   SDPA        attention_tkg   ratio
        4096   281 GB/s      143 GB/s      0.51x
        8192   138 GB/s      232 GB/s      1.68x
       10240    97 GB/s      287 GB/s      2.97x
       16384    68 GB/s      318 GB/s      4.70x

SDPA's throughput *falls* with context length while `attention_tkg`'s rises, so
the two cross somewhere between 4096 and 8192 and both providers are worth
keeping: at kv_len 4096 SDPA is still the faster one. `attention_tkg` also serves
`q_len > 1` (speculative decode), which the SDPA provider turns away because
`is_causal=True` in PyTorch aligns the mask to the top-left of a non-square score
matrix; here the active mask is built explicitly, and the kv_len 10240 / q_len 4
row measures 1.89x. Numbers from tools/probe_attention_tkg.py.

`attention_tkg` is not a launchable kernel -- it is an internal helper that takes a
`BufferManager` and a caller-allocated `out`, so the `@nki.jit` entry point below
is ours. Everything in it was pinned down by trial on device; the constraints that
are not obvious from the signature:

* `qk_in_sb=True` is mandatory whenever `fuse_rope=False` (`[NCC_INKI016]
  Currently only support skipping fusing RoPE when QK is in SBUF`), which is why
  q and k_active are staged into SBUF here instead of being handed over in HBM.
* Every `alloc_stack` -- including the two below -- has to sit inside
  `sbm.open_scope()` (`[SBM] Cannot allocate in stack without an open scope`).
* Block KV additionally requires `strided_mm1=False`, `tp_k_prior=True`, and an
  `active_blocks_table` whose row length times `block_len` equals `curr_sprior`.
* `curr_sprior % 128 == 0` is asserted inside the kernel. `curr_sprior` counts the
  active tokens, so it is `kv_len`, and the published `fa_linear_ops.json` decode
  rows do not qualify: `cache_len` 4096 with `q_len` 1 is a kv_len of 4097.
  workloads/fa_decode_tkg.json restates them one token shorter -- `cache_len` 4095
  -- so that kv_len lands on 4096. Both providers run that file, so the comparison
  stays exact; against the published table it is a 0.02% shorter context.
* GQA is expressed by folding the kv heads into the batch: the kernel's `bs` is
  `batch_size * kv_head_num` and its `q_head` is the group size. For a linear
  cache `[B, kv_h, max_kv_len, d]` that fold is a free view.
* `block_len` barely matters, because `resize_cache_block_len_for_attention_tkg_kernel`
  overrides it: at kv_len 10240 every one of 128/64/32/16 is reduced to 16 and all
  four land within 2% of each other. 128 is passed because it is both the fastest
  measured and a realistic page size.
"""
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.backends.NEURON.backend_neuron import (
    RUNTIME_EAGER,
    detect_neuron_runtime,
)

# Must match __init__.py, which is where load_plugin_package() reads the provider
# name from. Declared again rather than imported: the plugin loader builds the
# package with spec_from_file_location and no submodule_search_locations.
PROVIDER_NAME = "nkilib"

# nl.tile_size.pmax. Both the s_prior assert inside the kernel and the head_dim
# limit are expressed against it.
P_MAX = 128

# Page size handed to the kernel; see the module docstring on why it is advisory.
BLOCK_LEN = 128

try:
    if detect_neuron_runtime() != RUNTIME_EAGER:
        raise ImportError("nkilib attention_tkg requires the eager runtime")

    import nki
    import nki.isa as nisa
    import nki.language as nl
    from nkilib.core.attention.attention_tkg import attention_tkg
    from nkilib.core.attention.attention_tkg_utils import AttnTKGConfig
    from nkilib.core.utils.allocator import create_auto_alloc_manager
    from torch_neuronx.neuron_dynamo_backend import decompositions as _dc

    # The kernel's __getitem__ wants the bare int (`replace(self, lnc=lnc)`), not
    # the `grid=(lnc,)` tuple torch_neuronx hands its own wrapper -- a tuple raises
    # `NkiValidationError: NKI only supports LNC 1 or 2`. Omitting the subscript
    # entirely is worse than wrong-looking: it runs on one half of the LNC2 pair
    # and silently costs ~1.85x, which is the trap tools/probe_attention_kernel.py
    # documents for attention_cte.
    LNC = int(_dc.get_logical_neuron_cores())

    @nki.jit
    def tkg_block_kv(
        q, k_active, v_active, k_prior, v_prior, active_mask, pos_ids,
        blocks_table, bs, q_head, s_active, curr_sprior, d_head, block_len,
    ):
        """Entry point for nkilib's attention_tkg over a block KV cache.

        Shapes, all with `qk_in_sb=True` / `out_in_sb=False` / `tp_k_prior=True`:

            q            [d_head, bs * q_head * s_active]  pre-scaled by 1/sqrt(d)
            k_active     [d_head, bs * s_active]
            v_active     [bs, 1, s_active, d_head]
            k/v_prior    [bs * n_blocks, block_len, d_head]
            active_mask  [s_active, bs, q_head, s_active]  uint8, 1 == attend
            pos_ids      [bs, s_active]                    float32 cache length
            blocks_table [bs, n_blocks]                    int32
            out          [bs, q_head, d_head, s_active]

        q is pre-scaled on the host because `fuse_rope=False` leaves the kernel no
        place to apply `softmax_scale`.
        """
        out = nl.ndarray(
            (bs, q_head, d_head, s_active), dtype=q.dtype, buffer=nl.shared_hbm
        )
        sbm = create_auto_alloc_manager()
        sbm.open_scope(name="xpu-perf-tkg")

        q_sb = sbm.alloc_stack(
            (d_head, bs * q_head * s_active), dtype=q.dtype, buffer=nl.sbuf,
            name="q_sb",
        )
        nisa.dma_copy(q_sb, q)
        k_sb = sbm.alloc_stack(
            (d_head, bs * s_active), dtype=k_active.dtype, buffer=nl.sbuf,
            name="k_active_sb",
        )
        nisa.dma_copy(k_sb, k_active)

        attention_tkg(
            q=q_sb,
            k_active=k_sb,
            v_active=v_active,
            k_prior=k_prior,
            v_prior=v_prior,
            mask=active_mask,
            out=out,
            cfg=AttnTKGConfig(
                bs=bs,
                q_head=q_head,
                s_active=s_active,
                curr_sprior=curr_sprior,
                full_sprior=curr_sprior,
                d_head=d_head,
                block_len=block_len,
                tp_k_prior=True,
                strided_mm1=False,
                use_pos_id=True,
                fuse_rope=False,
                use_gpsimd_sb2sb=True,
                qk_in_sb=True,
                k_out_in_sb=False,
                out_in_sb=False,
                enable_fa_s_prior_tiling=True,
            ),
            sbm=sbm,
            rope_pos_ids=pos_ids,
            active_blocks_table=blocks_table,
        )
        sbm.close_scope()
        return out

    @ProviderRegistry.register_vendor_impl("flash_attention", PROVIDER_NAME)
    class NkiLibTkgFlashAttentionOp:
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            if self.attn_mode != "decode":
                # The `torch` provider reaches attention_cte for prefill only when
                # torch_neuronx's SDPA rewrite gate fires, and that gate also needs
                # head_dim <= 128. Above it there is no fused prefill path on this
                # backend at all -- measured at head_dim 256, prefill falls to
                # 4.6 TFLOPS and gets *worse* with sequence length -- so do not read
                # this message as a promise that prefill is covered.
                raise ValueError(
                    "attention_tkg is the token-generation kernel; prefill is "
                    "measured by the `torch` provider, which reaches a fused NKI "
                    f"kernel there (attention_cte) only for head_dim <= {P_MAX}; "
                    f"this case has head_dim {self.head_dim}."
                )

            if self.cache_type != "linear":
                raise ValueError(
                    "this provider reinterprets the linear cache as pages, so it "
                    f"needs cache_type=linear, not {self.cache_type}. A real paged "
                    "cache would also need the workload's block_table remapped "
                    "onto the kv-head fold, which is bookkeeping this measurement "
                    "does not model."
                )

            if not (
                self.dtype == "bfloat16"
                and self.dst_dtype == "bfloat16"
                and self.cache_dtype == "bfloat16"
                and self.qk_compute_dtype == "bfloat16"
                and self.pv_compute_dtype == "bfloat16"
            ):
                raise ValueError(
                    "attention_tkg is wired here for an all-bfloat16 dtype set; "
                    "its fp8 KV path needs a packed cache layout the op def does "
                    "not produce."
                )

            if self.q_head_num % self.kv_head_num != 0:
                raise ValueError(
                    f"q_head_num {self.q_head_num} must be a multiple of "
                    f"kv_head_num {self.kv_head_num}: GQA is expressed by folding "
                    "kv heads into the batch."
                )

            if self.head_dim > P_MAX:
                raise ValueError(
                    f"head_dim {self.head_dim} exceeds the {P_MAX} partitions the "
                    "kernel puts it on."
                )

            # kv_len is the kernel's curr_sprior (the active tokens live in the
            # last s_active slots of it), and `atp.s_prior % TC.p_max == 0` is
            # asserted inside the kernel.
            kv_len = self.kv_lens[0]
            if kv_len % P_MAX != 0:
                raise ValueError(
                    f"kv_len {kv_len} (= cache_len + q_len) must be a multiple of "
                    f"{P_MAX}; use workloads/fa_decode_tkg.json, which restates "
                    "the published decode rows with an aligned kv_len."
                )
            if self.max_kv_len % BLOCK_LEN != 0:
                raise ValueError(
                    f"max_kv_len {self.max_kv_len} must be a multiple of "
                    f"block_len {BLOCK_LEN} for the cache to be a free block view."
                )
            if kv_len != self.max_kv_len:
                raise ValueError(
                    f"kv_len {kv_len} must equal max_kv_len {self.max_kv_len}: the "
                    "block table covers the whole allocated cache."
                )

        def vendor_impl(self):
            super().vendor_impl()

            self.q_len = self.q_lens[0]
            self.kv_len = self.kv_lens[0]
            self.cache_len = self.cache_lens[0]
            self.group_size = self.q_head_num // self.kv_head_num
            # The kernel's batch: kv heads folded in.
            self.bs_folded = self.batch_size * self.kv_head_num
            self.n_blocks = self.kv_len // BLOCK_LEN

            device = self.backend.get_torch_device_name()

            # Built once, outside the timed region, because none of it depends on
            # the input tensors -- in a serving stack the page table and the mask
            # come from the scheduler, not from the attention kernel.
            #
            # active_mask[k, b, h, s] == 1 iff query s attends to active key k, so
            # causal within the active window is `k <= s`. The prior region is
            # masked in-kernel from pos_ids (`iota < pos_ids`), which is why
            # pos_ids carries cache_len rather than an absolute position.
            k_idx = torch.arange(self.q_len).view(-1, 1)
            s_idx = torch.arange(self.q_len).view(1, -1)
            causal = (k_idx <= s_idx).to(torch.uint8).view(
                self.q_len, 1, 1, self.q_len
            )
            self.active_mask = (
                causal.expand(self.q_len, self.bs_folded, self.group_size, self.q_len)
                .contiguous()
                .to(device)
            )
            self.pos_ids = torch.full(
                (self.bs_folded, self.q_len), float(self.cache_len),
                dtype=torch.float32,
            ).to(device)
            # Identity paging: block i of folded batch b is row b*n_blocks + i,
            # which is what reshaping the contiguous linear cache produces.
            self.blocks_table = (
                torch.arange(self.bs_folded * self.n_blocks, dtype=torch.int32)
                .reshape(self.bs_folded, self.n_blocks)
                .to(device)
            )

            self._run_func = self.tkg_run

        def tkg_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            b, s, hkv, g, d = (
                self.batch_size, self.q_len, self.kv_head_num,
                self.group_size, self.head_dim,
            )
            bf = self.bs_folded

            # (num_tokens, q_head_num, head_dim) -> [d, bs*q_head*s_active], free
            # axis indexed b*q_head*s_active + h*s_active + s. This is a transpose
            # of q only -- 320 KB at the published decode shapes against 268 MB of
            # cache traffic -- so it stays in the timed region rather than being
            # hoisted, which would measure a layout the op def does not hand over.
            q_tkg = (
                (q * self.softmax_scale)
                .view(b, s, hkv, g, d)
                .permute(4, 0, 2, 3, 1)
                .reshape(d, bf * g * s)
            )

            # curr_sprior includes the active tokens, and the kernel overwrites the
            # last s_active prior slots with k_active/v_active, so these have to be
            # the tail of the cache for the result to cover the whole of it.
            # `.t()` alone would hand the kernel a non-contiguous tensor; nki reads
            # the storage, not the strides.
            k_active = (
                k_cache[:, :, self.cache_len:, :].reshape(bf * s, d).t().contiguous()
            )
            v_active = v_cache[:, :, self.cache_len:, :].reshape(bf, 1, s, d)

            # Free views: the cache is contiguous and max_kv_len == n_blocks*block_len.
            k_prior = k_cache.reshape(bf * self.n_blocks, BLOCK_LEN, d)
            v_prior = v_cache.reshape(bf * self.n_blocks, BLOCK_LEN, d)

            out = tkg_block_kv[LNC](
                q_tkg, k_active, v_active, k_prior, v_prior,
                self.active_mask, self.pos_ids, self.blocks_table,
                bf, g, s, self.kv_len, d, BLOCK_LEN,
            )

            # [bs, q_head, d, s_active] -> the packed layout op_defs declares.
            return (
                out.reshape(b, hkv, g, d, s)
                .permute(0, 4, 1, 2, 3)
                .reshape(self.num_tokens, self.q_head_num, d)
            )

except Exception:
    pass
