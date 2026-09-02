import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry


@ProviderRegistry.register_vendor_impl("flash_attention", "torch")
class GPUSDPAFlashAttentionOp:
    """flash_attention via torch.nn.functional.scaled_dot_product_attention.

    This exists alongside the `fa2` / `fa3` providers rather than replacing them,
    for two reasons.

    The first is coverage. `fa2` accepts prefill only at `batch_size == 1` with a
    linear cache, and decode only with a *paged* cache. So of the ten cases in
    `workloads/llm/single_test_ops/fa_linear_ops.json` it measures four and turns
    away six -- every decode case, because that file's cache is linear, and both
    `batch_size: 4` prefills. This provider's envelope is the union that file
    actually needs.

    The second is that it is the only attention comparison against Neuron that is
    like-for-like: its reference numbers come from a provider that is this same
    code against the same op def. Comparing those to `flash_attn` would be
    comparing two different algorithms as well as two chips.

    Both sides are fused, though not symmetrically, and the asymmetry is in the
    Neuron numbers rather than in this file. `torch_neuronx` lowers SDPA to a NKI
    flash kernel inside its dynamo backend (`_can_use_nki_flash_attention` in
    `neuron_dynamo_backend/decompositions.py`, on by default via
    `TORCH_NEURONX_ENABLE_NKI_SDPA`) whenever `L % 512 == 0 and S % 512 == 0 and
    D <= 128 and B*H <= 512` with no attn_bias and no dropout. Every prefill case
    in `fa_linear_ops.json` satisfies that -- setting the variable to 0 takes the
    80/8/128 `q_len` 4096 case from 9,443 us to 60,500 us, 6.41x -- and no decode
    case can, since `q_len == 1` never satisfies `L % 512 == 0` and `B*H` is
    1280/5120 there. So Neuron prefill is a fused-kernel number and Neuron decode
    is not; the earlier claim in this docstring that the eager runtime had no fused
    kernel at all was wrong. What is true is the narrower statement it rested on:
    `neuronxcc.nki.kernels.attention.flash_fwd` is HLO-traced and only loads under
    torch_xla, so that particular kernel is unreachable there.

    That is *not* the same as saying this is a slow fallback on CUDA. SDPA
    dispatches to a fused backend -- FlashAttention or cuDNN, both of which do
    the online-softmax tiling -- and never materialises the score matrix. The
    measured numbers bear that out: prefill lands at 61-69% MFU on an H100, which
    an unfused implementation could not reach.

    `targets["kernels"]` does *not* identify which backend ran, despite the GPU
    backend profiling with `ProfilerActivity.CUDA`. It comes back empty here,
    because `core_perf` drops any kernel whose launch count differs from
    `prefer_iterations`, and a fused SDPA call launches more than one kernel per
    iteration. Use `torch.nn.attention.sdpa_kernel` or a standalone profile if
    the backend identity matters; do not read the empty list as "no kernel ran".
    Since the report cannot name the backend, `vendor_impl` pins it instead --
    see the comment there, and treat a ~2x jump in prefill with decode unchanged
    as the signature of that pinning having been defeated.

    Rejections match the NEURON provider exactly, so neither side reports a case
    the other skipped:

    * a paged cache -- gathering `block_table` rows into a contiguous k/v would be
      a copy inside the timed region, which is the cost the paged layout exists to
      avoid. `fa_ops.json` is the paged file and `fa2` is its provider;
    * anything but an all-bfloat16 dtype set;
    * chunked prefill (`cache_len > 0` with `q_len > 1`) and speculative decode
      (`q_len > 1`) -- both need the causal mask aligned to the bottom-right of a
      non-square score matrix, and `is_causal=True` aligns to the top-left.
      Passing it anyway does not fail, it silently attends to the wrong keys.

    GQA goes through `enable_gqa=True`, which broadcasts the kv heads inside the
    kernel. `repeat_interleave` on k and v would instead materialise a
    `q_head_num`-sized cache on every call, and that copy would land inside the
    timed region and dominate a decode step.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_parser(self):
        super().vendor_parser()

        if self.cache_type != "linear":
            raise ValueError(
                "SDPA attention only supports a linear cache, not "
                f"cache_type={self.cache_type}; the paged cases are the fa2 "
                "provider's."
            )

        if not (
            self.dtype == "bfloat16"
            and self.dst_dtype == "bfloat16"
            and self.cache_dtype == "bfloat16"
            and self.qk_compute_dtype == "bfloat16"
            and self.pv_compute_dtype == "bfloat16"
        ):
            raise ValueError(
                "SDPA attention only supports an all-bfloat16 dtype set."
            )

        if self.attn_mode == "prefill":
            if self.cache_lens[0] != 0:
                raise ValueError(
                    "SDPA attention prefill only supports cache_len == 0, got "
                    f"{self.cache_lens[0]}; chunked prefill needs a "
                    "bottom-right aligned causal mask."
                )
        else:
            if self.q_lens[0] != 1:
                raise ValueError(
                    "SDPA attention decode only supports q_len == 1, got "
                    f"{self.q_lens[0]}; a multi-token decode step needs a "
                    "bottom-right aligned causal mask."
                )

    def vendor_impl(self):
        super().vendor_impl()

        # Pin the fused-backend set. Which backend SDPA picks is process-global
        # state, and any *other* provider's import can change it: `import vllm`
        # calls torch.backends.cuda.enable_cudnn_sdp(False), and the provider
        # registry imports every vendor module, so installing vllm silently
        # reconfigures this op. It cost 1.9x on every prefill case here and left
        # no trace in the report -- cuDNN attention runs an 80/8 GQA prefill at
        # q_len 4096 in 584 us, and PyTorch's FLASH backend, next in the priority
        # order, takes 1,079 us for the same shape. Decode barely moves (1.1x),
        # which is what makes the symptom easy to misread as noise.
        #
        # Re-enabling all three restores PyTorch's own default priority whatever
        # else got imported, so a number here does not depend on which unrelated
        # packages happen to be installed. This is deliberately a global set once
        # per op rather than an `sdpa_kernel` context around the call: a context
        # manager inside _run_func would sit in the timed region.
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):  # torch >= 2.3
            torch.backends.cuda.enable_cudnn_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        self.q_len = self.q_lens[0]
        self.kv_len = self.kv_lens[0]

        self.sdpa_kwargs = {
            "scale": self.softmax_scale,
            # prefill: a square score matrix, so top-left and bottom-right
            # alignment coincide and is_causal is exact.
            # decode: one query row against the whole cache attends to all of it,
            # and is_causal would leave it seeing only key 0.
            "is_causal": self.attn_mode == "prefill" and self.is_causal,
        }
        if self.q_head_num != self.kv_head_num:
            self.sdpa_kwargs["enable_gqa"] = True

        self._run_func = self.sdpa_run

    def sdpa_run(self, tensor_mapping):
        q = tensor_mapping["q"]
        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]

        # (num_tokens, q_head_num, head_dim) -> (bs, heads, seq_q, head_dim)
        q_sdpa = q.view(
            self.batch_size, self.q_len, self.q_head_num, self.head_dim
        ).transpose(1, 2)

        # The caches are already (bs, heads, max_kv_len, head_dim) and
        # max_kv_len == kv_len here, so this slice is the whole tensor for
        # prefill and everything written so far for decode.
        k_sdpa = k_cache[:, :, : self.kv_len, :]
        v_sdpa = v_cache[:, :, : self.kv_len, :]

        out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, **self.sdpa_kwargs
        )

        # Back to the packed layout op_defs declares for "out".
        return out.transpose(1, 2).reshape(
            self.num_tokens, self.q_head_num, self.head_dim
        )
