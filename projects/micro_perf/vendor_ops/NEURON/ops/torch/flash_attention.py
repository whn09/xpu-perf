from xpu_perf.micro_perf.core.op import ProviderRegistry


from xpu_perf.micro_perf.backends.NEURON.backend_neuron import (
    RUNTIME_EAGER,
    detect_neuron_runtime,
)

try:
    # There is no NKI attention kernel for *this file* to call on the PyTorch-native
    # stack: the bundled neuronxcc.nki.kernels.attention.flash_fwd is HLO-traced and
    # only runs under torch_xla. Native scaled_dot_product_attention does dispatch to
    # the device, so it is what flash_attention is measured with here.
    #
    # That does not make these unfused numbers, and an earlier version of this
    # comment wrongly implied it did. torch_neuronx lowers SDPA to a NKI flash kernel
    # inside its own dynamo backend -- _can_use_nki_flash_attention in
    # neuron_dynamo_backend/decompositions.py, enabled by default through
    # TORCH_NEURONX_ENABLE_NKI_SDPA -- for any call with
    # `L % 512 == 0 and S % 512 == 0 and D <= 128 and B*H <= 512`, no attn_bias and
    # no dropout. Measured on one logical core with the 80/8/128 GQA prefill at
    # q_len 4096: 9,443 us by default against 60,500 us with the variable set to 0,
    # so prefill here is fused and worth 6.41x. Decode cannot reach it -- q_len 1
    # never satisfies `L % 512 == 0`, and B*H is 1280/5120, over the 512 limit --
    # which is what the decode rows' 15-33% of HBM peak reflects. nkilib (installed
    # in the beta images, so wrap_nki does have kernels to wrap now, contrary to what
    # this comment used to say) ships attention_tkg for exactly that case; wiring it
    # would be a second provider, not a change to this one.
    #
    # And the kernel this rewrite picks for prefill is already the best one nkilib
    # has to offer: decompositions.py line 24 imports nkilib's attention_cte and
    # line 71 wraps that same object, so `dc.attention_cte is attention_cte` holds.
    # Calling attention_cte by hand with the launch arguments used at line 935
    # (lnc=logical_neuron_cores, tp_k=True, KV left at its own head count) gives
    # 7,407.4 us against SDPA's 7,636.7 us -- 0.97x, bit-identical output, one
    # kernel reached two ways. So a hand-rolled NKI prefill provider would
    # re-derive this number rather than improve on it; see
    # tools/probe_attention_kernel.py. Note the trap it documents: omitting the lnc
    # subscript runs the kernel on one half of the LNC2 pair and costs 1.85x.
    #
    # This is registered only on the native runtime. On the XLA runtime the NKI
    # provider is the intended implementation, and adding a second one would
    # change results that have already been validated.
    if detect_neuron_runtime() != RUNTIME_EAGER:
        raise ImportError("native SDPA attention requires the eager runtime")

    import torch

    @ProviderRegistry.register_vendor_impl("flash_attention", "torch")
    class NeuronSDPAFlashAttentionOp:
        """flash_attention via torch.nn.functional.scaled_dot_product_attention.

        op_defs hands over

            q:               (num_tokens, q_head_num, head_dim)
            k_cache/v_cache: (batch_size, kv_head_num, max_kv_len, head_dim)

        and SDPA wants (batch, heads, seq, head_dim) throughout. Because
        `arg_type: llm` gives every sequence the same q_len and cache_len
        (`get_attn_info` in core/utils.py), that conversion is a view plus a
        transpose of q and a slice of the caches for both a full prefill and a
        one-token decode step -- no data movement that would distort the
        measurement, and no per-batch bookkeeping.

        What is rejected, and why (all reported as unsupported rather than
        failing later):

        * a paged cache -- gathering `block_table` rows into a contiguous k/v
          would be a copy inside the timed region, which is the cost the paged
          layout exists to avoid;
        * anything but an all-bfloat16 dtype set -- an int8 cache has to be
          dequantized, again a copy inside the timed region;
        * chunked prefill (`cache_len > 0` with `q_len > 1`) and speculative
          decode (`q_len > 1`) -- both need the causal mask aligned to the
          bottom-right of a non-square score matrix, and `is_causal=True` in
          PyTorch aligns to the top-left. Passing it anyway does not fail, it
          silently attends to the wrong keys, so these cases must be turned away
          rather than measured. An explicit `attn_mask` would express them, but a
          (q_len, kv_len) bool mask is itself tens of MB of extra HBM traffic per
          call and would be measuring something else.

        GQA is handled by `enable_gqa=True`, which lets SDPA broadcast the kv
        heads internally. The alternative -- `repeat_interleave` on k and v --
        materialises a q_head_num-sized cache on every call, and that copy would
        land inside the timed region and dominate a decode step.
        """

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            if self.cache_type != "linear":
                raise ValueError(
                    "native SDPA attention only supports a linear cache, not "
                    f"cache_type={self.cache_type} (drop block_size from the "
                    "workload to get one)."
                )

            if not (
                self.dtype == "bfloat16"
                and self.dst_dtype == "bfloat16"
                and self.cache_dtype == "bfloat16"
                and self.qk_compute_dtype == "bfloat16"
                and self.pv_compute_dtype == "bfloat16"
            ):
                raise ValueError(
                    "native SDPA attention only supports an all-bfloat16 dtype "
                    "set."
                )

            if self.attn_mode == "prefill":
                if self.cache_lens[0] != 0:
                    raise ValueError(
                        "native SDPA attention prefill only supports "
                        f"cache_len == 0, got {self.cache_lens[0]}; chunked "
                        "prefill needs a bottom-right aligned causal mask."
                    )
            else:
                if self.q_lens[0] != 1:
                    raise ValueError(
                        "native SDPA attention decode only supports q_len == 1, "
                        f"got {self.q_lens[0]}; a multi-token decode step needs a "
                        "bottom-right aligned causal mask."
                    )

        def vendor_impl(self):
            super().vendor_impl()

            self.q_len = self.q_lens[0]
            self.kv_len = self.kv_lens[0]

            # MHA keeps the exact call the earlier reference numbers were taken
            # with; only GQA adds the flag.
            self.sdpa_kwargs = {
                "scale": self.softmax_scale,
                # prefill: a square score matrix, so top-left and bottom-right
                # alignment coincide and is_causal is exact.
                # decode: one query row against the whole cache attends to all of
                # it, and is_causal would leave it seeing only key 0.
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

except Exception:
    pass
