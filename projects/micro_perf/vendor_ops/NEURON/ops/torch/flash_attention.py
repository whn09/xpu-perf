from xpu_perf.micro_perf.core.op import ProviderRegistry


from xpu_perf.micro_perf.backends.NEURON.backend_neuron import (
    RUNTIME_EAGER,
    detect_neuron_runtime,
)

try:
    # The PyTorch-native stack has no NKI attention kernel to offer: the bundled
    # neuronxcc.nki.kernels.attention.flash_fwd is HLO-traced and only runs under
    # torch_xla, and nki 0.6.0 ships no kernel library for torch_neuronx.wrap_nki
    # to wrap. Native scaled_dot_product_attention does dispatch to the device,
    # so it is what flash_attention can be measured with there.
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

        The accepted envelope deliberately mirrors the NKI flash_fwd provider so
        the two runtimes report the same cases, even though SDPA itself is more
        general: op_defs hands over

            q:               (num_tokens, q_head_num, head_dim)
            k_cache/v_cache: (batch_size, kv_head_num, max_kv_len, head_dim)

        and SDPA wants (batch, heads, seq, head_dim) throughout, which for a
        single-sequence prefill is a reshape and a transpose of q plus a slice of
        the caches -- no data movement that would distort the measurement.
        """

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            if self.attn_mode != "prefill":
                # A decode step reads a variable-length or paged cache, which
                # this single contiguous slice cannot express.
                raise ValueError(
                    "native SDPA attention only supports prefill, not "
                    f"attn_mode={self.attn_mode}."
                )

            if not (
                self.dtype == "bfloat16"
                and self.dst_dtype == "bfloat16"
                and self.cache_dtype == "bfloat16"
                and self.qk_compute_dtype == "bfloat16"
                and self.pv_compute_dtype == "bfloat16"
                and self.cache_type == "linear"
            ):
                raise ValueError(
                    "native SDPA attention only supports all-bfloat16 with a "
                    "linear cache."
                )

            if self.q_head_num != self.kv_head_num:
                # GQA would need the kv heads expanded, and repeat_interleave is
                # a real copy that would land inside the timed region.
                raise ValueError(
                    "native SDPA attention only supports MHA, so q_head_num "
                    f"must equal kv_head_num (got {self.q_head_num} vs "
                    f"{self.kv_head_num})."
                )

            if self.batch_size != 1:
                raise ValueError(
                    "native SDPA attention prefill only supports batch_size == 1."
                )

            if self.cache_lens[0] != 0:
                raise ValueError(
                    "native SDPA attention prefill only supports cache_lens[0] == 0."
                )

        def vendor_impl(self):
            super().vendor_impl()
            self._run_func = self.prefill_run

        def prefill_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            # (num_tokens, q_head_num, head_dim) -> (bs, heads, seq_q, head_dim)
            q_sdpa = q.view(
                self.batch_size, self.num_tokens, self.q_head_num, self.head_dim
            ).transpose(1, 2)

            # The caches are already (bs, heads, seq, head_dim); prefill attends
            # over exactly the tokens it just wrote.
            k_sdpa = k_cache[:, :, : self.num_tokens, :]
            v_sdpa = v_cache[:, :, : self.num_tokens, :]

            out = torch.nn.functional.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                is_causal=self.is_causal,
                scale=self.softmax_scale,
            )

            # Back to the packed layout op_defs declares for "out".
            return out.transpose(1, 2).reshape(
                self.num_tokens, self.q_head_num, self.head_dim
            )

except Exception:
    pass
