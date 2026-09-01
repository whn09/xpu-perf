from xpu_perf.micro_perf.core.op import ProviderRegistry


from xpu_perf.micro_perf.backends.NEURON.backend_neuron import (
    RUNTIME_XLA,
    detect_neuron_runtime,
)

try:
    # The bundled neuronxcc kernels are traced into HLO, so they only run on the
    # XLA runtime; on the PyTorch-native stack the call fails deep inside the
    # kernel with "No module named 'torch_neuronx.pyhlo'". Deciding that at
    # registration time means flash_attention is reported as having no
    # implementation there, rather than failing case by case. The native stack's
    # own NKI entry point (torch_neuronx.wrap_nki) expects kernels written
    # against the newer standalone `nki` package, which as of nki 0.6.0 ships no
    # attention kernel to point it at.
    if detect_neuron_runtime() != RUNTIME_XLA:
        raise ImportError("neuronxcc NKI kernels require the XLA runtime")

    from neuronxcc.nki.kernels.attention import flash_fwd, FlashConfig

    # Hardware kernel shipped with neuronx-cc, invoked through NKI.
    # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/
    @ProviderRegistry.register_vendor_impl("flash_attention", "nki")
    class NKIFlashAttentionOp:
        """flash_attention via the NKI flash_fwd kernel.

        flash_fwd expects channel-last-transposed layouts:
            q: (batch_size, q_head_num, head_dim, seq_q)
            k: (batch_size, kv_head_num, head_dim, seq_k)
            v: (batch_size, kv_head_num, head_dim, seq_v)  with should_transpose_v
            o: (batch_size, q_head_num, seq_q, head_dim)

        while op_defs hands over
            q:               (num_tokens, q_head_num, head_dim)
            k_cache/v_cache: (batch_size, kv_head_num, max_kv_len, head_dim)
        """

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            if self.attn_mode != "prefill":
                # flash_fwd takes a single contiguous K/V block, so it cannot
                # express a variable-length or paged KV cache.
                raise ValueError(
                    "NKI flash_fwd only supports prefill, not "
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
                    "NKI flash_fwd only supports all-bfloat16 with a linear cache."
                )

            if self.q_head_num != self.kv_head_num:
                raise ValueError(
                    "NKI flash_fwd only supports MHA, so q_head_num must equal "
                    f"kv_head_num (got {self.q_head_num} vs {self.kv_head_num})."
                )

            if self.batch_size != 1:
                raise ValueError("NKI flash_fwd prefill only supports batch_size == 1.")

            if self.cache_lens[0] != 0:
                raise ValueError("NKI flash_fwd prefill only supports cache_lens[0] == 0.")

            # The kernel tiles the sequence; the tile must divide the sequence.
            self._flash_config = FlashConfig(
                seq_tile_size=2048 if self.num_tokens >= 2048 else self.num_tokens,
                training=False,
                should_transpose_v=True,
            )

        def vendor_impl(self):
            super().vendor_impl()
            self._run_func = self.prefill_run

        def prefill_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            # (num_tokens, q_head_num, head_dim) -> (bs, q_head_num, head_dim, seq_q)
            q_nki = q.view(
                self.batch_size, self.num_tokens, self.q_head_num, self.head_dim
            ).permute(0, 2, 3, 1).contiguous()

            # (bs, kv_head_num, max_kv_len, head_dim) -> (bs, kv_head_num, head_dim, seq)
            k_nki = k_cache[:, :, :self.num_tokens, :].permute(0, 1, 3, 2).contiguous()
            v_nki = v_cache[:, :, :self.num_tokens, :].permute(0, 1, 3, 2).contiguous()

            return flash_fwd[self.batch_size, self.q_head_num](
                q_nki,
                k_nki,
                v_nki,
                seed=None,
                use_causal_mask=self.is_causal,
                mixed_precision=True,
                softmax_scale=self.softmax_scale,
                config=self._flash_config,
            )

except Exception:
    pass
