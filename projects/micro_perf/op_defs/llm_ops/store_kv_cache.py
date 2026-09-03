"""LLM op: store_kv_cache (base implementation)."""
from ._common import *

@ProviderRegistry.register_base_impl("store_kv_cache", "ComputeEngine")
class StoreKVCacheOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm", "batch_llm"]:
            raise ValueError
        
        # pre-defined attrs
        self.attn_mode = self.args_dict.get("attn_mode", "prefill")
        if not self.attn_mode in ["prefill", "decode"]:
            raise ValueError
        get_attn_info(self.arg_type, self.attn_mode, self.args_dict, self)
        

        """
        一般来说, 我们会同时存KV Cache, 但是有些场景只需要保存一个cache
        """
        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num
        self.head_dim = self.args_dict["head_dim"]

        self.store_mode = self.args_dict.get("store_mode", "both")
        if self.store_mode not in ["both", "k", "v"]:
            raise ValueError("store_mode must be either 'both', 'k', or 'v'")
        
        """
        实际上是这么排布数据的
        [q_head_num * head_dim, kv_head_num * head_dim + kv_head_num * head_dim]
        """
        self.total_dim = \
            self.q_head_num * self.head_dim \
            + self.kv_head_num * self.head_dim \
            + self.kv_head_num * self.head_dim

        self.k_dim_start = self.q_head_num * self.head_dim
        self.k_dim_end = self.q_head_num * self.head_dim + self.kv_head_num * self.head_dim
        self.v_dim_start = self.k_dim_end
        self.v_dim_end = self.total_dim
    

        # 以下参数决定当前 store_kv_cache 的具体数据类型
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        self.cache_dtype = self.args_dict.get("cache_dtype", "bfloat16")

    def vendor_parser(self):
        if self.dtype == "bfloat16" and self.cache_dtype == "bfloat16":
            self.use_quant = False
        elif self.dtype == "bfloat16" and self.cache_dtype == "int8":
            self.use_quant = True
        else:
            raise ValueError(f"dtype = {self.dtype}, cache_dtype = {self.cache_dtype} is not supported.")



    def vendor_impl(self):
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.cache_torch_dtype = get_torch_dtype(self.cache_dtype)
            
        self.input_tensor_info = {}
        self.output_tensor_info = {}

        """
        输入的 qkv, packed模式, unsplited模式
        """
        self.input_tensor_info["packed_qkv"] = OpTensorInfo(
            shape=[self.num_tokens, self.total_dim], 
            dtype=self.torch_dtype, 
            device=self.backend.get_torch_device_name(),
        )


        """
        确定当前的 num_tokens 中的具体的组成信息
        """
        self.attn_info_tensors = {
            "q_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device)
            ), 
            "cache_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device)
            ), 
            "kv_lens": OpTensorInfo(
                shape=[self.batch_size], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.kv_lens, dtype=dtype, device=device)
            ), 
            "accum_q_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device)
            ), 
            "accum_cache_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_cache_lens, dtype=dtype, device=device)
            ), 
            "accum_kv_lens": OpTensorInfo(
                shape=[self.batch_size + 1], 
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: torch.tensor(self.accum_kv_lens, dtype=dtype, device=device)
            ), 
        }
        self.input_tensor_info.update(self.attn_info_tensors)


        """
        kv cache信息, 根据 block_size 决定是 linear 还是 paged
        """
        if self.cache_type == "linear":
            self.input_tensor_info["slot_mapping"] = OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: \
                    torch.tensor(self.slot_mapping, dtype=dtype, device=device)
            )
            cache_shape = [self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim]
        elif self.cache_type == "paged":
            self.input_tensor_info["block_table"] = OpTensorInfo(
                shape=[self.target_batch_size, self.target_per_seq_num_block],
                dtype=torch.int32, 
                device=self.backend.get_torch_device_name(), 
                creator=lambda size, dtype, device: \
                    torch.tensor(self.block_table, dtype=dtype, device=device)
            )
            cache_shape = [self.total_cache_blocks, self.kv_head_num, self.block_size, self.head_dim]



        if self.store_mode == "both" or self.store_mode == "k":
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=cache_shape, 
                dtype=self.cache_torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
        if self.store_mode == "both" or self.store_mode == "v":
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=cache_shape, 
                dtype=self.cache_torch_dtype, 
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )

        """
        根据 kv cache 是否需要量化确定量化参数
        """
        if self.use_quant:
            quant_scale_shape = [self.kv_head_num, self.head_dim]

            if self.store_mode == "both" or self.store_mode == "k":
                self.input_tensor_info["k_scale"] = OpTensorInfo(
                    shape=quant_scale_shape, 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(), 
                    creator=torch.ones
                )
            if self.store_mode == "both" or self.store_mode == "v":
                self.input_tensor_info["v_scale"] = OpTensorInfo(
                    shape=quant_scale_shape, 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(), 
                    creator=torch.ones
                )



        # calculator
        self.input_tensor_size = sum(
            [calc_tensor_size(info) for info in self.input_tensor_info.values()]
        )
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size


        self.read_bytes = self.input_tensor_size
        self.read_bytes -= \
            calc_tensor_size(self.input_tensor_info["packed_qkv"]) \
            // self.total_head_num * self.q_head_num


        cache_bytes = 0
        if self.store_mode == "both" or self.store_mode == "k":
            cache_bytes += calc_tensor_size(self.input_tensor_info["k_cache"])
        if self.store_mode == "both" or self.store_mode == "v":
            cache_bytes += calc_tensor_size(self.input_tensor_info["v_cache"])

        self.read_bytes -= cache_bytes


        self.write_bytes = self.output_tensor_size
        if self.cache_type == "linear":
            self.write_bytes += \
                cache_bytes // self.batch_size // self.max_kv_len * self.num_tokens
        elif self.cache_type == "paged":
            self.write_bytes += \
                cache_bytes // self.total_cache_blocks * self.num_q_blocks

        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, tensor_mapping):
        # get pre-allocated input tensors
        packed_qkv = tensor_mapping["packed_qkv"]

        if self.cache_type == "linear":
            slot_mapping = tensor_mapping["slot_mapping"]
        elif self.cache_type == "paged":
            block_table = tensor_mapping["block_table"]

        """
        以下 tensor 在实际实现时作为输入对接到kernel
        下面只展示计算逻辑
        """
        cache_lens = tensor_mapping["cache_lens"]
        q_lens = tensor_mapping["q_lens"]
        kv_lens = tensor_mapping["kv_lens"]

        cache_lens = tensor_mapping["cache_lens"]
        accum_q_lens = tensor_mapping["accum_q_lens"]
        accum_kv_lens = tensor_mapping["accum_kv_lens"]
        
        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]

        k_scale = None if "k_scale" not in tensor_mapping else tensor_mapping["k_scale"]
        v_scale = None if "v_scale" not in tensor_mapping else tensor_mapping["v_scale"]

        """
        参考linear cache的实现改写成paged cache的实现
        """
        if self.cache_type == "paged":
            raise NotImplementedError("StoreKVCacheOp paged cache not implemented yet.")
        
        if self.cache_type == "linear":
            for batch_idx in range(self.batch_size):
                kv_slot_id = self.slot_mapping[batch_idx]

                q_len = self.q_lens[batch_idx]
                q_offset = self.accum_q_lens[batch_idx]
                cache_len = self.cache_lens[batch_idx]

                token_start = q_offset
                token_end = q_offset + q_len

                cache_start = cache_len
                cache_end = cache_len + q_len

                # The packed source is [q_len, kv_head_num * head_dim] and the cache
                # slice is [kv_head_num, q_len, head_dim], so the head dimension has to
                # be split out before the transpose: a plain transpose(0, 1) leaves
                # [kv_head_num * head_dim, q_len], which copy_ cannot broadcast onto the
                # destination at any shape. Quantisation happens first, while the tensor
                # is still token-major, because that is static_quant's contract --
                # [num_tokens, hidden_size] against a [1, hidden_size] scale.
                def _to_cache_layout(src, scale):
                    if self.use_quant:
                        src = static_quant(src, scale, self.cache_torch_dtype)
                    return src.contiguous().view(
                        q_len, self.kv_head_num, self.head_dim
                    ).transpose(0, 1)

                if self.store_mode == "both" or self.store_mode == "k":
                    src_k_data = packed_qkv[token_start:token_end, self.k_dim_start:self.k_dim_end]
                    dst_k_cache = k_cache[kv_slot_id, :, cache_start:cache_end, :]
                    dst_k_cache.copy_(_to_cache_layout(src_k_data, k_scale))
                if self.store_mode == "both" or self.store_mode == "v":
                    src_v_data = packed_qkv[token_start:token_end, self.v_dim_start:self.v_dim_end]
                    dst_v_cache = v_cache[kv_slot_id, :, cache_start:cache_end, :]
                    dst_v_cache.copy_(_to_cache_layout(src_v_data, v_scale))

        return k_cache, v_cache



