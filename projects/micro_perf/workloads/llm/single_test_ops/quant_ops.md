## quant_ops

独立的量化算子：

- `scale_dynamic_quant`：`per_token` 动态量化本身（bf16 -> int8），不融合任何
  其它计算，是上面那些 `*_dynamic_quant` 融合算子的对照组。
- `quant_group_gemm_reduce_sum`：sp 切分下的量化 gemm，`sp_size` 份
  `[num_tokens, hidden_size] x [hidden_size, new_hidden_size]` 算完后在 sp 维
  reduce sum，等价于 tp gemm + all_reduce 的单卡计算部分。
