## norm_ops

Transformer block 中的各类 `RMSNorm`：

- `add_rms_norm`：residual add + RMSNorm 的融合形态，attention/FFN 前后各一次。
- `add_rms_norm_dynamic_quant`：上者再融合 `per_token` 动态量化，直接产出下一个
  `quant_matmul` 的输入。`output_mode` 决定是否额外吐出 residual (`res`) 或
  bf16 的 norm 结果 (`norm`)，`none` 只留量化输出。
- `head_rms_norm` / `head_rms_norm_dynamic_quant`：按 head 归一化，作用在融合 qkv
  张量的一段 head 上（`norm_head_start` / `norm_head_num`），所以只读写这一段。
- `qk_rms_norm`：q 和 k 各自的 head norm，一次算子里做完。

`num_tokens` 从 1 扫到 32768：小 token 数量下这些算子完全被 kernel launch 开销
支配，读到的是 runtime 的 dispatch 时间而不是带宽；带宽要看 token 数大的那几行。
