## activation_ops

FFN 的激活部分：

- `swiglu`：输入 `[num_tokens, hidden_size * 2]`，chunk 成两半后 `silu(x1) * x2`。
- `swiglu_dynamic_quant`：上者融合 `per_token` 动态量化，产出 down_proj 的量化输入。
- `moe_swiglu` / `moe_swiglu_dynamic_quant`：MoE 版本，输入是 dispatch 之后的
  `[dispatch_tokens, hidden_size * 2]`，`dispatch_tokens` 由
  `num_tokens / num_experts / topk / ep_size` 推出，因此本地实际处理的 token 数
  远小于 `num_tokens`。

纯 memory bound，`hidden_size` 指的是输出宽度，输入是它的两倍。
