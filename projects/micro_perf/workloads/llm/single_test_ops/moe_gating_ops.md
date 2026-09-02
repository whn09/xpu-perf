## moe_gating_ops

MoE 路由的 `softmax + topk`，输入是 gating gemm 的输出 `[num_tokens, num_experts]`。

`compute_mode` 区分两种常见写法：`pre-softmax` 先 softmax 再取 topk，
`post-softmax` 先取 topk 再对这 `topk` 个 logit 做 softmax；后者的 softmax 只有
`topk` 宽，量级小一个数量级。规范要求 `dtype` 为 `float32`。
