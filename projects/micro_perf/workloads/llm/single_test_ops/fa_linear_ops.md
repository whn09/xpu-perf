## fa_linear_ops

和 `fa_ops.json` 同一批 head 配置，区别是**不带 `block_size`**，也就是 linear
(连续) kv cache 而不是 paged cache。`fa_ops.json` 里每一个 case 都设了
`block_size: 512`，所以任何只实现 linear cache 的 vendor 在那个文件里一个 case
都跑不到，GQA / decode 的数字也就无从谈起；这个文件补上这一块。

覆盖：
- prefill，`cache_len = 0`：GQA (80/8) 和 MHA (80/80) 各一组，`q_len` 4096 /
  10240，另加一个 `batch_size = 4` 的 batched prefill。
- decode，`q_len = 1`：GQA (80/8)，`batch_size` 16 / 64，`cache_len` 4096 / 10240。
  另外保留 `fa_ops.json` 里那个 `q_len = 4` 的 speculative decode case。

MHA 只在 prefill 出现：decode 的 linear cache 是
`[batch_size, kv_head_num, cache_len + q_len, head_dim]`，80 个 kv head 配
`batch_size = 16`、`cache_len = 10240` 就是 6.7 GB 的 k/v，超出单个逻辑 core 的
HBM 预算。

**`q_len = 32768` 曾经在这个文件里，后来被删掉了。** 它不是被 vendor op 拒绝，而是
在 NEURON eager 的 SDPA 上根本跑不完：单核上跑了 55 分钟，宿主机上一个线程 83%
CPU，`walrus_driver` 早就退出了（所以不是编译），也没有报任何错。同一条 provider
在 `q_len = 10240` 上只要 40.0 ms。eager stack 上没有 fused flash kernel 可以退到
（NKI 的 `flash_fwd` 只在 XLA runtime 下能用），所以长 context prefill 在这里没有
可测的路径。留一个跑一小时都不出结果、也不报错的 case 在 workload 里，只会让别人
以为自己的机器挂了，所以宁可删掉并把原因写在这。

没有 chunked prefill (`cache_len > 0` 且 `q_len > 1`) 的 case：它需要 bottom-right
对齐的 causal mask，而 PyTorch `is_causal=True` 给的是 top-left 对齐。同样的原因，
上面 `q_len = 4` 的 decode case 在 NEURON 的 SDPA provider 下会被拒绝并报出理由，
留在这里是为了让能自己下 mask 的 vendor 有得可跑（见 NEURON/README.md）。
