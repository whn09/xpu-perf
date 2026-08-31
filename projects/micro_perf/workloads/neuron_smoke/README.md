# neuron_smoke

One small shape per op, for checking the NEURON backend end to end without
waiting on a full neuronx-cc sweep. Shapes match the reference numbers in
`vendor_ops/NEURON/README.md`, so results here are directly comparable.

```bash
cd projects/micro_perf
python launch.py --backend NEURON --device 0 --task_dir workloads/neuron_smoke
```

`all_reduce.json` and `all_gather.json` are `world_size=2`, so they need
`--device 0,1`. Neuron requires the launched world size to equal the case's
world_size, so run the collectives separately from the single-core ops.

`flash_attention.json` targets the `nki` provider: bfloat16, linear cache, MHA
(`q_head_num == kv_head_num`), `batch_size=1`, `cache_len=0` — the only
combination `flash_fwd` accepts.
