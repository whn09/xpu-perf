"""`torch.topk` re-registered as an explicit provider, so adding a NKI one does not hide it.

This file adds no code path. It exists because of `core/op.py:153-155`:

    for op_name in cls.BASE_IMPL_MAPPING.keys():
        if op_name not in cls.OP_MAPPING:
            cls.OP_MAPPING[op_name] = {cls.BASE_PROVIDER: cls.BASE_IMPL_MAPPING[op_name]}

The `base` provider is inserted **only when an op has no vendor provider at all**. A
vendor provider therefore *replaces* the base implementation rather than being added
alongside it. So the moment `ops/nkilib/topk.py` registers, `torch.topk` stops being
measured entirely and `topk/base/topk-base.jsonl` stops being written -- which would
silently drop the baseline the `k` cliff was diagnosed from, and leave the sweep
reporting only the kernel that wins some of the rows.

Registering the inherited implementation under its own name restores it. Both providers
now run every case, `engine.py:128` iterating the pair, and the sweep writes
`topk/torch/` beside `topk/nkilib/`. Note the rename: these rows used to land under
`topk/base/`. That is the same convention `flash_attention` already follows, which is
why that op has both `ops/torch/flash_attention.py` and `ops/nkilib/flash_attention.py`.

Nothing is overridden below: `BasicOp.vendor_impl()` sets
`self._run_func = self.vendor_impl_run`, and `TopkOp.vendor_impl_run`
(`op_defs/basic_ops/vector_reduction_ops.py`) is the `torch.topk(..., sorted=False)`
call we want measured.
"""
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "torch"


@ProviderRegistry.register_vendor_impl("topk", PROVIDER_NAME)
class NeuronTorchTopkOp:
    pass
