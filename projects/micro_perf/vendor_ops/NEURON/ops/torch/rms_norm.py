"""`F.rms_norm` re-registered as an explicit provider, so adding a NKI one does not hide it.

This file adds no code path. It exists because of `core/op.py:153-155`:

    for op_name in cls.BASE_IMPL_MAPPING.keys():
        if op_name not in cls.OP_MAPPING:
            cls.OP_MAPPING[op_name] = {cls.BASE_PROVIDER: cls.BASE_IMPL_MAPPING[op_name]}

The `base` provider is inserted **only when an op has no vendor provider at all**. A
vendor provider therefore *replaces* the base implementation rather than being added
alongside it. So the moment `ops/nkilib/rms_norm.py` registers, the aten path stops
being measured entirely and `rms_norm/base/rms_norm-base.jsonl` stops being written --
which would drop the baseline every number in that kernel's docstring is quoted
against, and leave the sweep reporting only the kernel.

Registering the inherited implementation under its own name restores it. Both providers
then run every case, `engine.py:128` iterating the pair, and the sweep writes
`rms_norm/torch/` beside `rms_norm/nkilib/`. **Note the rename**: these rows used to
land under `rms_norm/base/`, so a comparison against results collected before this
change has to look there. Same convention `topk` and `flash_attention` already follow.

Nothing is overridden below: `BasicOp.vendor_impl()` sets
`self._run_func = self.vendor_impl_run`, and `RMSNormOp.vendor_impl_run`
(`op_defs/basic_ops/vector_norm_ops.py:124`) is the `torch.nn.functional.rms_norm` call
we want measured.

Keeping it is not just for the record. The kernel rejects the shapes it has no code
path for -- a ragged token count, a hidden size too wide to tile, `add_residual`, a
dtype it has not been verified on -- and this is what covers them.
"""
from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "torch"


@ProviderRegistry.register_vendor_impl("rms_norm", PROVIDER_NAME)
class NeuronTorchRMSNormOp:
    pass
