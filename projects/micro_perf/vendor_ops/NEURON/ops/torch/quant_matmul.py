"""Base `quant_matmul` re-registered as `torch`, only when the MX provider is enabled.

A vendor provider *replaces* an op's base implementation (core/op.py:153-155), so
turning on ops/nkilib/quant_matmul.py (`XPU_PERF_NEURON_MX=1`) would otherwise stop
the int8 rows from being measured at all. Same pattern as ops/torch/topk.py and
ops/torch/rms_norm.py. Gated on the same flag so a default sweep keeps writing its
rows under `quant_matmul/base/` exactly as before.

Nothing is overridden: QuantMatmulOp.vendor_impl_run (op_defs/llm_ops/quant_matmul.py)
is the call measured.
"""
import os

from xpu_perf.micro_perf.core.op import ProviderRegistry

PROVIDER_NAME = "torch"

if os.environ.get("XPU_PERF_NEURON_MX", "").strip().lower() in ("1", "true", "yes", "on"):
    @ProviderRegistry.register_vendor_impl("quant_matmul", PROVIDER_NAME)
    class NeuronTorchQuantMatmulOp:
        pass
