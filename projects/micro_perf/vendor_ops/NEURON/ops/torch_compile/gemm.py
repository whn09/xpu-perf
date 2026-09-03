"""An fp8 `gemm` that actually reaches Trainium2's tensor engines.

The `torch` provider next door measures `torch.matmul` on fp8 operands in eager
mode, and that lands at ~1.3% of the fp8 peak because eager has no fp8 matmul
lowering: it casts both operands up to bfloat16 first and the cast is the whole
cost. That number is worth publishing -- it is what the sweep runs today -- but it
is not an answer to "can this chip do fp8". This provider is.

Two things have to change to reach the engines, and both are expressed here rather
than in the op def, because neither is portable:

1. **The format.** `nki.isa.nc_matmul` takes `float8_e4m3` (legacy) and
   `float8_e5m2` on NeuronCore-v3, and adds OCP `float8_e4m3fn` only "starting
   NeuronCore-v4" -- v3 is Trn2, v4 is Trn3. `TORCH_DTYPE_MAPPING` resolves the
   workload string `float8_e4m3` to `torch.float8_e4m3fn`, i.e. to the encoding
   this chip cannot multiply, and neuronx-cc rejects it by name:
   `[NCC_EVRF051] Data type F8E4M3FN is not supported on TRN1/TRN2`. So the only
   fp8 string in the mapping that has a datapath here is `float8_e5m2`, and this
   provider accepts exactly that one.

2. **The path.** `torch.compile(backend="neuron")`. `dynamic=False` is not
   optional: compiling the same body at a second shape otherwise specialises it
   dynamically and neuronx-cc rejects the result with "Dynamic shape is not
   supported: ... shape 'bf16[?,?]'".

Measured this way on one logical NeuronCore at 4096^3: 559.8 us = 245.50 TFLOPS =
75.6% of the 324.75 TF per-core fp8 peak, 1.82x its own compiled bf16 (nominal
headroom 1.95x) and 115.7x the eager row. See tools/probe_fp8_datapath.py, which
is where those numbers come from and which also carries the bf16 and e4m3fn
controls this provider deliberately does not run.

Not registered unless `XPU_PERF_NEURON_GEMM_COMPILE=1`, on purpose. A registered
provider is instantiated once per case, and `gemm.json` holds 848 non-fp8 cases
that this one rejects -- each rejection printing a traceback. Keeping it opt-in
means a default sweep is byte-identical to the published one and its log stays
readable. Run it against vendor_ops/NEURON/workloads/gemm_fp8_compiled.json, which
pairs the shapes with the `dst_dtype` this provider requires.
"""
import os

import torch
import torch._dynamo

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.backends.NEURON.backend_neuron import (
    RUNTIME_EAGER,
    detect_neuron_runtime,
)


# Keep in sync with ops/torch/gemm.py: the fp8 workload strings TORCH_DTYPE_MAPPING
# resolves to a real torch fp8 type. `float8` and `float8_e4m3` both land on
# torch.float8_e4m3fn, which has no matmul datapath before NeuronCore-v4.
FP8_DTYPES = frozenset({"float8", "float8_e4m3", "float8_e5m2"})
COMPILABLE_FP8 = frozenset({"float8_e5m2"})

# Must match __init__.py, which is where load_plugin_package() reads the provider
# name from. Declared again here rather than imported: the plugin loader builds the
# package with spec_from_file_location and no submodule_search_locations, so
# relative imports inside a provider are not something to rely on.
PROVIDER_NAME = "torch_compile"

ENV_FLAG = "XPU_PERF_NEURON_GEMM_COMPILE"


def _enabled():
    if os.environ.get(ENV_FLAG, "").strip().lower() not in ("1", "true", "yes", "on"):
        return False
    # backend="neuron" is the torch_neuronx dynamo backend, which exists only on the
    # PyTorch-native stack. Under torch_xla compilation happens through XLA instead
    # and this provider would fail every case rather than skip them.
    return detect_neuron_runtime() == RUNTIME_EAGER


def _make_body(out_torch_dtype):
    """One closure per op instance, matching the shape the probe validated.

    The cast to `out_torch_dtype` is inside the compiled region on purpose: on
    NeuronCore-v3 `nc_matmul` writes an fp32 `dst`, so *something* has to convert,
    and letting the compiler fuse that conversion is what an fp8 inference path
    would do. Doing it in Python afterwards would put an M*N pass in the timed
    region and measure the cast instead of the matmul.
    """
    def body(a, b):
        return torch.matmul(a, b).to(out_torch_dtype)
    return body


class NeuronCompiledGemmOp:
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_parser(self):
        # fp8 only. The float formats are the `torch` provider's job, and running
        # them here would republish them under a second provider name.
        if self.dtype not in FP8_DTYPES:
            raise NotImplementedError(
                f"{PROVIDER_NAME} covers fp8 gemm only; dtype {self.dtype} is "
                "measured by the `torch` provider"
            )

        if self.dtype not in COMPILABLE_FP8:
            raise NotImplementedError(
                f"dtype {self.dtype} maps to torch.float8_e4m3fn (OCP), which "
                "nc_matmul supports only from NeuronCore-v4 (Trn3); neuronx-cc "
                "rejects it as NCC_EVRF051. Use float8_e5m2 on Trn2."
            )

        # The output dtype has to be stated by the workload rather than defaulted,
        # because the default is `dtype` and an fp8 output is a different question
        # from the one this provider answers -- it would also make write_bytes
        # claim 1 byte per element while the compiled graph emits 2 or 4.
        dst_dtype = self.args_dict.get("dst_dtype", self.dtype)
        if dst_dtype in FP8_DTYPES:
            raise NotImplementedError(
                "set an explicit non-fp8 dst_dtype (bfloat16 or float32): "
                "nc_matmul writes an fp32 dst on NeuronCore-v3, so an fp8 output "
                "measures an extra quantisation step this provider does not model"
            )

    def vendor_impl(self):
        super().vendor_impl()

        # A fresh compile per case. Every case is a distinct shape and dynamo's
        # default cache_size_limit is 8, so without the reset the 9th shape in a
        # sweep would silently fall back to eager -- which is exactly the number
        # this provider exists to avoid publishing.
        torch._dynamo.reset()
        self._compiled_run_func = torch.compile(
            _make_body(self.dst_torch_dtype), backend="neuron", dynamic=False
        )
        self._run_func = self.compiled_run

    def compiled_run(self, tensor_mapping):
        return self._compiled_run_func(tensor_mapping["a"], tensor_mapping["b"])


# The first of perf()'s two warm-up iterations pays the neuronx-cc compile for this
# shape; the timed loop that follows sees a cached graph. That is minutes per case,
# not the ~3 s of the eager per-shape warm-up -- size any watchdog accordingly.
if _enabled():
    NeuronCompiledGemmOp = ProviderRegistry.register_vendor_impl(
        "gemm", PROVIDER_NAME
    )(NeuronCompiledGemmOp)
