import os

import torch
import torch._dynamo

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.backends.NEURON.backend_neuron import RUNTIME_EAGER


# The fp8 dtype strings TORCH_DTYPE_MAPPING (core/utils.py) resolves to a real
# torch fp8 type. The `mxfloat8*` aliases are deliberately absent: they map to the
# same torch.float8_e4m3fn / float8_e5m2 with no block-scale tensor anywhere in
# the op def, so accepting them would republish the numbers below under a label
# that promises microscaling. See README, "fp8: the sweep measures the eager path,
# in a dtype this chip cannot multiply".
FP8_DTYPES = frozenset({"float8", "float8_e4m3", "float8_e5m2"})

# neuronx-cc's own switch for running fp32 matmuls on the TF32 datapath.
TF32_CC_FLAGS = "--auto-cast=matmult --auto-cast-type=tf32"


def _tf32_body(a, b):
    return torch.matmul(a, b)


@ProviderRegistry.register_vendor_impl("gemm", "torch")
class NeuronGemmOp:
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_parser(self):
        # int8 matmul is not lowered by neuronx-cc through torch.matmul. Reject it
        # before tensors are allocated so the case is reported as unsupported
        # rather than failing later inside the compiler.
        if self.dtype == "int8":
            raise NotImplementedError("Neuron does not support int8 gemm via torch.matmul")

        if self.dtype == "tfloat32":
            # TF32 is a Tensor Engine datatype on Trainium2 and Trainium3 (667 /
            # 671 TFLOPS per chip, the same as bf16), and an earlier version of this
            # file rejected it as "an NVIDIA format". It is not reachable the way
            # CUDA does it -- `torch.set_float32_matmul_precision("high")` changes
            # nothing here (measured: rel err 1.8e-7, i.e. exact fp32, 35.6 TFLOPS
            # at 4096^3) -- but neuronx-cc's auto-cast is: with
            # `NEURON_CC_FLAGS="--auto-cast=matmult --auto-cast-type=tf32"` the same
            # matmul gives rel err 1.5e-4 (TF32 rounding) at 106.8 TFLOPS, 3.0x fp32.
            #
            # The flag cannot simply be exported for the run. It is read per
            # compile, but the eager runtime's in-process graph cache is keyed by
            # shape and dtype only, *not* by compiler flags: whichever of an fp32
            # and a tf32 case of the same shape compiles first decides the
            # precision of both, and gemm.json has both at every shape. So the tf32
            # case goes through its own torch.compile graph (a separate cache
            # source; measured isolated from eager fp32 in either order), with the
            # flag set only for the call that compiles it. See vendor_impl.
            if self.backend.neuron_runtime != RUNTIME_EAGER:
                raise NotImplementedError(
                    "tfloat32 is wired through torch.compile(backend='neuron'), "
                    "which exists only on the PyTorch-native (eager) runtime"
                )
            return

        if self.dtype in FP8_DTYPES:
            # Rejected because the result is wrong, not because it is slow.
            # Re-validated on Neuron SDK 2.33 (torch-neuronx 2.14.3, neuronx-cc
            # 2.28.7631): eager `torch.matmul` on fp8 operands returns an fp8
            # tensor whose relative error against an fp32 matmul of the *same* fp8
            # values is 0.65 (e5m2) / 0.68 (e4m3fn), against 0.03-0.05 for merely
            # rounding the exact product to fp8 -- correlation with the true result
            # is ~0.8. It is not a transpose, nor a bit-format mix-up (inputs or
            # output read as the other fp8 format both fit worse), and the inputs
            # round-trip bit-exactly; casting to bf16 or fp32 *on device* before the
            # matmul gives the right answer. A latency for a kernel that returns the
            # wrong product is not a number to publish, so earlier rows labelled
            # "fp8 eager" should be discarded. The earlier SDK was only ever timed,
            # never checked, so whether it was already wrong there is unknown.
            #
            # fp8 is measured by the `torch_compile` provider instead
            # (XPU_PERF_NEURON_GEMM_COMPILE=1), which reaches the tensor engines and
            # passes the same check at bf16 rounding (rel 1.7e-3).
            # tools/verify_providers.py is the check.
            raise NotImplementedError(
                f"eager fp8 torch.matmul returns numerically wrong results on this "
                f"stack (rel err ~0.65 vs fp32); measure {self.dtype} with "
                "XPU_PERF_NEURON_GEMM_COMPILE=1 (torch_compile provider)"
            )

        super().vendor_parser()

    def vendor_impl(self):
        super().vendor_impl()
        if self.dtype == "tfloat32":
            # One graph per case, like the torch_compile provider: dynamo's
            # cache_size_limit is 8, and the 9th shape would otherwise silently fall
            # back to eager fp32.
            torch._dynamo.reset()
            self._tf32_fn = torch.compile(_tf32_body, backend="neuron", dynamic=False)
            self._tf32_compiled = False
            self._run_func = self.tf32_run

    def tf32_run(self, tensor_mapping):
        a, b = tensor_mapping["a"], tensor_mapping["b"]
        if self._tf32_compiled:
            return self._tf32_fn(a, b)
        # First call compiles. Scope the flag to exactly this compile so no other
        # op's graph in this process inherits it.
        old = os.environ.get("NEURON_CC_FLAGS")
        os.environ["NEURON_CC_FLAGS"] = f"{old} {TF32_CC_FLAGS}" if old else TF32_CC_FLAGS
        try:
            out = self._tf32_fn(a, b)
            self.backend.device_synchronize()
        finally:
            if old is None:
                os.environ.pop("NEURON_CC_FLAGS", None)
            else:
                os.environ["NEURON_CC_FLAGS"] = old
        self._tf32_compiled = True
        return out
