from xpu_perf.micro_perf.core.op import ProviderRegistry


# The fp8 dtype strings TORCH_DTYPE_MAPPING (core/utils.py) resolves to a real
# torch fp8 type. The `mxfloat8*` aliases are deliberately absent: they map to the
# same torch.float8_e4m3fn / float8_e5m2 with no block-scale tensor anywhere in
# the op def, so accepting them would republish the numbers below under a label
# that promises microscaling. See README, "fp8 runs but does not reach the fp8
# datapath".
FP8_DTYPES = frozenset({"float8", "float8_e4m3", "float8_e5m2"})


@ProviderRegistry.register_vendor_impl("gemm", "torch")
class NeuronGemmOp:
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_parser(self):
        # tfloat32 is an NVIDIA format; int8 matmul is not lowered by
        # neuronx-cc through torch.matmul. Reject both before tensors are
        # allocated so the case is reported as unsupported rather than failing
        # later inside the compiler.
        if self.dtype == "tfloat32":
            raise NotImplementedError("Neuron does not support tfloat32")
        if self.dtype == "int8":
            raise NotImplementedError("Neuron does not support int8 gemm via torch.matmul")

        if self.dtype in FP8_DTYPES:
            # The base op def gates dtype to the four float formats, so fp8 is
            # only measurable if a vendor accepts it. It does run here -- the
            # inherited torch.matmul dispatches to the device and returns an fp8
            # tensor -- but read the result as an fp8 *storage* number, not an
            # fp8 datapath number: it lands two orders of magnitude below the
            # published fp8 peak because the lowering casts both operands up to
            # bfloat16 first, and that conversion is the whole cost. The
            # measurement that pins it down is in the README.
            #
            # torch._scaled_mm is the API that would express a real fp8 gemm
            # (fp32 scales, bf16 accumulate), and it does dispatch to the device
            # at 64x64x64 -- but at 4096x4096x4096 it sat in neuronx-cc for over
            # 40 minutes without returning, the same wedge `gather` and `scatter`
            # hit, so it cannot be used for a sweep.
            return

        super().vendor_parser()
