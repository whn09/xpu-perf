from xpu_perf.micro_perf.core.op import ProviderRegistry


# The fp8 dtype strings TORCH_DTYPE_MAPPING (core/utils.py) resolves to a real
# torch fp8 type. The `mxfloat8*` aliases are deliberately absent: they map to the
# same torch.float8_e4m3fn / float8_e5m2 with no block-scale tensor anywhere in
# the op def, so accepting them would republish the numbers below under a label
# that promises microscaling. See README, "fp8: the sweep measures the eager path,
# in a dtype this chip cannot multiply".
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
            # tensor -- but read the result as a *storage* number, not an fp8
            # datapath number: it lands two orders of magnitude below the
            # published fp8 peak because the eager lowering casts both operands
            # up to bfloat16 first, and that conversion is the whole cost.
            #
            # That is a property of this path and not of the chip, which is why
            # the README no longer says Trainium2 has no fp8 gemm. Two things
            # would have to change to reach it, and the op def can express
            # neither. (1) The encoding: float8_e4m3 -> torch.float8_e4m3fn, the
            # OCP finite-only variant CUDA uses, which has no matmul datapath
            # before Trn3 -- nki.isa.nc_matmul takes legacy float8_e4m3 and
            # float8_e5m2 on NeuronCore-v3 and adds float8_e4m3fn only "starting
            # NeuronCore-v4", and the compiler agrees by name:
            # `[NCC_EVRF051] Data type F8E4M3FN is not supported on TRN1/TRN2`.
            # The workaround flag that error names *casts* to the legacy encoding
            # and does not exist in neuronx-cc 2.27.2878.0 anyway. e5m2 has no
            # such split. (2) The path: eager
            # has no fp8 gemm lowering for either encoding. Under
            # torch.compile(backend="neuron", dynamic=False), e5m2 reaches the
            # tensor engines at 245.50 TFLOPS, 75.6% of the 324.75 TF fp8 peak,
            # 115.7x this path at 4096^3. See tools/probe_fp8_datapath.py.
            #
            # torch._scaled_mm is the API that would express a real fp8 gemm
            # (fp32 scales, bf16 accumulate) and it does dispatch at 64x64x64,
            # but it is not usable on this backend: 690 ms at 512^3, ~5,500x the
            # bf16 matmul, and at 2048^3 one call did not return in 3.5 minutes
            # while holding 555% host CPU with the device idle. Earlier notes
            # blamed a neuronx-cc wedge; it is host execution, not compilation --
            # aten::_scaled_mm is absent from _NEURON_OPS_REGISTRY, and
            # TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1 does not fire.
            return

        super().vendor_parser()
