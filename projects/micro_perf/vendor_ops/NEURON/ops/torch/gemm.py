from xpu_perf.micro_perf.core.op import ProviderRegistry


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
        super().vendor_parser()
