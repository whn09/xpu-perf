from xpu_perf.micro_perf.core.op import ProviderRegistry


@ProviderRegistry.register_vendor_impl("all_gather", "torch")
class NeuronAllGatherOp:
    """all_gather via xm.all_gather.

    The base implementation uses dist.all_gather_into_tensor, which the xla
    process group backend does not implement.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        import torch_xla.core.xla_model as xm

        src = tensor_mapping["src"]
        return xm.all_gather(src, dim=0)
