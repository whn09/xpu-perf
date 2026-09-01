from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.backends.NEURON.backend_neuron import RUNTIME_EAGER


@ProviderRegistry.register_vendor_impl("all_gather", "torch")
class NeuronAllGatherOp:
    """all_gather via xm.all_gather on the XLA runtime.

    The base implementation uses dist.all_gather_into_tensor, which the xla
    process group backend does not implement. The native "neuron" backend does
    implement it, so on that runtime this reproduces the base behaviour rather
    than reaching for a torch_xla that is not installed.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]

        if self.backend.neuron_runtime == RUNTIME_EAGER:
            dst = tensor_mapping["dst"]
            self.backend.get_dist_module().all_gather_into_tensor(
                dst, src,
                group=self.op_group
            )
            return dst

        import torch_xla.core.xla_model as xm
        return xm.all_gather(src, dim=0)
