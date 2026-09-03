import pathlib
import torch
from functools import partial

from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size
from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry
from .vector_sfu_ops import CosOp


@ProviderRegistry.register_base_impl("gelu", "ComputeEngine")
class GeluOp(CosOp):
    """`gelu`, in either of the two formulations torch itself offers.

    `approximate` is optional and defaults to `"none"`, the erf form, so every
    workload written before this argument existed keeps measuring exactly what it
    measured. It is worth exposing because the two forms are not
    interchangeable on either axis:

    * **Numerically**, `"tanh"` is a different function -- ~0.016 max abs error
      against the erf form in bfloat16 -- and it is the one a large family of
      models specifies. HuggingFace calls it `gelu_pytorch_tanh` (also
      `gelu_new`), and Qwen3.5-27B's vision tower is one of many that ask for it,
      so measuring the erf form there measures a function the model never runs.
    * **In cost**, they are 3.75x apart on Trainium2, because `erf` has no fast
      lowering on that backend: at 16384x4304 bf16 on one logical core, `erf`
      alone is 6,387 us / 44.2 GB/s while `tanh`, `sigmoid` and `exp` all reach
      ~560 GB/s. See tools/probe_gelu_lowering.py under vendor_ops/NEURON.

    Do not "help" by hand-writing the tanh polynomial instead of passing the
    argument: the same probe measures the expanded form at 6,497 us against 1,796
    for the fused op, because each elementwise step is a separate device
    round trip on an eager backend.
    """

    APPROXIMATE_MODES = ("none", "tanh")

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )

    def prepare_args(self):
        super().prepare_args()
        self.approximate = self.args_dict.get("approximate", "none")
        if self.approximate not in self.APPROXIMATE_MODES:
            raise ValueError(
                f"gelu approximate must be one of {self.APPROXIMATE_MODES}, not "
                f"{self.approximate!r}; these are torch's own two modes and there "
                "is no third one to pass through."
            )

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.gelu(src, approximate=self.approximate)
        return dst



@ProviderRegistry.register_base_impl("silu", "ComputeEngine")
class SiluOp(CosOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )
    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = torch.nn.functional.silu(src)
        return dst

