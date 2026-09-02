import pathlib
import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry


# The fp8 dtype strings TORCH_DTYPE_MAPPING (core/utils.py) resolves to a real
# torch fp8 type. The `mxfloat8*` aliases are deliberately absent for the same
# reason as on the NEURON side: they map to the same torch.float8_e4m3fn /
# float8_e5m2 with no block-scale tensor anywhere in the op def, so accepting them
# would report plain-fp8 numbers under a label that promises microscaling.
FP8_DTYPES = frozenset({"float8", "float8_e4m3", "float8_e5m2"})


@ProviderRegistry.register_vendor_impl("gemm", "torch")
class GPUGemmOp:
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_parser(self):
        if self.dtype in FP8_DTYPES:
            # The base op def gates dtype to the four float formats, so fp8 is
            # only measurable if a vendor opts in. Unlike the NEURON provider,
            # opting in is not enough on CUDA: torch.matmul has no fp8 kernel at
            # all and raises, so this provider also has to replace the run
            # function with torch._scaled_mm. See vendor_impl below.
            #
            # cuBLAS will not multiply two e5m2 operands -- e5m2 is defined as a
            # gradient format and is only ever one side of a mixed pair, so the
            # only all-e5m2 kernel that could exist does not. Both operands here
            # carry `dtype`, so an e5m2 case has no kernel and is reported as
            # unsupported rather than raising mid-sweep. (Worth contrasting with
            # the NEURON result, where e5m2 is not merely supported but *faster*
            # than e4m3 -- which is itself the tell that no fp8 kernel is
            # involved there at all.)
            if self.dtype == "float8_e5m2":
                raise NotImplementedError(
                    "cuBLAS does not implement e5m2 x e5m2; e5m2 is only "
                    "supported as one side of a mixed-format fp8 matmul, which "
                    "this op def cannot express (a and b share one dtype)."
                )
            return

        super().vendor_parser()
        if self.dtype == "float32":
            torch.set_float32_matmul_precision("highest")
        elif self.dtype == "tfloat32":
            torch.set_float32_matmul_precision("high")

    def vendor_impl(self):
        super().vendor_impl()

        if self.dtype not in FP8_DTYPES:
            return

        # torch._scaled_mm is the only fp8 gemm on CUDA, and it requires the
        # second operand to be column-major. The base op def declares b as a
        # row-major [K, N], and transposing that inside the run function would
        # put a full K*N copy inside the timed region -- which is precisely the
        # cost we are trying to measure around. So declare b as [N, K] instead
        # and pass b.t(): a transposed view of a row-major [N, K] *is* a
        # column-major [K, N], for free. Element count is unchanged, so the
        # read_bytes / io_bytes / calc_flops the base def already computed stay
        # correct.
        self.input_tensor_info["b"].shape = [self.N, self.K]

        # Per-tensor scales, fixed at 1.0. A real pipeline derives these from the
        # data, but computing them here would time a reduction over both operands
        # rather than the gemm. Held as attributes so the allocation is outside
        # the timed region.
        device = self.backend.get_torch_device_name()
        self.scale_a = torch.ones((), dtype=torch.float32, device=device)
        self.scale_b = torch.ones((), dtype=torch.float32, device=device)

        # fp8 out needs fast accumulate; bf16 out does not and is what an actual
        # fp8 inference path emits.
        self.use_fast_accum = self.dst_dtype in FP8_DTYPES

        self._run_func = self.scaled_mm_run

    def scaled_mm_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        return torch._scaled_mm(
            a,
            b.t(),
            scale_a=self.scale_a,
            scale_b=self.scale_b,
            out_dtype=self.dst_torch_dtype,
            use_fast_accum=self.use_fast_accum,
        )

    def __del__(self):
        torch.set_float32_matmul_precision("highest")
        getattr(super(), "__del__", lambda: None)()
