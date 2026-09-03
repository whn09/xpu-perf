"""Vocabulary `topk` through nkilib's `rotational_topk`, which is flat in `k`.

Why a second provider rather than a change to the base op def:

`torch.topk` on this backend has a small-`k` path and a general sort above it, and the
step between them is enormous. Measured on one Trn2 logical core, bf16, `sorted=False`
(which is what `op_defs/basic_ops/vector_reduction_ops.py` asks for):

    vocab 248320, batch 1     k=1      k=8      k=50
      torch.topk             587.7    588.5   5355.5 us
      rotational_topk        674.8    653.1    651.8 us

The nkilib kernel is **flat in k** -- it is a multi-stage rotational reduction, so `k`
changes how much is carried between stages rather than which algorithm runs -- and at
`k=50` that is **8.22x**. The H100 is likewise flat in `k` (90-193 us throughout), so
this closes a gap that exists only on this backend. Values match `torch.topk` exactly
(0.0000 max abs error at every shape measured).

Neither provider wins everywhere, which is why this is added alongside rather than
instead -- and "alongside" needs `ops/torch/topk.py` to exist, because a vendor
provider *replaces* the base implementation instead of joining it. See that file.
Full measurement from tools/probe_topk_rotational.py:

    vocab     B    k   torch us   nkilib us   gain
    248320    1    1      587.7       674.8   0.88x
    248320    1    8      588.5       653.1   0.90x
    248320    1   50     5355.5       651.8   8.22x
    248320   64    1      640.7      2289.4   0.28x
    248320   64    8      630.2      2294.5   0.28x
    248320   64   50     5410.6      2274.9   2.38x
     62080    1    1      195.1       256.8   0.77x
     62080    1    8      193.3       273.9   0.75x
     62080    1   50     1401.6       240.4   5.83x
     62080   64    1      218.5       259.6   0.84x
     62080   64    8      194.2       253.3   0.77x
     62080   64   50     1383.8       389.5   3.55x

So the crossover is at `k` between 8 and 50, and `torch` keeps the small-`k` rows.
Batch also matters: this kernel's cost grows with `BxS` (652 us at batch 1 against
2275 us at batch 64 over the same vocabulary) where `torch.topk` is nearly flat in
batch, so at batch 64 it only wins the `k=50` row. `k=50` is what typical top-k
sampling uses, and batch 1 with the full vocabulary is the latency-critical case, so
the row it wins by 8.22x is the one that matters most.

`sorted` makes no measurable difference on either side (651.8 vs 657.8 us here), so the
cliff is not the sort. This provider passes `sorted=False` to match the op def.

Two things a caller beyond this benchmark needs to know:

* the returned indices carry nkilib's own `index_dtype` (`HW_PARAMS.index_dtype`), not
  the `torch.int64` the op def declares in `output_tensor_info`. That costs nothing
  here because `TopkOp` sets `create_outputs=False` and the run function returns its
  own tensors, but a sampler that feeds these into a gather has to cast.
* `num_programs` is handed `LNC` and the kernel lowers it to 1 by itself when
  `BxS == 1`, logging that it did. Passing 1 directly would be equivalent; passing
  `LNC` keeps the batched rows sharded.
"""
from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.backends.NEURON.backend_neuron import (
    RUNTIME_EAGER,
    detect_neuron_runtime,
)

# Must match __init__.py, which is where load_plugin_package() reads the provider
# name from. Declared again rather than imported: the plugin loader builds the
# package with spec_from_file_location and no submodule_search_locations.
PROVIDER_NAME = "nkilib"

try:
    if detect_neuron_runtime() != RUNTIME_EAGER:
        raise ImportError("nkilib rotational_topk requires the eager runtime")

    import nki.language as nl
    from nkilib.core.topk.rotational_topk import rotational_topk
    from nkilib.core.topk.rotational_topk_utils import (
        create_rotational_topk_config,
        create_topk_config,
    )
    from torch_neuronx.neuron_dynamo_backend import decompositions as _dc

    # Same subscript requirement as ops/nkilib/flash_attention.py: the kernel's
    # __getitem__ wants the bare int, and omitting it runs on one half of the LNC2
    # pair. See that file's docstring for the 1.85x trap this avoids.
    LNC = int(_dc.get_logical_neuron_cores())

    # Both entries must come from `nl`, not numpy. nki rejects a numpy dtype
    # outright -- `error: numpy dtypes are not supported as arguments; use
    # nki.language.float32 instead` -- and it does so at *lowering* time, inside
    # the kernel call, not when create_topk_config stores it. So np.float32 here
    # cost all 18 float32 cases with a RuntimeError from NkiDispatch while the 18
    # bfloat16 cases passed, which reads like a shape limitation rather than a
    # one-word type bug.
    _NKI_DTYPE = {
        "bfloat16": nl.bfloat16,
        "float32": nl.float32,
    }

    @ProviderRegistry.register_vendor_impl("topk", PROVIDER_NAME)
    class NkiLibRotationalTopkOp:
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            if self.dtype not in _NKI_DTYPE:
                raise ValueError(
                    f"rotational_topk is wired here for {sorted(_NKI_DTYPE)}, not "
                    f"{self.dtype}. float16 is left out because nothing in this "
                    "repo has verified it on this kernel, not because it is known "
                    "to fail."
                )

            # The factory asserts 2-D itself; check here so the message names the
            # op def's arguments rather than the kernel's internals.
            if self.k > self.dim_size:
                raise ValueError(
                    f"k {self.k} exceeds dim_size {self.dim_size}."
                )

        def vendor_impl(self):
            super().vendor_impl()

            shape = (self.batch_size, self.dim_size)
            # sorted=False matches TopkOp.vendor_impl_run, which asks torch for an
            # unsorted top-k. Measured to make no difference to this kernel either
            # way, so the choice is about comparing like with like.
            topk_config = create_topk_config(
                shape,
                _NKI_DTYPE[self.dtype],
                self.k,
                sorted=False,
                num_programs=LNC,
            )
            self.rotational_config = create_rotational_topk_config(
                shape, topk_config
            )

            self._run_func = self.rotational_topk_run

        def rotational_topk_run(self, tensor_mapping):
            src = tensor_mapping["src"]
            value, indice = rotational_topk[LNC](src, self.rotational_config)
            return value, indice

except Exception:
    pass
