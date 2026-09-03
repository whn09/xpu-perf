"""`rms_norm` as one fused NKI kernel, because the aten lowering costs three passes.

`rms_norm` is a streaming op: read `[T, H]` plus a `[H]` weight, write `[T, H]`. On one
Trn2 logical core that should land wherever the other streaming ops land -- `silu`
reaches 451-475 GB/s and a bare `clone()` 579 -- and instead the op def's
`torch.nn.functional.rms_norm` sits at 184-186 GB/s, 26% of the ~725 GB/s a quarter
chip gets. Against an H100 (`vllm` provider, 2638 GB/s, i.e. 79% of its 3.35 TB/s) that
was 13.8x at `T=10240` bf16 and 14.4x at fp32.

Nothing here is the op def's fault, unlike `gelu`: it already calls the single fused
aten op (`op_defs/basic_ops/vector_norm_ops.py:130`), not a hand-rolled decomposition.
And no torch-level rewrite helps -- writing the same arithmetic out is 1.9x *worse* in
the input dtype and 3.5x worse with an fp32 reduction, because each intermediate is a
whole tensor through HBM. Measured at `T=10240, H=5120` bf16:

    F.rms_norm (what the op def calls)     1127.3 us   186.0 GB/s
    hand-written, native dtype            2099.6 us    99.9 GB/s
    hand-written, fp32 reduction          3824.7 us    54.8 GB/s
    reduce only: (x*x).mean(-1)            934.7 us   112.2 GB/s
    square only: x*x                       533.3 us   393.2 GB/s
    control: silu                          464.2 us   451.8 GB/s
    control: x * w                         564.1 us   371.8 GB/s
    control: x.clone()                     362.1 us   579.2 GB/s

The reduce-only row is the finding: **83% of the whole op's time is the row reduction**,
which by itself moves the input once and writes `[T, 1]`, and still only gets 112 GB/s.
So the cost is not the arithmetic and not the elementwise half -- it is that the
reduction is a separate pass over HBM, and the multiply is another one after it.

A kernel fixes exactly that: load a tile of rows into SBUF once, and do the square, the
row sum, the rsqrt and both multiplies on-chip before storing. Two nisa instructions do
most of it:

* `nisa.activation` squares **and** reduces along the free axis in the same pass
  (`reduce_op`/`reduce_res`), so `x^2` is never read back to be summed;
* `nisa.scalar_tensor_tensor` applies `(x * inv_rms) * gamma` in one pass, with the
  per-partition `[p, 1]` broadcast free.

Both providers through the harness, one `ONLY=qwen3_5_27b_norm` run of
../../../../workloads/models/qwen3_5_27b/norm_ops.json, against the best of the H100's
two providers on the same shape:

    dtype   T       torch     nkilib    GB/s    H100      was       now
    bf16        1    64.6 us    88.9 us    0.3   46.5 us   1.39x    1.91x
    bf16       16    58.2 us    85.5 us    4.0   36.0 us   1.62x    2.38x
    bf16       64    59.7 us    90.4 us   14.6   35.9 us   1.66x    2.52x
    bf16     1024    92.2 us    92.2 us  227.5   41.7 us   2.21x    2.21x
    bf16     4096   282.5 us   203.2 us  412.8   41.2 us   6.86x    4.93x
    bf16    10240  1095.3 us   475.8 us  440.7   79.5 us  13.78x    5.98x
    fp32        1    62.0 us    88.0 us    0.7   41.2 us   1.50x    2.14x
    fp32       16    66.1 us    83.3 us    8.1   32.4 us   2.04x    2.57x
    fp32       64    66.2 us    90.2 us   29.3   31.8 us   2.08x    2.84x
    fp32     1024   119.4 us   103.7 us  404.5   36.5 us   3.27x    2.84x
    fp32     4096   501.2 us   308.9 us  543.2   71.3 us   7.03x    4.33x
    fp32    10240  2178.5 us   763.0 us  549.8  150.9 us  14.44x    5.06x

440.7 GB/s is 98% of what `silu` reaches on the same core at the same dtype (451.8), so
this is the streaming roofline: what is left is the memory system, not the kernel. The
fp32 rows go higher still, 549.8. Divided by 4 for the per-chip claim the rest of these
READMEs make, 5.98x and 5.06x are 1.50x and 1.26x -- next to the 1.16x an HBM-bound op
should show (2.9 TB/s against 3.35).

**Below 1024 tokens this kernel loses**, by a flat 20-30 us: at 1-64 tokens the op moves
0.03-1.3 MB and both providers are measuring fixed cost, of which the nki path has more
(kernel launch, and the weight broadcast that is hoisted per call rather than per tile).
The crossover is at 1024 -- a tie in bf16, 1.15x in fp32 -- and it is left visible rather
than papered over with a `vendor_parser` rejection: both providers run every case, so the
results say where each wins. `topk` is registered the same way for the same reason.

Two details that are easy to get wrong:

* **the grid is not optional.** `logical-neuroncore-config` is 2 on a trn2.3xlarge, so
  one logical core is two physical halves. At `T=10240` bf16, `grid=1` gives 815.7 us
  and `grid=2` gives 502.8 -- 1.62x, for free, from a launch subscript. Same trap as
  ops/nkilib/flash_attention.py documents.
* **`nl.rms_norm` cannot be used.** nki 0.6.0 ships it as a tile primitive, but it
  hands its `[p, 1]` rsqrt straight to `nisa.tensor_tensor`, which rejects it: `'dst'
  free total elements 5120 != 'rhs' free total elements 1`. It fails at *lowering*
  time, inside the kernel call. So the broadcast has to be spelled out regardless,
  which is why the body below is explicit rather than a one-liner.
* **most of `nl` cannot be touched from `vendor_parser`.** It runs on the host, with
  no kernel being traced, and every `nl.tile_size` *property* resolves the NeuronCore
  generation through an nki backend that does not exist yet. Reading one there fails
  every case of the op before it runs, with an error that names none of this. See
  `_SBUF_BYTES_PER_PARTITION` below.

Numerics match aten to rounding: max abs error against an fp32 host reference is
0.01563 in bf16, where `F.rms_norm` itself gets 0.01562, and 0.00001 in fp32 against
its 0.00000. The reduction accumulates in fp32 here, as aten's does; the one place
that was worth 20x of fp32 error is commented at the rsqrt below.

This provider is added *alongside* the inherited implementation, not instead of it,
which needs `ops/torch/rms_norm.py` to exist -- a vendor provider replaces the base
one rather than joining it. See that file.
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
        raise ImportError("the fused rms_norm kernel requires the eager runtime")

    import nki
    import nki.isa as nisa
    import nki.language as nl
    from torch_neuronx.neuron_dynamo_backend import decompositions as _dc

    # Same subscript requirement as ops/nkilib/flash_attention.py, and worth 1.64x
    # here: see the docstring above.
    LNC = int(_dc.get_logical_neuron_cores())

    _NKI_DTYPE = {
        "bfloat16": nl.bfloat16,
        "float32": nl.float32,
    }

    # SBUF held while a tile is live, in bytes per element of one *row*: the
    # input, the fp32 squared scratch, the output, and the replicated weight.
    # SBUF is per-partition storage and a row sits inside one partition, so the
    # row length is what has to fit -- the 128 rows of a tile occupy 128
    # partitions, not 128 times one partition's budget. Used only to reject a
    # hidden size this kernel cannot tile, rather than letting the allocation
    # fail from inside the kernel.
    def _sbuf_bytes_per_elem(itemsize):
        return itemsize * 3 + 4

    # `nl.tile_size.total_available_sbuf_size` is the right constant and cannot be
    # read from here: it resolves the NeuronCore generation through the active nki
    # backend, which exists only while a kernel is being traced. Reading it on the
    # host raises "No backend set. Call _activate_backend() before using nki.isa
    # operations." (nki/_backends/__init__.py:38) -- from vendor_parser, i.e. at op
    # construction, so every case fails before it runs. `nl.tile_size.pmax` is a
    # plain class attribute and is safe; only the properties are not.
    #
    # So this is the gen2 entry of that module's own `_SBUF_FMAX_BYTES` table: the
    # smallest of the three generations, and therefore conservative on the
    # Trainium2 (gen3, 212984 B) these numbers were taken on. The margin only
    # matters for a hidden size between the two, which no shape in this repo has.
    _SBUF_BYTES_PER_PARTITION = 192 * 1024 - 16384

    def _shard(T, n_prg):
        """(chunk, p, n_full) for the token axis. All Python ints.

        `num_programs` is fixed at trace time, so every bound here is static and
        only the tile offset is symbolic. Deliberately raise-free: nki traces the
        source of everything a kernel calls and rejects a `raise` outright ("NKI
        does not support 'raise' statements"), so the evenness check that belongs
        with this arithmetic lives in `_splits_evenly`, which only the host calls.
        """
        chunk = T // n_prg
        p = min(nl.tile_size.pmax, chunk)
        return chunk, p, chunk // p

    def _splits_evenly(T, n_prg):
        """Whether `_shard` covers all of T with equal tiles. Host side only.

        The divisibility test has to come first, and not as a tidiness matter:
        `_shard(1, 2)` gets `chunk = 0`, hence `p = 0`, and divides by it. That is
        `num_tokens: 1` in every norm workload here, and it surfaced as
        "integer division or modulo by zero" at op construction.
        """
        if n_prg < 1 or T % n_prg != 0:
            return False
        chunk, p, n_full = _shard(T, n_prg)
        return n_full * p == chunk

    def _pick_grid(T):
        """Largest exact grid: the LNC halves if they divide T, else one.

        A ragged split would need a second, differently-shaped copy of the loop
        body -- nki refuses to trace a call to an inner function, so it cannot be
        factored out -- and at the token counts where the split is ragged the op
        is at fixed overhead anyway.
        """
        if LNC > 1 and _splits_evenly(T, LNC):
            return LNC
        return 1

    @nki.jit
    def rms_norm_kernel(src, gamma, eps):
        """`out[t, :] = src[t, :] * rsqrt(mean(src[t, :]^2) + eps) * gamma`.

        src:   [T, H] in HBM      gamma: [H] in HBM      out: [T, H] in HBM

        Rows go on the 128 partitions and H stays in the free dimension. nkilib's
        own `core/subkernels/rmsnorm_tkg.py` does the opposite -- H on partitions,
        returning `[128, T, H//128]` -- because it is written to feed a sharded
        matmul and wants that layout anyway. Here the consumer is the harness,
        which declares `dst` as `[T, H]`, so that layout would only move the cost
        into a transpose the timed region would then pay. Rows on partitions also
        keeps the reduction on the free axis, which is the cheap direction, and
        both DMAs contiguous.
        """
        T, H = src.shape
        out = nl.ndarray((T, H), dtype=src.dtype, buffer=nl.shared_hbm)

        n_prg = nl.num_programs(0)
        pid = nl.program_id(0)
        chunk, p, n_full = _shard(T, n_prg)

        # Hoisted: the weight is read and replicated once per call, not per tile.
        # Partition-dim broadcast is a real copy -- `NkiTensor.broadcast()` is
        # stride-0 but refuses dim 0 -- so this is p x H of SBUF, paid once.
        gb = nl.broadcast_to(nl.load(gamma.reshape((1, H))), (p, H))

        for i in nl.affine_range(n_full):
            off = pid * chunk + i * p
            x = nl.load(src[nl.ds(off, p), :])

            # `sq` is a required dst but its value is dead; only `ss` is read.
            # Accumulating the sum in fp32 is what aten does too, and it is what
            # keeps the error at bf16 rounding rather than above it.
            sq = nl.ndarray((p, H), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((p, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(sq, op=nl.square, data=x,
                            reduce_op=nl.add, reduce_res=ss)

            # `nisa.activation(op=nl.rsqrt, scale=1/H, bias=eps)` computes the
            # same thing in one instruction on the Scalar engine, and measured
            # 2.8% faster in bf16 and 0.8% in fp32 -- but its activation table
            # costs accuracy: max abs error against an fp32 host reference goes
            # from 0.00001 to 0.00021 in fp32, and from 0.01563 (which is exactly
            # what `F.rms_norm` itself gets) to 0.01575 in bf16. This tile is 128
            # values against 655360, so the pass is not where the time is, and a
            # provider that diverges from the reference by more than rounding is a
            # worse artifact than a 2% slower one.
            # `ss * (1.0 / H)` does not trace: nki's operator overloads want two
            # scalars or two tiles, not a tile and a float.
            inv = nl.rsqrt(nl.add(nl.multiply(ss, 1.0 / H), eps))

            y = nl.ndarray((p, H), dtype=src.dtype, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(y, data=x, op0=nl.multiply, operand0=inv,
                                      op1=nl.multiply, operand1=gb)
            nl.store(out[nl.ds(off, p), :], value=y)

        return out

    @ProviderRegistry.register_vendor_impl("rms_norm", PROVIDER_NAME)
    class NkiLibRMSNormOp:
        """Rejections are reported as unsupported rather than failing later.

        Every one of them is a case this kernel would have to grow a second code
        path for, and the `torch` provider already covers all of them.
        """

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def vendor_parser(self):
            super().vendor_parser()

            if self.dtype not in _NKI_DTYPE:
                raise ValueError(
                    f"the fused rms_norm kernel is wired here for "
                    f"{sorted(_NKI_DTYPE)}, not {self.dtype}. float16 is left out "
                    "because nothing in this repo has verified it, not because it "
                    "is known to fail."
                )

            if self.dst_dtype != self.dtype:
                raise ValueError(
                    f"the kernel writes its output in the input dtype, so "
                    f"dst_dtype {self.dst_dtype} != dtype {self.dtype} would need "
                    "a cast inside the timed region."
                )

            if self.add_residual:
                raise ValueError(
                    "add_residual is not fused in. `add_rms_norm` is a separate op "
                    "(op_defs/llm_ops/add_rms_norm.py) and should get its own "
                    "kernel rather than an extra input on this one."
                )

            self.grid = _pick_grid(self.batch_size)
            if not _splits_evenly(self.batch_size, self.grid):
                raise ValueError(
                    f"num_tokens {self.batch_size} does not split into equal "
                    f"{nl.tile_size.pmax}-row tiles; the kernel has no masked tail "
                    "tile."
                )

            itemsize = 2 if self.dtype == "bfloat16" else 4
            need = self.dim_size * _sbuf_bytes_per_elem(itemsize)
            if need > _SBUF_BYTES_PER_PARTITION:
                raise ValueError(
                    f"one row of hidden_size {self.dim_size} needs "
                    f"{need / 1024:.1f} KiB of the "
                    f"{_SBUF_BYTES_PER_PARTITION / 1024:.1f} KiB SBUF each "
                    "partition has; this kernel does not split the hidden "
                    "dimension."
                )

        def vendor_impl(self):
            super().vendor_impl()
            self._run_func = self.nki_rms_norm_run

        def nki_rms_norm_run(self, tensor_mapping):
            src = tensor_mapping["src"]
            weight = tensor_mapping["weight"]
            return rms_norm_kernel[self.grid](src, weight, self.epsilon)

except Exception:
    pass
