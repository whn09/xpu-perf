"""Where `rms_norm`'s 13.8-14.4x goes on Trainium2, and what recovers it.

Background is in ../../../workloads/models/qwen3_5_27b/README.md. `rms_norm` is a
pure streaming op -- read `[T, H]` plus a `[H]` weight, write `[T, H]` -- so on one
Trn2 logical core it should land wherever `silu` lands, and `silu` lands at 475-489
GB/s. Measured through the harness at `T=10240, H=5120`:

    bf16   Trn2  1098.6 us  =  190.9 GB/s      H100 (vllm)  79.5 us  =  2638 GB/s
    fp32   Trn2  2177.8 us  =  192.5 GB/s      H100 (vllm) 175.8 us  =  2385 GB/s

The H100 side is at 71-79% of its 3.35 TB/s, so it is not the interesting half. The
Trn2 side is at 26% of the ~725 GB/s a quarter-chip gets, and 40% of what the same
core reaches on `silu`. That residue is what this probe is for.

Note that unlike `gelu`, there is nothing to blame on the op def: it already calls
`torch.nn.functional.rms_norm`, i.e. the single fused aten op, not a hand-rolled
decomposition (`op_defs/basic_ops/vector_norm_ops.py:130`). So the question is not
"is the benchmark asking for the slow formulation" -- it is "what does this one op
lower to, and can any spelling of the same arithmetic beat it".

The candidates, in increasing order of how much they cost to adopt:

* a different **spelling in torch**. `aten::rms_norm` upcasts to fp32 internally for
  a reduced-precision input; if that upcast is materialised, a bf16 tensor becomes a
  4-byte one and the traffic roughly doubles before anything else happens. Writing
  the reduction out by hand in bf16 tests that.
* a different **place to reduce**. `mean(-1)` is a free-axis reduction, which is the
  cheap direction on this hardware; `sum(x*x)` materialising `x*x` first is not.
* a **fused NKI kernel**. nki 0.6.0 ships `nl.rms_norm` as a tile primitive, so this
  is about 15 lines, keeps the `[T, H]` layout on both sides, and loads each byte
  exactly once. This is the one that should hit the streaming roofline if the
  lowering is the problem.

The controls matter as much as the candidates: `silu` and a bare broadcast multiply
put a ceiling on the shape, and the reduce-only and square-only rows say which half
of the op the cost is in.

Run inside the eager image, on a machine with a free logical core:

    docker run --rm -it --privileged -v $PWD:/w -w /w \
        -e PYTHONPATH=/xpu-perf/src xpu-perf-eager:latest \
        python3 vendor_ops/NEURON/tools/probe_rms_norm_lowering.py

`NEURON_RT_VISIBLE_CORES=<n>` pins it to one core if someone else holds the others.
`--dtype float32` repeats in fp32; `--shapes` overrides the `T,H` list.
"""
import argparse
import math
import time

import torch
import torch_neuronx  # noqa: F401  (registers the neuron device)

# Qwen3.5-27B's hidden_size is 5120 and the token counts are the ones
# workloads/models/qwen3_5_27b/norm_ops.json sweeps. 5120 = 40 * 128, so it tiles
# on the partition dimension without padding, which is worth knowing before
# reading any NKI number below.
SHAPES = [
    (1024, 5120),
    (4096, 5120),
    (10240, 5120),
]

EPS = 1e-5


def bench(fn, iters=10, warmup=3):
    """Timed loop with the sync that makes the number mean anything.

    `torch.neuron.synchronize()` is not optional: without it this measures enqueue
    cost. Three warmups because the eager runtime compiles per *shape* on first
    sight (~2.9 s) and every formulation below is a new graph.
    """
    for _ in range(warmup):
        out = fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6, out


# ---------------------------------------------------------------------------
# The fused NKI candidate.
#
# Partitioning is on the token axis, not the hidden axis. nkilib's own
# `core/subkernels/rmsnorm_tkg.py` does the opposite -- it puts H on the 128
# partitions and returns `[128, T, H//128]` -- because it is written to feed a
# sharded matmul and wants that layout anyway. Here the consumer is the harness,
# which declares `dst` as `[T, H]`, so producing the transposed layout would only
# move the cost into a transpose that the timed region would then have to pay.
# Tokens on partitions keeps the reduction on the free axis, which is the cheap
# direction, and both DMAs contiguous.
# ---------------------------------------------------------------------------
NKI_AVAILABLE = True
try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    from torch_neuronx.neuron_dynamo_backend import decompositions as _dc

    # `logical-neuroncore-config: 2` on a trn2.3xlarge, so one logical core is two
    # physical halves and a grid of 1 would leave half of it idle. Same reason
    # ../../ops/nkilib/topk.py passes this as `num_programs`.
    LNC = int(_dc.get_logical_neuron_cores())

    def _shard(T, n_prg):
        """(chunk, p, n_full) for the token axis. All Python ints.

        `num_programs` is fixed at trace time, so every bound here is static and
        only the tile offset is symbolic. Deliberately raise-free: nki traces the
        source of everything a kernel calls and rejects a `raise` outright ("NKI
        does not support 'raise' statements"), so the evenness check that belongs
        with this arithmetic lives in `splits_evenly` instead, which only the host
        calls. A ragged split would need a second, differently-shaped copy of the
        loop body -- nki also refuses to trace a call to an inner function, so it
        cannot be factored out -- and `pick_grid` never asks for one.
        """
        chunk = T // n_prg
        p = min(nl.tile_size.pmax, chunk)
        return chunk, p, chunk // p

    def splits_evenly(T, n_prg):
        """Whether `_shard` covers all of T with equal tiles. Host-side only."""
        chunk, p, n_full = _shard(T, n_prg)
        return chunk * n_prg == T and n_full * p == chunk

    @nki.jit
    def rms_norm_nl(src, gamma, eps):
        """`out[t, :] = src[t, :] * rsqrt(mean(src[t, :]^2) + eps) * gamma`.

        The readable version, written with `nki.language` only. `nl.rms_norm`
        itself cannot be used at nki 0.6.0: it hands a `[p, 1]` rsqrt straight to
        `nisa.tensor_tensor`, which rejects it -- `'dst' free total elements 5120
        != 'rhs' free total elements 1` -- so the broadcast has to be spelled out
        here either way.

        src:   [T, H] in HBM      gamma: [1, H] in HBM      out: [T, H] in HBM
        """
        T, H = src.shape
        out = nl.ndarray((T, H), dtype=src.dtype, buffer=nl.shared_hbm)

        n_prg = nl.num_programs(0)
        pid = nl.program_id(0)
        chunk, p, n_full = _shard(T, n_prg)

        # Hoisted: the weight is read and replicated once per call, not per tile.
        # Partition-dim broadcast is a real copy -- `NkiTensor.broadcast()` is
        # stride-0 but refuses dim 0 -- so this is 128 x 10 KB of SBUF, once.
        gb = nl.broadcast_to(nl.load(gamma), (p, H))

        for i in nl.affine_range(n_full):
            off = pid * chunk + i * p
            x = nl.load(src[nl.ds(off, p), :])
            sq = nl.square(x, dtype=nl.float32)
            ss = nl.sum(sq, axis=[1], keepdims=True)
            # `ss * (1.0 / H)` does not trace -- nki's operator overloads want
            # two scalars or two tiles, not a tile and a float.
            inv = nl.rsqrt(nl.add(nl.multiply(ss, 1.0 / H), eps))
            y = nl.multiply(nl.multiply(x, inv.broadcast(1, H)), gb,
                            dtype=src.dtype)
            nl.store(out[nl.ds(off, p), :], value=y)

        return out

    @nki.jit
    def rms_norm_nisa(src, gamma, eps):
        """Same arithmetic, three instructions per tile instead of five.

        Two things `nki.language` cannot express, and both matter for a streaming
        op where every avoidable pass over SBUF is the whole cost:

        * `nisa.activation` squares *and* reduces along the free axis in one pass
          (`reduce_op`/`reduce_res`), so `x^2` is never read back to be summed.
        * `nisa.scalar_tensor_tensor` applies `(x * inv) * gamma` in one pass,
          with the per-partition `[p, 1]` broadcast free.

        src:   [T, H] in HBM      gamma: [1, H] in HBM      out: [T, H] in HBM
        """
        T, H = src.shape
        out = nl.ndarray((T, H), dtype=src.dtype, buffer=nl.shared_hbm)

        n_prg = nl.num_programs(0)
        pid = nl.program_id(0)
        chunk, p, n_full = _shard(T, n_prg)

        gb = nl.broadcast_to(nl.load(gamma), (p, H))

        for i in nl.affine_range(n_full):
            off = pid * chunk + i * p
            x = nl.load(src[nl.ds(off, p), :])

            # `sq` is a required dst but its value is dead; only `ss` is read.
            sq = nl.ndarray((p, H), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((p, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(sq, op=nl.square, data=x,
                            reduce_op=nl.add, reduce_res=ss)

            # activation computes op(scale * data + bias), so this is
            # rsqrt(sum/H + eps) with no extra instruction for the divide.
            inv = nl.ndarray((p, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(inv, op=nl.rsqrt, data=ss, scale=1.0 / H, bias=eps)

            y = nl.ndarray((p, H), dtype=src.dtype, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(y, data=x, op0=nl.multiply, operand0=inv,
                                      op1=nl.multiply, operand1=gb)
            nl.store(out[nl.ds(off, p), :], value=y)

        return out

    @nki.jit
    def rms_norm_nisa_nl_rsqrt(src, gamma, eps):
        """`rms_norm_nisa`, but the rsqrt comes from `nl` instead of the Scalar engine.

        Same two fused passes over the `[p, H]` tile; only the `[p, 1]` reciprocal
        square root changes hands. That tile is 128 values against 655360, so if
        this costs nothing and is more accurate, it is strictly better.
        """
        T, H = src.shape
        out = nl.ndarray((T, H), dtype=src.dtype, buffer=nl.shared_hbm)

        n_prg = nl.num_programs(0)
        pid = nl.program_id(0)
        chunk, p, n_full = _shard(T, n_prg)

        gb = nl.broadcast_to(nl.load(gamma), (p, H))

        for i in nl.affine_range(n_full):
            off = pid * chunk + i * p
            x = nl.load(src[nl.ds(off, p), :])

            sq = nl.ndarray((p, H), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((p, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(sq, op=nl.square, data=x,
                            reduce_op=nl.add, reduce_res=ss)

            inv = nl.rsqrt(nl.add(nl.multiply(ss, 1.0 / H), eps))

            y = nl.ndarray((p, H), dtype=src.dtype, buffer=nl.sbuf)
            nisa.scalar_tensor_tensor(y, data=x, op0=nl.multiply, operand0=inv,
                                      op1=nl.multiply, operand1=gb)
            nl.store(out[nl.ds(off, p), :], value=y)

        return out

    NKI_KERNELS = [("NKI, nki.language", rms_norm_nl),
                   ("NKI, nisa fused", rms_norm_nisa),
                   ("NKI, nisa fused + nl rsqrt", rms_norm_nisa_nl_rsqrt)]

    def pick_grid(T):
        """Largest exact grid: the LNC halves if they divide T, else one.

        At the token counts where the split is ragged (T < 256) the whole op is
        at fixed overhead anyway, so there is nothing to win by masking a tail.
        """
        if LNC > 1 and splits_evenly(T, LNC):
            return LNC
        return 1

except Exception as exc:  # pragma: no cover - reported, not raised
    NKI_AVAILABLE = False
    NKI_ERROR = exc


def torch_forms(dtype):
    """(label, fn(x, w), io_multiplier) for every torch-level spelling.

    `io_multiplier` is how many tensor-sized streams the row *must* move at
    minimum: 2 for a 1-in-1-out map (the weight is 1/T of a row and rounds away),
    1 for the reduce-only rows, which write only `[T, 1]`.
    """
    forms = [
        # What the harness measures today.
        ("F.rms_norm (what the op def calls)",
         lambda x, w: torch.nn.functional.rms_norm(
             x, [x.shape[-1]], weight=w, eps=EPS), 2),
        # Same arithmetic, written out, staying in the input dtype throughout. If
        # this wins, the aten op's internal fp32 upcast is the cost.
        ("hand-written, native dtype",
         lambda x, w: x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * w, 2),
        # Same, but with the reduction accumulated in fp32 -- the numerically
        # honest version, and the one that shows what the upcast costs when it is
        # confined to the reduction instead of the whole tensor.
        ("hand-written, fp32 reduction",
         lambda x, w: (
             x * torch.rsqrt(
                 (x.float() * x.float()).mean(-1, keepdim=True) + EPS
             ).to(x.dtype) * w), 2),
        # torch.mean over the last axis of x*x, then scale -- separates "the
        # reduction is slow" from "materialising x*x is slow".
        ("reduce only: (x*x).mean(-1)",
         lambda x, w: (x * x).mean(-1, keepdim=True), 1),
        ("square only: x*x",
         lambda x, w: x * x, 2),
        # Ceilings. Same bytes in, same bytes out, no reduction at all.
        ("control: silu",
         lambda x, w: torch.nn.functional.silu(x), 2),
        ("control: x * w (broadcast multiply)",
         lambda x, w: x * w, 2),
        ("control: x.clone() (pure copy)",
         lambda x, w: x.clone(), 2),
    ]
    return forms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--shapes", default=None,
                    help="semicolon-separated T,H pairs, e.g. '4096,5120;10240,5120'")
    args = ap.parse_args()

    shapes = SHAPES
    if args.shapes:
        shapes = [tuple(int(v) for v in s.split(",")) for s in args.shapes.split(";")]

    dtype = getattr(torch, args.dtype)
    itemsize = torch.tensor([], dtype=dtype).element_size()

    dev = "neuron"
    print(f"dtype={args.dtype}  itemsize={itemsize}  device={dev}")
    if not NKI_AVAILABLE:
        print(f"NKI kernel unavailable: {NKI_ERROR!r}")

    for T, H in shapes:
        x = torch.randn(T, H, dtype=dtype).to(dev)
        w = torch.ones(H, dtype=dtype).to(dev)
        w2d = w.view(1, H)

        # Reference on the host, fp32, so every row below can be checked rather
        # than assumed. bf16 rounding alone is ~4e-3 here.
        xc = x.cpu().float()
        ref = (xc * torch.rsqrt(xc.pow(2).mean(-1, keepdim=True) + EPS)) * w.cpu().float()

        nominal = T * H * itemsize
        print(f"\n=== T={T} H={H}  ({nominal / (1 << 20):.1f} MiB per stream) ===")
        print(f"{'form':<40} {'us':>9} {'GB/s':>8} {'vs F.rms_norm':>14} {'max abs err':>12}")

        base_us = None
        for label, fn, mult in torch_forms(dtype):
            try:
                us, out = bench(lambda: fn(x, w), iters=args.iters)
            except Exception as exc:
                print(f"{label:<40} {'FAILED':>9}  {exc}")
                continue
            if base_us is None:
                base_us = us
            gbs = nominal * mult / (us * 1e-6) / 1e9
            err = "-"
            if mult == 2 and out.shape == x.shape:
                err = f"{(out.cpu().float() - ref).abs().max().item():.5f}"
            print(f"{label:<40} {us:>9.1f} {gbs:>8.1f} "
                  f"{base_us / us:>13.2f}x {err:>12}")

        if NKI_AVAILABLE:
            for name, kern in NKI_KERNELS:
                for grid in sorted({1, pick_grid(T)}):
                    label = f"{name}, grid={grid}"
                    try:
                        us, out = bench(
                            lambda: kern[grid](x, w2d, EPS), iters=args.iters)
                        gbs = nominal * 2 / (us * 1e-6) / 1e9
                        err = (out.cpu().float() - ref).abs().max().item()
                        print(f"{label:<40} {us:>9.1f} {gbs:>8.1f} "
                              f"{base_us / us:>13.2f}x {err:>12.5f}")
                    except Exception as exc:
                        print(f"{label:<40} {'FAILED':>9}  {type(exc).__name__}: "
                              f"{str(exc)[:160]}")


if __name__ == "__main__":
    main()
