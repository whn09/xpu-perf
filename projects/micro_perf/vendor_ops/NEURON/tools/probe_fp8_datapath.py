"""Trainium2 does have an fp8 gemm. The published fp8 rows do not reach it.

The `gemm` fp8 rows in the README land at 1.2-1.5% of the 324.75 TFLOPS per-core
fp8 peak, which the README used to explain as "the lowering casts up to bf16 and
the cast dominates". The cast measurement was right and the conclusion drawn from
it was too broad: it is a property of the *eager* path, not of the hardware.

Two independent things are going on, and this probe separates them.

1. **Format.** There are two e4m3 encodings. `torch.float8_e4m3fn` is the OCP
   finite-only one -- no infinities, one NaN pattern -- and it is what CUDA uses
   and what `TORCH_DTYPE_MAPPING` gives `float8_e4m3`. Trainium1/2 multiply the
   *other* one, legacy `f8e4m3`: `nki.isa.nc_matmul` takes `float8_e4m3` and
   `float8_e5m2` on NeuronCore-v3 and adds `float8_e4m3fn` only "starting
   NeuronCore-v4" (v3 is Trn2, v4 is Trn3), and the two cannot be mixed in one
   matmul. The compiler says the same by name rather than failing obscurely:

       [NCC_EVRF051] Data type F8E4M3FN is not supported on TRN1/TRN2. Target
       TRN3 or later hardware, or use the --experimental-unsafe-fp8e4m3fn-as-fp8e4m3
       flag to cast F8E4M3FN to F8E4M3.

   (That flag *casts* OCP into the legacy encoding rather than reaching an OCP
   datapath -- hence `unsafe`, since inf/NaN and range differ -- and it does not
   exist in neuronx-cc 2.27.2878.0 either: `compile --help` has no fp8 options at
   all. So on this chip e4m3fn has no route to the tensor engines, and a newer SDK
   is not expected to change that.) `torch.float8_e5m2` has no such split and is supported directly.

2. **Eager vs compiled.** Eager has no fp8 gemm lowering for *either* format, so
   both fall onto the same software path at ~1 TFLOPS. Under
   `torch.compile(backend="neuron")` e5m2 reaches the tensor engines.

So the fp8 comparison in the README is not measuring what its heading claims. The
number to quote for "can Trainium2 do fp8" is the compiled e5m2 row.

Run inside the eager container with one Neuron device:

    sudo docker run --rm --device /dev/neuron0 -v "$PWD":/w -w /w \
        xpu-perf-eager:latest \
        python3 vendor_ops/NEURON/tools/probe_fp8_datapath.py

About 5 minutes. The e4m3fn compile is *expected* to fail; the failure is the
finding, so it is caught and printed rather than raised -- and it runs last,
because the failed compile latches an error that the next device op inherits.

The x4 to a per-chip figure is not covered here, since it needs four processes.
It checks out: four concurrent single-core runs of the compiled e5m2 4096^3 case
give 610.7 / 613.3 / 619.2 / 622.5 us against 611.6 us alone.

    for i in 0 1 2 3; do
      sudo docker run --rm --device /dev/neuron0 -e NEURON_RT_VISIBLE_CORES=$i \
        -v "$PWD":/w -w /w xpu-perf-eager:latest \
        python3 vendor_ops/NEURON/tools/probe_fp8_datapath.py &
    done; wait
"""
import time

import torch
import torch_neuronx  # noqa: F401

SIZES = (2048, 4096)
BF16_PEAK = 166.75      # TFLOPS per logical NeuronCore
FP8_PEAK = 324.75
DTYPES = (("bf16", torch.bfloat16, BF16_PEAK),
          ("e5m2", torch.float8_e5m2, FP8_PEAK),
          ("e4m3fn", torch.float8_e4m3fn, FP8_PEAK))


def log(*a):
    print(*a, flush=True)


def bench(fn, a, b, iters=5, warmup=2):
    for _ in range(warmup):
        fn(a, b)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(a, b)
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def body(x, y):
    """bf16 out, so every row is asked for the same thing the GPU provider's
    `torch._scaled_mm(..., out_dtype=bfloat16)` is asked for."""
    return torch.matmul(x, y).to(torch.bfloat16)


log("Square matmul, one logical NeuronCore, bf16 output.")
log(f"peaks: bf16 {BF16_PEAK} TF/core, fp8 {FP8_PEAK} TF/core")
log("")

results = {}
# e4m3fn's compiled row is deliberately last: the failed compile latches an error
# in the runtime that surfaces on the *next* device op, so running it mid-loop
# takes the rest of the sweep down with it.
for S in SIZES:
    flops = 2 * S ** 3
    log(f"{'=' * 78}")
    log(f"=== {S}x{S}x{S}  ({flops / 1e9:.1f} GFLOP)")
    log(f"{'=' * 78}")
    torch._dynamo.reset()
    a_bf = torch.randn(S, S, dtype=torch.bfloat16, device="neuron")
    b_bf = torch.randn(S, S, dtype=torch.bfloat16, device="neuron")

    for name, dt, peak in DTYPES:
        a = a_bf if dt is torch.bfloat16 else a_bf.to(dt)
        b = b_bf if dt is torch.bfloat16 else b_bf.to(dt)

        for mode in ("eager", "compiled"):
            if mode == "compiled" and name == "e4m3fn":
                continue
            # dynamic=False matters: compiling the same `body` at a second size
            # otherwise specialises it dynamically, and the Neuron compiler rejects
            # the result with "Dynamic shape is not supported: ... shape 'bf16[?,?]'".
            fn = (body if mode == "eager"
                  else torch.compile(body, backend="neuron", dynamic=False))
            t0 = time.perf_counter()
            out = fn(a, b)
            torch.neuron.synchronize()
            first = (time.perf_counter() - t0) * 1e3
            us = bench(fn, a, b)
            tf = flops / us / 1e6
            results[(S, name, mode)] = us
            log(f"  {name:<7} {mode:<9} {us:>10.1f} us  {tf:>7.2f} TF  "
                f"{tf / peak * 100:>5.1f}% of {peak:.0f}   "
                f"(first call {first:.0f} ms, out {out.dtype})")
    log("")

log(f"{'=' * 78}")
log("=== what this means for the published rows")
log(f"{'=' * 78}")
for S in SIZES:
    eb = results.get((S, "bf16", "eager"))
    ce = results.get((S, "e5m2", "compiled"))
    cb = results.get((S, "bf16", "compiled"))
    ee = results.get((S, "e5m2", "eager"))
    log(f"  {S}^3:")
    if ee and ce:
        log(f"    e5m2 eager -> compiled      {ee / ce:>6.1f}x   "
            "the published fp8 rows are the eager number")
    if cb and ce:
        log(f"    compiled e5m2 vs bf16       {cb / ce:>6.2f}x   "
            f"nominal fp8/bf16 headroom is {FP8_PEAK / BF16_PEAK:.2f}x")
    if eb and ce:
        log(f"    compiled e5m2 vs bf16 eager {eb / ce:>6.2f}x")
log("")
log(f"{'=' * 78}")
log("=== and the format fact, kept for last because it poisons the runtime")
log(f"{'=' * 78}")
S = SIZES[0]
a8 = torch.randn(S, S, dtype=torch.bfloat16, device="neuron").to(torch.float8_e4m3fn)
b8 = torch.randn(S, S, dtype=torch.bfloat16, device="neuron").to(torch.float8_e4m3fn)
try:
    torch.compile(body, backend="neuron", dynamic=False)(a8, b8)
    torch.neuron.synchronize()
    log("  e4m3fn compiled: it worked. That is new -- check the compiler version.")
except Exception as exc:  # noqa: BLE001
    key = "is not supported on"
    hit = next((ln.strip() for ln in str(exc).splitlines() if key in ln), None)
    log(f"  e4m3fn compiled -> {type(exc).__name__}")
    log(f"  {(hit or str(exc).splitlines()[0])[:200]}")

log("")
log("  e4m3fn does not compile at all: the encoding is not implemented on TRN1/TRN2.")
log("  So the honest cross-backend statement is that the H100's fp8 number is an")
log("  e4m3fn number, Trainium2 has no e4m3fn at all, and its fp8 number has to be")
log("  quoted in e5m2 -- which `torch._scaled_mm` in turn rejects on CUDA.")
