"""Why `gelu` is ~52x off on Trainium while `silu` is at parity, in one runnable file.

Background is in ../../../workloads/models/qwen3_5_27b/README.md, section "`gelu` is
pinned at ~40 GB/s". The two op defs are byte-for-byte identical apart from the
function they call (`op_defs/basic_ops/vector_activation_ops.py`), both run the `base`
provider on both backends, and on one Trn2 logical core `silu` reaches 475-489 GB/s
while `gelu` is flat at 31-42 GB/s regardless of size. Nothing about the silicon
explains a 12-15x spread between two activations that read and write the same bytes,
so the difference has to be in what each one lowers to.

The hypothesis this probe tests: `torch.nn.functional.gelu` defaults to
`approximate='none'`, which is the **erf** formulation, and `erf` has no fast path on
this backend. `silu` is `x * sigmoid(x)`, and `sigmoid` does. If that is right then:

* `erf` alone should be as slow as `gelu`, and `sigmoid`/`tanh` alone should not;
* `gelu(approximate='tanh')` should be fast, which matters beyond benchmarking --
  Qwen3.5-27B's config says `hidden_act: gelu_pytorch_tanh`, so the tanh form is the
  one the model actually executes and the erf form is measuring a function no one
  runs.

Run inside the eager image, on a machine with a free logical core:

    docker run --rm -it --privileged -v $PWD:/w -w /w \
        -e PYTHONPATH=/xpu-perf/src xpu-perf-eager:latest \
        python3 vendor_ops/NEURON/tools/probe_gelu_lowering.py

`NEURON_RT_VISIBLE_CORES=<n>` pins it to one core if someone else holds the others.
Add `--dtype float32` to repeat in fp32; the default covers bf16 only, which is what
the vision tower runs.
"""
import argparse
import math
import time

import torch
import torch_neuronx  # noqa: F401  (registers the neuron device)

# The shapes the qwen3_5_27b comparison flagged. dim 4304 is the vision tower's
# intermediate_size and 1076 is its TP=4 share; batch is the token count.
SHAPES = [
    (1024, 1076),
    (1024, 4304),
    (4096, 1076),
    (4096, 4304),
    (16384, 1076),
    (16384, 4304),
]

SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


def bench(fn, iters=10, warmup=3):
    """Timed loop with the sync that makes the number mean anything.

    `torch.neuron.synchronize()` is not optional: without it this measures enqueue
    cost. Three warmups rather than two because the eager runtime compiles per
    *shape* on first sight (~2.9 s), and each formulation below is a new graph.
    """
    for _ in range(warmup):
        out = fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6, out


# Each entry is (label, fn, io_multiplier). The multiplier is how many tensor-sized
# reads+writes the formulation costs at minimum, so GB/s is comparable across rows:
# every one of these is a 1-in 1-out elementwise map, so it is always 2.
FORMS = [
    # What the harness measures today.
    ("gelu (erf, default)", lambda x: torch.nn.functional.gelu(x)),
    # What Qwen3.5-27B actually runs: config hidden_act = gelu_pytorch_tanh.
    ("gelu (approximate=tanh)",
     lambda x: torch.nn.functional.gelu(x, approximate="tanh")),
    # The tanh form written out, to separate "the fused op is slow" from "the
    # primitives it is built from are slow".
    ("tanh-gelu, hand-written",
     lambda x: 0.5 * x * (1.0 + torch.tanh(
         SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))),
    # Sigmoid-only formulation. Not numerically the same function -- max abs error
    # ~0.02 against erf-gelu -- but it is what several inference stacks ship as
    # "quick gelu", and it isolates whether sigmoid is the fast primitive.
    ("quick-gelu, x*sigmoid(1.702x)", lambda x: x * torch.sigmoid(1.702 * x)),
    # The fast control: same op def, same shapes, same provider.
    ("silu (control)", lambda x: torch.nn.functional.silu(x)),
    # Bare primitives, to locate the cost rather than infer it.
    ("erf alone", lambda x: torch.erf(x)),
    ("tanh alone", lambda x: torch.tanh(x)),
    ("sigmoid alone", lambda x: torch.sigmoid(x)),
    ("exp alone", lambda x: torch.exp(x)),
    ("mul alone (floor)", lambda x: x * 1.0001),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float32"])
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"=== gelu formulations and their primitives, {args.dtype}, "
          f"one logical core ===")
    print("GB/s counts one read + one write of the tensor, so every row is "
          "directly comparable.\n")

    hdr = f"{'shape':>14} " + "".join(f"{lbl.split(',')[0][:15]:>17}"
                                      for lbl, _ in FORMS)
    results = {}
    for rows, dim in SHAPES:
        x = torch.randn(rows, dim, dtype=dt, device="neuron")
        io_bytes = x.element_size() * rows * dim * 2
        line = {}
        for lbl, fn in FORMS:
            try:
                lat, _ = bench(lambda: fn(x), iters=args.iters)
                line[lbl] = (lat, io_bytes / lat / 1e3)
            except Exception as exc:  # a formulation the backend rejects is data
                line[lbl] = (float("nan"), float("nan"))
                print(f"  !! {lbl} at {rows}x{dim}: "
                      f"{type(exc).__name__}: {str(exc)[:90]}")
        results[(rows, dim)] = line

    # Per-shape table, one column per formulation, latency then GB/s.
    for lbl, _ in FORMS:
        print(f"\n--- {lbl}")
        print(f"{'shape':>14} {'latency us':>12} {'GB/s':>9} "
              f"{'x silu':>8}")
        for shape in SHAPES:
            lat, bw = results[shape][lbl]
            slat, _ = results[shape]["silu (control)"]
            print(f"{shape[0]:>7}x{shape[1]:<6} {lat:>12.1f} {bw:>9.1f} "
                  f"{lat / slat:>8.2f}")

    # The one-line answer.
    print("\n=== summary at the worst shape "
          f"({SHAPES[-1][0]}x{SHAPES[-1][1]}) ===")
    worst = results[SHAPES[-1]]
    base = worst["silu (control)"][0]
    print(f"{'formulation':<32} {'latency us':>12} {'GB/s':>9} {'x silu':>8}")
    for lbl, _ in FORMS:
        lat, bw = worst[lbl]
        print(f"{lbl:<32} {lat:>12.1f} {bw:>9.1f} {lat / base:>8.2f}")

    # Numerics, so a "fix" that changes the function is not mistaken for a free win.
    print("\n=== numerics vs gelu(erf) at 4096x1076, max abs error ===")
    x = torch.randn(4096, 1076, dtype=dt, device="neuron")
    ref = torch.nn.functional.gelu(x).to(torch.float32).cpu()
    for lbl, fn in FORMS[:5]:
        try:
            got = fn(x).to(torch.float32).cpu()
            print(f"  {lbl:<32} {(got - ref).abs().max().item():.5f}")
        except Exception as exc:
            print(f"  {lbl:<32} {type(exc).__name__}")


if __name__ == "__main__":
    main()
