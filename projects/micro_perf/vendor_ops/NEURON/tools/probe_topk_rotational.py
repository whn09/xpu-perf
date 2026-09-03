"""Why `topk` costs 9.5x more at `k=50` than at `k=8` on Trainium, and whether nkilib fixes it.

Background is in ../../../workloads/models/qwen3_5_27b/README.md, section "`topk` falls
off a cliff above `k = 8`". Measured on one Trn2 logical core over a 248320-entry
vocabulary, `torch.topk` costs 557 us at k=1, 558 us at k=8 and **5312 us at k=50** --
a 9.5x step -- at every vocabulary size and batch, while the H100 is flat in `k`
(90-193 us throughout). So there is a small-k path and a general sort above it.

`nkilib` ships a kernel aimed at exactly this shape: `core/topk/rotational_topk.py`,
whose own docstring names the reduced dimension "V: Vocabulary size". It offers three
methods (SCANNING, CASCADED, ROTATIONAL) behind one config factory. This probe asks
whether it clears the cliff, at the shapes Qwen3.5-27B's sampling step actually runs:
vocab 248320 at TP=1 and 62080 at TP=4, batch 1 and 64, k in {1, 8, 50}.

Two things it deliberately checks beyond latency:

* **Whether values *and* indices agree with `torch.topk`.** A sampling step needs the
  indices; a kernel that returns the right values against permuted indices is not a
  drop-in. Ties make exact index agreement unreasonable to demand, so the check
  compares gathered values rather than raw index equality.
* **`sorted=True` vs `sorted=False`.** Top-k sampling needs the set, not the order,
  and if the sort is what costs, that is a cheaper fix than a new kernel.

Run inside the eager image. Pass `--core N` to pin a specific logical core:

    docker run --rm -it --privileged -v ~/xpu-perf:/xpu-perf \\
        -w /xpu-perf/projects/micro_perf -e PYTHONPATH=/xpu-perf/src \\
        -e NEURON_RT_VISIBLE_CORES=1 xpu-perf-eager:latest \\
        python3 vendor_ops/NEURON/tools/probe_topk_rotational.py

Nothing here is wired into a provider yet -- this is the measurement that decides
whether it is worth wiring.
"""
import argparse
import time

import numpy as np
import torch
import torch_neuronx  # noqa: F401  (registers the neuron device)

VOCABS = [248320, 62080]
BATCHES = [1, 64]
KS = [1, 8, 50]


def bench(fn, iters=10, warmup=3):
    """`torch.neuron.synchronize()` is mandatory -- without it this times enqueue."""
    for _ in range(warmup):
        out = fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    try:
        import nki  # noqa: F401
        import nki.language as nl
        from nkilib.core.topk.rotational_topk import rotational_topk
        from nkilib.core.topk.rotational_topk_utils import (
            create_rotational_topk_config,
            create_topk_config,
        )
        from torch_neuronx.neuron_dynamo_backend import decompositions as _dc
        LNC = int(_dc.get_logical_neuron_cores())
        have_nki = True
    except Exception as exc:
        print(f"nkilib rotational_topk unavailable: {type(exc).__name__}: {exc}")
        have_nki = False
        LNC = 1

    # nki wants a numpy-side dtype; bfloat16 is not a numpy native, so nl supplies it.
    nki_dtype = nl.bfloat16 if args.dtype == "bfloat16" else np.float32

    print(f"=== topk over a vocabulary, {args.dtype}, one Trn2 logical core "
          f"(LNC={LNC}) ===\n")
    print(f"{'vocab':>8} {'B':>4} {'k':>4} {'torch us':>10} "
          f"{'nki sorted':>11} {'nki unsorted':>13} {'best gain':>10} {'match':>7}")

    for vocab in VOCABS:
        for B in BATCHES:
            x = torch.randn(B, vocab, dtype=dt, device="neuron")
            for k in KS:
                t_lat, (t_val, t_idx) = bench(
                    lambda: torch.topk(x, k, dim=-1), iters=args.iters)

                results = {}
                for want_sorted in (True, False):
                    if not have_nki:
                        results[want_sorted] = (float("nan"), None)
                        continue
                    try:
                        tc = create_topk_config(
                            (B, vocab), nki_dtype, k,
                            sorted=want_sorted, num_programs=LNC)
                        rc = create_rotational_topk_config((B, vocab), tc)
                        lat, out = bench(
                            lambda: rotational_topk[LNC](x, rc), iters=args.iters)
                        results[want_sorted] = (lat, out)
                    except Exception as exc:
                        results[want_sorted] = (float("nan"), None)
                        if k == KS[0] and B == BATCHES[0]:
                            print(f"  !! nki sorted={want_sorted} "
                                  f"vocab={vocab} k={k}: "
                                  f"{type(exc).__name__}: {str(exc)[:100]}")

                s_lat = results[True][0]
                u_lat = results[False][0]
                cand = [c for c in (s_lat, u_lat) if c == c]  # drop NaN
                gain = t_lat / min(cand) if cand else float("nan")

                # Correctness: compare the *values* the returned indices point at,
                # sorted descending, so ties do not make a correct kernel look wrong.
                match = "-"
                out = results[True][1]
                if out is not None:
                    try:
                        n_val = out[0].to(torch.float32).cpu().reshape(B, k)
                        ref = t_val.to(torch.float32).cpu().reshape(B, k)
                        err = (n_val.sort(dim=-1, descending=True).values
                               - ref.sort(dim=-1, descending=True).values
                               ).abs().max().item()
                        match = f"{err:.4f}"
                    except Exception as exc:
                        match = type(exc).__name__[:7]

                print(f"{vocab:>8} {B:>4} {k:>4} {t_lat:>10.1f} "
                      f"{s_lat:>11.1f} {u_lat:>13.1f} {gain:>9.2f}x {match:>7}")

    print("\nReading this: the torch column should show the k=8 -> k=50 cliff. "
          "If the nki columns are flat in k, the kernel is the fix; if only the "
          "unsorted column is flat, the sort is.")


if __name__ == "__main__":
    main()
