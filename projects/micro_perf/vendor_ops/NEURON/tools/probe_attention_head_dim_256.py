"""Can `head_dim 256` prefill be rescued without writing a NKI kernel?

Background is in ../../../workloads/models/qwen3_5_27b/README.md, section
"`head_dim: 256` has no fused attention path on Neuron at all". Both of Trainium2's
fused attention paths are gated on `head_dim <= 128` -- `nkilib` puts head_dim on 128
partitions (`P_MAX`), and torch_neuronx's SDPA rewrite gate requires `D <= 128` -- so
a model with `head_dim 256` (Qwen3.5-27B) misses both and SDPA falls to its math
decomposition. Measured cost of that: 13.7 ms at `q_len` 4096 and 282.6 ms at 10240,
against the H100's 364 us and 1836 us.

Splitting head_dim into two halves does not help, because softmax sits between the two
matmuls: `QK^T` can be accumulated as `Q1@K1^T + Q2@K2^T`, but the softmax needs the
whole score row before `P@V` can start, so there is no way to make two `D=128` fused
calls compose into one `D=256` result.

What might help instead is **tiling over the query axis**, which needs no kernel at
all. The observation that motivates it: cost grows 20.6x for a 2.5x length increase
(2.5^3.4), and pure O(n^2) traffic would predict 6.25x. Something is scaling worse than
the algorithm, and the candidates are (a) the score matrix itself -- 24 x 10240 x 10240
x 2 B = **5.03 GB** at the worst shape, against 24 GB of usable HBM per core -- and
(b) a materialised causal mask of the same shape in bool. Tiling the query axis caps
both at `tile / q_len` of that.

This probe measures the current path, the tiled path at several tile sizes, and two
controls that bound the answer from either side: the same shape at `head_dim 128`
(which reaches the fused kernel, so it is the target) and an explicitly-masked
unfused version (which is what the fallback is presumed to be doing).

Correctness is checked against the unfused reference, not asserted. Note that
`is_causal=True` cannot be reused per tile: PyTorch aligns the implied mask to the
**top-left** of a non-square score matrix, and a query tile against the whole prefix
needs bottom-right alignment, so the tiled path builds its mask explicitly.

Run inside the eager image, on a machine with a free logical core:

    docker run --rm -it --privileged -v ~/xpu-perf:/xpu-perf \
        -w /xpu-perf/projects/micro_perf -e PYTHONPATH=/xpu-perf/src \
        -e NEURON_RT_VISIBLE_CORES=0 xpu-perf-eager:latest \
        python3 vendor_ops/NEURON/tools/probe_attention_head_dim_256.py
"""
import argparse
import math
import time

import torch
import torch_neuronx  # noqa: F401  (registers the neuron device)
import torch.nn.functional as F

# Qwen3.5-27B full-attention layer at TP=1 and TP=4.
HEAD_SETS = [(24, 4, 256), (6, 1, 256)]
Q_LENS = [4096, 10240]
TILES = [512, 1024, 2048]


def bench(fn, iters=5, warmup=2):
    """`torch.neuron.synchronize()` is mandatory -- without it this times enqueue."""
    for _ in range(warmup):
        out = fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6, out


def sdpa_causal(q, k, v):
    """What the `torch` provider does today: one call, is_causal, D=256."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def sdpa_tiled(q, k, v, tile):
    """Tile the query axis; each tile attends to the prefix it is allowed to see.

    q/k/v are [B, H, S, D]. For query tile [t0, t1) the causal prefix ends at t1, so
    K/V are sliced to [0, t1) and the mask is causal only over the last
    `t1 - t0` columns -- everything before t0 is unconditionally visible. That is the
    bottom-right alignment `is_causal=True` would get wrong.
    """
    B, H, S, D = q.shape
    outs = []
    for t0 in range(0, S, tile):
        t1 = min(t0 + tile, S)
        qt = q[:, :, t0:t1, :]
        kt, vt = k[:, :, :t1, :], v[:, :, :t1, :]
        qi = torch.arange(t0, t1, device=q.device).view(-1, 1)
        ki = torch.arange(t1, device=q.device).view(1, -1)
        mask = ki <= qi
        outs.append(F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask))
    return torch.cat(outs, dim=2)


def unfused_ref(q, k, v):
    """The presumed shape of the fallback, written out, as a reference and a control."""
    B, H, S, D = q.shape
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)
    qi = torch.arange(S, device=q.device).view(-1, 1)
    ki = torch.arange(S, device=q.device).view(1, -1)
    scores = scores.masked_fill(ki > qi, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def flops(B, H, S, D):
    """Causal attention: 2 matmuls, 2 FLOP each, halved by causality."""
    return 2 * 2 * B * H * S * S * D * 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--skip-unfused", action="store_true",
                    help="the reference materialises 5 GB at the worst shape")
    args = ap.parse_args()

    print("=== head_dim 256 causal prefill on one Trn2 logical core, bf16 ===")
    print("peak per logical core = 166.75 TFLOPS; a chip is 4 of these.\n")
    print(f"{'heads':>10} {'q_len':>7} {'method':>16} {'latency us':>12} "
          f"{'TFLOPS':>8} {'MFU':>7} {'vs SDPA':>8}")

    for (qh, kvh, d) in HEAD_SETS:
        for S in Q_LENS:
            B = 1
            q = torch.randn(B, qh, S, d, dtype=torch.bfloat16, device="neuron")
            k = torch.randn(B, kvh, S, d, dtype=torch.bfloat16, device="neuron")
            v = torch.randn(B, kvh, S, d, dtype=torch.bfloat16, device="neuron")
            # SDPA broadcasts GQA itself only on some backends; expand explicitly so
            # every method below sees identical inputs.
            g = qh // kvh
            ke = k.repeat_interleave(g, dim=1)
            ve = v.repeat_interleave(g, dim=1)
            fl = flops(B, qh, S, d)

            rows = [("sdpa is_causal", lambda: sdpa_causal(q, ke, ve))]
            for t in TILES:
                if t < S:
                    rows.append((f"tiled q={t}",
                                 lambda t=t: sdpa_tiled(q, ke, ve, t)))
            if not args.skip_unfused:
                rows.append(("unfused ref", lambda: unfused_ref(q, ke, ve)))

            base = None
            for label, fn in rows:
                try:
                    lat, out = bench(fn, iters=args.iters)
                except Exception as exc:
                    print(f"{f'{qh}/{kvh}/{d}':>10} {S:>7} {label:>16} "
                          f"  !! {type(exc).__name__}: {str(exc)[:60]}")
                    continue
                if base is None:
                    base = lat
                tf = fl / lat / 1e6
                print(f"{f'{qh}/{kvh}/{d}':>10} {S:>7} {label:>16} {lat:>12.1f} "
                      f"{tf:>8.1f} {tf / 166.75:>7.3f} {base / lat:>7.2f}x")

    # The target: the same shapes at head_dim 128, where the fused kernel is legal.
    print("\n=== control: head_dim 128, same shapes, fused path is reachable ===")
    print(f"{'heads':>10} {'q_len':>7} {'latency us':>12} {'TFLOPS':>8} {'MFU':>7}")
    for (qh, kvh, _) in HEAD_SETS:
        for S in Q_LENS:
            d = 128
            q = torch.randn(1, qh, S, d, dtype=torch.bfloat16, device="neuron")
            k = torch.randn(1, qh, S, d, dtype=torch.bfloat16, device="neuron")
            v = torch.randn(1, qh, S, d, dtype=torch.bfloat16, device="neuron")
            lat, _ = bench(lambda: sdpa_causal(q, k, v), iters=args.iters)
            tf = flops(1, qh, S, d) / lat / 1e6
            print(f"{f'{qh}/{kvh}/{d}':>10} {S:>7} {lat:>12.1f} {tf:>8.1f} "
                  f"{tf / 166.75:>7.3f}")

    # Correctness of the tiled path, at a size the reference can afford.
    print("\n=== tiled vs unfused reference, 6/1/256 at q_len 2048 ===")
    S, qh, kvh, d = 2048, 6, 1, 256
    q = torch.randn(1, qh, S, d, dtype=torch.bfloat16, device="neuron")
    k = torch.randn(1, kvh, S, d, dtype=torch.bfloat16, device="neuron")
    v = torch.randn(1, kvh, S, d, dtype=torch.bfloat16, device="neuron")
    ke, ve = k.repeat_interleave(qh // kvh, 1), v.repeat_interleave(qh // kvh, 1)
    ref = unfused_ref(q, ke, ve).to(torch.float32).cpu()
    for t in (512, 1024):
        got = sdpa_tiled(q, ke, ve, t).to(torch.float32).cpu()
        print(f"  tile {t:>5}: max abs err {(got - ref).abs().max().item():.5f}")
    got = sdpa_causal(q, ke, ve).to(torch.float32).cpu()
    print(f"  sdpa      : max abs err {(got - ref).abs().max().item():.5f}")


if __name__ == "__main__":
    main()
