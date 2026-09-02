"""Are `store_kv_cache`'s and `rotary_embedding`'s in-place writes actually in-place?

Background in ../README.md, sections "Writes into a strided slice view are not
in-place" and "`rotary_embedding` is not a Neuron result at all".

Two op defs write into a *strided slice view* of a large pre-allocated tensor and
count only the slice in `write_bytes`:

    dst_k_cache = k_cache[kv_slot_id, :, cache_start:cache_end, :]  # store_kv_cache.py:276
    dst_k_cache.copy_(src_k_data)                                   # store_kv_cache.py:280
    packed_qkv[t0:t1, qk0:qk1, d0:d1].copy_(rotate(...))            # rotary_embedding.py:173

`store_kv_cache` says of itself "This operator is inplace." If the eager stack
functionalises that into "copy the whole buffer, then write the slice", latency is
O(whole tensor) while io_bytes is O(slice), so mem_bw is understated by the ratio
and the op looks like a Neuron deficit when it is a semantics mismatch.

Part 1 is the slope test, which needs no profiler: hold the update size fixed and
scan the buffer size. True in-place is a flat line, a full-buffer copy is linear.
`add_` is the positive control -- it is *supposed* to be O(buffer), so its slope is
the yardstick for what a degraded case looks like.

Part 2 decomposes `rotary_embedding` at the shape the README publishes, because the
slope test cannot be applied to it directly: its prefill cases have
`q_len == num_tokens`, so buffer and update grow together and O(buffer) shows up as
a merely constant bandwidth. Part 2 is what establishes that the in-place problem is
only 15% of that op's cost and `rotate()` is 78%.

Run inside the eager image, on a machine with a free logical core:

    docker run --rm -it --device /dev/neuron0 \
        -v $PWD:/w -w /w xpu-perf-eager:latest \
        python3 vendor_ops/NEURON/tools/probe_inplace_write.py

`NEURON_RT_VISIBLE_CORES=<n>` pins it to one core if someone else holds the others.
Part 2 allocates a 240 MB tensor plus temporaries; it needs a few GB of HBM free.
"""
import time

import torch
import torch_neuronx  # noqa: F401  (registers the neuron device)

Q_HEADS, KV_HEADS, HEAD_DIM = 80, 8, 128
TOTAL_HEADS = Q_HEADS + 2 * KV_HEADS        # 96, as the op defs build it
QK_HEADS = Q_HEADS + KV_HEADS               # 88, the range rotary writes
DT = torch.bfloat16

# k_cache is [1, KV_HEADS, max_kv_len, HEAD_DIM] bf16 = 2048 B per cache token.
CACHE_LENS = [4096, 16384, 65536]           # 8 MB -> 128 MB, 16x
UPD_TOKENS = 128                            # fixed 256 KB of new K
PEAK_GBS = 725.0                            # a quarter of the chip's 2.9 TB/s


def bench(fn, iters=10, warmup=3):
    """Timed loop with the syncs that make the number mean anything.

    Without `torch.neuron.synchronize()` this reports enqueue cost, not work.
    """
    for _ in range(warmup):
        fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3      # ms


def cache_mb(n_tokens):
    return n_tokens * KV_HEADS * HEAD_DIM * 2 / 2**20


def report(name, lats):
    """Slope over the 16x buffer range, with a verdict.

    A true in-place write is flat. A full-buffer copy would be 16x if it were
    purely bandwidth-bound; it is not, because there is ~0.1 ms of fixed overhead
    at these sizes, so the threshold is set well below 16 rather than at it.
    """
    slope = lats[-1] / lats[0]
    verdict = ("flat = true in-place" if slope < 2.0
               else "scales with the BUFFER, not the update")
    cells = "  ".join(f"{v:>9.3f}" for v in lats)
    print(f"  {name:<44} {cells}   {slope:>6.2f}x  {verdict}")


print("=" * 100)
print("PART 1 -- slope test: fixed update, growing buffer")
print("=" * 100)
print("cache sizes: " + ", ".join(f"{cache_mb(n):.0f} MB" for n in CACHE_LENS)
      + f"  ({CACHE_LENS[-1] // CACHE_LENS[0]}x)   fixed update = "
      f"{UPD_TOKENS * KV_HEADS * HEAD_DIM * 2 / 1024:.0f} KB")
print(f"{'':<46} " + "  ".join(f"{cache_mb(n):>7.0f}MB" for n in CACHE_LENS)
      + "   slope   verdict")
print()
print("--- store_kv_cache's exact write: strided 4-D slice view + copy_ ---")

lats = []
for n in CACHE_LENS:
    k_cache = torch.empty(1, KV_HEADS, n, HEAD_DIM, dtype=DT, device="neuron")
    # src built the way the op def builds it: slice packed qkv, contiguous, transpose
    packed = torch.randn(UPD_TOKENS, TOTAL_HEADS * HEAD_DIM, dtype=DT,
                         device="neuron")
    k0 = Q_HEADS * HEAD_DIM
    k1 = k0 + KV_HEADS * HEAD_DIM
    cs, ce = n // 2, n // 2 + UPD_TOKENS    # non-zero start, as during decode

    def step(k_cache=k_cache, packed=packed, cs=cs, ce=ce):
        src = packed[:, k0:k1].contiguous().view(
            UPD_TOKENS, KV_HEADS, HEAD_DIM).permute(1, 0, 2)
        k_cache[0, :, cs:ce, :].copy_(src)

    lats.append(bench(step))
report("k_cache[0,:,cs:ce,:].copy_(src)", lats)

print()
print("--- controls ---")

lats = []
for n in CACHE_LENS:
    k_cache = torch.empty(1, KV_HEADS, n, HEAD_DIM, dtype=DT, device="neuron")
    lats.append(bench(lambda c=k_cache: c.add_(1.0)))
report("cache.add_(1.0)  [O(buffer) by design]", lats)

lats = []
for n in CACHE_LENS:
    k_cache = torch.empty(1, KV_HEADS, n, HEAD_DIM, dtype=DT, device="neuron")
    cs, ce = n // 2, n // 2 + UPD_TOKENS
    lats.append(bench(lambda c=k_cache, cs=cs, ce=ce:
                      c[0, :, cs:ce, :].contiguous()))
report("read the slice, write nothing", lats)

lats = []
for n in CACHE_LENS:
    k_cache = torch.empty(1, KV_HEADS, n, HEAD_DIM, dtype=DT, device="neuron")
    src = torch.randn(KV_HEADS, UPD_TOKENS, HEAD_DIM, dtype=DT, device="neuron")
    lats.append(bench(lambda c=k_cache, s=src: c[0, :, 0:UPD_TOKENS, :].copy_(s)))
report("the same copy_ but at offset 0", lats)

lats = []
for n in CACHE_LENS:
    # 2-D contiguous destination: the simplest possible slice write, and the
    # reference point for what the strided 4-D view itself costs.
    flat = torch.empty(n, KV_HEADS * HEAD_DIM, dtype=DT, device="neuron")
    src = torch.randn(UPD_TOKENS, KV_HEADS * HEAD_DIM, dtype=DT, device="neuron")
    cs, ce = n // 2, n // 2 + UPD_TOKENS
    lats.append(bench(lambda f=flat, s=src, cs=cs, ce=ce: f[cs:ce].copy_(s)))
report("2-D CONTIGUOUS dst, flat[cs:ce].copy_(src)", lats)

print()
print("--- does torch.compile(backend='neuron') establish the aliasing? ---")
print("Dynamo input/output aliasing is the usual advice for this. Note where it")
print("stops working.")
for n in CACHE_LENS:
    k_cache = torch.empty(1, KV_HEADS, n, HEAD_DIM, dtype=DT, device="neuron")
    src = torch.randn(KV_HEADS, UPD_TOKENS, HEAD_DIM, dtype=DT, device="neuron")
    cs, ce = n // 2, n // 2 + UPD_TOKENS

    def f(cache, s, cs=cs, ce=ce):
        cache[0, :, cs:ce, :].copy_(s)
        return cache

    try:
        cf = torch.compile(f, backend="neuron")
        lat = bench(lambda: cf(k_cache, src))
        print(f"  compile, cache {cache_mb(n):>7.0f} MB: {lat:>8.3f} ms")
    except Exception as exc:  # noqa: BLE001
        print(f"  compile, cache {cache_mb(n):>7.0f} MB: {type(exc).__name__}: "
              f"{str(exc)[:80]}")

print()
print("=" * 100)
print("PART 2 -- rotary_embedding decomposed at the published shape")
print("=" * 100)
print("The slope test does not apply here: the prefill cases have")
print("q_len == num_tokens, so buffer and update grow together and an O(buffer)")
print("cost shows up as a merely constant bandwidth. So time the body instead.")
print()
print("Note what is NOT in the timed region: cos and sin are precomputed by")
print("precompute_freqs_cis when the tensors are created, and rotate() is only")
print("cat/mul/add -- no trig runs here at all.")
print()

N = 10240      # the published case: batch_size 1, cache_len 0, q_len 10240
packed = torch.randn(N, TOTAL_HEADS, HEAD_DIM, dtype=DT, device="neuron")
cos = torch.randn(N, HEAD_DIM, dtype=DT, device="neuron")
sin = torch.randn(N, HEAD_DIM, dtype=DT, device="neuron")

slice_mb = N * QK_HEADS * HEAD_DIM * 2 / 2**20
full_mb = N * TOTAL_HEADS * HEAD_DIM * 2 / 2**20
one_pass_us = slice_mb * 2**20 / (PEAK_GBS * 1e9) * 1e6
print(f"packed_qkv = {full_mb:.1f} MB, the qk slice written = {slice_mb:.1f} MB")
print(f"one full pass over the slice at {PEAK_GBS:.0f} GB/s = {one_pass_us:.0f} us")
print()


def rotate(qk, c, s):
    """core/utils.py:587, inlined so this file is self-contained."""
    rope_dim = qk.size(-1)
    left_part = qk[:, :, :rope_dim // 2]
    right_part = qk[:, :, rope_dim // 2:]
    return (torch.cat([left_part, right_part], dim=-1) * c.unsqueeze(1)
            + torch.cat([-right_part, left_part], dim=-1) * s.unsqueeze(1))


def body():
    """rotary_embedding.py:171-175, one batch."""
    target_qk = packed[0:N, 0:QK_HEADS, 0:HEAD_DIM].contiguous()
    packed[0:N, 0:QK_HEADS, 0:HEAD_DIM].copy_(rotate(target_qk, cos, sin))


contig = packed[0:N, 0:QK_HEADS, 0:HEAD_DIM].contiguous()
flat_dst = torch.empty(N, QK_HEADS, HEAD_DIM, dtype=DT, device="neuron")

steps = [
    ("whole vendor_impl_run body", body),
    (".contiguous() on the strided slice",
     lambda: packed[0:N, 0:QK_HEADS, 0:HEAD_DIM].contiguous()),
    ("rotate() on already-contiguous input", lambda: rotate(contig, cos, sin)),
    ("copy_ back INTO the strided slice view",
     lambda: packed[0:N, 0:QK_HEADS, 0:HEAD_DIM].copy_(contig)),
    ("  -- same bytes into a contiguous dst (control)",
     lambda: flat_dst.copy_(contig)),
    ("one elementwise pass, reference (mul)", lambda: contig * 2.0),
]

print(f"{'':<46} {'us':>10} {'GB/s over the slice':>21}")
results = {}
for name, fn in steps:
    lat_us = bench(fn, iters=5, warmup=2) * 1e3
    results[name] = lat_us
    bw = slice_mb * 2**20 * 2 / lat_us / 1e3    # read + write of the slice
    print(f"{name:<46} {lat_us:>10.1f} {bw:>21.1f}")

total = results["whole vendor_impl_run body"]
for name in (".contiguous() on the strided slice",
             "rotate() on already-contiguous input",
             "copy_ back INTO the strided slice view"):
    print(f"  {name:<52} {results[name] / total * 100:>5.1f}% of the body")

print()
print("--- is it the strided destination, or something else about the write? ---")
print("packed_qkv has 96 heads and the op writes heads 0:88, so the destination")
print("view skips 8 heads and is not contiguous. Size the buffer to exactly 88")
print("heads and the identical write becomes contiguous.")
exact = torch.randn(N, QK_HEADS, HEAD_DIM, dtype=DT, device="neuron")
for label, dst in (("96-head buffer, strided dst", packed),
                   ("88-head buffer, contiguous dst", exact)):
    lat_us = bench(lambda d=dst: d[0:N, 0:QK_HEADS, 0:HEAD_DIM].copy_(contig),
                   iters=5, warmup=2) * 1e3
    print(f"  {label:<34} {lat_us:>10.1f} us  "
          f"{slice_mb * 2**20 * 2 / lat_us / 1e3:>7.1f} GB/s")
