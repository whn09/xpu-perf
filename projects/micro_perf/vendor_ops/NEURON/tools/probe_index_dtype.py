"""Why `gather` is ~450x off on Trainium, in one runnable file.

Background is in ../README.md, section "Two index ops are pathologically slow, and
one is an int64 index away from parity". Short version: the device has no int64. An
int64 index is converted on the way into the graph by a NKI custom call, and that
conversion *materialises* the index -- which destroys the stride-0 broadcast that
`GatherOp.create_tensors` builds with `.expand()`, and the fast lowering needs to
see that broadcast intact. Constructing the same index as int32 keeps it a view and
a direct graph argument, and gather gets 291-731x faster.

This probe measures that, plus the two controls that identify *layout* rather than
dtype as the proximate cause, plus the implementation-priority inspection that
explains why the NKI gather kernel does not win even when the index is right.

Run inside the eager image, on a machine with a free logical core:

    docker run --rm -it --device /dev/neuron0 \
        -v $PWD:/w -w /w xpu-perf-eager:latest \
        python3 vendor_ops/NEURON/tools/probe_index_dtype.py

`NEURON_RT_VISIBLE_CORES=<n>` pins it to one core if someone else holds the others.

Do not read this as a proposed fix to the benchmark. `torch.gather` requires an
int64 index on CPU and CUDA, so the op def keeps int64 and the published rows stay
as the honest cross-backend measurement. This is what the hardware can do, for
model code that owns its own indices.
"""
import time

import torch
import torch_neuronx  # noqa: F401  (registers the neuron device)

N = 1024
# 65536 is deliberately absent: ../README.md records `gather` wedging neuronx-cc
# for hours at that size, and a compile hang is not something a timeout survives.
DIMS = [128, 512, 1024, 4096, 8192, 32768]
DTYPES = [("float32", torch.float32), ("bfloat16", torch.bfloat16)]


def bench(fn, iters=10, warmup=2):
    """Timed loop with the syncs that make the number mean anything.

    Without `torch.neuron.synchronize()` this reports enqueue cost: an early
    version of this probe had `torch.gather` "taking" 59.5 us against a real
    6,343 us.
    """
    for _ in range(warmup):
        out = fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6, out


def make_index(dtype, dim, contiguous=False):
    idx = torch.arange(N, dtype=dtype, device="neuron").view(N, 1).expand(N, dim)
    return idx.contiguous() if contiguous else idx


print("=== gather: int64 index vs int32 index, same values, same shapes ===")
print(f"{'dtype':<9} {'dim':>6} {'int64 us':>11} {'int32 us':>11} {'speedup':>8} "
      f"{'int32 GB/s':>11} {'match':>6}")
for dname, dt in DTYPES:
    for dim in DIMS:
        src = torch.randn(N, dim, dtype=dt, device="neuron")
        io_bytes = src.element_size() * N * dim * 2
        lat, out = {}, {}
        for label, idt in (("int64", torch.int64), ("int32", torch.int32)):
            idx = make_index(idt, dim)
            try:
                lat[label], out[label] = bench(
                    lambda idx=idx: torch.gather(src, 0, idx)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {dname} dim={dim} {label} raised "
                      f"{type(exc).__name__}: {str(exc)[:70]}")
        if lat.get("int64") and lat.get("int32"):
            match = torch.allclose(
                out["int64"].to("cpu").float(), out["int32"].to("cpu").float()
            )
            print(f"{dname:<9} {dim:>6} {lat['int64']:>11.1f} {lat['int32']:>11.1f} "
                  f"{lat['int64'] / lat['int32']:>7.1f}x "
                  f"{io_bytes / lat['int32'] / 1e3:>11.2f} {str(match):>6}")

print()
print("=== control: is it the dtype or the layout? ===")
print("int32 + .contiguous() destroys the same broadcast the int64 conversion does,")
print("so if layout is what matters this row is slow again despite being int32.")
print(f"{'dtype':<9} {'dim':>6} {'int32 view us':>14} {'int32 contig us':>16} "
      f"{'penalty':>8}")
for dname, dt in DTYPES:
    for dim in [1024, 8192]:
        src = torch.randn(N, dim, dtype=dt, device="neuron")
        try:
            lv, _ = bench(
                lambda i=make_index(torch.int32, dim):
                torch.gather(src, 0, i)
            )
            lc, _ = bench(
                lambda i=make_index(torch.int32, dim, contiguous=True):
                torch.gather(src, 0, i)
            )
            print(f"{dname:<9} {dim:>6} {lv:>14.1f} {lc:>16.1f} "
                  f"{lc / lv:>7.1f}x")
        except Exception as exc:  # noqa: BLE001
            print(f"  {dname} dim={dim} raised {type(exc).__name__}: "
                  f"{str(exc)[:70]}")

print()
print("=== control: 1-D-index ops have nothing to materialise, so are unaffected ===")
for dname, dt in DTYPES:
    dim = 8192
    src = torch.randn(N, dim, dtype=dt, device="neuron")
    for label, idt in (("int64", torch.int64), ("int32", torch.int32)):
        idx1d = torch.arange(N, dtype=idt, device="neuron")
        try:
            lat, _ = bench(lambda i=idx1d: torch.index_select(src, 0, i))
            io_bytes = src.element_size() * N * dim * 2
            print(f"  index_select {dname:<9} {label:<6} {lat:>10.1f} us  "
                  f"{io_bytes / lat / 1e3:>8.2f} GB/s")
        except Exception as exc:  # noqa: BLE001
            print(f"  index_select {dname} {label} raised "
                  f"{type(exc).__name__}: {str(exc)[:70]}")

print()
print("=== scatter_add_ : the other 2-D-index op that has an NKI kernel ===")
print("(plain `scatter` has none -- aten::scatter_.src is unimplemented, which is")
print(" why its 621x is real and an int32 index does not move it)")
print(f"{'dtype':<9} {'dim':>6} {'int64 us':>11} {'int32 us':>11} {'speedup':>8}")
for dname, dt in DTYPES:
    for dim in [1024, 8192]:
        src = torch.randn(N, dim, dtype=dt, device="neuron")
        lat = {}
        for label, idt in (("int64", torch.int64), ("int32", torch.int32)):
            dst = torch.zeros(N, dim, dtype=dt, device="neuron")
            idx = make_index(idt, dim)
            try:
                lat[label], _ = bench(
                    lambda d=dst, i=idx: d.scatter_add_(0, i, src)
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {dname} dim={dim} {label} raised "
                      f"{type(exc).__name__}: {str(exc)[:70]}")
        if lat.get("int64") and lat.get("int32"):
            print(f"{dname:<9} {dim:>6} {lat['int64']:>11.1f} "
                  f"{lat['int32']:>11.1f} {lat['int64'] / lat['int32']:>7.1f}x")

print()
print("=== the second defect: @neuron_op(priority=) is dropped ===")
print("Declared priorities say the NKI impl should win. The registered instances all")
print("report the class default of 50, so ties fall to import order instead.")
from torch_neuronx.python_ops import auto_registration as ar  # noqa: E402

for op in ("aten::gather", "aten::scatter_add", "aten::scatter_add_",
           "aten::contiguous", "aten::copy_", "aten::_to_copy"):
    entry = ar._NEURON_OPS_REGISTRY.get(op)
    if not entry:
        print(f"  {op}: not registered")
        continue
    parts = []
    for obj in entry["implementations"]:
        cls = obj if isinstance(obj, type) else type(obj)
        declared = getattr(cls, "_auto_priority", None)
        try:
            inst = cls() if isinstance(obj, type) else obj
            prio = getattr(inst, "priority", None)
            effective = prio() if callable(prio) else prio
        except Exception:  # noqa: BLE001
            effective = "?"
        parts.append(f"{cls.__name__}(declared={declared}, effective={effective})")
    print(f"  {op}: " + ", ".join(parts))
