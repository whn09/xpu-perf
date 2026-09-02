"""Which kernel backs the prefill attention rows, and is there a faster one?

The README once said the Neuron eager runtime has no fused flash kernel. That was
false: torch_neuronx rewrites SDPA to a NKI kernel in its own dynamo backend, and
disabling the rewrite costs 6.41x. The follow-up question is whether the kernel it
picks is the *best* one available -- nkilib ships `attention_cte`, a
context-encoding attention kernel, and that is the obvious candidate.

This probe answers it by identity rather than by benchmark. torch_neuronx's
`decompositions.py` does:

    from nkilib.core.attention.attention_cte import attention_cte   # line 24
    wrapped_flash_fwd = wrap_nki(attention_cte)                     # line 71
    ...
    grid = (int(get_logical_neuron_cores()),)                       # line 894
    output = wrapped_flash_fwd[grid](q_nki, k_nki, v_nki,
                                     tp_q=True, tp_k=True, scale=scale,
                                     causal_mask=is_causal,
                                     cache_softmax=False)           # line 935

so the published prefill number already *is* `attention_cte`. Section 1 checks
that by object identity against the kernel imported straight from nkilib, instead
of inferring it from timings.

Sections 2-4 then time three ways of reaching that kernel at the shape the README
publishes, because two of them are traps:

  - SDPA, which is what the op def calls.
  - `attention_cte` launched the way torch_neuronx launches it: `lnc` set to
    `logical_neuron_cores`, `tp_k=True`, and KV left at its own head count so the
    kernel does the GQA replication.
  - `attention_cte` launched the obvious way -- `lnc` left at its default,
    `tp_k=False`, KV pre-expanded to the query head count. This is the trap:
    without `lnc` the kernel runs on one half of the LNC2 pair, and the 1.85x that
    costs is easy to misread as "the nkilib kernel is slower than what SDPA gets".
    Note the subscript takes the bare int: `attention_cte[2]`, since
    `attention_cte[(2,)]` raises `NkiValidationError: NKI only supports LNC 1 or
    2, but got (2,)`. torch_neuronx passes a tuple because its own `wrap_nki`
    wrapper unpacks it.

Run inside the eager container with one Neuron device:

    sudo docker run --rm --device /dev/neuron0 -v "$PWD":/w -w /w \
        xpu-perf-eager:latest \
        python3 vendor_ops/NEURON/tools/probe_attention_kernel.py

Roughly 4 minutes, most of it the first-call kernel compile.
"""
import math
import time

import torch
import torch_neuronx  # noqa: F401

B, HQ, HKV, L, D = 1, 80, 8, 4096, 128
SCALE = 1.0 / math.sqrt(D)
# Causal attention does half the dense work. Both are printed, because a flash
# kernel that skips fully-masked tiles should be scored causal and a naive one
# that computes them should be scored dense, and the ratio between the two
# columns is exactly 2x either way.
FLOPS_CAUSAL = 2 * B * HQ * L * L * D
FLOPS_DENSE = 4 * B * HQ * L * L * D
PEAK = 166.75      # bf16 TFLOPS per logical NeuronCore


def log(*a):
    print(*a, flush=True)


def bench(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.neuron.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def report(label, us):
    log(f"  {label:<46} {us:>9.1f} us  "
        f"{FLOPS_CAUSAL / us / 1e6:>6.2f} TF causal ({FLOPS_CAUSAL / us / 1e6 / PEAK * 100:>4.1f}% of {PEAK:.0f})  "
        f"{FLOPS_DENSE / us / 1e6:>6.2f} TF dense")


log(f"B={B} HQ={HQ} HKV={HKV} L={L} D={D}, bf16, causal -- the GQA prefill row")
log(f"causal FLOPs {FLOPS_CAUSAL / 1e9:.1f} G, dense {FLOPS_DENSE / 1e9:.1f} G")
log("")

log("=== 1. is the SDPA rewrite attention_cte? (identity, not inference) ===")
from torch_neuronx.neuron_dynamo_backend import decompositions as dc
from nkilib.core.attention.attention_cte import attention_cte


# decompositions.py line 24 imports the symbol into its own namespace and line 71
# wraps that object, so this is an object-identity check on the kernel itself, not
# a name comparison and not an inference from timings.
log(f"  decompositions.attention_cte is nkilib's attention_cte: "
    f"{dc.attention_cte is attention_cte}")
log(f"  the object it wraps it into : {type(dc.wrapped_flash_fwd).__name__}"
    " (a NKI higher-order-op caller)")
inner = [v for v in vars(dc.wrapped_flash_fwd).values() if v is attention_cte]
log(f"  and that caller still holds the same object: {bool(inner)}")
log("  -> the published prefill rows are an attention_cte score already; the"
    " remaining\n     gap to the H100 is this kernel against cuDNN/FA, not a"
    " missing kernel.")

# torch_neuronx passes grid=(lnc,) to its wrapper; the raw nki kernel's __getitem__
# wants the bare int (`replace(self, lnc=lnc)`), and a tuple raises
# NkiValidationError: NKI only supports LNC 1 or 2.
lnc = int(dc.get_logical_neuron_cores())
log(f"  logical_neuron_cores = {lnc}, so torch_neuronx launches with grid=({lnc},)")

log("")
torch.manual_seed(0)
q_s = torch.randn(B, HQ, L, D, dtype=torch.bfloat16, device="neuron")
k_s = torch.randn(B, HKV, L, D, dtype=torch.bfloat16, device="neuron")
v_s = torch.randn(B, HKV, L, D, dtype=torch.bfloat16, device="neuron")
# The flash_attention op def expands KV to the query head count before calling
# SDPA, so the baseline has to do the same to stay comparable.
k_e = k_s.repeat_interleave(HQ // HKV, dim=1)
v_e = v_s.repeat_interleave(HQ // HKV, dim=1)

log("=== 2. SDPA, which is what the op def calls ===")
sdpa = torch.nn.functional.scaled_dot_product_attention


def run_sdpa():
    return sdpa(q_s, k_e, v_e, is_causal=True, scale=SCALE)


ref = run_sdpa()
torch.neuron.synchronize()
us_sdpa = bench(run_sdpa)
report("SDPA (kv pre-expanded to 80 heads)", us_sdpa)


def check(label, out):
    o = out["out"] if isinstance(out, dict) else out
    o = o[0] if isinstance(o, (tuple, list)) else o
    got = o.reshape(B, HQ, L, D).float().cpu()
    want = ref.float().cpu()
    err = (got - want).abs()
    rel = (err.mean() / want.abs().mean().clamp_min(1e-6)).item()
    log(f"  {label}: max abs {err.max().item():.4f}  rel {rel:.5f}  "
        f"{'OK' if rel < 0.02 else '*** MISMATCH, timing below is meaningless ***'}")


log("")
log("=== 3. attention_cte launched the way torch_neuronx launches it ===")
q_n = q_s.reshape(B * HQ, L, D)
k_n = k_s.reshape(B * HKV, L, D)          # tp_k=True, so no manual transpose
v_n = v_s.reshape(B * HKV, L, D)
log(f"  q {tuple(q_n.shape)}  k {tuple(k_n.shape)}  v {tuple(v_n.shape)}  "
    f"lnc={lnc} tp_k=True")


def run_cte_grid():
    return attention_cte[lnc](q_n, k_n, v_n, tp_q=True, tp_k=True, scale=SCALE,
                              causal_mask=True, cache_softmax=False)


t0 = time.perf_counter()
out = run_cte_grid()
torch.neuron.synchronize()
log(f"  first call (incl. kernel compile): {(time.perf_counter() - t0) * 1e3:.1f} ms")
check("vs SDPA", out)
us_grid = bench(run_cte_grid)
report("attention_cte[lnc], tp_k=True, kernel GQA", us_grid)

log("")
log("=== 4. the trap: no lnc, tp_k=False, KV pre-expanded ===")
q_t = q_s.reshape(B * HQ, L, D).contiguous()
k_t = k_s.reshape(B * HKV, L, D).transpose(1, 2).contiguous()   # tp_k=False
v_t = v_s.reshape(B * HKV, L, D).contiguous()


def run_cte_nogrid():
    return attention_cte(q_t, k_t, v_t, scale=SCALE, causal_mask=True,
                         tp_q=True, tp_k=False, tp_out=False)


t0 = time.perf_counter()
out = run_cte_nogrid()
torch.neuron.synchronize()
log(f"  first call (incl. kernel compile): {(time.perf_counter() - t0) * 1e3:.1f} ms")
check("vs SDPA", out)
us_nogrid = bench(run_cte_nogrid)
report("attention_cte, default lnc", us_nogrid)

log("")
log("=== summary ===")
report("SDPA (the published path)", us_sdpa)
report("attention_cte[lnc] (the same kernel, by hand)", us_grid)
report("attention_cte, default lnc", us_nogrid)
log("")
log(f"  lnc vs SDPA    : {us_grid / us_sdpa:.2f}x  "
    "(same kernel, so ~1.00x is the expected answer)")
log(f"  no-lnc penalty : {us_nogrid / us_grid:.2f}x  "
    f"(passing lnc={lnc} is not optional on an LNC{lnc} core)")
