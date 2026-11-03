"""Numerical check of every NEURON vendor provider against a CPU fp32 reference.

Why this exists: `launch.py` only times. Nothing in micro_perf compares an output to
anything, so a provider that returns garbage -- a kernel that lost a transpose under a
new nki, a torch_neuronx lowering that changed -- still produces a clean latency row.
And every provider file wraps itself in `try: ... except Exception: pass`, so an
import that breaks under a new SDK does not fail either: the provider just stops
being registered and the sweep quietly measures one fewer column.

So this does two things, in order:

1. **Registration.** Lists which providers the plugin loader actually registered and
   compares against what this runtime should have. A missing one is reported with the
   exception its import raised, re-run outside the swallowing `try`.
2. **Numerics.** Builds each provider through the real `BackendNEURON` (same
   construction as `probe_case_overhead.py`), runs one case on device, and compares
   against fp32 on the host from the *same* inputs.

The pass criterion is not a hand-picked tolerance. For each case it also computes the
same reference on the host in the case's own low-precision dtype, which is the error a
correct implementation in that dtype is expected to have. A device result passes if
its relative error is within `--slack` times that floor (or under an absolute floor
for fp32, where the host floor is ~0 and any device accumulation order exceeds it).

Run from `projects/micro_perf`, on a free logical core:

    python vendor_ops/NEURON/tools/verify_providers.py --device 16

`XPU_PERF_NEURON_GEMM_COMPILE=1` is set by this script so the fp8 `torch_compile`
provider is registered too; its first case pays a neuronx-cc compile (minutes).
"""
import argparse
import importlib.metadata
import json
import os
import pathlib
import sys
import time
import traceback

os.environ.setdefault("XPU_PERF_NEURON_GEMM_COMPILE", "1")

import torch

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent.parent.parent
OPS_DIR = FILE_DIR.parent / "ops"

from xpu_perf.micro_perf.core.common_utils import get_submodules


# What should register on the PyTorch-native (eager) runtime. `nki` is the
# torch_xla-only provider and is expected to be absent here.
EXPECTED_EAGER = {
    "gemm": {"torch", "torch_compile"},
    "flash_attention": {"torch", "torch_tiled", "nkilib"},
    "rms_norm": {"torch", "nkilib"},
    "topk": {"torch", "nkilib"},
    "all_gather": {"torch"},
}

# provider directory -> the file defining it, for re-running a swallowed import.
PROVIDER_FILES = {
    ("flash_attention", "torch"): "torch/flash_attention.py",
    ("flash_attention", "torch_tiled"): "torch_tiled/flash_attention.py",
    ("flash_attention", "nkilib"): "nkilib/flash_attention.py",
    ("rms_norm", "nkilib"): "nkilib/rms_norm.py",
    ("topk", "nkilib"): "nkilib/topk.py",
    ("gemm", "torch_compile"): "torch_compile/gemm.py",
}


def attn(bs, cache_len, q_len, hq, hkv, d, mode):
    return {
        "arg_type": "llm", "attn_mode": mode,
        "dtype": "bfloat16", "cache_dtype": "bfloat16",
        "qk_compute_dtype": "bfloat16", "pv_compute_dtype": "bfloat16",
        "dst_dtype": "bfloat16",
        "q_head_num": hq, "kv_head_num": hkv, "head_dim": d,
        "batch_size": bs, "cache_len": cache_len, "q_len": q_len,
    }


# (op, provider, case). Shapes are the ones the README publishes or the Qwen3.5-27B
# workloads run, cut down where the host reference would otherwise dominate.
CASES = [
    # gemm, eager torch.matmul
    ("gemm", "torch", {"arg_type": "default", "dtype": "float32", "M": 1024, "K": 4096, "N": 4096}),
    ("gemm", "torch", {"arg_type": "default", "dtype": "float16", "M": 1024, "K": 4096, "N": 4096}),
    ("gemm", "torch", {"arg_type": "default", "dtype": "bfloat16", "M": 1024, "K": 4096, "N": 4096}),
    ("gemm", "torch", {"arg_type": "default", "dtype": "tfloat32", "M": 1024, "K": 4096, "N": 4096}),
    ("gemm", "torch", {"arg_type": "default", "dtype": "tfloat32", "M": 32768, "K": 1024, "N": 8192}),
    # eager fp8 returns wrong products (rel ~0.65); expected REJECTED, not run
    ("gemm", "torch", {"arg_type": "default", "dtype": "float8_e5m2", "dst_dtype": "bfloat16", "M": 1024, "K": 1024, "N": 1024}),
    # gemm, compiled fp8 (the only path that reaches the fp8 tensor engines)
    ("gemm", "torch_compile", {"arg_type": "default", "dtype": "float8_e5m2", "dst_dtype": "bfloat16", "M": 2048, "K": 2048, "N": 2048}),
    # rms_norm
    ("rms_norm", "torch", {"arg_type": "llm", "dtype": "bfloat16", "num_tokens": 4096, "hidden_size": 5120}),
    ("rms_norm", "torch", {"arg_type": "llm", "dtype": "float32", "num_tokens": 4096, "hidden_size": 5120}),
    ("rms_norm", "nkilib", {"arg_type": "llm", "dtype": "bfloat16", "num_tokens": 4096, "hidden_size": 5120}),
    ("rms_norm", "nkilib", {"arg_type": "llm", "dtype": "float32", "num_tokens": 4096, "hidden_size": 5120}),
    ("rms_norm", "nkilib", {"arg_type": "llm", "dtype": "bfloat16", "num_tokens": 1, "hidden_size": 5120}),
    ("rms_norm", "nkilib", {"arg_type": "llm", "dtype": "bfloat16", "num_tokens": 10240, "hidden_size": 5120}),
    # topk
    ("topk", "torch", {"arg_type": "default", "dtype": "bfloat16", "batch_size": 1, "dim_size": 248320, "k": 50}),
    ("topk", "torch", {"arg_type": "default", "dtype": "float32", "batch_size": 16, "dim_size": 62080, "k": 8}),
    ("topk", "nkilib", {"arg_type": "default", "dtype": "bfloat16", "batch_size": 1, "dim_size": 248320, "k": 50}),
    ("topk", "nkilib", {"arg_type": "default", "dtype": "bfloat16", "batch_size": 64, "dim_size": 62080, "k": 8}),
    ("topk", "nkilib", {"arg_type": "default", "dtype": "float32", "batch_size": 16, "dim_size": 62080, "k": 50}),
    ("topk", "nkilib", {"arg_type": "default", "dtype": "float32", "batch_size": 1, "dim_size": 248320, "k": 1}),
    # flash_attention, torch (SDPA; prefill hits the fused attention_cte rewrite)
    ("flash_attention", "torch", attn(1, 0, 2048, 8, 8, 128, "prefill")),
    ("flash_attention", "torch", attn(1, 0, 4096, 40, 8, 128, "prefill")),
    ("flash_attention", "torch", attn(16, 4095, 1, 80, 8, 128, "decode")),
    ("flash_attention", "torch", attn(1, 0, 2048, 24, 4, 256, "prefill")),
    # flash_attention, nkilib attention_tkg (decode, incl. the q_len 4 row)
    ("flash_attention", "nkilib", attn(16, 4095, 1, 80, 8, 128, "decode")),
    ("flash_attention", "nkilib", attn(16, 10239, 1, 80, 8, 128, "decode")),
    ("flash_attention", "nkilib", attn(16, 10236, 4, 80, 8, 128, "decode")),
    # flash_attention, torch_tiled (head_dim 256 prefill; needs >= 1 GiB of scores)
    ("flash_attention", "torch_tiled", attn(1, 0, 6144, 24, 4, 256, "prefill")),
]


# ---------------------------------------------------------------- references

def ref_gemm(op, x, dt):
    out = torch.matmul(x["a"].to(dt), x["b"].to(dt))
    return out


def ref_rms_norm(op, x, dt):
    src = x["src"].to(dt)
    w = x["weight"].to(dt)
    if dt != torch.float32:
        # what aten does for reduced precision: accumulate in fp32, round at the end
        out = torch.nn.functional.rms_norm(src.float(), [op.dim_size], w.float(), op.epsilon)
        return out.to(dt)
    return torch.nn.functional.rms_norm(src, [op.dim_size], w, op.epsilon)


def ref_attention(op, x, dt):
    b, s, hq, hkv, d = op.batch_size, op.q_lens[0], op.q_head_num, op.kv_head_num, op.head_dim
    kv_len, cache_len = op.kv_lens[0], op.cache_lens[0]
    q = x["q"].to(dt).view(b, s, hq, d).transpose(1, 2)              # b,hq,s,d
    k = x["k_cache"][:, :, :kv_len].to(dt)                            # b,hkv,kv,d
    v = x["v_cache"][:, :, :kv_len].to(dt)
    g = hq // hkv
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    out = torch.empty(b, hq, s, d, dtype=dt)
    # query i sits at absolute position cache_len + i: bottom-right aligned causal.
    qi = torch.arange(s).view(-1, 1) + cache_len
    ki = torch.arange(kv_len).view(1, -1)
    mask = ki <= qi
    # chunk heads so a 6144^2 x 24-head fp32 score matrix does not need 3.6 GB at once
    for h0 in range(0, hq, 4):
        h1 = min(h0 + 4, hq)
        sc = torch.matmul(q[:, h0:h1].float(), k[:, h0:h1].float().transpose(-1, -2)) * op.softmax_scale
        sc = sc.masked_fill(~mask, float("-inf"))
        p = torch.softmax(sc, dim=-1).to(dt)
        out[:, h0:h1] = torch.matmul(p, v[:, h0:h1]).to(dt)
    return out.transpose(1, 2).reshape(op.num_tokens, hq, d)


def compare(got, want):
    got = got.float()
    want = want.float()
    diff = (got - want)
    finite = bool(torch.isfinite(got).all())
    rel = (diff.norm() / want.norm().clamp_min(1e-30)).item()
    return {"finite": finite, "max_abs": diff.abs().max().item(), "rel": rel}


def check_topk(op, x, out):
    """Values must be the true top-k set; indices must point at those values."""
    val, idx = out
    val = val.cpu().float()
    idx = idx.cpu().long()
    src = x["src"].float()
    want = torch.topk(src, op.k, dim=-1, largest=True, sorted=True).values
    got_sorted = val.sort(dim=-1, descending=True).values
    in_range = bool(((idx >= 0) & (idx < op.dim_size)).all())
    gathered = src.gather(-1, idx.clamp(0, op.dim_size - 1))
    return {
        "finite": bool(torch.isfinite(val).all()),
        "value_max_abs": (got_sorted - want).abs().max().item(),
        "index_in_range": in_range,
        "index_value_max_abs": (gathered - val).abs().max().item(),
        "index_dtype": str(out[1].dtype),
        "unique_indices": bool(all(len(set(r.tolist())) == op.k for r in idx)),
    }


# ---------------------------------------------------------------- driver

def build_backend():
    names, mods = get_submodules("xpu_perf.micro_perf.backends")
    vendor_ops_dir = MICRO_PERF_DIR / "vendor_ops"
    env_path = vendor_ops_dir / "NEURON" / "env.json"
    return getattr(mods["NEURON"], "BackendNEURON")(
        backend="NEURON",
        env_file=str(env_path) if env_path.exists() else None,
        op_defs=MICRO_PERF_DIR / "op_defs",
        vendor_ops=[(vendor_ops_dir / "NEURON" / "ops").absolute()],
    )


def diagnose_missing(op, prov):
    """Re-run the provider file's imports without its blanket except."""
    rel = PROVIDER_FILES.get((op, prov))
    if rel is None:
        return "no diagnosis available"
    src = (OPS_DIR / rel).read_text()
    lines = [l.strip() for l in src.splitlines()
             if l.strip().startswith(("import ", "from ")) and "xpu_perf" not in l]
    for l in lines:
        try:
            exec(l, {})
        except Exception as e:
            return f"`{l}` -> {type(e).__name__}: {e}"
    return "all imports succeed; the provider's own module-level code raised"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=16, help="logical core index")
    ap.add_argument("--slack", type=float, default=3.0,
                    help="pass if device rel err <= slack x host same-dtype rel err")
    ap.add_argument("--only", default=None, help="comma list of op[:provider] to run")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    backend = build_backend()
    backend.load_all_ops()
    backend.set_device(args.device)
    runtime = getattr(backend, "neuron_runtime", "?")

    versions = {}
    for p in ("neuronx-cc", "torch-neuronx", "torch", "nki", "nkilib", "numpy"):
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = "-"
    print(f"runtime={runtime}  core={args.device}  " +
          "  ".join(f"{k}={v}" for k, v in versions.items()))

    # ---- 1. registration
    print("\n=== registration ===")
    reg_ok = True
    for op, want in EXPECTED_EAGER.items():
        have = set(backend.op_mapping.get(op, {}).keys())
        missing = want - have
        extra = have - want
        status = "OK" if not missing else "MISSING"
        reg_ok &= not missing
        print(f"{op:<16} have={sorted(have)}  {status}"
              + (f" missing={sorted(missing)}" if missing else "")
              + (f" extra={sorted(extra)}" if extra else ""))
        for m in sorted(missing):
            print(f"    {m}: {diagnose_missing(op, m)}")

    # ---- 2. numerics
    print("\n=== numerics (device vs host fp32; floor = host in case dtype) ===")
    only = None
    if args.only:
        only = {tuple(s.split(":")) if ":" in s else (s,) for s in args.only.split(",")}
    records = []
    for op_name, prov, case in CASES:
        if only and (op_name,) not in only and (op_name, prov) not in only:
            continue
        label = f"{op_name}/{prov} " + ",".join(
            f"{k}={v}" for k, v in case.items()
            if k not in ("arg_type", "cache_dtype", "qk_compute_dtype", "pv_compute_dtype", "dst_dtype"))
        rec = {"op": op_name, "provider": prov, "case": case}
        op_cls = backend.op_mapping.get(op_name, {}).get(prov)
        if op_cls is None:
            rec["status"] = "NOT_REGISTERED"
            print(f"NOT_REGISTERED  {label}")
            records.append(rec)
            continue
        try:
            op = op_cls(dict(case), backend)
        except (NotImplementedError, ValueError) as e:
            rec["status"] = "REJECTED"
            rec["reason"] = str(e)
            print(f"REJECTED        {label}\n    {e}")
            records.append(rec)
            continue

        try:
            torch.manual_seed(0)
            tensors = op.create_tensors(1)[0]
            # caches are created with torch.empty; give them real data
            for key in ("k_cache", "v_cache"):
                if key in tensors:
                    t = tensors[key]
                    t.copy_(torch.randn(t.shape, dtype=torch.float32).to(t.dtype).to(t.device))
            backend.device_synchronize()
            host_in = {k: v.detach().cpu() for k, v in tensors.items() if torch.is_tensor(v)}

            t0 = time.perf_counter()
            out = op.core_run(tensors)
            backend.device_synchronize()
            # second call: a stale/aliased output or an in-place clobber shows up here
            out2 = op.core_run(tensors)
            backend.device_synchronize()
            rec["first_call_s"] = round(time.perf_counter() - t0, 2)

            if op_name == "topk":
                r = check_topk(op, host_in, out)
                r2 = check_topk(op, host_in, out2)
                ok = (r["finite"] and r["value_max_abs"] == 0.0 and r["index_in_range"]
                      and r["index_value_max_abs"] == 0.0 and r["unique_indices"]
                      and r2["value_max_abs"] == 0.0)
                rec.update(r)
            else:
                ref_fn = {"gemm": ref_gemm, "rms_norm": ref_rms_norm,
                          "flash_attention": ref_attention}[op_name]
                low = op.torch_dtype if hasattr(op, "torch_dtype") else torch.bfloat16
                if low in (torch.float8_e5m2, torch.float8_e4m3fn):
                    low = torch.bfloat16      # fp8 inputs are exact in bf16; the
                                              # rounding is the bf16 output
                want = ref_fn(op, host_in, torch.float32)
                floor_t = ref_fn(op, host_in, low) if low != torch.float32 else want
                got = out.detach().cpu()
                got2 = out2.detach().cpu()
                r = compare(got, want)
                rf = compare(floor_t, want)
                r["floor_rel"] = rf["rel"]
                r["repeat_identical"] = bool(torch.equal(got, got2))
                r["shape_ok"] = list(got.shape) == list(want.shape)
                # fp32: host floor is 0, device sums in a different order (and may
                # use reduced-precision passes); 1e-3 is far below any layout bug,
                # which shows up as O(1).
                limit = max(args.slack * rf["rel"], 1e-3 if low == torch.float32 else 0.0)
                r["limit"] = limit
                ok = r["finite"] and r["shape_ok"] and r["rel"] <= limit
                if case.get("dtype") == "tfloat32":
                    # must be TF32 rounding (~1e-4), not exact fp32 (~1e-7): an
                    # exact result means the TF32 datapath was not engaged and the
                    # row would be an fp32 number under a tf32 label
                    r["tf32_engaged"] = r["rel"] > 1e-5
                    ok = ok and r["tf32_engaged"]
                rec.update(r)
            rec["status"] = "PASS" if ok else "FAIL"
        except Exception as e:
            rec["status"] = "ERROR"
            rec["reason"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc(limit=6)

        keys = [k for k in ("rel", "floor_rel", "tf32_engaged", "max_abs", "value_max_abs",
                            "index_value_max_abs", "index_dtype", "repeat_identical",
                            "first_call_s", "reason") if k in rec]
        print(f"{rec['status']:<15} {label}\n    " +
              "  ".join(f"{k}={rec[k]:.3g}" if isinstance(rec[k], float) else f"{k}={rec[k]}"
                        for k in keys))
        if rec["status"] == "ERROR":
            print(rec["trace"])
        records.append(rec)
        try:
            del tensors, out, out2
        except NameError:
            pass
        backend.empty_cache()

    print("\n=== summary ===")
    counts = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"registration {'OK' if reg_ok else 'MISSING PROVIDERS'};  " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"versions": versions, "runtime": runtime, "records": records}, f, indent=1)
    bad = (not reg_ok) or any(r["status"] in ("FAIL", "ERROR", "NOT_REGISTERED") for r in records)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
