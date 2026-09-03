"""Where a micro_perf case's wall clock goes: device time vs. harness setup.

Cross-backend on purpose -- `--backend NEURON` and `--backend GPU` both work, and
the point is to run the same probe on both. It lives here rather than under
`GPU/tools/` for the same reason `analyze_sweep.py` does: the Neuron side is where
the comparison tooling grew up.

Why this exists
---------------
A full `basic/tensor_gemm_ops/gemm.json` sweep takes ~403 s on one H100 and
~4,200 s on one logical NeuronCore, which invites "Trainium is 10x slower at
gemm". It is not what that ratio measures. Two things are in the way:

1. The per-case latencies in the two reports sum to 1.8 s (GPU) and 6.9 s
   (Neuron). Whatever the rest of the wall clock is, it is not the timed region.
2. The two backends do not run the same policy. `core/backend.py:334-336` gives
   each case a 50 ms / 10-iteration timed loop; `backends/NEURON/backend_neuron.py:595-597`
   gives it a **1 s / 50-iteration** one, and `:651` sleeps 0.2 s per case where
   `core/backend.py:367` sleeps 0.1 s. The Neuron backend also caps the tensor
   copies at 4 (`backend_neuron.py:619`) where the GPU path will build up to
   1 GiB worth (`core/backend.py:346-349`). So sweep durations were never
   comparable, by construction.

Rather than argue from the source, this replays the **real** `Backend.perf()` on
chosen cases from a real workload file, with its phases instrumented, and prints
the split. Every number below is measured; the only inference is the `other`
column, which is the residual.

Phases reported, in the order `perf()` runs them:

    construct   op_cls(case, backend) -- arg parsing, vendor_impl, any compile
    create      create_tensors(max_data_cnt): copy 1 via the creator (for float
                dtypes `core/utils.py:47` = CPU fp32 randn -> H2D -> on-device
                cast), copies 2..N via on-device clone (`core/op.py:274`)
    sync        the device_synchronize() calls perf() makes outside core_perf
    calib       core_perf(warmup=2, iters=2) -- the run that picks prefer_iters
    sleep       the fixed time.sleep between the two core_perf calls
    timed       core_perf(warmup=2, iters=prefer_iters) -- the reported number
    cache       empty_cache()
    other       wall clock minus all of the above (get_mem_info, summary, GC)

`calib + timed` is the only device-execution time in a case, and it is
`(6 + prefer_iters)` executions of the op -- not one.

Usage
-----
Neuron, inside the eager image, on a free core:

    docker run --rm -it --device /dev/neuron0 -v $PWD:/w -w /w \
        -e PYTHONPATH=/w/../../src xpu-perf-eager:latest \
        python3 vendor_ops/NEURON/tools/probe_case_overhead.py \
            --backend NEURON --workload workloads/basic/tensor_gemm_ops/gemm.json

H100, against the DLAMI interpreter:

    PYTHONPATH=$HOME/xpudeps:$(cd ../.. && pwd)/src /opt/pytorch/bin/python \
        vendor_ops/NEURON/tools/probe_case_overhead.py \
            --backend GPU --workload workloads/basic/tensor_gemm_ops/gemm.json

Both default to 8 cases spread evenly across the file, which is ~1 min on Neuron
and a few seconds on an H100. `--limit 0` runs every case, i.e. the whole sweep.
Run it from `projects/micro_perf`, like `launch.py`.

Read the totals line, not individual cases: `create` is dominated by the largest
tensor in the case, so a file whose shapes span two orders of magnitude has cases
that look nothing like each other.
"""
import argparse
import json
import pathlib
import resource
import statistics
import sys
import time

import torch

FILE_DIR = pathlib.Path(__file__).parent.absolute()
# vendor_ops/NEURON/tools -> projects/micro_perf
MICRO_PERF_DIR = FILE_DIR.parent.parent.parent

from xpu_perf.micro_perf.core.common_utils import get_submodules, parse_workload
from xpu_perf.micro_perf.core.utils import (
    CREATOR_MAPPING, default_creator, float_creator, get_numa_info,
)


PHASES = ["construct", "create", "sync", "calib", "sleep", "timed", "cache", "other"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose per-case wall clock of a micro_perf workload."
    )
    parser.add_argument("--backend", type=str, default="NEURON")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--workload",
        type=str,
        default="workloads/basic/tensor_gemm_ops/gemm.json",
        help="Same file launch.py --workload takes.",
    )
    parser.add_argument(
        "--op",
        type=str,
        default=None,
        help="Which op in the file. Default: every op it defines.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Which vendor provider. Default: every registered one, which is "
             "what the sweep does.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Sample this many cases, spread evenly across the file. 0 = all.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Comma-separated case indices, overriding --limit.",
    )
    parser.add_argument(
        "--no-stages",
        action="store_true",
        help="Skip the extra float_creator stage breakdown (randn / H2D / cast).",
    )
    parser.add_argument(
        "--no-affinity",
        action="store_true",
        help="Do not pin to numa node 0's cores. The real run pins "
             "(core/backend.py:405), and CPU-side phases care.",
    )
    parser.add_argument("--json", type=str, default=None, help="Dump raw records here.")
    return parser.parse_args()


def build_backend(backend_type):
    """Same construction XpuPerfServer does, minus the process spawning."""
    backend_name_list, backend_mod_list = get_submodules("xpu_perf.micro_perf.backends")
    if backend_type not in backend_name_list:
        sys.exit(f"unknown backend {backend_type!r}; have {backend_name_list}")

    vendor_ops_dir = MICRO_PERF_DIR.joinpath("vendor_ops")
    env_path = vendor_ops_dir.joinpath(backend_type, "env.json")
    backend_class = getattr(backend_mod_list[backend_type], "Backend" + backend_type)
    return backend_class(
        backend=backend_type,
        env_file=str(env_path) if env_path.exists() else None,
        op_defs=MICRO_PERF_DIR.joinpath("op_defs"),
        vendor_ops=[vendor_ops_dir.joinpath(backend_type, "ops").absolute()],
    )


def pin_to_numa0():
    """Reproduce compute_infer_loop's affinity, or say why it could not."""
    try:
        import psutil

        _, numa_configs = get_numa_info()
        if not numa_configs:
            return "no numa info"
        cores = numa_configs[0]
        psutil.Process().cpu_affinity(cores)
        return f"{len(cores)} cores on numa node 0"
    except Exception as exc:  # a container without the sysfs, most likely
        return f"failed ({exc})"


def tensor_bytes(info):
    numel = 1
    for dim in info.shape:
        numel *= dim
    return numel * torch.empty(0, dtype=info.dtype).element_size()


def effective_creator(info):
    """Which creator actually runs for this tensor.

    Almost no op def sets `creator=` -- `OpTensorInfo` defaults to
    `default_creator` (`core/utils.py:124`), which dispatches on the torch dtype
    through `CREATOR_MAPPING`. So `gemm`'s bf16 operands do go through
    `float_creator`, but you cannot see that by comparing the `creator` field.
    """
    if info.creator is default_creator:
        return CREATOR_MAPPING.get(info.dtype)
    return info.creator


def biggest_input(op_instance):
    """The tensor whose creation dominates `create`, or None."""
    candidates = list(op_instance.input_tensor_info.items())
    candidates += list(op_instance.output_tensor_info.items())
    candidates = [(k, v) for k, v in candidates if v.device != "cpu"]
    if not candidates:
        return None
    return max(candidates, key=lambda kv: tensor_bytes(kv[1]))


def stage_breakdown(backend, info, repeats=3):
    """Time float_creator's three stages separately.

    `core/utils.py:47` chains them with no sync in between, so on a backend with
    async copies the real split is blurrier than this -- the totals still add up,
    but treat the H2D/cast columns as indicative. A `.to()` that is a no-op
    (already the target dtype) shows up as ~0, which is itself worth seeing.
    """
    out = {"randn_ms": [], "h2d_ms": [], "cast_ms": [], "clone_ms": []}
    for _ in range(repeats):
        t0 = time.perf_counter()
        host = torch.randn(size=info.shape, dtype=torch.float32)
        t1 = time.perf_counter()
        on_device = host.to(device=info.device)
        backend.device_synchronize()
        t2 = time.perf_counter()
        cast = on_device.to(dtype=info.dtype)
        backend.device_synchronize()
        t3 = time.perf_counter()
        # What copies 2..max_data_cnt cost each (core/op.py:274). Device-local,
        # so this is the one stage that says whether max_data_cnt matters.
        extra = cast.clone()
        backend.device_synchronize()
        t4 = time.perf_counter()
        del host, on_device, cast, extra
        backend.empty_cache()
        out["randn_ms"].append((t1 - t0) * 1e3)
        out["h2d_ms"].append((t2 - t1) * 1e3)
        out["cast_ms"].append((t3 - t2) * 1e3)
        out["clone_ms"].append((t4 - t3) * 1e3)
    return {k: min(v) for k, v in out.items()}


class InstrumentedPerf:
    """Wrap the phases of `Backend.perf` without changing what it does.

    The wrappers are instance attributes shadowing the class methods, so the
    `self.core_perf(...)` / `self.device_synchronize()` calls inside `perf()`
    land here. `device_synchronize` is also called *from inside* `core_perf`
    (`core/backend.py:301,305,309`), so a depth counter keeps those out of the
    `sync` column -- they belong to the core_perf measurement that contains them.

    `time.sleep` is patched on the `time` module, which is the same object the
    backend imported.
    """

    def __init__(self, backend):
        self.backend = backend
        self.orig = {}
        self.depth = 0
        self.rec = {}

    def __enter__(self):
        b = self.backend
        self.orig = {
            "core_perf": b.core_perf,
            "device_synchronize": b.device_synchronize,
            "empty_cache": b.empty_cache,
            "sleep": time.sleep,
        }

        def core_perf(op_instance, warmup, iters, tensor_list, **kwargs):
            self.depth += 1
            t0 = time.perf_counter()
            try:
                return self.orig["core_perf"](
                    op_instance, warmup, iters, tensor_list, **kwargs
                )
            finally:
                self.depth -= 1
                self.rec["core_perf_calls"].append(
                    {
                        "warmup": warmup,
                        "iters": iters,
                        "wall_ms": (time.perf_counter() - t0) * 1e3,
                    }
                )

        def device_synchronize(*a, **kw):
            if self.depth:
                return self.orig["device_synchronize"](*a, **kw)
            t0 = time.perf_counter()
            try:
                return self.orig["device_synchronize"](*a, **kw)
            finally:
                self.rec["sync"] += (time.perf_counter() - t0) * 1e3

        def empty_cache(*a, **kw):
            t0 = time.perf_counter()
            try:
                return self.orig["empty_cache"](*a, **kw)
            finally:
                self.rec["cache"] += (time.perf_counter() - t0) * 1e3

        def sleep(seconds):
            t0 = time.perf_counter()
            try:
                return self.orig["sleep"](seconds)
            finally:
                self.rec["sleep"] += (time.perf_counter() - t0) * 1e3

        b.core_perf = core_perf
        b.device_synchronize = device_synchronize
        b.empty_cache = empty_cache
        time.sleep = sleep
        return self

    def __exit__(self, *exc):
        b = self.backend
        for name in ("core_perf", "device_synchronize", "empty_cache"):
            try:
                delattr(b, name)
            except AttributeError:
                pass
        time.sleep = self.orig["sleep"]
        return False

    def run(self, op_instance):
        """One instrumented `perf()`. Returns (record, target_dict)."""
        self.rec = {"sync": 0.0, "cache": 0.0, "sleep": 0.0, "create": 0.0,
                    "core_perf_calls": [], "copies": 0}

        orig_create = op_instance.create_tensors

        def create_tensors(instance_num):
            t0 = time.perf_counter()
            try:
                out = orig_create(instance_num)
            finally:
                self.rec["create"] += (time.perf_counter() - t0) * 1e3
            self.rec["copies"] = instance_num
            return out

        op_instance.create_tensors = create_tensors

        t0 = time.perf_counter()
        target_dict = self.backend.perf(op_instance)
        perf_ms = (time.perf_counter() - t0) * 1e3

        calls = self.rec["core_perf_calls"]
        rec = {
            # `construct` happens before perf() and is added to wall_ms by the
            # caller, which is the only phase not inside this measurement.
            "perf_ms": perf_ms,
            "create": self.rec["create"],
            "sync": self.rec["sync"],
            "sleep": self.rec["sleep"],
            "cache": self.rec["cache"],
            "calib": calls[0]["wall_ms"] if len(calls) > 0 else 0.0,
            "timed": calls[1]["wall_ms"] if len(calls) > 1 else 0.0,
            "copies": self.rec["copies"],
            "prefer_iters": calls[1]["iters"] if len(calls) > 1 else 0,
            "n_core_perf": len(calls),
        }
        return rec, target_dict


def pick_indices(n_cases, limit, explicit):
    if explicit:
        return [int(x) for x in explicit.replace(",", " ").split()]
    if limit <= 0 or limit >= n_cases:
        return list(range(n_cases))
    # Evenly spaced, endpoints included: gemm.json is ordered by shape then
    # dtype, so a contiguous head would be all-small and all-one-dtype.
    step = (n_cases - 1) / (limit - 1) if limit > 1 else 1
    return sorted({int(round(i * step)) for i in range(limit)})


def main():
    args = parse_args()

    affinity = "not pinned"
    if not args.no_affinity:
        affinity = pin_to_numa0()

    backend = build_backend(args.backend)
    backend.load_all_ops()
    backend.set_device(args.device)

    total_mem, avail_mem = backend.get_mem_info()
    print("")
    print(f"backend        : {args.backend}  device {args.device}")
    print(f"device_name    : {backend.get_device_name(0)}")
    if args.backend == "NEURON":
        print(f"neuron_runtime : {backend.neuron_runtime}")
    print(f"get_mem_info   : avail {avail_mem / 2**30:.1f} GiB of "
          f"{total_mem / 2**30:.1f} GiB  (drives max_data_cnt)")
    print(f"cpu affinity   : {affinity}")
    print(f"torch threads  : {torch.get_num_threads()}")
    print("")

    task_dict = parse_workload(args.workload)
    if args.op:
        task_dict = {k: v for k, v in task_dict.items() if k == args.op}
    if not task_dict:
        sys.exit(f"no cases in {args.workload} (op filter {args.op!r})")

    # Host CPU consumed while probing, which is how a per-shape cost that is
    # *compilation* separates from one that is a device-side load: run the probe
    # over N shapes and then N+1, and see whether the extra wall clock came with
    # matching user CPU time. Sampled around the case loop only, so process
    # startup and runtime init are excluded.
    ru_before = resource.getrusage(resource.RUSAGE_SELF)
    loop_start = time.perf_counter()

    records = []
    with InstrumentedPerf(backend) as probe:
        for op_name, cases in task_dict.items():
            providers = backend.op_mapping.get(op_name, {})
            if args.provider:
                providers = {k: v for k, v in providers.items() if k == args.provider}
            if not providers:
                print(f"[skip] {op_name}: no provider registered on {args.backend}")
                continue

            indices = pick_indices(len(cases), args.limit, args.cases)
            print(f"=== {op_name}: {len(cases)} cases in file, probing "
                  f"{len(indices)} of them, providers {list(providers)} ===")
            print(f"{'idx':>4} {'prov':<8} {'tensors':>9} {'cps':>3} {'iters':>5} "
                  f"{'lat us':>10} {'wall ms':>9} " +
                  " ".join(f"{p:>8}" for p in PHASES))

            # The stage breakdown runs once per op, on whichever probed case has
            # the largest tensor -- doing it per case would double the run, and
            # anchoring it to the last index breaks whenever that index is one of
            # the cases a provider rejects (the fp8 tail of gemm.json, for one).
            stage_target = None

            for provider, op_cls in providers.items():
                for idx in indices:
                    case = cases[idx]
                    t0 = time.perf_counter()
                    try:
                        op_instance = op_cls(case, backend)
                        op_instance.is_concurrent = False
                    except Exception as exc:
                        print(f"{idx:>4} {provider:<8} construct failed: {exc}")
                        continue
                    construct_ms = (time.perf_counter() - t0) * 1e3

                    rec, target = probe.run(op_instance)
                    rec.update(
                        op_name=op_name, provider=provider, case_index=idx,
                        construct=construct_ms,
                        wall_ms=construct_ms + rec["perf_ms"],
                        tensor_size=op_instance.tensor_size,
                        latency_us=target.get("latency(us)", 0.0),
                        case=case,
                    )
                    rec["other"] = rec["wall_ms"] - sum(
                        rec[p] for p in PHASES if p != "other"
                    )

                    print(f"{idx:>4} {provider:<8} "
                          f"{rec['tensor_size'] / 2**20:>8.1f}M "
                          f"{rec['copies']:>3} {rec['prefer_iters']:>5} "
                          f"{rec['latency_us']:>10.1f} {rec['wall_ms']:>9.1f} " +
                          " ".join(f"{rec[p]:>8.1f}" for p in PHASES))
                    records.append(rec)

                    if not args.no_stages:
                        info = biggest_input(op_instance)
                        # Only float_creator has these stages; int_creator
                        # generates int8 on the host and casts, which is a
                        # different (and much cheaper) shape of cost.
                        if info is not None and effective_creator(info[1]) is float_creator:
                            size = tensor_bytes(info[1])
                            if stage_target is None or size > stage_target[2]:
                                stage_target = (info[0], info[1], size)

                    del op_instance
                    backend.empty_cache()

            if stage_target is not None:
                name, tinfo, size = stage_target
                stages = stage_breakdown(backend, tinfo)
                print(f"     float_creator stages, largest tensor probed: {name} "
                      f"{list(tinfo.shape)} {tinfo.dtype} ({size / 2**20:.1f} MiB)")
                print(f"     cpu randn fp32 {stages['randn_ms']:.1f} ms | "
                      f"h2d {stages['h2d_ms']:.1f} ms | "
                      f"cast {stages['cast_ms']:.1f} ms | "
                      f"device clone {stages['clone_ms']:.1f} ms "
                      f"(x max_data_cnt-1)")
            print("")

    if not records:
        sys.exit("nothing measured")

    wall = sum(r["wall_ms"] for r in records)
    print(f"=== totals over {len(records)} cases: {wall / 1e3:.1f} s wall clock ===")
    print(f"{'phase':<12} {'total s':>9} {'% wall':>8} {'mean ms':>9} {'max ms':>9}")
    for phase in PHASES:
        vals = [r[phase] for r in records]
        print(f"{phase:<12} {sum(vals) / 1e3:>9.2f} {100 * sum(vals) / wall:>7.1f}% "
              f"{statistics.mean(vals):>9.1f} {max(vals):>9.1f}")

    device_ms = sum(r["calib"] + r["timed"] for r in records)
    reported_ms = sum(r["latency_us"] for r in records) / 1e3
    execs = sum(6 + r["prefer_iters"] for r in records)
    print("")
    print(f"device execution (calib+timed) : {device_ms / 1e3:.2f} s "
          f"= {100 * device_ms / wall:.1f}% of wall clock")
    # Deliberately in ms: the whole point is that this is far below `device_ms`,
    # because a reported latency is *one* iteration of a warm loop while the case
    # paid for `6 + prefer_iters` executions, the first of which is cold.
    print(f"sum of *reported* latencies    : {reported_ms:.1f} ms "
          f"({execs} op executions across all cases, "
          f"{execs / len(records):.1f} per case)")
    print(f"harness setup (everything else): "
          f"{(wall - device_ms) / 1e3:.2f} s = {100 * (wall - device_ms) / wall:.1f}%")
    print(f"mean per case                  : {wall / len(records) / 1e3:.2f} s")

    ru = resource.getrusage(resource.RUSAGE_SELF)
    loop_wall = time.perf_counter() - loop_start
    user_s = ru.ru_utime - ru_before.ru_utime
    sys_s = ru.ru_stime - ru_before.ru_stime
    print("")
    print(f"host CPU over the case loop    : user {user_s:.2f} s + sys {sys_s:.2f} s "
          f"over {loop_wall:.2f} s wall = {100 * (user_s + sys_s) / loop_wall:.0f}% "
          f"of one core")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(records, indent=2, default=str))
        print(f"\nraw records -> {args.json}")


if __name__ == "__main__":
    main()
