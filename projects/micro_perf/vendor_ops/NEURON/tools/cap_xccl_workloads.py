"""Write size-capped copies of workloads/xccl_ops/ that fit one NeuronCore.

Every file in that directory sweeps batch_size to 2,097,152 x dim_size 1024,
which is 8 GiB at fp32. A logical NeuronCore on a trn2 at LNC=2 gets 24 GiB, and
neuronx-cc budgets I/O plus an equal scratchpad, so an 8 GiB buffer asks for
32 GiB and fails:

    [ERROR] [NCC_EOOM001] Maximum peak HBM usage of 32.00GB exceeds HBM limit of
    24.00GB for Trn2.

That failure is not survivable in place: XCCLEngine has no liveness check on its
workers, so the OOMing rank dies, becomes a zombie, and the launcher waits for
its result forever. Cap instead. 262,144 x 1024 fp32 is 1 GiB, well past the
point where every collective here has reached its bandwidth plateau.

Also drops world_size values other than the one being launched -- perf() skips
mismatched cases anyway, and leaving them in only pads the enumerated case list.

    python cap_xccl_workloads.py workloads/xccl_ops /tmp/xccl_ws4 --world-size 4
"""
import argparse
import json
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("src", type=pathlib.Path)
parser.add_argument("dst", type=pathlib.Path)
parser.add_argument("--world-size", type=int, default=4)
parser.add_argument("--max-batch-size", type=int, default=262144)
args = parser.parse_args()

args.dst.mkdir(parents=True, exist_ok=True)

for src in sorted(args.src.glob("*.json")):
    spec = json.loads(src.read_text())
    dropped = 0
    for case in spec["cases"]:
        before = case.get("batch_size", [])
        case["batch_size"] = [b for b in before if b <= args.max_batch_size]
        dropped += len(before) - len(case["batch_size"])
        # device2device is not a collective and carries no world_size.
        if "world_size" in case:
            case["world_size"] = [args.world_size]
    kept = max((max(c["batch_size"]) for c in spec["cases"] if c.get("batch_size")),
               default=None)
    (args.dst / src.name).write_text(json.dumps(spec, indent=4) + "\n")
    print(f"{src.name}: dropped {dropped} batch_size values, largest kept {kept}")
