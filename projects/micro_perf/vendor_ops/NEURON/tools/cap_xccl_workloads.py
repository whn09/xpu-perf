"""Write size-capped, world_size-pinned copies of the collective workloads.

Handles both collective workload layouts, and takes either a directory or a
single .json file as `src`:

  * `workloads/xccl_ops/*.json` -- the `{"cases": [...]}` layout, sweeping
    `batch_size` x `dim_size`;
  * `workloads/llm/single_test_ops/ccl_ops.json` -- the `{"<op>": [...]}`
    layout, with `num_tokens` x `hidden_size` and a *scalar* `world_size`.

Both need capping for the same reason. xccl_ops sweeps batch_size to 2,097,152 x
dim_size 1024, which is 8 GiB at fp32; ccl_ops.json asks for num_tokens
1,048,576 x hidden_size 1024, which is 4 GiB at fp32 and, with all_reduce
holding a separate output, 8 GiB of I/O. A logical NeuronCore on a trn2 at LNC=2
gets 24 GiB, and neuronx-cc budgets I/O plus an equal scratchpad, so an 8 GiB
buffer asks for 32 GiB and fails:

    [ERROR] [NCC_EOOM001] Maximum peak HBM usage of 32.00GB exceeds HBM limit of
    24.00GB for Trn2.

That failure is not survivable in place: XCCLEngine has no liveness check on its
workers, so the OOMing rank dies, becomes a zombie, and the launcher waits for
its result forever. Cap instead. 262,144 x 1024 fp32 is 1 GiB, well past the
point where every collective here has reached its bandwidth plateau.

Also rewrites world_size to the one being launched -- perf() skips mismatched
cases, so ccl_ops.json's hardcoded 8 is why that file had no runnable case at
all on a 4-core trn2.3xlarge, and leaving extra values in the xccl_ops sweep
only pads the enumerated case list.

    python cap_xccl_workloads.py workloads/xccl_ops /tmp/xccl_ws4 --world-size 4
    python cap_xccl_workloads.py workloads/llm/single_test_ops/ccl_ops.json \
        /tmp/ccl_ws8 --world-size 8
"""
import argparse
import json
import pathlib

# The count dimension, whichever layout names it. dim_size / hidden_size are
# left alone: at 1024 they are what makes 262,144 come to exactly 1 GiB at fp32.
COUNT_KEYS = ("batch_size", "num_tokens")

parser = argparse.ArgumentParser()
parser.add_argument("src", type=pathlib.Path, help="workload dir or single .json")
parser.add_argument("dst", type=pathlib.Path)
parser.add_argument("--world-size", type=int, default=4)
parser.add_argument("--max-batch-size", type=int, default=262144)
parser.add_argument(
    "--dtypes", default=None,
    help="comma-separated dtypes to keep, e.g. bfloat16. Default: keep all. "
         "Used by the world_size scaling sweep, where the question is how "
         "bus_bw moves with rank count and one dtype answers it in a quarter "
         "of the cases.",
)
args = parser.parse_args()
keep_dtypes = args.dtypes.split(",") if args.dtypes else None

args.dst.mkdir(parents=True, exist_ok=True)


def cap_case(case):
    """Cap the count dimension in place; return how many values were dropped.

    A scalar stays a scalar and a list stays a list, because the two layouts
    disagree and get_cartesian_product() treats a scalar as a one-element list
    anyway -- rewriting the type would work but makes the diff unreadable.
    """
    dropped = 0
    if keep_dtypes and "dtype" in case:
        before = case["dtype"]
        if isinstance(before, list):
            case["dtype"] = [d for d in before if d in keep_dtypes] or before[:1]
    for key in COUNT_KEYS:
        if key not in case:
            continue
        before = case[key]
        if isinstance(before, list):
            # Clamp rather than drop, then dedupe preserving order. On the
            # xccl_ops sweep the two are the same thing -- 262,144 is already in
            # the list, so the four larger values collapse onto it. On
            # ccl_ops.json they are not: num_tokens is [8, 1048576], and
            # dropping would leave only the 32 KB latency-floor point and no
            # bandwidth point at all.
            after = []
            for b in before:
                b = min(b, args.max_batch_size)
                if b not in after:
                    after.append(b)
            case[key] = after
            dropped += len(before) - len(after)
        elif before > args.max_batch_size:
            case[key] = args.max_batch_size
            dropped += 1
    # device2device is not a collective and carries no world_size.
    if "world_size" in case:
        case["world_size"] = (
            [args.world_size] if isinstance(case["world_size"], list)
            else args.world_size
        )
    return dropped


def counts(case):
    for key in COUNT_KEYS:
        if key in case:
            value = case[key]
            return value if isinstance(value, list) else [value]
    return []


srcs = sorted(args.src.glob("*.json")) if args.src.is_dir() else [args.src]

for src in srcs:
    spec = json.loads(src.read_text())
    # {"cases": [...]} for xccl_ops, {"<op>": [...]} for ccl_ops.json.
    groups = [spec["cases"]] if "cases" in spec else list(spec.values())
    all_cases = [case for group in groups for case in group]

    dropped = sum(cap_case(case) for case in all_cases)
    kept = max((c for case in all_cases for c in counts(case)), default=None)

    (args.dst / src.name).write_text(json.dumps(spec, indent=4) + "\n")
    print(f"{src.name}: dropped {dropped} count values, largest kept {kept}, "
          f"world_size={args.world_size}")
