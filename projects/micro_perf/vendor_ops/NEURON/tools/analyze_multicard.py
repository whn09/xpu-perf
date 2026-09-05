"""Summarize a run_multicard_sweep.sh results tree.

analyze_sweep.py reads the sweep *log* and accounts for time per label; this
reads the *reports* and answers the two questions the multi-chip labels exist
for: how a collective's bandwidth moves with rank count, and what crossing a
chip boundary costs at a fixed rank count.

    python analyze_multicard.py /tmp/mc                 # every label found
    python analyze_multicard.py /tmp/mc --curve xccl_ws8  # full size sweep

Reports live at <results>/<label>/<backend>/<sku>/<op>/<provider>/<op>-<provider>.jsonl.
Collective reports carry bus_bw(GB/s); device2device and the compute labels carry
mem_bw(GB/s) instead, so the metric is picked per record rather than assumed.
"""
import argparse
import collections
import glob
import json
import os
import pathlib

# bus_bw is the collective metric: algo_size scaled by the ring's 2(n-1)/n, i.e.
# bytes actually crossing a link. mem_bw is what the copy and compute ops report.
METRICS = ("bus_bw(GB/s)", "mem_bw(GB/s)", "algo_bw(GB/s)")

# The count dimension, whichever layout the workload used.
COUNT_KEYS = ("batch_size", "num_tokens")


def count_of(arguments):
    for key in COUNT_KEYS:
        if key in arguments:
            return arguments[key]
    return 0


def bytes_of(record):
    """Payload bytes, from whichever size field the op reports."""
    targets = record["targets"]
    for key in ("algo_size(B)", "io_bytes(B)", "tensor_size(B)"):
        if key in targets:
            return targets[key]
    return None


def metric_of(record):
    for name in METRICS:
        if name in record["targets"]:
            return name, record["targets"][name]
    return None, None


def load(label_dir):
    """[(op, provider, arguments, targets)] for one label."""
    out = []
    for path in sorted(glob.glob(f"{label_dir}/**/*.jsonl", recursive=True)):
        for line in open(path):
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


parser = argparse.ArgumentParser()
parser.add_argument("results", type=pathlib.Path)
parser.add_argument("--curve", default=None,
                    help="print the full size sweep for this label instead")
args = parser.parse_args()

labels = sorted(
    d.name for d in args.results.iterdir()
    if d.is_dir() and glob.glob(f"{d}/**/*.jsonl", recursive=True)
)
if not labels:
    raise SystemExit(f"no reports under {args.results}")

if args.curve:
    records = load(args.results / args.curve)
    by_op = collections.defaultdict(dict)
    for record in records:
        by_op[record["op_name"]][count_of(record["arguments"])] = record
    ops = sorted(by_op)
    print(f"{args.curve}: full size sweep, "
          f"{next(iter(records))['arguments'].get('dtype')} unless mixed\n")
    header = f"{'count':>10s} {'payload(B)':>12s}"
    for op in ops:
        header += f" | {op[:25]:>25s}"
    print(header)
    for count in sorted({c for op in ops for c in by_op[op]}):
        first = next(by_op[op][count] for op in ops if count in by_op[op])
        row = f"{count:>10d} {bytes_of(first) or 0:>12.0f}"
        for op in ops:
            record = by_op[op].get(count)
            if record is None:
                row += f" | {'-':>25s}"
                continue
            _, value = metric_of(record)
            row += f" | {value:>10.1f} GB/s {record['targets']['latency(us)']:>7.0f}us"
        print(row)
    raise SystemExit(0)

# Peak and floor per (label, op). "Floor" is the latency of the smallest case,
# which for a collective is pure launch + sync cost and is the number that
# decides whether a small tensor-parallel all_reduce is worth doing at all.
print(f"{'label':<16s} {'op':<16s} {'dtype':<9s} {'peak':>10s} {'at bytes':>12s} "
      f"{'floor(us)':>10s} {'cases':>6s}")
for label in labels:
    rows = collections.defaultdict(list)
    for record in load(args.results / label):
        rows[(record["op_name"], record["arguments"].get("dtype", "-"))].append(record)
    for (op, dtype), records in sorted(rows.items()):
        peak = max(records, key=lambda r: metric_of(r)[1] or 0)
        floor = min(records, key=lambda r: count_of(r["arguments"]))
        _, value = metric_of(peak)
        print(f"{label:<16s} {op:<16s} {dtype:<9s} {value:>10.1f} "
              f"{bytes_of(peak) or 0:>12.0f} {floor['targets']['latency(us)']:>10.1f} "
              f"{len(records):>6d}")
