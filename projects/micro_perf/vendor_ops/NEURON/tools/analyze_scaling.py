"""Join a 1-core report against a 4-core report to test whether "x4" is real.

Every per-chip Trainium2 figure in ../README.md and ../../GPU/README.md is a
one-logical-core number multiplied by four. That is only legitimate if four cores
each do what one core alone does, so the claim is checked directly: run a workload
file on `--device 0`, run it again on `--device 0,1,2,3`, and compare shape by
shape. In the 4-core run micro_perf keeps four *different* cases in flight, so each
case's latency is still a single-core latency -- measured while the other three
cores compete for the same HBM. A ratio of 1.0 means contention is free and x4 is
exact.

Two things this reports that a single summary number would hide:

* the median and the tail separately. The medians are 1.00-1.03 and the worst cases
  are 5-11x, because the smallest shapes are dominated by dispatch and
  synchronisation rather than arithmetic and contend on something the large shapes
  never touch. A per-chip figure quoted from a small shape is wrong.
* the ratio *at the peak shape*, which is the cell a README actually publishes.
  The median over all shapes is not the right check for it.

Usage:

    python3 analyze_scaling.py <one-core-report-root> <four-core-report-root>
    python3 analyze_scaling.py /tmp/sweep_results/basic_tensor_gemm_ops \
                               /tmp/sweep_results/chip4_gemm

Report roots are whatever `--report_dir` produced -- the script walks for *.jsonl
and takes the op name from the filename stem, so it does not care about the
BACKEND/instance/op/provider nesting in between.
"""
import collections
import glob
import json
import os
import sys

# Compute-bound ops report tflops, memory-bound ops report mem_bw, and a few
# report both. Preference order, first hit wins, per op.
METRICS = ("calc_flops_power(tflops)", "mem_bw(GB/s)")


def load(root):
    """{(op, frozen args): (args, targets)} for every case under a report root."""
    out = {}
    for path in glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True):
        # gemm-torch.jsonl -> gemm ; topk-base.jsonl -> topk
        op = os.path.basename(path).rsplit("-", 1)[0]
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                args = rec.get("arguments", {})
                key = (op, tuple(sorted((k, str(v)) for k, v in args.items())))
                out[key] = (args, rec.get("targets", {}))
    return out


def metric_of(targets):
    for name in METRICS:
        val = targets.get(name)
        if val:
            return name, val
    return None, None


def main(one_root, four_root):
    c1, c4 = load(one_root), load(four_root)
    common = [k for k in c1 if k in c4]
    print(f"1-core cases={len(c1)}  4-core cases={len(c4)}  joined={len(common)}")
    only1 = len(c1) - len(common)
    only4 = len(c4) - len(common)
    if only1 or only4:
        print(f"  (unjoined: {only1} only in the 1-core run, {only4} only in the "
              f"4-core run -- usually a case one run rejected and the other did not)")
    print()

    by_op = collections.defaultdict(list)
    for key in common:
        a1, t1 = c1[key]
        _, t4 = c4[key]
        l1, l4 = t1.get("latency(us)"), t4.get("latency(us)")
        if not l1 or not l4:
            continue
        mname, m1 = metric_of(t1)
        m4 = t4.get(mname) if mname else None
        by_op[key[0]].append((l4 / l1, l1, l4, m1, m4, mname, a1))

    if not by_op:
        print("nothing joined -- are both paths report roots from the same workload?")
        return 1

    def summarise(title, groups):
        print(f"{title:<28} {'n':>4} {'median 4c/1c':>13} {'p90':>8} {'worst':>8} "
              f"{'effective x':>12}")
        for name, rows in sorted(groups.items()):
            ratios = sorted(r[0] for r in rows)
            n = len(ratios)
            med = ratios[n // 2]
            p90 = ratios[int(n * 0.9)] if n > 1 else ratios[0]
            print(f"{name:<28} {n:>4} {med:>13.3f} {p90:>8.3f} {ratios[-1]:>8.3f} "
                  f"{4.0 / med:>11.2f}x")

    summarise("op", by_op)

    # fp8 gemm is the one dtype whose median does not scale, and an op-level
    # rollup hides that, so split when an op spans dtypes.
    by_dtype = collections.defaultdict(list)
    for op, rows in by_op.items():
        for row in rows:
            by_dtype[f"{op} / {row[6].get('dtype', '?')}"].append(row)
    if len(by_dtype) > len(by_op):
        print()
        summarise("op / dtype", by_dtype)

    print()
    print("worst shape per op -- the small-shape contention tail:")
    for op, rows in sorted(by_op.items()):
        ratio, l1, l4, _, _, _, args = max(rows)
        print(f"  {op:<20} 1c={l1:>9.1f}us 4c={l4:>9.1f}us ratio={ratio:>6.2f}  "
              f"{args}")

    print()
    print("peak 1-core shape per op, and the same shape under four cores -- this is")
    print("the ratio that justifies (or does not) a published per-chip cell:")
    for op, rows in sorted(by_op.items()):
        scored = [r for r in rows if r[3]]
        if not scored:
            print(f"  {op:<20} no metric reported")
            continue
        ratio, l1, l4, m1, m4, mname, args = max(scored, key=lambda r: r[3])
        line = (f"  {op:<20} 1c={m1:>10.2f} {mname}  4c/core={m4 or 0:>10.2f}  "
                f"ratio={ratio:.3f} -> x{4.0 / ratio:.2f}")
        print(line)
        print(f"  {'':<20} per chip = {(m4 or 0) * 4:>10.2f}   at {args}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
