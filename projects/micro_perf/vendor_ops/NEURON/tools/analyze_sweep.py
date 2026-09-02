"""Per-run accounting for a micro_perf sweep log.

For each `########## RUN <label>` section: how many cases the launcher tried,
how many produced a measurement, and -- for the rest -- the exact reason,
grouped. A rejection is not a failure here; most of them are an op definition
declining a dtype it never implemented, which is the thing worth reporting.
"""
import collections
import json
import re
import sys

lines = open(sys.argv[1], errors="replace").read().split("\n")

RUN_RE = re.compile(r"^########## RUN (\S+)\s+(tmo=|exit=|\d{4}-)")
ERR_RE = re.compile(r"^([A-Za-z_]*(?:Error|Exception)): (.*)$")


def parse_targets(idx):
    """Read the pretty-printed targets dict that follows an arguments line."""
    depth, buf, j = 0, [], idx
    while j < len(lines):
        buf.append(lines[j])
        depth += lines[j].count("{") - lines[j].count("}")
        j += 1
        if depth == 0:
            break
    try:
        return json.loads("\n".join(buf)), j
    except json.JSONDecodeError:
        return None, idx


runs = []           # (label, results, errors, ops)
label = "(whole log)"
results, errors, ops = [], collections.Counter(), collections.Counter()
op_name = None

i = 0
while i < len(lines):
    line = lines[i]

    m = RUN_RE.match(line)
    if m and m.group(2) != "exit=":
        if label:
            runs.append((label, results, errors, ops))
        label = m.group(1)
        results, errors, ops = [], collections.Counter(), collections.Counter()

    m = re.match(r"\| op_name\s+\| (\S+)\s+\|", line)
    if m:
        op_name = m.group(1)

    m = ERR_RE.match(line)
    if m:
        # Tracebacks repeat the message; count one per "Failed to create op".
        pass
    if line.startswith("Failed to create op "):
        m2 = re.match(r"Failed to create op (\S+) with provider (\S+) .* with error (.*)$", line)
        if m2:
            errors[(m2.group(1), m2.group(3).strip())] += 1

    if line.startswith('{"arg_type"') and line.rstrip().endswith("}"):
        try:
            args = json.loads(line)
        except json.JSONDecodeError:
            i += 1
            continue
        if i + 1 < len(lines) and lines[i + 1].rstrip() == "{":
            targets, j = parse_targets(i + 1)
            if targets and "latency(us)" in targets:
                results.append((op_name, args, targets))
                ops[op_name] += 1
                i = j
                continue
    i += 1

if label:
    runs.append((label, results, errors, ops))

DETAIL = sys.argv[2] if len(sys.argv) > 2 else None

if DETAIL:
    # Per-case dump for one op, for reading a scaling curve rather than a summary.
    for label, results, errors, ops in runs:
        rows = [(a, t) for o, a, t in results if o == DETAIL]
        if not rows:
            continue
        print(f"\n=== {label} / {DETAIL}   {len(rows)} cases")
        for args, targets in rows:
            shape = " ".join(f"{k}={v}" for k, v in args.items() if k != "arg_type")
            bw = targets.get("mem_bw(GB/s)") or targets.get("bus_bw(GB/s)") or 0
            print(f"  {targets['latency(us)']:12.1f} us  {bw:8.2f} GB/s   {shape}")
    sys.exit(0)

for label, results, errors, ops in runs:
    tried = len(results) + sum(errors.values())
    print(f"\n=== {label}   tried={tried}  measured={len(results)}  rejected={sum(errors.values())}")
    for op, n in sorted(ops.items()):
        rows = [t for o, a, t in results if o == op]
        tf = max((t.get("calc_flops_power(tflops)") or 0 for t in rows), default=0)
        # "mfu absent" and "mfu rounds to zero" mean different things: the first
        # says the backend publishes no peak for the dtype, the second says the
        # op is memory-bound. Keep them apart.
        # Communication ops report bandwidth, not arithmetic; TFLOPS is
        # meaningless for them and an MFU is never emitted.
        bws = [t[k] for t in rows for k in ("bus_bw(GB/s)", "algo_bw(GB/s)", "mem_bw(GB/s)")
               if k in t]
        if any(k in rows[0] for k in ("bus_bw(GB/s)", "algo_bw(GB/s)")):
            print(f"    measured {op:34s} {n:5d}  best={max(bws):8.1f} GB/s    bandwidth op")
            continue

        mfus = [t["mfu"] for t in rows if "mfu" in t]
        if mfus:
            note = f"mfu {max(mfus):.2%} (n={len(mfus)}/{n})"
        elif all((t.get("calc_flops") or 0) == 0 for t in rows):
            mbw = max((t.get("mem_bw(GB/s)") or 0 for t in rows), default=0)
            note = f"no arithmetic reported; best mem_bw {mbw:.1f} GB/s"
        else:
            note = "no peak published for this dtype"
        print(f"    measured {op:34s} {n:5d}  best={tf:8.1f} TFLOPS  {note}")
    for (op, msg), n in errors.most_common():
        print(f"    rejected {op:34s} {n:5d}  {msg[:110]}")
