"""Reconstruct micro_perf results from a sweep log.

micro_perf writes its CSV/jsonl reports only when a launch finishes, so a run
killed mid-sweep leaves nothing on disk. It does however print every case as it
completes -- a prettytable header naming op/provider, then the arguments dict,
then the targets dict -- so a killed run's results are recoverable from stdout.

Emits one CSV per op/provider, same shape as the real reporter.
"""
import csv
import json
import pathlib
import re
import sys

LOG = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
SKU = sys.argv[3] if len(sys.argv) > 3 else "trn2.3xlarge"

lines = LOG.read_text(errors="replace").split("\n")

# Per-device peaks, mirroring NEURON_CHIP_PEAK_TFLOPS in
# backends/NEURON/backend_neuron.py divided by the 4 logical cores a trn2 exposes
# at LNC=2. Only used to backfill logs from before the backend emitted `mfu`
# itself, and only correct for that SKU -- adjust the divisor for trn1/inf2.
PEAK = {"float32": 45.25, "tfloat32": 166.75, "float16": 166.75, "half": 166.75,
        "bfloat16": 166.75, "float8": 324.75, "float8_e4m3": 324.75,
        "float8_e5m2": 324.75, "mxfloat8": 324.75, "mxfloat8_e4m3": 324.75,
        "mxfloat8_e5m2": 324.75}

records = []
i = 0
op = provider = None
while i < len(lines):
    line = lines[i]
    m = re.match(r"\| op_name\s+\| (\S+)\s+\|", line)
    if m:
        op = m.group(1)
    m = re.match(r"\| op_provider\s+\| (\S+)\s+\|", line)
    if m:
        provider = m.group(1)

    # An arguments dict is a complete single-line JSON object.
    if line.startswith('{"') and line.rstrip().endswith("}"):
        try:
            args = json.loads(line)
        except json.JSONDecodeError:
            i += 1
            continue
        # The targets dict follows, pretty-printed across several lines.
        if i + 1 < len(lines) and lines[i + 1].rstrip() == "{":
            depth = 0
            buf = []
            j = i + 1
            while j < len(lines):
                buf.append(lines[j])
                depth += lines[j].count("{") - lines[j].count("}")
                j += 1
                if depth == 0:
                    break
            try:
                targets = json.loads("\n".join(buf))
            except json.JSONDecodeError:
                i += 1
                continue
            if "latency(us)" in targets:
                records.append((op, provider, args, targets))
                i = j
                continue
    i += 1

print(f"recovered {len(records)} cases from {LOG}")

by_key = {}
for op, provider, args, targets in records:
    by_key.setdefault((op, provider), []).append((args, targets))

for (op, provider), rows in sorted(by_key.items()):
    # Backfill mfu for runs that predate the metric.
    for args, targets in rows:
        if "mfu" in targets or not targets.get("calc_flops_power(tflops)"):
            continue
        dtype = next((args[k] for k in ("compute_dtype", "qk_compute_dtype", "dtype")
                      if args.get(k)), None)
        peak = PEAK.get(dtype)
        if peak:
            targets["peak_tflops"] = peak
            targets["mfu"] = round(targets["calc_flops_power(tflops)"] / peak, 4)

    arg_keys, target_keys = [], []
    for args, targets in rows:
        for k in args:
            if k not in arg_keys:
                arg_keys.append(k)
        for k in targets:
            if k not in target_keys:
                target_keys.append(k)

    out_dir = OUT / op / provider
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{op}-{provider}.csv"
    with open(out_file, "w", newline="") as f:
        fields = ["sku_name", "op_name", "provider"] + arg_keys + target_keys
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for args, targets in rows:
            row = {"sku_name": SKU, "op_name": op, "provider": provider}
            row.update(args)
            row.update(targets)
            writer.writerow(row)
    print(f"  {out_file}  {len(rows)} rows")
