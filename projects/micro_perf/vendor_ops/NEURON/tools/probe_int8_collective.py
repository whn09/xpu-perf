"""Why is an int8 collective ~200x slower than the same collective in bf16?

Measured on a trn2.48xlarge at world_size 8 (workloads/llm/single_test_ops/
ccl_ops.json, num_tokens 262,144 x hidden_size 1024):

    all_gather  bf16   512 MiB     5.76 ms    81.6 GB/s
    all_gather  int8   256 MiB  1307.00 ms     0.2 GB/s     227x slower
    all_to_all  bf16   512 MiB    16.42 ms    28.5 GB/s
    all_to_all  int8   256 MiB  2002.03 ms     0.1 GB/s     122x slower

It is not a bytes effect -- per *element* int8 is 110-230x worse too -- and it is
not only a fixed overhead: the 8-token case takes 2407 us against bf16's 147.
Nothing is logged: no warning, no exception, no fallback notice. Both facts are
consistent with the dtype not being supported by the device collective and
something else quietly picking up the work.

This settles it the same way the fp8 datapath question was settled: run the
collective and watch what is busy underneath it. bf16 runs first and int8 second
for each op, so the sample series carries its own control.

Three signals, because for a collective no one of them is enough:

  * `neuroncore_utilization` -- the compute engines. A device collective is mostly
    DMA, so this is *low even when the collective is on the device*; what it can
    show is whether the device is busy for the whole phase or only a slice of it.
  * the 8 worker runtimes' own `vcpu_usage.user`, summed -- 100% here is one
    saturated vCPU. This is the scale a per-rank host fallback would show at, and
    it is what settled the fp8 question (one core pinned, device idle).
  * instance-wide `vcpu_usage.average_usage.user` -- of all 192 vCPUs, for
    context; on its own it cannot separate these processes from anything else.

Sampling is 1s (neuron-monitor's default is 5s, which gave 8 samples for the whole
run the first time this was written and could not resolve a phase).

neuron-monitor is safe to run alongside a workload: it opens /dev/neuron*
read-only and reserves no core.

Run from projects/micro_perf on an *idle* trn2.48xlarge (it needs 8 cores):

    sudo -v && python3 vendor_ops/NEURON/tools/probe_int8_collective.py

Options: --image, --device, --out, --docker. Writes the sample series to
/tmp/int8_probe/util.log (elapsed, max NeuronCore util, instance user%, summed
worker cpu%) and the harness output to /tmp/int8_probe/run.log. Read the series,
not only the per-phase summary: neuron-monitor's 1s aggregation lags the phase
boundary by up to one sample, so a bf16 spike can land in the next window's max.
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import threading
import time

NOISE = re.compile(r"NRT:nrta_tensor|_warn_trace_cache_once|NKI_ENABLE_TRACE_CACHE")

MONITOR_CONFIG = {
    "period": "1s",
    "neuron_runtimes": [
        {"tag_filter": ".*", "metrics": [{"type": "neuroncore_counters"},
                                        {"type": "neuron_runtime_vcpu_usage"}]}
    ],
    "system_metrics": [{"type": "vcpu_usage"}],
}

parser = argparse.ArgumentParser()
parser.add_argument("--image", default="xpu-perf-eager:latest")
parser.add_argument("--device", default="0,1,2,3,4,5,6,7")
parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("/tmp/int8_probe"))
parser.add_argument("--docker", default="sudo docker")
args = parser.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
# .../<repo>/projects/micro_perf/vendor_ops/NEURON/tools/this file
repo = pathlib.Path(__file__).resolve().parents[5]

stop = threading.Event()
samples = []


def max_utilization(blob):
    """Highest neuroncore_utilization over every core neuron-monitor reports.

    Max rather than mean: one busy core is enough to prove the work is on the
    device, and averaging over 64 cores when 8 are in use would hide it.
    """
    values = []
    for runtime in blob.get("neuron_runtime_data", []):
        counters = runtime.get("report", {}).get("neuroncore_counters", {})
        for core in counters.get("neuroncores_in_use", {}).values():
            values.append(core.get("neuroncore_utilization", 0))
    return max(values) if values else None


def host_user(blob):
    """Instance-wide user CPU, in percent of all 192 vCPUs."""
    usage = blob.get("system_data", {}).get("vcpu_usage", {})
    average = usage.get("average_usage")
    return average.get("user") if average else None


def worker_cpu(blob):
    """The 8 worker runtimes' own CPU, summed, in percent of one vCPU each.

    Instance-wide user% cannot tell a host fallback in *these* processes from
    unrelated load, and 100% here means one saturated vCPU rather than 1/192 of
    the box, which is the scale a per-rank fallback would show up at.
    """
    total = 0.0
    seen = False
    for runtime in blob.get("neuron_runtime_data", []):
        report = runtime.get("report", {})
        usage = report.get("neuron_runtime_vcpu_usage", {}).get("vcpu_usage", {})
        value = usage.get("user")
        if value is not None:
            total += value
            seen = True
    return total if seen else None


def sampler():
    """One `neuron-monitor` for the whole run; it emits one JSON line per period."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(MONITOR_CONFIG, handle)
        config = handle.name
    proc = subprocess.Popen(
        ["neuron-monitor", "-c", config], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True,
    )
    t0 = time.time()
    try:
        for line in proc.stdout:
            if stop.is_set():
                break
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                blob = json.loads(line)
            except json.JSONDecodeError:
                continue
            samples.append((round(time.time() - t0, 1), max_utilization(blob),
                            host_user(blob), worker_cpu(blob)))
    finally:
        proc.terminate()


thread = threading.Thread(target=sampler, daemon=True)
thread.start()

cmd = args.docker.split() + [
    "run", "--rm", "--name", "xpu_probe_int8", "--privileged",
    "-v", f"{repo}:/xpu-perf", "-w", "/xpu-perf/projects/micro_perf",
    "-e", "PYTHONPATH=/xpu-perf/src", args.image,
    "env", "XPU_PERF_ENGINES=XCCLEngine", "XPU_PERF_XCCL_READY_TIMEOUT_S=3600",
    "python", "launch.py", "--backend", "NEURON", "--device", args.device,
    "--numa", "-1", "--report_dir", "/tmp/int8_probe_report",
    "--workload", "vendor_ops/NEURON/workloads/ccl_int8_probe.json",
]
print(" ".join(cmd), flush=True)

t_start = time.time()
marks = []
with open(args.out / "run.log", "w") as log:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        if NOISE.search(line):
            continue
        log.write(line)
        # The harness prints the case dict together with that case's result, i.e.
        # *after* measuring it, so a mark is the END of a phase and phase i runs
        # from mark i-1 to mark i. (Read as starts, the first version of this
        # script attributed every phase to the dtype of the previous one.)
        if '"dtype"' in line and "num_tokens" in line:
            case = json.loads(line.strip())
            marks.append((round(time.time() - t_start, 1), case["dtype"]))
            print(f"[{marks[-1][0]:>7.1f}s] {case['dtype']}", flush=True)
        elif '"latency(us)"' in line:
            print(f"          {line.strip()}", flush=True)
    proc.wait()

stop.set()
thread.join(timeout=5)

with open(args.out / "util.log", "w") as log:
    for elapsed, util, user, workers in samples:
        log.write(f"{elapsed}\t{util}\t{user}\t{workers}\n")

if not samples:
    sys.exit("no neuron-monitor samples: is neuron-monitor on PATH?")

# The workload is all_gather bf16, all_gather int8, all_to_all bf16, all_to_all
# int8. A mark is the end of its case, so case i owns (mark i-1, mark i]; the
# first case owns everything before mark 0, which also contains process startup
# and is therefore only a bound.
print("\nper phase, over the samples inside it. NeuronCore is the max over cores,")
print("host user% is instance-wide over all vCPUs, both as sampled at 1s:")
print(f"  {'window':>17}  {'n':>3}  {'NC max':>7}  {'NC mean':>7}  "
      f"{'host user%':>10}  {'workers cpu%':>12}  dtype")
starts = [0.0] + [t for t, _ in marks[:-1]]


def peak(values):
    return max(values) if values else float("nan")


for start, (end, dtype) in zip(starts, marks):
    window = [s[1:] for s in samples if start <= s[0] < end]
    nc = [u for u, _, _ in window if u is not None]
    cpu = [c for _, c, _ in window if c is not None]
    workers = [w for _, _, w in window if w is not None]
    print(f"  {start:>7.1f}-{end:>7.1f}s  {len(window):>3d}  "
          f"{peak(nc):>7.1f}  {(sum(nc) / len(nc)) if nc else float('nan'):>7.2f}  "
          f"{peak(cpu):>10.1f}  {peak(workers):>12.1f}  {dtype}")
print(f"\nsamples: {args.out / 'util.log'}   harness output: {args.out / 'run.log'}")
