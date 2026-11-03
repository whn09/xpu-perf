# NEURON troubleshooting

## Troubleshooting

### A run that looks hung is usually compiling

On the XLA runtime this is normal: every distinct shape is compiled by
`neuronx-cc` on first use, 5-15 minutes per op, hours for a sweep. **On eager it
also happens** — "eager" is not a promise that `neuronx-cc` never runs. Most ops
dispatch to a prebuilt kernel, but one the runtime has no kernel for falls back
to the full XLA-era compile: `gather` at bf16 / `dim_size=8192` sat in
`neuronx-cc compile module.mlir --framework XLA --target trn2 --lnc 2 -O1` for
over two hours at 199% CPU before it was killed. So check for a `neuronx-cc` or
`walrus_driver` process before assuming a deadlock, and budget a per-shape
timeout on both runtimes.

`find /var/tmp/neuron-compile-cache -name "*.neff" | wc -l` shows cache growth.
Run under tmux or screen so an SSH drop does not kill a compile.

`flash_attention` on the eager runtime is the case most likely to be mistaken for
a hang, because it is slow exactly once. A cold `80/8/128` GQA prefill at
`q_len: 4096` spent **over 13 minutes** in `walrus_driver --optlevel 2` and had
produced no result when it was killed; the same case measured **9,512.6 us** on
the next run, because `NKI_ENABLE_TRACE_CACHE=1` (the default) persists the kernel
cache across processes and the first run had populated it. A bare
`scaled_dot_product_attention` at that shape is ~33 ms, so if the first case of
`fa_linear_ops.json` has been silent for ten minutes, that is a compile and not a
broken vendor op — the check is a live `walrus_driver`, not elapsed time. Two
practical consequences: give an FA sweep a per-case budget in hours rather than
minutes, and warm each distinct shape once before timing anything you intend to
publish.

### A sweep is taking all day but nothing is hung

Expected, and it is mostly not the ops. Every distinct tensor shape costs ~2.9 s
of one-time warm-up on the eager runtime — cached in the process, not on disk — and
a workload file with 856 distinct shapes pays it 856 times. Measured breakdown and
the evidence that this is *not* `neuronx-cc` (8-17% host CPU, nothing written to
disk) are under [Sweep wall clock is not a performance
comparison](READING_THE_NUMBERS.md#sweep-wall-clock-is-not-a-performance-comparison).

The distinguishing check against a real compile: a `neuronx-cc` or `walrus_driver`
process at ~199% CPU means compiling; a python process at ~10% with the core
allocated means warming up, and it will finish in seconds per case.

### `timeout docker run` does not bound anything

It signals the docker *client*; the container is a child of the daemon, so the
launch is orphaned and keeps holding the core. Use a watchdog that runs
`docker kill <name>` against the container — `tools/run_full_sweep.sh` does.

### A collective run hangs forever after a rank dies

`XCCLEngine` has no liveness check and no timeout on the results it waits for, so
when a worker OOMs during compilation the launcher blocks indefinitely: `ps`
shows the workers in state `Z` and the launcher in `sleep`, with nothing on
stdout after the last completed case. It sat 35 minutes before being killed from
outside. Bound collective runs externally, and cap sizes so the OOM never
happens (see the 24 GiB ceiling above).

### An op produced no output, no error, and exit code 0

Its engine is not in `XPU_PERF_ENGINES`. See the note in
[Quick start](README.md#3-run) — `device2device` is the one that catches people.

### `❌ 导入失败 ._<something>：attempted relative import beyond top-level package`

macOS AppleDouble sidecars in the synced tree, not a code problem. `rsync`/`scp`
from a Mac leaves a 220-byte `._foo.py` next to every `foo.py` it copied, and
`parse_vendor_ops` (`core/common_utils.py`) imports every `*.py` it finds, so it
tries `..foo` and gets two leading dots. The ops themselves still register —
check the `Provider: torch` table for the real answer. `ls` hides these files
because the name starts with a dot:

```bash
find ~/xpu-perf -name "._*" -delete
rsync -a --exclude="._*" ...        # or pass --iconv=. / use COPYFILE_DISABLE=1
```

### A killed run still holds the cores

```bash
pkill -9 -f multiprocessing && pkill -9 -f neuronx-cc
find /var/tmp/neuron-compile-cache -name "*.lock" -delete   # killed compilers leave locks
neuron-ls   # confirm the cores are free
```

A timed-out collective launch is the common case: the parent exits but its
non-daemon children do not. Note that `pkill -f launch.py` matches its own
command line if the invoking shell's arguments contain that text, so bracket the
pattern:

```bash
ps -o pid=,cmd= -u "$USER" | grep -E "[l]aunch\.py|[s]pawn_main"
pkill -9 -f "[l]aunch\.py"; pkill -9 -f "[s]pawn_main"
```

### `NRT_FAILURE` / `Logical Neuron Core(s) not available ... (cores busy, ret=-16)`

Another process owns the cores — or, on a multi-device launch, one of the
single-process-per-core constraints (see
[IMPLEMENTATION.md](IMPLEMENTATION.md#one-process-per-neuroncore)). `neuron-ls`
attributes cores to a PID but can miss a holder in another container, so check
directly:

```bash
sudo bash -c 'for p in /proc/[0-9]*; do for fd in "$p"/fd/*; do
  case "$(readlink "$fd" 2>/dev/null)" in /dev/neuron*)
    echo "$(basename $p) $(tr -d "\0" < $p/cmdline)"; break;; esac; done; done'
sudo docker ps        # a second container is an easy holder to miss
```

Match `/dev/neuron*`, not any path containing `neuron` — the looser pattern also
matches a process whose *own executable* lives under `/opt/aws/neuron`, which
says nothing about the device. And `neuron-top` / `neuron-monitor` appear in this
list: they open `/dev/neuron0` to read counters without reserving a core, so
watching the device looks identical to using it. Filter them out before treating
a non-empty list as busy, or waiting for an idle machine will wait forever.

### Suspiciously fast results

A gemm reporting ~1,900 TFLOPS on the XLA runtime is a pruned graph, not a fast
gemm — see
[IMPLEMENTATION.md](IMPLEMENTATION.md#xla-compilation-dominates-first-run-time).
`torch_neuronx.get_fallback_ops()` lists ops that silently ran on CPU (across a
basic sweep only `aten::normal_` falls back), and `get_backend_info()` reports
which runtime was selected.
