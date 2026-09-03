#!/bin/bash
# Drive a full micro_perf sweep on the eager (PyTorch-native) Neuron stack.
#
# This produced the "Eager runtime, full sweep" numbers in ../README.md. Run it
# from projects/micro_perf on a trn2.3xlarge with the image from
# Dockerfile.eager. Expect the better part of a day.
#
# `python launch.py --task all` does not survive this workload set, for four
# reasons that all cost real time to find:
#
# 1. `timeout 7200 docker run ...` cannot bound a launch, and neither can
#    `sudo timeout ... docker run`. timeout signals the docker *client*; the
#    container is a child of the daemon, so killing the client orphans it and it
#    keeps holding the NeuronCore. The only bound that works is a watchdog
#    running `docker kill <name>` against the container.
# 2. micro_perf writes its reports only when a launch finishes, so one launch
#    for everything means a single wedged op loses every result behind it. Each
#    workload file therefore gets its own launch and its own --report_dir, and
#    results land incrementally.
# 3. Some ops wedge neuronx-cc indefinitely rather than failing: `gather` at
#    float32/dim_size=65536 and `scatter` at float32/dim_size=4096 sit at ~200%
#    CPU forever. One gather case ran two hours before being killed. They are
#    run last, separately, and after the other three index ops have their
#    numbers.
# 4. Engines and device counts are per-file, not global. Collectives need
#    XPU_PERF_ENGINES=XCCLEngine (otherwise ComputeEngine workers race them for
#    cores), but `device2device` lives in the same directory and is registered
#    under ComputeEngine, so under that setting it measures nothing and reports
#    success. It gets its own launch with the default engine set.
#
# Results: $RESULTS/<label>/. Log: $LOG. Feed the log to analyze_sweep.py for the
# per-run accounting, and to recover_from_log.py if a run was killed before it
# could write its CSVs.
#
# You usually do not want the whole thing. Two knobs make it reproducible one row
# at a time, which is what checking a single README figure needs:
#
#   LIST=1 ./run_full_sweep.sh              # print every label and its launch
#                                           # arguments; touches no device
#   ONLY=single_gemm_ops ./run_full_sweep.sh # run just that label
#   ONLY=basic_index_ok,basic_index_slow ... # or several, comma-separated
#
# Under ONLY the log and $RESULTS layout are unchanged, so analyze_sweep.py works
# on a one-label log exactly as on a full one. Every label except xccl4, d2d,
# chip4_gemm and chip4_mem runs on a single logical NeuronCore (`--device 0`); see
# ../README.md, "Reproduce one row at a time".
set -u

IMAGE=${IMAGE:-xpu-perf-eager:latest}
REPO=${REPO:-$(cd ../.. && pwd)}          # repo root, mounted at /xpu-perf
LOG=${LOG:-/tmp/neuron_sweep.log}
RESULTS=${RESULTS:-/tmp/sweep_results}
WS4=${WS4:-/tmp/xccl_ws4}                 # capped copies of workloads/xccl_ops
DOCKER=${DOCKER:-sudo docker}
WAIT_BUDGET_S=${WAIT_BUDGET_S:-1800}
TOOLS=$(cd "$(dirname "$0")" && pwd)
ONLY=${ONLY:-}
LIST=${LIST:-}

# LIST writes to the terminal, not the log: its whole purpose is to be read now.
if [ -z "$LIST" ]; then
    mkdir -p "$RESULTS"
    exec >>"$LOG" 2>&1
    echo "=============== sweep started $(date -Is) ==============="
fi

# ONLY=<label>[,<label>...] or ONLY="<label> <label>". Both separators work: this
# script and run_new_workloads.sh would otherwise disagree on the delimiter, which
# is a trap worth spending one substitution on. The match is on a whole delimited
# word, so ONLY=gemm cannot also select single_gemm_ops.
want() {
    [ -z "$ONLY" ] && return 0
    case " ${ONLY//,/ } " in *" $1 "*) return 0;; esac
    return 1
}

# Noise filter: three lines per case, thousands of cases.
NOISE="NRT:nrta_tensor|_warn_trace_cache_once|NKI_ENABLE_TRACE_CACHE|^\\\\_NEFF"

# Anything holding /dev/neuron* other than the monitoring tools, which open it
# read-only and do not reserve a core.
count_holders() {
    sudo bash -c '
      n=0
      for p in /proc/[0-9]*; do
        comm=$(cat "$p/comm" 2>/dev/null)
        case "$comm" in neuron-monitor|neuron-top|neuron-ls|neuron-profile) continue;; esac
        for fd in "$p"/fd/*; do
          case "$(readlink "$fd" 2>/dev/null)" in
            /dev/neuron*) n=$((n+1)); break;;
          esac
        done
      done
      echo $n'
}

# Containers with a python process in them. Plain `docker ps -q | wc -l` is the
# wrong test: an interactive `docker run -it` container that someone left open
# after their sweep finished holds no NeuronCore and blocks nothing, but it would
# make the count non-zero forever. A container about to grab a core, on the other
# hand, already has python running before it opens /dev/neuron*, so this catches
# the race that count_holders alone would miss.
count_busy_containers() {
    local c n=0
    for c in $($DOCKER ps -q); do
        # `-o args` alone is rejected: docker top insists the ps format include
        # pid ("Couldn't find PID field in ps output").
        if $DOCKER top "$c" -o pid,args 2>/dev/null | grep -q "[p]ython"; then
            n=$((n + 1))
        fi
    done
    echo $n
}

# Raise WAIT_BUDGET_S when someone else's job is already on the chip: the
# default 30 min is enough to outlast a stuck teardown, not a neighbour's sweep.
wait_for_free() {
    local waited=0 holders others
    while true; do
        holders=$(count_holders)
        others=$(count_busy_containers)
        if [ "$holders" -eq 0 ] && [ "$others" -eq 0 ]; then return 0; fi
        if [ "$waited" -ge "$WAIT_BUDGET_S" ]; then
            echo "[$(date -Is)] STILL BUSY after ${waited}s (holders=$holders busy_containers=$others)"
            return 1
        fi
        sleep 30
        waited=$((waited + 30))
    done
}

# run_one <label> <timeout_s> <device_list> <env_prefix> <launch args...>
# env_prefix is passed to `env` verbatim and split on spaces; use "-" for none.
# Both are positional rather than inherited from the environment on purpose: a
# `VAR=x run_one ...` prefix on a *function* call leaves VAR set afterwards in
# bash, so the setting would silently leak into every later run.
run_one() {
    local label="$1"; shift
    local tmo="$1"; shift
    local dev="$1"; shift
    local env_prefix="$1"; shift
    if [ -n "$LIST" ]; then
        printf '%-26s dev=%-8s budget=%-7s %s\n' \
            "$label" "$dev" "${tmo}s" "$*"
        return 0
    fi
    want "$label" || return 0
    [ "$env_prefix" = "-" ] && env_prefix=""
    local cname="xpu_sweep_$label"
    local start rc runpid wdpid
    start=$(date +%s)
    echo ""
    echo "########## RUN $label   tmo=${tmo}s   dev=$dev   $(date -Is)"
    wait_for_free || { echo "########## RUN $label SKIPPED (busy)"; return 1; }

    $DOCKER rm -f "$cname" >/dev/null 2>&1

    $DOCKER run --rm --name "$cname" --privileged \
        -v "$REPO":/xpu-perf -v "$RESULTS":/results -v "$WS4":/xccl_ws4 \
        -w /xpu-perf/projects/micro_perf \
        -e PYTHONPATH=/xpu-perf/src "$IMAGE" \
        env $env_prefix python launch.py --backend NEURON --device "$dev" \
        --report_dir "/results/$label" "$@" \
        > >(grep -vE "$NOISE") 2>&1 &
    runpid=$!   # docker itself; note there is no pipeline here, see reason 1

    ( sleep "$tmo"; $DOCKER kill "$cname" >/dev/null 2>&1 \
        && echo "########## RUN $label KILLED by watchdog after ${tmo}s" ) &
    wdpid=$!

    wait "$runpid"; rc=$?
    kill "$wdpid" 2>/dev/null
    wait "$wdpid" 2>/dev/null

    $DOCKER rm -f "$cname" >/dev/null 2>&1
    echo "########## RUN $label exit=$rc elapsed=$(( $(date +%s) - start ))s $(date -Is)"
}

W=workloads

# The capped copies are cheap to make and xccl4/d2d cannot run without them, so
# this runs for any ONLY selection -- but not for LIST, which must not write.
if [ -z "$LIST" ]; then
    python3 "$TOOLS/cap_xccl_workloads.py" $W/xccl_ops "$WS4" --world-size 4
fi

# 1. The quantized ops first: they are the slowest per case and the most often
#    asked about, and burying them behind ~2000 basic cases means a wedge
#    upstream costs them. 1,380 + 736 cases, of which only the int8 slice runs.
run_one moe_quant_group_gemm 14400 0 - --workload $W/llm/vendor_test/moe_quant_group_gemm.json
run_one quant_matmul         18000 0 - --workload $W/llm/vendor_test/quant_matmul.json
run_one fa_vendor_test       3600  0 - --workload $W/llm/vendor_test/flash_attention.json

run_one demo_moe_quant_group_gemm 7200 0 - --workload $W/llm/vendor_test_demo/moe_quant_group_gemm.json
run_one demo_quant_matmul         3600 0 - --workload $W/llm/vendor_test_demo/quant_matmul.json
run_one demo_flash_attention      3600 0 - --workload $W/llm/vendor_test_demo/flash_attention.json

# 2. Single-op LLM workloads. ccl_ops.json is excluded on purpose: every case
#    asks for world_size 8 and a trn2.3xlarge has 4 logical cores, so there is
#    nothing in it this instance can run.
for wl in gemm_ops fa_ops fa_linear_ops pre_fa_ops moe_dispatch_ops moe_combine_ops \
          norm_ops activation_ops moe_gating_ops quant_ops; do
    run_one "single_$wl" 5400 0 - --workload "$W/llm/single_test_ops/$wl.json"
done

# 3. Basic ops, one directory per launch.
for d in tensor_gemm_ops vector_linear_ops vector_activation_ops vector_sfu_ops \
         vector_norm_ops vector_reduction_ops; do
    run_one "basic_$d" 5400 0 - --task_dir "$W/basic/$d" --task all
done

# 4. Index ops, split. --task takes a comma-separated list (parse_tasks in
#    core/common_utils.py), so the three well-behaved ops go first and get their
#    numbers before the two compiler hazards can eat the budget.
run_one basic_index_ok   5400 0 - --task_dir $W/basic/vector_index_ops \
    --task embedding,index_select,index_add
run_one basic_index_slow 3600 0 - --task_dir $W/basic/vector_index_ops \
    --task gather,scatter

# 5. Collectives, all four logical cores. world_size 4 is the only size that
#    runs every op -- all_to_all rejects 2 -- and every other collective is
#    faster at 4 than at 2. The ready timeout has to fit a cold-cache compile of
#    the warmup all_reduce; the upstream default of 60 s does not.
run_one xccl4 10800 0,1,2,3 \
    "XPU_PERF_ENGINES=XCCLEngine XPU_PERF_XCCL_READY_TIMEOUT_S=2400" \
    --task_dir /xccl_ws4 \
    --task all_gather,all_reduce,reduce_scatter,all_to_all,device2host,host2device

# 6. device2device separately, under ComputeEngine. See reason 4.
run_one d2d 5400 0,1 "XPU_PERF_ENGINES=ComputeEngine" \
    --workload /xccl_ws4/device2device.json

# 7. The whole chip: the same compute-bound and memory-bound cases with all four
#    logical cores working at once.
#
#    Read this carefully before comparing it to the --device 0 runs above. A
#    ComputeEngine spawns one worker per device and they all pull from a single
#    shared input queue (ComputeEngine.start / BaseEngine.run in core/engine.py),
#    so this is NOT four cores cooperating on one case -- four different cases
#    run concurrently, and every latency reported is still a single-core latency,
#    just measured while three other cores compete for the same HBM. That makes
#    the diff against the single-core runs the measurement of interest: a per-chip
#    2.9 TB/s over 4 logical cores is ~725 GB/s each, and one core alone already
#    reaches 648, so if these numbers hold up the chip really does deliver ~2.6
#    TB/s in aggregate, and if they fall towards a quarter then one core was
#    already taking the whole budget.
#
#    XPU_PERF_ENGINES is required, not optional: with 4 devices and no filter the
#    launch also starts an XCCLEngine worker on every core, and the second set
#    cannot get cores.
run_one chip4_gemm 5400 0,1,2,3 "XPU_PERF_ENGINES=ComputeEngine" \
    --task_dir $W/basic/tensor_gemm_ops --task all
run_one chip4_mem  5400 0,1,2,3 "XPU_PERF_ENGINES=ComputeEngine" \
    --task_dir $W/basic/vector_linear_ops --task all

if [ -z "$LIST" ]; then
    echo ""
    echo "=============== sweep finished $(date -Is) ==============="
    echo "Now: python3 $TOOLS/analyze_sweep.py $LOG"
fi
