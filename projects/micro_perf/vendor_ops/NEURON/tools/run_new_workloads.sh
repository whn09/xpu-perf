#!/bin/bash
# Run only the workload files added after the 2026-09-01 full sweep, plus the
# whole-chip aggregate runs. Same watchdog and same accounting as
# run_full_sweep.sh -- see that script's header for why each launch is separate
# and why `timeout docker run` does not work here -- but it finishes in about an
# hour instead of most of a day.
#
# Covers:
#   * the four op families that had an op_def but no workload json anywhere in
#     the repo (add_rms_norm / head_rms_norm / swiglu / moe_softmax_topk and
#     their _dynamic_quant siblings, plus qk_rms_norm and the two standalone
#     quant ops);
#   * flash_attention over a linear cache, which is the only way to reach GQA and
#     decode -- every case in fa_ops.json sets block_size, i.e. a paged cache;
#   * gemm again, for the fp8 cases added to
#     workloads/basic/tensor_gemm_ops/gemm.json;
#   * tensor_gemm_ops and vector_linear_ops on all four logical cores at once.
#
# ONLY=<labels> restricts the run to a subset, which is what you want most of the
# time after the first pass -- re-running one file because its op changed should not
# mean sitting through gemm's 856 cases again. Labels are the first argument to
# run_one; either separator works, matching run_full_sweep.sh:
#
#   ONLY="chip4_gemm chip4_mem" ./run_new_workloads.sh
#   ONLY=single_norm_ops,single_quant_ops ./run_new_workloads.sh
#   LIST=1 ./run_new_workloads.sh     # print every label and its launch arguments
#                                     # to the terminal; touches no device
#
# See ../README.md, "Reproduce one row at a time".
set -u

IMAGE=${IMAGE:-xpu-perf-eager:latest}
REPO=${REPO:-$(cd ../.. && pwd)}
LOG=${LOG:-/tmp/neuron_new.log}
RESULTS=${RESULTS:-/tmp/new_results}
DOCKER=${DOCKER:-sudo docker}
WAIT_BUDGET_S=${WAIT_BUDGET_S:-1800}
ONLY=${ONLY:-}
LIST=${LIST:-}
TOOLS=$(cd "$(dirname "$0")" && pwd)

# LIST writes to the terminal, not the log: its whole purpose is to be read now.
if [ -z "$LIST" ]; then
    mkdir -p "$RESULTS"
    exec >>"$LOG" 2>&1
    echo "=============== new-workload sweep started $(date -Is) ==============="
fi

NOISE="NRT:nrta_tensor|_warn_trace_cache_once|NKI_ENABLE_TRACE_CACHE|^\\\\_NEFF"

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
    [ "$env_prefix" = "-" ] && env_prefix=""
    local cname="xpu_new_$label"
    local start rc runpid wdpid
    # Whole-word match on either separator, so ONLY=single_norm_ops does not also
    # select single_norm_ops_something later.
    if [ -n "$ONLY" ]; then
        case " ${ONLY//,/ } " in
            *" $label "*) ;;
            *) return 0;;
        esac
    fi
    start=$(date +%s)
    echo ""
    echo "########## RUN $label   tmo=${tmo}s   dev=$dev   $(date -Is)"
    wait_for_free || { echo "########## RUN $label SKIPPED (busy)"; return 1; }

    $DOCKER rm -f "$cname" >/dev/null 2>&1

    $DOCKER run --rm --name "$cname" --privileged \
        -v "$REPO":/xpu-perf -v "$RESULTS":/results \
        -w /xpu-perf/projects/micro_perf \
        -e PYTHONPATH=/xpu-perf/src "$IMAGE" \
        env $env_prefix python launch.py --backend NEURON --device "$dev" \
        --report_dir "/results/$label" "$@" \
        > >(grep -vE "$NOISE") 2>&1 &
    runpid=$!

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

for wl in norm_ops activation_ops moe_gating_ops quant_ops; do
    run_one "single_$wl" 5400 0 - --workload "$W/llm/single_test_ops/$wl.json"
done

# fa_linear_ops gets its own budget because it is dominated by compilation, not by
# execution: a cold 80/8/128 GQA prefill at q_len 4096 sat in walrus_driver for
# over 13 minutes and then measured 9.5 ms. Twelve cases at 5-15 minutes of
# neuronx-cc each does not fit the 5400 s the other files use, and a watchdog kill
# here loses the whole file's report rather than one case.
run_one single_fa_linear_ops 21600 0 - \
    --workload "$W/llm/single_test_ops/fa_linear_ops.json"

# fp8 gemm: gemm.json enumerates 856 cases (dtype x K.N x M over both blocks) and
# the 24 fp8 ones are last, each 20-570 s, so a budget sized for the float cases
# cuts the run before it reaches them. Copy out just the fp8 block if that is all
# you want -- the float cases here re-run as a regression check against the
# 2026-09-01 table.
run_one gemm_fp8 28800 0 - --workload $W/basic/tensor_gemm_ops/gemm.json

# A one-core run of the *same task_dir*, which is the only thing chip4_gemm can be
# compared against per shape. The 2026-09-01 sweep's per-case gemm jsonl no longer
# exists on disk, and comparing chip4's peak to a published peak only shows that the
# best case is unchanged -- it cannot show whether some particular shape degrades.
# Skip it only if you already have a matched single-core tree.
run_one core1_gemm 28800 0 - --task_dir $W/basic/tensor_gemm_ops --task all

# All four logical cores. See run_full_sweep.sh section 7 for what this does and
# does not measure -- in short, four independent cases in flight, so these are
# single-core latencies under three-way HBM contention.
run_one chip4_gemm 7200 0,1,2,3 "XPU_PERF_ENGINES=ComputeEngine" \
    --task_dir $W/basic/tensor_gemm_ops --task all
run_one chip4_mem  5400 0,1,2,3 "XPU_PERF_ENGINES=ComputeEngine" \
    --task_dir $W/basic/vector_linear_ops --task all

# Selection and sorting, one core and four, because these are the two ops where a
# Neuron core is within 8-32% of a whole H100 (see ../../GPU/README.md) and the
# per-chip claim therefore rests entirely on whether they scale. chip4_mem shows
# elementwise work scales at 1.00-1.02x, but topk and moe_softmax_topk are not
# elementwise -- they are the ops most likely to serialise on something shared, so
# the x4 has to be measured rather than carried over. Both are cheap: the whole
# 56-case moe_gating file takes 123 s here (the same file takes an H100 1,191 s).
run_one core1_reduction 5400 0       - --task_dir $W/basic/vector_reduction_ops --task all
run_one chip4_reduction 5400 0,1,2,3 "XPU_PERF_ENGINES=ComputeEngine" \
    --task_dir $W/basic/vector_reduction_ops --task all
run_one chip4_moe       5400 0,1,2,3 "XPU_PERF_ENGINES=ComputeEngine" \
    --workload "$W/llm/single_test_ops/moe_gating_ops.json"

if [ -z "$LIST" ]; then
    echo ""
    echo "=============== new-workload sweep finished $(date -Is) ==============="
    echo "Now: python3 $TOOLS/analyze_sweep.py $LOG"
fi
