#!/bin/bash
# Multi-chip sweep for the eager (PyTorch-native) Neuron stack on a
# trn2.48xlarge: 16 trn2 devices x 4 logical NeuronCores = 64 devices as
# micro_perf counts them.
#
# ../RESULTS.md's numbers were all taken on a trn2.3xlarge, which is one device
# and four logical cores. Everything about crossing a chip boundary was therefore
# unmeasurable there, and three claims in the docs were left open:
#
#   * workloads/llm/single_test_ops/ccl_ops.json asks for world_size 8 in every
#     case and so had *no runnable case at all* -- the only file in the repo of
#     which that was true;
#   * the collective rows stopped at world_size 4, which on a 3xlarge is four
#     cores inside one chip, i.e. the NeuronLink numbers were not NeuronLink
#     numbers, they were on-package;
#   * "one chip is 4x one logical core" was measured at 4 cores and asserted no
#     further.
#
# This script is deliberately separate from run_full_sweep.sh rather than more
# labels inside it. run_full_sweep.sh is a single-device script -- 33 of its 39
# labels are `--device 0` -- and it is the reproduction recipe for a published
# table taken on a 3xlarge. Folding 64-core labels in would mean every label had
# to grow a "which instance type" condition, and the note at the top of that
# script about run_new_workloads.sh is about the opposite failure (two files with
# overlapping label lists that drift). There is no overlap here: no label below
# exists there, and every label below needs more than one chip.
#
# Same conventions as run_full_sweep.sh, and for the same hard-won reasons -- see
# its header for all four:
#   * one launch per workload file, each with its own --report_dir, because
#     reports are written only when a launch finishes;
#   * a `docker kill` watchdog, because `timeout docker run` bounds nothing;
#   * XPU_PERF_ENGINES set explicitly on every multi-device label, because
#     otherwise a ComputeEngine worker and an XCCLEngine worker both try to
#     reserve every core and the second set cannot get one;
#   * --numa -1. On this instance the default (--numa unset) is *wrong*: it
#     starts one process group per NUMA node and splits the device list between
#     them in index order, but the devices are not laid out that way -- devices
#     0-3 and 12-15 are on NUMA node 1 and devices 4-11 are on node 0, so an
#     index-order split pins most workers to the wrong socket. --numa -1 asks
#     for no pinning at all, which is the honest default until someone measures
#     the pinned variants (see the numa_pin labels below, which do exactly that
#     for one case).
#
# Usage, from projects/micro_perf on a trn2.48xlarge:
#
#   LIST=1 vendor_ops/NEURON/tools/run_multicard_sweep.sh    # print, touch nothing
#   RESULTS=/tmp/mc LOG=/tmp/mc.log vendor_ops/NEURON/tools/run_multicard_sweep.sh
#   ONLY=ccl8_llm ...                                        # one label
#   ONLY=xccl ...                                            # a prefix group
#
# Expect several hours. Results: $RESULTS/<label>/. Log: $LOG.
set -u

IMAGE=${IMAGE:-xpu-perf-eager:latest}
REPO=${REPO:-$(cd ../.. && pwd)}
LOG=${LOG:-/tmp/neuron_multicard.log}
RESULTS=${RESULTS:-/tmp/multicard_results}
CAPPED=${CAPPED:-/tmp/xccl_capped}        # capped workload copies, per world_size
DOCKER=${DOCKER:-sudo docker}
WAIT_BUDGET_S=${WAIT_BUDGET_S:-1800}
TOOLS=$(cd "$(dirname "$0")" && pwd)
ONLY=${ONLY:-}
LIST=${LIST:-}

# Cores per device, so the device lists below are derived rather than hardcoded:
# at the default LNC=2 a trn2 chip is 4 logical cores, at LNC=1 it is 8 and every
# list here would be wrong by a factor of two.
CORES_PER_CHIP=${CORES_PER_CHIP:-4}

# --numa for every launch. -1 is "no CPU pinning"; see section 6 for why that is
# the default here and how to measure the alternative.
NUMA=${NUMA:--1}

# seq_dev <n>            -> "0,1,...,n-1"          (n cores, filling chips in order)
# stride_dev <n> <step>  -> "0,step,2*step,..."    (n cores, one per chip at LNC=2)
seq_dev()    { local i s=""; for ((i=0; i<$1; i++)); do s+="${i},"; done; echo "${s%,}"; }
stride_dev() { local i s=""; for ((i=0; i<$1; i++)); do s+="$((i * $2)),"; done; echo "${s%,}"; }

if [ -z "$LIST" ]; then
    mkdir -p "$RESULTS"
    exec >>"$LOG" 2>&1
    echo "=============== multicard sweep started $(date -Is) ==============="
    neuron-ls 2>/dev/null | head -30
fi

want() {
    [ -z "$ONLY" ] && return 0
    local tok
    for tok in ${ONLY//,/ }; do
        [ "$tok" = "$1" ] && return 0
        case "$1" in "$tok"_*) return 0;; esac
    done
    return 1
}

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

count_busy_containers() {
    local c n=0
    for c in $($DOCKER ps -q); do
        if $DOCKER top "$c" -o pid,args 2>/dev/null | grep -q "[p]ython"; then
            n=$((n + 1))
        fi
    done
    echo $n
}

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
        printf '%-22s ranks=%-3s dev=%s\n                       budget=%-7s %s\n' \
            "$label" "$(($(tr -cd , <<<"$dev" | wc -c) + 1))" "$dev" "${tmo}s" "$*"
        return 0
    fi
    want "$label" || return 0
    [ "$env_prefix" = "-" ] && env_prefix=""
    local cname="xpu_mc_$label"
    local start rc runpid wdpid
    start=$(date +%s)
    echo ""
    echo "########## RUN $label   tmo=${tmo}s   dev=$dev   $(date -Is)"
    wait_for_free || { echo "########## RUN $label SKIPPED (busy)"; return 1; }

    $DOCKER rm -f "$cname" >/dev/null 2>&1

    $DOCKER run --rm --name "$cname" --privileged \
        -v "$REPO":/xpu-perf -v "$RESULTS":/results -v "$CAPPED":/capped \
        -w /xpu-perf/projects/micro_perf \
        -e PYTHONPATH=/xpu-perf/src "$IMAGE" \
        env $env_prefix python launch.py --backend NEURON --device "$dev" \
        --numa "$NUMA" --report_dir "/results/$label" "$@" \
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
XCCL="XPU_PERF_ENGINES=XCCLEngine XPU_PERF_XCCL_READY_TIMEOUT_S=3600"
COMPUTE="XPU_PERF_ENGINES=ComputeEngine"

# Capped copies. Same 24 GiB-per-core reason as run_full_sweep.sh; see
# cap_xccl_workloads.py. One directory per world_size because the cap script
# pins world_size, which is what makes ccl_ops.json runnable here at all.
if [ -z "$LIST" ]; then
    python3 "$TOOLS/cap_xccl_workloads.py" \
        $W/llm/single_test_ops/ccl_ops.json "$CAPPED/ccl_ws8" --world-size 8
    for ws in 2 4 8 16 32 64; do
        python3 "$TOOLS/cap_xccl_workloads.py" $W/xccl_ops \
            "$CAPPED/ws$ws" --world-size "$ws" --dtypes bfloat16
    done
fi

# 1. The headline gap-closer: the one workload file with no runnable case on a
#    3xlarge. world_size 8 spans two chips, so this is also the first NeuronLink
#    measurement in the repo. All four ops, all three dtypes it asks for.
run_one ccl8_llm 7200 "$(seq_dev 8)" "$XCCL" --workload /capped/ccl_ws8/ccl_ops.json

# 2. world_size scaling, bf16 only. The question here is how bus_bw moves as
#    ranks are added, and one dtype answers it in a quarter of the cases -- the
#    dtype dependence is already covered at world_size 4 in ../RESULTS.md.
#
#    ws4 is included even though run_full_sweep.sh has an xccl4: this one is the
#    same 4 ranks on *this* instance, which is the control the ws8/16/32/64 rows
#    need. Comparing them to a number taken on a different instance type would
#    confound rank count with everything else that differs.
#
#    device2host/host2device are excluded on purpose: with 64 ranks they all
#    share one host DMA path and measure that contention, not the link. Same
#    reason the world_size 4 host-transfer rows in ../RESULTS.md are flagged as
#    not comparable to the world_size 2 ones.
#
#    broadcast is in the task list from ws4 up. It was registered in
#    op_defs/basic_ops/xccl_ops.py with no workload file anywhere in the repo, so
#    it had never once run on any backend -- and when workloads/xccl_ops/
#    broadcast.json was added it turned out its summary() override had drifted
#    twice from BasicOp.summary and killed every case. Worth its place here rather
#    than for completeness: it is the one collective whose cost is (n-1) hops of
#    the *whole* payload on a ring and log(n) on a tree, so it is where the
#    runtime's algorithm choice is most visible -- and on this instance algorithm
#    choice is not free (see section 6).
for ws in 4 8 16 32 64; do
    run_one "xccl_ws$ws" 14400 "$(seq_dev "$ws")" "$XCCL" \
        --task_dir "/capped/ws$ws" \
        --task all_gather,all_reduce,reduce_scatter,all_to_all,broadcast
done

# 2b. all_reduce_h2d, the third op that had no workload file. It is not another
#     all_reduce: it issues one with async_op=True and then does a host-to-device
#     copy while it is in flight, so it measures *overlap* of the collective with
#     the host DMA -- the single most load-bearing overlap in TP inference, and
#     unmeasured on both backends until now.
#
#     Its own label because the op def issues the all_reduce on the *default*
#     group rather than op_group, so the group is always every rank the launch
#     started and `world_size` in the file only feeds the bus_size formula. 8
#     devices, and the file's ws=8 cases, so the two agree.
#
#     Read it against xccl_ws8's all_reduce and against host2device at the same
#     rank count -- latency near max(the two) means they overlap, near the sum
#     means they do not. Do not read it alone: 8 ranks each copying the same
#     payload share one host link, so the H2D half is contended by construction
#     and the uncontended host2device number is not the right control.
run_one ccl8_h2d 7200 "$(seq_dev 8)" "$XCCL" \
    --workload /capped/ws8/all_reduce_h2d.json

# 2c. p2p -- the only op in the repo that measures a real point-to-point transfer
#     (dist.isend on one rank, dist.irecv on another), and the gap left by
#     device2device turning out to be a local copy.
#
#     **Expected to fail on this backend today**, and kept for that reason: the
#     native runtime's torch.distributed backend has no send/recv at all --
#     "RuntimeError: Backend neuron does not support send" / "... does not support
#     recv" -- so these three labels are a 1-minute check of whether a Neuron
#     release has gained it, and the file's real first use is a multi-GPU box.
#     Placement follows the d2d labels exactly, so if it ever runs the hop ladder
#     is already here.
run_one p2p_hop0 900 0,1 "$XCCL" --workload /capped/ws2/p2p.json
run_one p2p_hop1 900 "0,$CORES_PER_CHIP" "$XCCL" \
    --workload /capped/ws2/p2p.json
run_one p2p_hop2 900 "0,$((2 * CORES_PER_CHIP))" "$XCCL" \
    --workload /capped/ws2/p2p.json

# 3. Where the rank count is held fixed and only the topology changes -- the
#    comparison a 3xlarge cannot express at all. Four ranks, twice:
#      hop0: cores 0-3, all four inside chip 0, no NeuronLink hop;
#      hop1: cores 0,4,8,12, one core in each of chips 0,1,2,3.
#    On the 4x4 torus chip 0's neighbours are 1, 3, 4 and 12, so chips 0-3 is a
#    row of the torus: 0-1 and 1-2 and 2-3 are one hop, 0-2 is two. A ring
#    all_reduce over that row therefore pays one hop per step and this is the
#    cheapest inter-chip case, not the average one.
run_one topo_hop0 10800 "$(seq_dev 4)" "$XCCL" \
    --task_dir /capped/ws4 \
    --task all_gather,all_reduce,reduce_scatter,all_to_all,broadcast
run_one topo_hop1 10800 "$(stride_dev 4 "$CORES_PER_CHIP")" "$XCCL" \
    --task_dir /capped/ws4 \
    --task all_gather,all_reduce,reduce_scatter,all_to_all,broadcast

# 4. device2device, which is a copy and not a collective, so ComputeEngine and
#    only two devices. Three variants, and the pair of them is the point:
#      intra: cores 0,1 -- same chip, on-package, this is the 648 GB/s row in
#             ../RESULTS.md;
#      hop1:  cores 0,4 -- chip 0 to chip 1, adjacent on the torus;
#      hop2:  cores 0,8 -- chip 0 to chip 2, two hops (0's neighbours are
#             1,3,4,12; 2 is not among them).
run_one d2d_intra 5400 0,1 "$COMPUTE" --workload /capped/ws4/device2device.json
run_one d2d_hop1  5400 "0,$CORES_PER_CHIP"       "$COMPUTE" \
    --workload /capped/ws4/device2device.json
run_one d2d_hop2  5400 "0,$((2 * CORES_PER_CHIP))" "$COMPUTE" \
    --workload /capped/ws4/device2device.json

# 5. Concurrency at 64 cores, which extends run_full_sweep.sh's chip4_* labels.
#    Read the same caveat as there: a ComputeEngine spawns one worker per device
#    and they all pull from one shared queue, so this is 64 *different* cases
#    running at once, every reported latency is still a single-core latency, and
#    the diff against the single-core run is the measurement. What it can show
#    that chip4 cannot: whether the per-core number holds when all 16 chips are
#    loaded, i.e. whether "x4 per chip" extends to "x64 per instance" or whether
#    something off-chip (host, PCIe, power) caps it first.
#    The conc1_* labels are the control, and they are not optional: the ratio has
#    to be against one core of *this* instance, measured the same way. The
#    single-core numbers in ../RESULTS.md are from a trn2.3xlarge, and dividing a
#    64-core number here by a 3xlarge number there would fold every difference
#    between the two instances into a figure reported as concurrency loss.
run_one conc64_gemm 14400 "$(seq_dev 64)" "$COMPUTE" \
    --task_dir $W/basic/tensor_gemm_ops --task all
run_one conc64_mem  10800 "$(seq_dev 64)" "$COMPUTE" \
    --task_dir $W/basic/vector_linear_ops --task all
#    conc1_gemm runs a *subset* of the gemm cases -- vendor_ops/NEURON/workloads/
#    conc_control/gemm.json, whose header says which shapes and why. The full 648
#    cases take minutes at 64-way and hours serially, and a run the watchdog kills
#    writes no report at all. The comparison is per case, not peak-to-peak, so a
#    subset costs nothing as long as it contains the shapes the 64-core run is
#    read at.
run_one conc1_gemm  14400 0 "$COMPUTE" \
    --task_dir vendor_ops/NEURON/workloads/conc_control --task all
run_one conc1_mem   10800 0 "$COMPUTE" \
    --task_dir $W/basic/vector_linear_ops --task all

# 6. Does it matter *which* chips you pick? Three more 8-rank runs of the same
#    file as ccl8_llm, differing only in the device list. Read `neuron-ls
#    --topology` alongside this: the 16 chips are a 4x4 torus in row-major order,
#    so chip 0's neighbours are 1, 3 (wrap), 4 and 12 (wrap), and chips 3 and 4
#    are *two* hops apart even though their indices are adjacent.
#      ccl8_llm:   devices 0-7   = chips 0,1  -- torus-adjacent, both NUMA node 1
#      numa_node0: devices 16-23 = chips 4,5  -- torus-adjacent, both NUMA node 0
#      numa_cross: devices 0-3 + 16-19 = chips 0,4 -- torus-adjacent, *different*
#                  NUMA nodes (chips 0-3 and 12-15 are node 1, 4-11 are node 0).
#                  This is the NUMA test: adjacency is held fixed, so a difference
#                  against the two above is host NUMA and nothing else.
#      topo_far8:  devices 12-19 = chips 3,4  -- two torus hops apart. Originally
#                  written as a NUMA straddle, on the mistaken belief that chips
#                  3 and 4 are neighbours; it is worth keeping because of what it
#                  found. A ring cannot be laid over 8 cores whose two chips are
#                  not adjacent, the runtime falls back to its Mesh algorithm, and
#                  Mesh caps a collective at 40 MiB:
#                    Collective operation ALLREDUCE with total_size 268435456 ...
#                    is currently not supported ... because message size is larger
#                    than 41943040 bytes
#                  so all_reduce, all_gather and reduce_scatter *fail outright* at
#                  the 512 MiB case here while all_to_all not only survives but
#                  goes 2.2x faster than on an adjacent pair. Which chips you pick
#                  therefore changes which collectives are legal, not just how
#                  fast they are.
#
#    CPU pinning is a *different* knob and is not tested here: run_one passes
#    --numa $NUMA, default -1 (unpinned), so a second invocation with
#    `NUMA=0 ONLY=xccl_ws8` measures pinning against the unpinned run of the same
#    label. Do not set NUMA for a run whose device list spans both nodes: the
#    launcher splits the device list between NUMA processes in index order, which
#    on this instance does not match where the devices actually are.
run_one numa_node0 7200 16,17,18,19,20,21,22,23 "$XCCL" \
    --workload /capped/ccl_ws8/ccl_ops.json
run_one numa_cross 7200 0,1,2,3,16,17,18,19 "$XCCL" \
    --workload /capped/ccl_ws8/ccl_ops.json
run_one topo_far8  7200 12,13,14,15,16,17,18,19 "$XCCL" \
    --workload /capped/ccl_ws8/ccl_ops.json

if [ -z "$LIST" ]; then
    echo ""
    echo "=============== multicard sweep finished $(date -Is) ==============="
    echo "Now: python3 $TOOLS/analyze_sweep.py $LOG"
fi
