#!/usr/bin/env bash
#
# The GPU half of a Trainium-vs-GPU comparison: exactly the workload files the
# NEURON eager numbers were taken with, so the two report trees line up file for
# file and op for op.
#
# What is deliberately NOT here:
#
#   * workloads/xccl_ops/{all_reduce,all_gather,reduce_scatter,all_to_all,
#     device2device}.json and llm/single_test_ops/ccl_ops.json -- a single-GPU box
#     has world_size 1, so there is no collective to measure, and device2device
#     needs two devices. Same reason ccl_ops.json has no runnable case on a
#     trn2.3xlarge either (it asks for 8).
#   * --task all over a whole directory. Independent per-file launches mean one
#     op wedging the machine costs one file's report, not the sweep's.
#
# Also not here: workloads/xccl_ops/{device2host,host2device}.json. These look
# runnable on one GPU -- both files list a world_size 1 case -- and are not. Both
# ops live on XCCLEngine, and perf_engine.py:173 skips that engine when
# `len(device_ids) * node_world_size <= 1`, so a one-GPU launch writes
# `note: engine_not_started` for every case and exits 0. XPU_PERF_ENGINES=XCCLEngine
# does not override it; the env filter runs before that guard, not instead of it.
#
# TODO -- these three have Neuron numbers, are NOT blocked on a p5.4xlarge, and are
# missing only because nobody has run them. See GPU/README.md, "What this table does
# not cover yet", for what each one would settle:
#
#   * workloads/llm/single_test_ops/gemm_ops.json -- moe_gating_gemm (48% MFU on
#     Neuron and no GPU control), quant_matmul, moe_quant_group_gemm.
#   * workloads/llm/single_test_ops/moe_dispatch_ops.json -- moe_scatter_dynamic_quant,
#     0.7 GB/s on Neuron, i.e. gather/scatter territory; the GPU column is what
#     would say whether that is the op def or the lowering.
#   * workloads/llm/single_test_ops/moe_combine_ops.json -- moe_gather (9.2 GB/s
#     on Neuron, 68x off index_select on the same chip) and
#     moe_quant_group_gemm_combine.
#
# They have labels in section 5 below, reachable only by naming them with ONLY, so
# that a default run stays a description of what the published numbers were actually
# taken with.
#
# Read the results with the same analyzer as the Neuron side:
#   python3 vendor_ops/NEURON/tools/analyze_sweep.py <log>
# and compare MFU columns, not latencies -- see NEURON/README.md, "MFU". One
# H100 is a whole accelerator; one Neuron device is a *logical NeuronCore*, a
# quarter of a Trainium2 chip at the default LNC=2. Per-device latency ratios
# between the two are therefore meaningless on their own.
#
# Two execution modes, because a DLAMI already has a working CUDA PyTorch and
# building an image on top of it buys nothing:
#
#   MODE=host   (default) run against an existing interpreter. This is what the
#               published p5.4xlarge numbers were taken with -- PYTHON points at
#               the DLAMI's /opt/pytorch/bin/python and EXTRA_PYTHONPATH at a
#               `pip install --target` overlay holding the few reporting deps
#               (jsonlines, prettytable), so the user's environment is never
#               mutated. Here `timeout` does bound the run, since the process is
#               a direct child.
#   MODE=docker build Dockerfile.cuda first. Worth it only if you need
#               flash_attn / flashinfer / vllm providers, which are what that
#               image exists for.
#
# Usage:
#   RESULTS=/tmp/gpu_results ./run_comparison_sweep.sh 2>&1 | tee /tmp/gpu_sweep.log
#   MODE=docker RESULTS=/tmp/gpu_results ./run_comparison_sweep.sh 2>&1 | tee ...
#
# One row at a time, which is what checking a single README figure needs:
#
#   LIST=1 ./run_comparison_sweep.sh          # print every label; runs nothing
#   ONLY=gemm ./run_comparison_sweep.sh       # run just that label
#   ONLY=gemm,single_fa_linear_ops ...        # or several; a space-separated
#                                             # "gemm single_fa_linear_ops" also works
#   ONLY=qwen3_5_27b ./run_comparison_sweep.sh # or a whole prefix group: every
#                                             # qwen3_5_27b_* label, one command
#
# The model-shaped set in one tree, to be compared against the Neuron tree the same
# ONLY produces there:
#
#   RESULTS=/tmp/qwen3_5_27b_gpu ONLY=qwen3_5_27b ./run_comparison_sweep.sh \
#       2>&1 | tee /tmp/qwen3_5_27b_gpu.log
#
# qwen3_5_27b_ccl is in that group and needs 4 GPUs; on a one-GPU box it writes
# nothing, which is expected and not a failure of the rest.
#
# The log and $RESULTS layout are unchanged under ONLY, so the Neuron side's
# analyze_sweep.py works on a one-label log exactly as on a full one. Every label
# here is single-GPU; see ../README.md, "Reproduce one row at a time".

set -u

MODE=${MODE:-host}
PYTHON=${PYTHON:-/opt/pytorch/bin/python}
EXTRA_PYTHONPATH=${EXTRA_PYTHONPATH:-$HOME/xpudeps}
DOCKER=${DOCKER:-sudo docker}
IMAGE=${IMAGE:-xpu-perf-cuda:latest}
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)}
RESULTS=${RESULTS:-/tmp/gpu_results}
DEVICE=${DEVICE:-0}
ONLY=${ONLY:-}
LIST=${LIST:-}
# Passed to `env` verbatim, so `EXTRA_ENV="A=1 B=2"` -- an escape hatch for one-off
# reruns (e.g. XPU_PERF_ENGINES=, TORCH_LOGS=) without editing this file.
EXTRA_ENV=${EXTRA_ENV:-}

[ -z "$LIST" ] && mkdir -p "$RESULTS"

W=workloads

# ONLY=<label>[,<label>...] or ONLY="<label> <label>". Both separators work: this
# script and the NEURON sweep script would otherwise disagree on the delimiter, which
# is a trap worth spending one substitution on. The match is on a whole delimited
# word, so ONLY=gemm cannot also select single_gemm_ops.
#
# A token may also name a *prefix group*: it selects every label that begins with it
# followed by an underscore, so `ONLY=qwen3_5_27b` runs the whole model-shaped set in
# section 6 with one word. Kept identical to the Neuron script's `want()` on purpose
# -- the two sides of a comparison have to be selectable the same way. Whole-word
# matching still wins first, so a group name cannot shadow a label of the same name.
want() {
    [ -z "$ONLY" ] && return 0
    local tok
    for tok in ${ONLY//,/ }; do
        [ "$tok" = "$1" ] && return 0
        case "$1" in "$tok"_*) return 0;; esac
    done
    return 1
}

# run_one <label> <timeout_s> <launch-args...>
run_one() {
    local label="$1" tmo="$2"; shift 2
    local start rc
    if [ -n "$LIST" ]; then
        printf '%-24s budget=%-7s %s\n' "$label" "${tmo}s" "$*"
        return 0
    fi
    want "$label" || return 0
    start=$(date +%s)

    echo "########## RUN $label   tmo=${tmo}s   dev=$DEVICE   mode=$MODE   $(date -Is)"

    if [ "$MODE" = host ]; then
        ( cd "$REPO/projects/micro_perf" \
          && PYTHONPATH="$EXTRA_PYTHONPATH:$REPO/src" CUDA_VISIBLE_DEVICES="$DEVICE" \
             env $EXTRA_ENV timeout "$tmo" "$PYTHON" -u launch.py \
                 --backend GPU --device 0 \
                 --report_dir "$RESULTS/$label" "$@" 2>&1 )
        rc=$?
        # 124 is the shell convention for "timeout fired", worth distinguishing
        # from the op itself failing.
        [ "$rc" = 124 ] && echo "########## RUN $label KILLED by timeout after ${tmo}s"
    else
        local cname="xpu_gpu_$label"
        $DOCKER rm -f "$cname" >/dev/null 2>&1

        # --gpus and --ipc=host: flash_attn and vllm both want a large shm
        # segment, and the default 64 MB makes a dataloader-free benchmark fail
        # in confusing ways.
        $DOCKER run -d --name "$cname" \
            --gpus "device=$DEVICE" --ipc=host \
            -v "$REPO":/xpu-perf -v "$RESULTS":/results \
            -w /xpu-perf/projects/micro_perf \
            -e PYTHONPATH=/xpu-perf/src \
            "$IMAGE" \
            env $EXTRA_ENV python -u launch.py --backend GPU --device "$DEVICE" \
                --report_dir "/results/$label" "$@" >/dev/null

        # A `timeout docker run` would signal the client, not the container,
        # which is a child of the daemon -- same trap as on the Neuron side.
        ( sleep "$tmo"; $DOCKER kill "$cname" >/dev/null 2>&1 \
            && echo "########## RUN $label KILLED by watchdog after ${tmo}s" ) &
        local wdpid=$!

        $DOCKER wait "$cname" >/dev/null 2>&1
        rc=$($DOCKER inspect -f '{{.State.ExitCode}}' "$cname" 2>/dev/null || echo "?")
        $DOCKER logs "$cname" 2>&1
        kill "$wdpid" 2>/dev/null
        wait "$wdpid" 2>/dev/null
        $DOCKER rm -f "$cname" >/dev/null 2>&1
    fi

    echo "########## RUN $label exit=$rc elapsed=$(( $(date +%s) - start ))s $(date -Is)"
}

# 1. The headline compute number, and the fp8 contrast. gemm.json enumerates 856
#    cases including the 24 fp8 ones, which are last. On an H100 these are fast
#    (a real fp8 tensor-core path, unlike Neuron's software up-cast), so unlike
#    the Neuron run this does not need splitting out.
run_one gemm 14400 --workload $W/basic/tensor_gemm_ops/gemm.json

# 2. Memory-bound ops -- the same six basic/ directories the Neuron memory-bound
#    table covers. The interesting column is GB/s against 3.35 TB/s (H100 SXM)
#    vs ~725 GB/s (one logical NeuronCore at LNC=2).
#
#    vector_index_ops is last on purpose: `gather` and `scatter` are the two ops
#    that wedged neuronx-cc for hours on the Neuron side, so keep them where a
#    watchdog kill costs least. It turns out to be cheap insurance rather than a
#    real risk -- the guess that the *base op def* is partly to blame (GatherOp
#    builds an output-shaped index where IndexSelectOp passes a 1-D one) does not
#    survive measurement: on CUDA `gather` runs at 2,805 GB/s against
#    `index_select`'s 2,862, i.e. the index construction costs nothing and the
#    Neuron result is entirely a lowering problem.
for d in vector_linear_ops vector_activation_ops vector_norm_ops \
         vector_reduction_ops vector_sfu_ops vector_index_ops; do
    run_one "basic_$d" 7200 --task_dir "$W/basic/$d" --task all
done

# 3. The four op families this port wrote workload files for.
for wl in norm_ops activation_ops moe_gating_ops quant_ops; do
    run_one "single_$wl" 5400 --workload "$W/llm/single_test_ops/$wl.json"
done

# 4. Attention. fa_linear_ops is the linear-cache file this port added, and is
#    what the Neuron GQA/decode numbers come from; the `torch` SDPA provider is
#    what makes it comparable, since `fa2` accepts only 4 of its 10 cases (no
#    linear decode, no batch_size > 1 prefill).
#
#    fa_ops is the paged-cache file, and it has no runnable case on *either*
#    backend -- installing flash_attn does not change that, so do not read the
#    empty single_fa_ops report as a missing dependency. All 11 cases set
#    block_size: 512, which is what makes cache_type "paged"
#    (core/utils.py:427). fa2/fa3 reject them from both ends:
#      * its 9 prefill cases, because FA2Op.vendor_parser demands
#        cache_type == "linear" for prefill (flash_attn_func takes no block
#        table; only flash_attn_with_kvcache does, and that is the decode path);
#      * its 2 decode cases, which are paged as the decode path wants, because
#        the same parser demands an all-bfloat16 dtype set and both carry
#        cache_dtype: int8.
#    So the file needs an int8-KV paged kernel, which no provider here has --
#    ops/flashinfer and ops/vllm hold only rms_norm.py. Filling this row means
#    writing that provider (or relaxing fa_ops to a bf16 cache), not pip install.
#    Under MODE=host with no flash_attn it still measures nothing and exits 0,
#    which is the trap the Dockerfile's `import flash_attn` assertion catches --
#    that assertion is about the *other* labels, not this one.
run_one single_fa_linear_ops 10800 --workload $W/llm/single_test_ops/fa_linear_ops.json
run_one single_fa_ops        10800 --workload $W/llm/single_test_ops/fa_ops.json

# 5. Not part of the published sweep, and skipped unless named with ONLY.
#
#    These are the files with a Trainium2 number and no GPU column that are NOT
#    hardware-blocked on a one-GPU box -- see GPU/README.md, "What this table does
#    not cover yet", for what each would settle. They exist as labels so that
#    `ONLY=single_gemm_ops ...` just works, and are gated so a default run remains
#    a description of what the published numbers were actually taken with.
#
#
#    single_pre_fa_ops is in this block for a different reason: the published
#    rotary_embedding row was measured by hand rather than through this script, so
#    the file has a number but no reproducible label. Running it would also pick up
#    store_kv_cache, which has no runnable case on any backend.
if [ -n "$ONLY" ] || [ -n "$LIST" ]; then
    run_one single_gemm_ops         5400 --workload $W/llm/single_test_ops/gemm_ops.json
    run_one single_moe_dispatch_ops 5400 --workload $W/llm/single_test_ops/moe_dispatch_ops.json
    run_one single_moe_combine_ops  5400 --workload $W/llm/single_test_ops/moe_combine_ops.json
    run_one single_pre_fa_ops       5400 --workload $W/llm/single_test_ops/pre_fa_ops.json
fi

# 6. Model-shaped workloads, also gated. Not part of the published comparison
#    table: every other label above sweeps powers of two, and these sweep one real
#    model's config.json instead, so mixing them into the default run would blur
#    what the table means. See workloads/models/qwen3_5_27b/README.md for where each
#    shape comes from.
#
#    qwen3_5_27b_ccl needs 4 devices and does nothing on a one-GPU box.
if [ -n "$ONLY" ] || [ -n "$LIST" ]; then
    Q=$W/models/qwen3_5_27b
    run_one qwen3_5_27b_gemm          7200 --workload $Q/gemm_ops.json
    run_one qwen3_5_27b_attention     5400 --workload $Q/attention_ops.json
    run_one qwen3_5_27b_norm          3600 --workload $Q/norm_ops.json
    run_one qwen3_5_27b_activation    3600 --workload $Q/activation_ops.json
    run_one qwen3_5_27b_pre_attention 3600 --workload $Q/pre_attention_ops.json
    run_one qwen3_5_27b_sampling      3600 --workload $Q/sampling_ops.json
    run_one qwen3_5_27b_deltanet      3600 --workload $Q/deltanet_ops.json
    run_one qwen3_5_27b_ccl           1800 --workload $Q/ccl_ops.json
fi

if [ -z "$LIST" ]; then
    echo ""
    echo "=============== GPU comparison sweep finished $(date -Is) ==============="
fi
