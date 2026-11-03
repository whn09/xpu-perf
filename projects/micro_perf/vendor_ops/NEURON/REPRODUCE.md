# Reproducing NEURON results

## Reproduce one row at a time

The full sweep is most of a day, and checking one figure in the tables above does
not need it. Every row came from exactly one `launch.py` call, and each of those
calls has a label in `run_full_sweep.sh`:

```bash
cd projects/micro_perf

# What is there. Prints label, device list, budget and launch arguments; touches
# no device, so it is safe on a machine someone else is using.
LIST=1 vendor_ops/NEURON/tools/run_full_sweep.sh

# One row.
ONLY=single_gemm_ops IMAGE=xpu-perf-eager:latest \
    vendor_ops/NEURON/tools/run_full_sweep.sh
tail -f /tmp/neuron_sweep.log        # the script logs to a file, not the terminal

# Several. Commas or spaces, either works.
ONLY=basic_index_ok,basic_index_slow IMAGE=xpu-perf-eager:latest \
    vendor_ops/NEURON/tools/run_full_sweep.sh

# A whole family. A token also matches labels that start with it plus "_", so this
# runs all eight model-shaped files into one tree.
RESULTS=/tmp/qwen3_5_27b_neuron LOG=/tmp/qwen3_5_27b_neuron.log \
    ONLY=qwen3_5_27b vendor_ops/NEURON/tools/run_full_sweep.sh
```

`ONLY` matches whole labels, or a label prefix ending at an underscore — so
`ONLY=gemm` still selects nothing rather than quietly running `single_gemm_ops`, while
`ONLY=qwen3_5_27b` and `ONLY=chip4` select their families. Everything else is unchanged:
the log format and
the `$RESULTS/<label>/` layout are identical to a full run, so
`analyze_sweep.py` reads a one-label log the same way. The
[waiting-for-an-idle-chip behaviour](#reproducing-the-full-sweep) still applies to
each run — that is the point of going through the script rather than calling
`launch.py` by hand.

**25 of the 31 labels in `run_full_sweep.sh` are single-core** (`--device 0`, one
logical NeuronCore, a quarter of a chip at LNC=2). Elapsed times are from the logs
the published numbers came from, on an otherwise idle trn2.3xlarge:

| Label | Workload | Dev | Elapsed | Covers |
|---|---|---|---|---|
| `basic_tensor_gemm_ops` | `basic/tensor_gemm_ops` (all) | 0 | 4,939 s | [gemm](RESULTS.md#eager-runtime-full-sweep) — 856 cases, the longest single-core run. 4,939 s is the float cases only; the 24 fp8 ones are last at 20–570 s each, which is why this label's budget is 28,800 s and not the 5,400 the other directories get |
| `basic_vector_linear_ops` | `basic/vector_linear_ops` (all) | 0 | not recorded | [memory-bound](RESULTS.md#memory-bound-ops) |
| `basic_vector_activation_ops` | `basic/vector_activation_ops` (all) | 0 | not recorded | " |
| `basic_vector_sfu_ops` | `basic/vector_sfu_ops` (all) | 0 | not recorded | " |
| `basic_vector_norm_ops` | `basic/vector_norm_ops` (all) | 0 | 413 s | " |
| `basic_vector_reduction_ops` | `basic/vector_reduction_ops` (all) | 0 | 591 s | " |
| `basic_index_ok` | `basic/vector_index_ops`, `embedding,index_select,index_add` | 0 | not recorded | the three index ops that are fine |
| `basic_index_slow` | same dir, `gather,scatter` | 0 | **killed at 1,240 s** | [the two that are not](READING_THE_NUMBERS.md#two-index-ops-are-pathologically-slow-and-one-is-an-int64-index-away-from-parity). Expect a kill; results come back with `recover_from_log.py` |
| `single_gemm_ops` | `llm/single_test_ops/gemm_ops.json` | 0 | 182 s | `moe_gating_gemm` (48% MFU), `quant_matmul`, `moe_quant_group_gemm` |
| `single_fa_linear_ops` | `llm/single_test_ops/fa_linear_ops.json` | 0 | **killed at 784 s** | [attention](RESULTS.md#eager-runtime) — prefill, decode, GQA. It is compile-bound, not execution-bound: one cold GQA prefill sat in `walrus_driver` over 13 minutes and then measured 9.5 ms. Budget is now 21,600 s for that reason |
| `single_fa_decode_tkg` | `NEURON/workloads/fa_decode_tkg.json` | 0 | ~9 min | [decode through `attention_tkg`](RESULTS.md#decode-the-nkilib-provider-attention_tkg) — the same decode rows on a 128-aligned `kv_len`, extended to 16,384. Both the `torch` and `nkilib` providers run it, which is the comparison |
| `single_fa_ops` | `llm/single_test_ops/fa_ops.json` | 0 | 15 s | **0 cases** — paged, and no Neuron provider takes a block table. Not a Neuron-only gap: the GPU side is empty too, and `flash_attn` does not fix it ([why](../GPU/README.md#what-this-table-does-not-cover-yet)) |
| `single_pre_fa_ops` | `llm/single_test_ops/pre_fa_ops.json` | 0 | 177 s | `rotary_embedding`; `store_kv_cache` runs 0 cases |
| `single_moe_dispatch_ops` | `llm/single_test_ops/moe_dispatch_ops.json` | 0 | 32 s | `moe_scatter_dynamic_quant` |
| `single_moe_combine_ops` | `llm/single_test_ops/moe_combine_ops.json` | 0 | 61 s | `moe_gather` |
| `single_norm_ops` | `llm/single_test_ops/norm_ops.json` | 0 | 873 s | [norm/activation/MoE](READING_THE_NUMBERS.md#the-_dynamic_quant-family-is-90-180x-off-and-it-is-one-shared-helper) |
| `single_activation_ops` | `llm/single_test_ops/activation_ops.json` | 0 | 312 s | " |
| `single_moe_gating_ops` | `llm/single_test_ops/moe_gating_ops.json` | 0 | 123 s | " — and 9.7x faster than the same file on an H100 (1,191 s) |
| `single_quant_ops` | `llm/single_test_ops/quant_ops.json` | 0 | 172 s | " |
| `quant_matmul` | `llm/vendor_test/quant_matmul.json` | 0 | **killed at 1,988 s** | 85 of 184 cases; recover from the log |
| `moe_quant_group_gemm` | `llm/vendor_test/moe_quant_group_gemm.json` | 0 | **killed at 565 s** | 15 of 276 — [~16 min of wall clock per 2.7 s measurement](READING_THE_NUMBERS.md#most-workloadsllm-cases-do-not-run) |
| `fa_vendor_test` | `llm/vendor_test/flash_attention.json` | 0 | 16 s | **0 cases** |
| `demo_quant_matmul` | `llm/vendor_test_demo/quant_matmul.json` | 0 | 41 s | the small versions of the three above |
| `demo_moe_quant_group_gemm` | `llm/vendor_test_demo/moe_quant_group_gemm.json` | 0 | 38 s | " |
| `demo_flash_attention` | `llm/vendor_test_demo/flash_attention.json` | 0 | 17 s | " (0 cases) |
| `xccl4` | capped copies of `xccl_ops/` | 0,1,2,3 | 762 s | the 4 collectives plus `device2host`/`host2device`. Needs `XPU_PERF_ENGINES=XCCLEngine`, which the script sets |
| `d2d` | capped `xccl_ops/device2device.json` | 0,1 | 142 s | `device2device`, 648.7 GB/s |
| `chip4_gemm` | `basic/tensor_gemm_ops` (all) | 0,1,2,3 | 1,480 s | whether "x4" is real. Joins per shape against `basic_tensor_gemm_ops` |
| `chip4_mem` | `basic/vector_linear_ops` (all) | 0,1,2,3 | 187 s | " — against `basic_vector_linear_ops` |
| `chip4_reduction` | `basic/vector_reduction_ops` (all) | 0,1,2,3 | not recorded | " — against `basic_vector_reduction_ops`. `topk` is where a Neuron core is closest to a whole H100, so the x4 there has to be measured rather than carried over |
| `chip4_moe` | `llm/single_test_ops/moe_gating_ops.json` | 0,1,2,3 | not recorded | " — against `single_moe_gating_ops`, for the same reason |

"not recorded" means that label ran under an earlier version of the script that did
not print an elapsed line, not that it failed. **A kill is not a failure either**:
micro_perf writes its CSVs only when a launch finishes, but every case is printed as
it completes, so `recover_from_log.py <log> <outdir>` rebuilds the reports — several
of the published numbers came through that path. Whether a 137 was the watchdog or a
hand kill to free the chip is not recoverable from the log, so treat these elapsed
figures as "how long it ran", not "how long it needs".

There used to be a second script here, `run_new_workloads.sh`, holding "just the
workload files added after the 2026-09-01 sweep" so that a re-run took an hour
instead of a day. `ONLY=` already does that, and maintaining two label lists meant
they drifted in ways that cost results: `fa_linear_ops` had a 21,600 s budget in one
and a fatal 5,400 s in the other, `gemm`'s fp8 tail was reachable only from the
newer script, and `attention_tkg` was added only to the older one. It has been
folded in — `chip4_reduction` and `chip4_moe` are now section 7, and its
`core1_gemm`/`core1_reduction` were `basic_tensor_gemm_ops` and
`basic_vector_reduction_ops` under different names.

The GPU side is label-for-label the same idea and much cheaper — 13 labels, all
single-GPU, 42 minutes end to end. See
[GPU/README.md, Reproduce one row at a time](../GPU/README.md#reproduce-one-row-at-a-time);
two labels are spelled differently there (`gemm` for `basic_tensor_gemm_ops`, and
one `basic_vector_index_ops` instead of the `basic_index_ok`/`basic_index_slow`
split, which exists here only because `gather`/`scatter` need their own watchdog).

## Reproducing the full sweep

`tools/` holds what produced the eager full-sweep numbers. `launch.py --task all`
does not survive this workload set; the comment block at the top of
`run_full_sweep.sh` lists the four separate reasons.

```bash
cd projects/micro_perf

# 1. Build the image (as in Quick start).
docker build -f vendor_ops/NEURON/tools/Dockerfile.eager \
    -t xpu-perf-eager:latest vendor_ops/NEURON/tools

# 2. Sweep, on an idle trn2.3xlarge. Budget a day.
IMAGE=xpu-perf-eager:latest setsid nohup vendor_ops/NEURON/tools/run_full_sweep.sh &

# 2b. Or a subset -- ONLY takes any set of labels, so "everything I changed since
#     the last sweep" is a list, not a second script. About an hour for this one.
ONLY="single_norm_ops,single_activation_ops,single_quant_ops,single_fa_decode_tkg" \
    IMAGE=xpu-perf-eager:latest vendor_ops/NEURON/tools/run_full_sweep.sh

# 3. Per-run accounting: cases tried, measured, and grouped rejection reasons.
python3 vendor_ops/NEURON/tools/analyze_sweep.py /tmp/neuron_sweep.log

# 4. One op's full scaling curve instead of a summary line.
python3 vendor_ops/NEURON/tools/analyze_sweep.py /tmp/neuron_sweep.log all_to_all

# 5. Is "x4" real? Join a 1-core report against a 4-core one, shape by shape.
#    Only matched pairs over the same task_dir are comparable -- comparing a chip4
#    peak against a published peak shows the best case is unchanged but cannot show
#    that some particular shape degrades, and the tail is where they do.
ONLY="basic_tensor_gemm_ops chip4_gemm basic_vector_reduction_ops chip4_reduction \
      single_moe_gating_ops chip4_moe" \
    IMAGE=xpu-perf-eager:latest vendor_ops/NEURON/tools/run_full_sweep.sh
python3 vendor_ops/NEURON/tools/analyze_scaling.py \
    /tmp/sweep_results/basic_tensor_gemm_ops /tmp/sweep_results/chip4_gemm

# 6. The gather int64-vs-int32 finding, on device, in one file (~2 min).
sudo docker run --rm --privileged -v "$PWD":/w -w /w xpu-perf-eager:latest \
    python3 vendor_ops/NEURON/tools/probe_index_dtype.py

# 7. The in-place-write finding: the slope test, plus rotary_embedding
#    decomposed at the shape the table publishes (~4 min, needs a few GB HBM).
sudo docker run --rm --privileged -v "$PWD":/w -w /w xpu-perf-eager:latest \
    python3 vendor_ops/NEURON/tools/probe_inplace_write.py

# 8. Where a case's wall clock goes, phase by phase (~1 min). Cross-backend:
#    --backend GPU runs the same probe on an H100. Repeating a case index
#    (--cases 0,0,0) is what shows the per-shape warm-up. See "Sweep wall clock
#    is not a performance comparison". Unlike 6 and 7 this one imports xpu_perf,
#    so it needs the repo root mounted, not just projects/micro_perf.
sudo docker run --rm --device /dev/neuron0 -v "$(cd ../.. && pwd)":/xpu-perf \
    -w /xpu-perf/projects/micro_perf -e PYTHONPATH=/xpu-perf/src \
    -e NEURON_RT_VISIBLE_CORES=0 xpu-perf-eager:latest \
    python3 -u vendor_ops/NEURON/tools/probe_case_overhead.py --backend NEURON \
        --workload workloads/basic/tensor_gemm_ops/gemm.json --limit 8
```

`IMAGE`, `REPO`, `LOG`, `RESULTS`, `DOCKER` and `WAIT_BUDGET_S` all come from the
environment, so a different image or a non-`sudo` docker needs no edit. Three
things to know:

- **Both scripts refuse to start a run while anything else holds a NeuronCore**,
  and wait `WAIT_BUDGET_S` (default 1800) for it to clear before skipping that
  run. Raise it when you are queueing behind someone else's sweep. What they wait
  on is a process with `/dev/neuron*` open, plus any container with a `python` in
  it — an idle `docker run -it` shell left open after a sweep does not count.

- **A killed run writes no report, but its results are recoverable.** micro_perf
  writes CSV/jsonl only when a launch finishes, so a watchdog kill loses
  everything on disk — but every case is printed to stdout as it completes.
  `recover_from_log.py <log> <outdir>` rebuilds the CSVs from that. Several of
  the numbers above came through this path.
- **Measured counts are not enumerated counts.** Compare `analyze_sweep.py`'s
  `measured` against the table in
  [Most `workloads/llm/` cases do not run](READING_THE_NUMBERS.md#most-workloadsllm-cases-do-not-run),
  not against what the launcher printed at startup.

The script deliberately skips `workloads/llm/single_test_ops/ccl_ops.json` and
caps `workloads/xccl_ops/` sizes. Both exclusions are load-bearing, not tidying:
the second is what keeps a rank from OOMing, which would hang the launch forever.

