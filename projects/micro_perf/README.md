# `xpu-perf`：micro_perf

## Introduction
micro_perf is a part of xpu-perf, which is mainly used to evaluate the performance of frequent computation and communication operators in mainstream deep learning models on new emerging heterogeneous hardwares. The main characteristics are as follows:

- Easy and quick access for diverse heterogeneous hardware
- Evaluation process fitting realistic business scenarios
- Coverage of frequent operators across multiple categories

## Quickstart

### 1. Prepare running environment
```
git clone git@github.com:bytedance/xpu-perf.git
pip3 install -e .
cd projects/micro_perf
pip3 install -r ./requirements.txt
```



### 2. (new) A quick start for remote bench
#### 2.1 Commands
```bash
# start bench server
python3 ./server.py --backend GPU --device 0,1,2,3

# send bench request
python3 ./client.py --task_dir ./workloads/basic --task add
python3 ./client.py --task_dir ./workloads/xccl_ops --task all_reduce
```


### 3. (old) A quick start for local bench 
#### 3.1 Commands
```bash
# local bench
python3 launch.py --backend GPU --device 0,1,2,3 --task_dir ./workloads/basic --task add
python3 launch.py --backend GPU --device 0,1,2,3 --task_dir ./workloads/xccl_ops --task all_reduce
```


## Expected Output
By default, reports are saved in the reports/ directory, and the specific parameters and performance metrics of the current test operator will also be printed to the terminal.

For different types of operators (Compute-bound / Memory-bound), we adopt various metrics to comprehensively evaluate the performance of the operator. Regarding the various metrics, the explanations are as follows:

### for computation ops
| Metric            | Unit          | Description |
| --------          | -------       | ------- |
| latency           | us            | kernel device e2e latency    |
| read_bytes        | B             | bytes read from memory |
| write_bytes       | B             | bytes write to memory |
| io_bytes          | B             | bytes read from memory and write to memory |
| mem_bw            | GB/s          | kernel memory bandwidth |
| calc_flops_power  | TFLOPS / TOPS | testing kernel computing power |
| calc_mem_ratio    | FLOPS / Byte  | algorithm roofline model |
| peak_tflops       | TFLOPS / TOPS | nominal computing power of one device for this compute dtype |
| mfu               | ratio         | `calc_flops_power / peak_tflops` |

`peak_tflops` and `mfu` are only reported where the backend publishes a nominal
figure for the op's compute dtype (`Backend.get_peak_tflops`). They are omitted,
not zeroed, otherwise -- a backend with no spec table, or a dtype the vendor has
never published a peak for, would otherwise look like an op that achieved
nothing. The dtype used is the compute dtype (`compute_dtype`, or
`qk_compute_dtype` for flash_attention), not the storage dtype, so a w8a8 matmul
is scored against the datapath that does the multiplying. For a memory-bound op
the MFU is genuinely near zero — read `mem_bw` there instead.

Example:
```
{
    "op_name": "gemm", 
    "sku_name": "NVIDIA A800-SXM4-80GB", 
    "provider": "default", 
    "arguments": {
        "arg_type": "default", 
        "dtype": "bfloat16", 
        "M": 32768, 
        "K": 8192, 
        "N": 8192
    }, 
    "targets": {
        "latency(us)": 14852.47, 
        "read_bytes(B)": 671088640, 
        "write_bytes(B)": 536870912, 
        "io_bytes(B)": 1207959552, 
        "mem_bw(GB/s)": 81.331, 
        "calc_flops_power(tflops)": 296.116, 
        "calc_mem_ratio": 3640.889
    }, 
    "kernels": [
        "ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_64x3_nn"
    ]
}
```


### for communication ops
| Metric            | Unit          | Description |
| --------          | -------       | ------- |
| latency           | us            | kernel device e2e latency    |
| algo size         | B             | algorithm communication size |
| bus size          | B             | bus communication size |
| algo_bw           | GB/s          | algorithm communication bandwidth |
| bus_bw            | GB/s          | bus communication bandwidth |
| latency_list      | list[us]      | latency for each rank |
| algo_bw_list      | list[GB/s]    | algorithm communication bandwidth for each rank |
| bus_bw_list       | list[GB/s]    | bus communication bandwidth for each rank |

Example:
```
{
    "op_name": "all_reduce", 
    "sku_name": "NVIDIA A800-SXM4-80GB", 
    "provider": "default", 
    "arguments": {
        "arg_type": "default", 
        "world_size": 8, 
        "dtype": "float32", 
        "batch_size": 131072, 
        "dim_size": 1024
        }, 
    "targets": {
        "latency(us)": 6281.726, 
        "algo_size(B)": 536870912, 
        "bus_size(B)": 939524096.0, 
        "algo_bw(GB/s)": 85.466, 
        "bus_bw(GB/s)": 149.565, 
        "algo_bw_sum(GB/s)": 681.339, 
        "bus_bw_sum(GB/s)": 1192.343, 
        "latency_list(us)": [6299.236, 6314.211, 6312.343, 6311.595, 6304.193, 6301.111, 6305.494, 6281.726], "algo_bw_list(GB/s)": [85.228, 85.026, 85.051, 85.061, 85.161, 85.203, 85.143, 85.466], 
        "bus_bw_list(GB/s)": [149.149, 148.795, 148.839, 148.857, 149.032, 149.105, 149.001, 149.565]
    }, 
    "kernels": [
        "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevComm*, unsigned long, ncclWork*)"
    ]
}
```

## Trouble Shooting
For more details, you can visit our offical website here: [xpu-perf.ai](https://xpu-perf.ai/). Please let us know if you need any help or have additional questions and issues!
