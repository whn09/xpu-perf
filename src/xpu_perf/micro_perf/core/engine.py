import os
import sys
import pathlib
import traceback
import threading
from datetime import timedelta
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable



FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

from xpu_perf.micro_perf.core.backend import Backend
from xpu_perf.micro_perf.core.utils import logger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class BaseEngine(ABC):
    def __init__(
        self, 
        backend_instance: Backend = None, 
        **kwargs
    ):
        self.backend_instance: Backend = backend_instance

        self.device_name = self.backend_instance.backend_info["device_name"]
        self.device_count = self.backend_instance.backend_info["device_count"]

        self.numa_configs = self.backend_instance.common_info["numa_configs"]

        self.node_world_size = kwargs.get("node_world_size", 1)
        self.node_rank = kwargs.get("node_rank", 0)

        self.master_addr = kwargs["master_addr"]
        self.host_port = kwargs["host_port"]
        self.device_port = kwargs["device_port"]

        # 根据numa配置决定要起多少个
        self.numa_num = kwargs.get("numa_num", 1)
        self.numa_order = kwargs.get("numa_order", [-1])
        self.device_mapping = kwargs.get("device_mapping", {})
        self.device_ids = kwargs.get("device_ids", [])
        
        self.device_num = len(self.device_ids)


        # 每一个numa process对应1个device, 不会创建多余的process
        if self.device_num <= self.numa_num:
            self.numa_num = self.device_num
            self.numa_order = self.numa_order[:self.numa_num]
            self.device_num_per_process = 1
        # 每一个numa process对应多个device, 且device数相等
        elif self.device_num % self.numa_num == 0:
            self.device_num_per_process = self.device_num // self.numa_num
        else:
            logger.error(f"device_num({self.device_num}) must be divisible by numa_num({self.numa_num})")



        self.process_mapping = []
        for process_id in range(self.device_num):
            device_id = self.device_ids[process_id]
            numa_id = self.numa_order[process_id // self.device_num_per_process]            
            numa_cores = self.numa_configs[numa_id] if numa_id != -1 else []
            self.process_mapping.append({
                "device_id": device_id,
                "numa_id": numa_id,
                "numa_cores": numa_cores,
            })

        self.is_running = False

        self.dispatch_lock = threading.Lock()

        self.subprocess_procs = []
        self.subprocess_pids = []

        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()


    def __del__(self):
        if self.node_world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        self.stop()


    def init_host_dist_env(self):
        # 如果使用多个node参与计算，需要初始化host端通信
        if self.node_world_size > 1:
            os.environ["WORLD_SIZE"] = str(self.node_world_size)
            os.environ["RANK"] = str(self.node_rank)
            os.environ["MASTER_ADDR"] = self.master_addr
            os.environ["MASTER_PORT"] = self.host_port

            dist.init_process_group(backend="gloo")


    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError

    def dispatch(
        self,
        test_cases: Dict[str, List],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        engine_name: str = "",
    ):
        index_mapping = {}
        with self.dispatch_lock:
            task_idx = 0
            task_meta: Dict[int, Dict[str, Any]] = {}
            for op_name, cases in test_cases.items():
                index_mapping[op_name] = []
                n_prov = len(self.backend_instance.op_mapping[op_name])
                op_total_steps = len(cases) * n_prov
                for case_i, case in enumerate(cases):
                    case_mapping = {}
                    for op_provider, op_cls in self.backend_instance.op_mapping[op_name].items():
                        case_mapping[op_provider] = task_idx
                        task_meta[task_idx] = {
                            "op_name": op_name,
                            "op_provider": op_provider,
                            "case_index": case_i,
                            "cases_for_op": len(cases),
                            "op_total_steps": op_total_steps,
                        }
                        self.input_queue.put((task_idx, (op_name, op_provider, op_cls), case))
                        task_idx += 1
                    index_mapping[op_name].append(case_mapping)

            all_results = {}
            for _ in range(task_idx):
                result_idx, result_dict = self.output_queue.get()
                all_results[result_idx] = result_dict
                if progress_callback:
                    meta = task_meta.get(result_idx, {})
                    progress_callback(
                        {
                            "event": "task_done",
                            "engine": engine_name,
                            "op_name": meta.get("op_name"),
                            "op_provider": meta.get("op_provider"),
                            "case_index": meta.get("case_index"),
                            "cases_for_op": meta.get("cases_for_op"),
                        }
                    )

            for op_name in index_mapping:
                for case_mapping in index_mapping[op_name]:
                    for op_provider, index in case_mapping.items():
                        case_mapping[op_provider] = all_results.get(index, {})

        return index_mapping


class ComputeEngine(BaseEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def start(self):
        # 创建子进程, 子进程各自独立汇报状态
        try:
            self.subprocess_procs = mp.spawn(
                self.backend_instance.compute_infer_loop, 
                args=(
                    self.process_mapping, 
                    self.input_queue, 
                    self.output_queue,
                ), 
                nprocs=self.device_num,
                join=False, 
                daemon=False
            )
            for proc in self.subprocess_procs.processes:
                self.subprocess_pids.append(proc.pid)
            logger.info(f"spawn compute infer loop success, pids: {self.subprocess_pids}")

            for _ in range(self.device_num):
                try:
                    signal = self.output_queue.get(timeout=30)
                    if signal != "success":
                        logger.error(f"compute infer loop failed, signal: {signal}")
                        sys.exit(-1)
                except Exception as e:
                    logger.error(f"compute infer loop timeout, error: {e}")
                    sys.exit(-1)

            self.is_running = True
            logger.info(f"all subprocesses are ready")

        except Exception as e:
            logger.exception(f"failed to spawn compute infer loop: {e}")
            traceback.print_exc()
            sys.exit(-1)
            


    def stop(self):
        if self.subprocess_procs:
            for _ in self.subprocess_procs.processes:
                self.input_queue.put(None)

            kill_flag = False
            for subprocess in self.subprocess_procs.processes:
                subprocess.join(timeout=10)
                if subprocess.is_alive():
                    kill_flag = True
                    break
            
            if kill_flag:
                for subprocess in self.subprocess_procs.processes:
                    subprocess.kill()

            self.subprocess_procs = []
            self.subprocess_pids = []
        
        self.is_running = False


class XCCLEngine(BaseEngine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        os.environ["MASTER_ADDR"] = str(self.master_addr)
        os.environ["MASTER_PORT"] = str(self.device_port)




    def start(self):
        # 创建子进程, 由一个子进程汇报状态
        try:
            self.subprocess_procs = mp.spawn(
                self.backend_instance.xccl_infer_loop, 
                args=(
                    self.process_mapping, 
                    self.input_queue, 
                    self.output_queue,
                    self.master_addr, 
                    self.device_port, 
                    self.node_world_size, 
                    self.node_rank,
                ), 
                nprocs=self.device_num,
                join=False, 
                daemon=False
            )
            for proc in self.subprocess_procs.processes:
                self.subprocess_pids.append(proc.pid)
            logger.info(f"spawn xccl infer loop success, pids: {self.subprocess_pids}")

            # Rank 0 only reports ready after the warmup all_reduce in
            # xccl_infer_loop, and on backends that compile ahead of execution
            # (Neuron) that first collective can take many minutes against a
            # cold cache. 60s is fine for GPU but times out there, so allow an
            # override.
            ready_timeout = int(
                os.environ.get("XPU_PERF_XCCL_READY_TIMEOUT_S", "60")
            )
            try:
                signal = self.output_queue.get(timeout=ready_timeout)
                if signal != "success":
                    logger.error(f"xccl infer loop failed, signal: {signal}")
                    sys.exit(-1)
            except Exception as e:
                logger.error(f"xccl infer loop timeout, error: {e}")
                sys.exit(-1)

            self.is_running = True

            logger.info(f"all subprocesses are ready")

        except Exception as e:
            logger.exception(f"failed to spawn xccl infer loop: {e}")
            traceback.print_exc()
            sys.exit(-1)
                

    def stop(self):
        if self.subprocess_procs:
            if self.node_rank == 0:
                self.input_queue.put(None)

            kill_flag = False
            for subprocess in self.subprocess_procs.processes:
                subprocess.join(timeout=10)
                if subprocess.is_alive():
                    kill_flag = True
                    break
            
            if kill_flag:
                for subprocess in self.subprocess_procs.processes:
                    subprocess.kill()

            self.subprocess_procs = []
            self.subprocess_pids = []
        
        self.is_running = False
            
