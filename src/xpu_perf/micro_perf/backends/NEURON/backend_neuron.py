"""micro_perf backend for AWS Trainium / Inferentia (Neuron SDK).

torch_xla is never imported at module level. Importing it initialises the PJRT
runtime, which claims the NeuronCores visible to the current process; if that
happens in the parent it leaves nothing for the spawned workers. Every
torch_xla import therefore lives inside a method that only runs in a child
process, after ``set_device`` has narrowed ``NEURON_RT_VISIBLE_CORES``.
"""
import os
import json
import math
import time
import fcntl
import random
import pathlib
import traceback
import subprocess
import contextlib
import importlib.metadata
from datetime import timedelta

import torch
import torch.distributed as dist

from xpu_perf.micro_perf.core.backend import Backend
from xpu_perf.micro_perf.core.utils import logger


class BackendNEURON(Backend):
    def __init__(self, **kwargs):
        # Patch pin_memory before any tensor work: Neuron hosts have no NVIDIA
        # driver, so pin_memory() on a CPU tensor raises instead of pinning.
        self._patch_pin_memory()

        # Set by set_device() in the worker process.
        self._device_index = None
        # (rank, world_size) recorded by initialize_ccl(), consumed by
        # set_device(). See _deferred_ccl_init().
        self._pending_ccl = None
        # group_size -> gloo process group, for Python object exchange.
        self._cpu_group_mapping = {}

        super().__init__(**kwargs)

    @staticmethod
    def _patch_pin_memory():
        original_pin_memory = torch.Tensor.pin_memory

        def safe_pin_memory(tensor, device=None):
            try:
                return original_pin_memory(tensor, device=device)
            except Exception:
                return tensor

        torch.Tensor.pin_memory = safe_pin_memory

    """
    neuron-ls helpers
    """
    def _get_neuron_ls_data(self):
        try:
            result = subprocess.run(
                ["neuron-ls", "-j"],
                capture_output=True, text=True, timeout=10
            )
            return json.loads(result.stdout)
        except Exception:
            return []

    def _get_instance_type(self):
        for device in self._get_neuron_ls_data():
            instance_type = device.get("instance_type")
            if instance_type:
                return instance_type
        return "unknown"

    """
    获取和backend相关的信息
    """
    def get_backend_info(self):
        info_dict = {}

        neuron_data = self._get_neuron_ls_data()
        nc_count = sum(device.get("nc_count", 0) for device in neuron_data)
        total_memory = sum(device.get("memory_size", 0) for device in neuron_data)

        # 一个 NeuronCore 视作一个 device
        info_dict["device_name"] = self._get_instance_type()
        info_dict["device_count"] = nc_count
        info_dict["device_memory_mb"] = \
            total_memory / nc_count / (1024 ** 2) if nc_count > 0 else 0

        info_dict["neuron_device_count"] = len(neuron_data)
        info_dict["neuron_core_count"] = nc_count
        if neuron_data:
            # trn2 defaults to LNC=2: two physical cores per logical core.
            info_dict["logical_neuroncore_config"] = \
                neuron_data[0].get("logical_neuroncore_config", 1)

        info_dict["torch_version"] = torch.__version__
        for key, package in (
            ("torch_xla_version", "torch-xla"),
            ("torch_neuronx_version", "torch-neuronx"),
            ("neuronx_cc_version", "neuronx-cc"),
        ):
            try:
                info_dict[key] = importlib.metadata.version(package)
            except Exception:
                info_dict[key] = "unknown"

        return info_dict

    def clean_extra_files(self):
        pass

    """
    device management related
    """
    def get_torch_device_name(self):
        return "xla"

    def get_device_name(self, index: int = 0):
        return self.backend_info.get("device_name", "unknown")

    def get_device_properties(self, index: int = 0):
        return {
            "name": self.backend_info.get("device_name", "unknown"),
            "total_memory": int(self.backend_info.get("device_memory_mb", 0) * (1024 ** 2)),
        }

    def get_mem_info(self, index: int = 0):
        # Neuron exposes no per-core allocator counters, so free is reported as
        # total. perf()'s memory guard is therefore advisory only: an op that
        # overcommits HBM fails at compile/execute time rather than being
        # skipped up front.
        total_memory = int(self.backend_info.get("device_memory_mb", 0) * (1024 ** 2))
        return (total_memory, total_memory)

    def get_device_count(self):
        device_count = self.backend_info.get("device_count", 0)
        return device_count, list(range(device_count))

    def set_device(self, index: int):
        self._device_index = index

        # Must be set before torch_xla initialises PJRT, otherwise this process
        # claims every core instead of the one it was assigned.
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(index)

        # torch_xla picks its backend at import time and, if it cannot find the
        # Neuron PJRT plugin, silently falls back to CPU with nothing but a
        # logging warning. A benchmark that reports CPU numbers under a NEURON
        # label is worse than one that fails, so pin the device and then verify.
        os.environ.setdefault("PJRT_DEVICE", "NEURON")

        # nrt_init() must not run concurrently with another rank's. Workers are
        # spawned together, so without this lock every rank reserving a core on
        # the same NeuronDevice races and *all* of them fail with
        # "Logical Neuron Core(s) not available ... (cores busy, ret=-16)".
        # Serialising the reservation is safe: nrt_init only claims cores, and
        # the XCCL rendezvous happens later, on first collective execution.
        with self._nrt_init_lock():
            import torch_xla  # noqa: F401

            # Also forces PJRT initialisation inside the lock, so the core is
            # actually reserved before the next rank starts trying.
            self._assert_running_on_neuron()

        self._deferred_ccl_init()

    @staticmethod
    @contextlib.contextmanager
    def _nrt_init_lock():
        """Serialise NeuronCore reservation across the worker processes.

        A plain file lock in a fixed location: the racing processes are
        siblings on one host, so they only need to agree on a path.
        """
        lock_path = os.environ.get(
            "XPU_PERF_NEURON_INIT_LOCK", "/tmp/xpu_perf_neuron_init.lock"
        )
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    @staticmethod
    def _assert_running_on_neuron():
        """Fail loudly if torch_xla did not actually bind to a NeuronCore.

        The plugin lives in libneuronxla, which is pulled in by torch-neuronx.
        Stacks that ship only libtorch-neuronx-lite (the vLLM inference venv,
        for one) have no plugin at all and resolve every device to CPU.
        """
        import torch_xla.core.xla_model as xm

        device_kind = str(xm.xla_device())
        if "CPU" in device_kind.upper():
            raise EnvironmentError(
                f"torch_xla bound to {device_kind!r} instead of a NeuronCore, so "
                "every measurement would be a CPU measurement. The Neuron PJRT "
                "plugin (libneuronxla/libneuronpjrt.so) is missing -- install "
                "torch-neuronx:\n"
                "  pip install torch-neuronx "
                "--extra-index-url=https://pip.repos.neuron.amazonaws.com"
            )

    def get_device(self):
        import torch_xla.core.xla_model as xm
        return xm.xla_device()

    def device_synchronize(self):
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.wait_device_ops()

    def empty_cache(self):
        pass

    """
    ccl related
    """
    def get_dist_module(self):
        return dist

    def get_dist_backend(self):
        return "xla"

    def initialize_ccl(self, rank: int, world_size: int):
        """Record the request; the real init happens in set_device().

        xccl_infer_loop calls initialize_ccl() before set_device(), but the
        "xla" process group backend needs torch_xla imported, and torch_xla
        must not be imported before NEURON_RT_VISIBLE_CORES is narrowed. So the
        init is deferred by one step. Both calls happen on every rank, in the
        same order, so the process group is still formed collectively.
        """
        self._pending_ccl = (rank, world_size)
        if self._device_index is not None:
            # set_device() already ran (single-process path): init immediately.
            self._deferred_ccl_init()
        return True

    def _deferred_ccl_init(self):
        if self._pending_ccl is None:
            return
        rank, world_size = self._pending_ccl
        self._pending_ccl = None

        # Registers the "xla" backend with torch.distributed.
        import torch_xla.distributed.xla_backend  # noqa: F401

        dist.init_process_group(
            backend=self.get_dist_backend(),
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=1800),
        )

        # Pre-create one gloo group per possible group_size. Python object
        # exchange (all_gather_object) hangs on the xla backend, so it has to
        # run over gloo. new_group() must be called by every rank in the
        # default group -- including non-members -- so they are all built here,
        # while every rank is still in lockstep, rather than lazily inside
        # perf() where only the active ranks are running.
        self._cpu_group_mapping = {}
        for group_size in range(2, world_size + 1):
            self._cpu_group_mapping[group_size] = dist.new_group(
                ranks=list(range(group_size)), backend="gloo"
            )

    def op_group_barrier(self, op_group=None, group_size=1):
        import torch_xla.core.xla_model as xm
        if dist.is_initialized() and group_size > 1:
            dist.all_reduce(
                torch.tensor([1], dtype=torch.int32, device=self.get_torch_device_name()),
                op=dist.ReduceOp.SUM,
                group=op_group
            )
            xm.mark_step()
            xm.wait_device_ops()

    """
    perf related
    """
    def perf(self, op_instance):
        import torch_xla.core.xla_model as xm

        # XLA collectives on a sub-group whose members are only part of the
        # world do not complete on Neuron, so a case narrower than the launched
        # world is reported as skipped instead of being run. Returning a
        # zero-latency summary yields an empty result dict, which the infer
        # loops already treat as "this case produced nothing" -- and the loops
        # exchange that verdict over gloo, so no rank is left waiting.
        # Bench one world_size per launch, e.g. --device 0,1 for world_size=2.
        if op_instance.group_size > 1 and dist.is_initialized():
            world_size = dist.get_world_size()
            if op_instance.group_size != world_size:
                logger.warning(
                    f"Skipping case with world_size={op_instance.group_size}: "
                    f"launched world_size is {world_size}, and Neuron requires "
                    f"them to be equal."
                )
                return op_instance.summary(0., {})

        tensor_size = op_instance.tensor_size

        device_mem_info = self.get_mem_info()
        avail_memory = device_mem_info[0]

        assume_avail_bytes = int(avail_memory * 0.9)
        assume_cache_size = 1 * (1024 ** 3)

        latency_us = 0.
        kernel_mapping = {}

        try:
            # Each XLA graph has to be compiled by neuronx-cc before it runs,
            # which dominates wall clock. Favour a few long iterations over
            # many short ones: a 1 s budget capped at 10 iterations.
            min_test_iters = 2
            max_test_iters = 10
            max_test_time = 1e6     # 1 s

            max_data_cnt = 1
            if not op_instance.is_concurrent:
                if tensor_size > assume_avail_bytes:
                    raise RuntimeError("Not enough memory to run the op")
                elif 2 * tensor_size > assume_avail_bytes:
                    max_data_cnt = 1
                elif tensor_size > assume_cache_size:
                    max_data_cnt = 2
                else:
                    max_data_cnt = min(
                        math.floor(max(assume_avail_bytes, assume_cache_size) / tensor_size),
                        math.floor(assume_cache_size / tensor_size)
                    )

            # create_tensors() builds each extra copy with clone(), so a large
            # count produces one long clone chain in the HLO. At the GPU
            # default of up to 256 copies that graph exceeds 10 MB and takes
            # neuronx-cc over five minutes to compile. Four copies is plenty
            # here anyway -- the copies exist to defeat a CPU cache that
            # NeuronCores do not have.
            max_data_cnt = min(max_data_cnt, 4)

            tensor_list = op_instance.create_tensors(max_data_cnt)
            random.shuffle(tensor_list)

            # Materialise the tensors now so they stop being pending lazy ops.
            # Otherwise the first clone carries "empty -> clone -> op" while
            # later ones carry "empty -> op": different graphs, so every few
            # warmup iterations triggers a fresh compile, and neuronx-cc
            # intermittently fails with "type must be number, but is null".
            xm.mark_step()
            xm.wait_device_ops()

            latency_us, _ = self.core_perf(op_instance, 2, 2, tensor_list, profiling=False)

            if latency_us >= max_test_time:
                prefer_iters = min_test_iters
            else:
                prefer_iters = min(
                    max(math.ceil(max_test_time / latency_us), min_test_iters),
                    max_test_iters
                )

            if op_instance.group_size > 1:
                # Over gloo, not op_instance.op_group: all_gather_object on the
                # xla backend hangs. See _deferred_ccl_init().
                cpu_group = self._cpu_group_mapping.get(op_instance.group_size)
                prefer_iters_list = [None for _ in range(op_instance.group_size)]
                dist.all_gather_object(prefer_iters_list, prefer_iters, group=cpu_group)
                prefer_iters = max(prefer_iters_list)

            time.sleep(0.2)

            # No kernel-level profiler is available on Neuron (see README), so
            # core_perf never collects a kernel breakdown.
            latency_us, kernel_mapping = self.core_perf(
                op_instance, 2, prefer_iters, tensor_list, profiling=False
            )

            del tensor_list
            self.empty_cache()
        except Exception as e:
            traceback.print_exc()

        return op_instance.summary(latency_us, kernel_mapping)

    def core_perf(
        self, op_instance,
        warmup_iterations, prefer_iterations,
        tensor_list,
        profiling=True
    ):
        import torch_xla.core.xla_model as xm

        op_group = op_instance.op_group
        group_size = op_instance.group_size

        self.op_group_barrier(op_group=op_group, group_size=group_size)
        self.device_synchronize()

        # Extra warmup so the neuronx-cc compile of the single-op graph lands
        # before the timed loop.
        effective_warmup = max(warmup_iterations, 4)
        # Keep the last result reachable across mark_step. A lazy tensor that is
        # dropped before the graph is cut is dead code, and XLA prunes it -- an
        # op whose output goes nowhere compiles to nothing and "runs" in the
        # time it takes to launch an empty graph. Warmup retains it too, so it
        # compiles the same graph the timed loop below executes.
        keepalive = None
        try:
            for i in range(effective_warmup):
                keepalive = op_instance.core_run(tensor_list[i % len(tensor_list)])
                xm.mark_step()
            xm.wait_device_ops()
        except Exception:
            # Flush pending lazy ops, otherwise peer ranks waiting inside a
            # collective on this rank block forever.
            try:
                xm.mark_step()
            except Exception:
                pass
            raise

        self.op_group_barrier(op_group=op_group, group_size=group_size)
        xm.wait_device_ops()

        start_time = time.perf_counter_ns()
        try:
            for i in range(prefer_iterations):
                keepalive = op_instance.core_run(tensor_list[i % len(tensor_list)])
                # One mark_step per iteration reuses the graph compiled during
                # warmup instead of fusing the loop into a new one.
                xm.mark_step()
            xm.wait_device_ops()
        except Exception:
            try:
                xm.mark_step()
            except Exception:
                pass
            raise
        end_time = time.perf_counter_ns()

        latency_us = (end_time - start_time) / 1e3 / prefer_iterations
        del keepalive
        return latency_us, []
