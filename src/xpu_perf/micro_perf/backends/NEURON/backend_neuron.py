"""micro_perf backend for AWS Trainium / Inferentia (Neuron SDK).

Neuron ships two mutually exclusive PyTorch integrations, and this backend
drives either one -- see ``_detect_runtime``:

``xla``
    The classic stack. Ops are traced into an XLA graph and compiled by
    neuronx-cc via the PJRT plugin in libneuronxla.
``eager``
    The PyTorch-native stack (torch-neuronx >= 2.12). Ops dispatch eagerly to a
    privateuse1 device named ``neuron``; there is no torch_xla at all.

Neither runtime module is imported at module level. Importing torch_xla
initialises PJRT, which claims the NeuronCores visible to the current process;
if that happens in the parent it leaves nothing for the spawned workers. The
native stack is equally eager to bind a core, and additionally refuses to
initialise a process group once the runtime is up. Every import of either
therefore lives inside a method that only runs in a child process, after
``set_device`` has narrowed ``NEURON_RT_VISIBLE_CORES``.
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
import importlib.util
import importlib.metadata
from datetime import timedelta

import torch
import torch.distributed as dist

from xpu_perf.micro_perf.core.backend import Backend
from xpu_perf.micro_perf.core.utils import logger


# The two PyTorch integrations Neuron offers. See detect_neuron_runtime().
RUNTIME_XLA = "xla"
RUNTIME_EAGER = "eager"


def detect_neuron_runtime():
    """Pick which Neuron PyTorch stack to drive.

    Detection is by import *availability* only: find_spec locates a module
    without executing it, so calling this cannot claim a NeuronCore or drag
    torch_xla into a process that is about to fork workers.

    A stack carrying torch_xla is treated as an XLA stack, which keeps the
    behaviour of every previously validated environment unchanged; the native
    path is only taken where torch_xla genuinely does not exist. Override with
    ``XPU_PERF_NEURON_RUNTIME=xla|eager``.
    """
    requested = os.environ.get("XPU_PERF_NEURON_RUNTIME", "auto").strip().lower()
    if requested in (RUNTIME_XLA, RUNTIME_EAGER):
        return requested
    if requested not in ("", "auto"):
        raise ValueError(
            f"XPU_PERF_NEURON_RUNTIME={requested!r} is not recognised; "
            f"expected one of 'auto', {RUNTIME_XLA!r}, {RUNTIME_EAGER!r}."
        )
    if importlib.util.find_spec("torch_xla") is not None:
        return RUNTIME_XLA
    return RUNTIME_EAGER


class BackendNEURON(Backend):
    def __init__(self, **kwargs):
        # Patch pin_memory before any tensor work: Neuron hosts have no NVIDIA
        # driver, so pin_memory() on a CPU tensor raises instead of pinning.
        self._patch_pin_memory()

        # Which Neuron PyTorch stack to drive. Resolved before super().__init__
        # because get_backend_info() reports it.
        self._runtime = detect_neuron_runtime()

        # Set by set_device() in the worker process.
        self._device_index = None
        # (rank, world_size) recorded by initialize_ccl(), consumed by
        # set_device(). See _deferred_ccl_init().
        self._pending_ccl = None
        # group_size -> gloo process group, for Python object exchange.
        self._cpu_group_mapping = {}

        super().__init__(**kwargs)

    @property
    def neuron_runtime(self):
        """``RUNTIME_XLA`` or ``RUNTIME_EAGER``; read by the vendor ops."""
        return self._runtime

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
        info_dict["neuron_runtime"] = self._runtime
        for key, package in (
            ("torch_xla_version", "torch-xla"),
            ("torch_neuronx_version", "torch-neuronx"),
            ("neuronx_cc_version", "neuronx-cc"),
            ("nki_version", "nki"),
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
        if self._runtime == RUNTIME_EAGER:
            # Deliberately unindexed. In a distributed run the native runtime
            # sets each rank's local device start index to its local rank, so
            # rank 1's only valid index is 1: "neuron:0" raises there for
            # torch.empty/torch.randn -- and torch.full silently returns a
            # neuron:1 tensor instead, which is worse than raising. Bare
            # "neuron" always resolves to the current device.
            return "neuron"
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

        # Must be set before either runtime binds a core, otherwise this process
        # claims every core instead of the one it was assigned. A bare index is
        # also exactly what the native runtime's resolve_visible_cores() passes
        # through untouched, so each rank keeps the core it was given.
        os.environ["NEURON_RT_VISIBLE_CORES"] = str(index)

        # initialize_ccl() runs before set_device(), so the world is already
        # known here -- which both runtimes need before they initialise.
        multi_process = (
            self._pending_ccl is not None and self._pending_ccl[1] > 1
        )

        if self._runtime == RUNTIME_EAGER:
            self._set_device_eager(multi_process)
        else:
            self._set_device_xla(multi_process)

    def _set_device_eager(self, multi_process: bool):
        """Bring up the PyTorch-native runtime on the assigned core.

        The ordering here is the opposite of the XLA path's. The native
        distributed backend asserts the Neuron runtime is *not* yet initialised
        when init_process_group is called -- it wants to assign cores, set
        NEURON_RT_ROOT_COMM_ID from the store and run an nrt barrier itself. So
        nothing may touch the device until the process group exists, and the
        verification that would normally come first has to come after.
        """
        if multi_process:
            # With both of these set, _set_rt_visible_cores() takes its
            # deterministic branch instead of inferring local rank by
            # rendezvousing on IP addresses through the store.
            os.environ.setdefault("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))

        # Same rationale as the XLA path: independent workers race in nrt_init
        # and must be serialised, while ranks of one collective are brought up
        # together by the runtime and would deadlock behind the lock.
        init_context = (
            contextlib.nullcontext() if multi_process else self._nrt_init_lock()
        )
        with init_context:
            self._deferred_ccl_init()
            self._assert_running_on_neuron()

    def _set_device_xla(self, multi_process: bool):
        # torch_xla picks its backend at import time and, if it cannot find the
        # Neuron PJRT plugin, silently falls back to CPU with nothing but a
        # logging warning. A benchmark that reports CPU numbers under a NEURON
        # label is worse than one that fails, so pin the device and then verify.
        os.environ.setdefault("PJRT_DEVICE", "NEURON")

        if multi_process:
            self._configure_pjrt_topology(*self._pending_ccl)

        # nrt_init() must not run concurrently with another rank's. Independent
        # workers are spawned together, so without a lock every rank reserving a
        # core on the same NeuronDevice races and *all* of them fail with
        # "Logical Neuron Core(s) not available ... (cores busy, ret=-16)".
        #
        # Once the topology above is set the plugin assigns cores across the
        # ranks itself and brings them up together, so there the lock would
        # deadlock instead of helping -- rank 0 would hold it while PJRT waits
        # for rank 1, which is waiting for the lock.
        init_context = (
            contextlib.nullcontext() if multi_process else self._nrt_init_lock()
        )
        with init_context:
            import torch_xla  # noqa: F401

            # Also forces PJRT initialisation inside the lock, so the core is
            # actually reserved before the next rank starts trying.
            self._assert_running_on_neuron()

        if multi_process:
            self._enable_replication()

        self._deferred_ccl_init()

    @staticmethod
    def _enable_replication():
        """Make torch_xla report the real world size.

        ``xr.world_size()`` reports 1 unless the runtime has replication devices
        configured, and ``ProcessGroupXla`` takes its size from there. Without
        this, torch.distributed believes the world is 1 process wide -- whatever
        world_size ``init_process_group`` was given -- and rejects every wider
        group with "the new group's world size should be less or equal to the
        world size set by init_process_group".

        ``xr.world_size()`` caches its result on first call, so this has to run
        before anything asks for it.
        """
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        xm.set_replication(device, [device])

    @staticmethod
    def _configure_pjrt_topology(rank: int, world_size: int):
        """Describe the multi-process world to the Neuron PJRT plugin.

        Each worker narrows NEURON_RT_VISIBLE_CORES to one core, so without
        this the plugin sees a single-device, single-process world and
        ProcessGroupXla reports a world size of 1 -- whatever world_size
        init_process_group was given. torch.distributed then rejects any group
        wider than 1 with "the new group's world size should be less or equal
        to the world size set by init_process_group".

        Must run before torch_xla initialises PJRT.
        """
        os.environ.setdefault("NEURON_PJRT_PROCESS_INDEX", str(rank))
        os.environ.setdefault("NEURON_PJRT_WORLD_SIZE", str(world_size))
        # One NeuronCore per micro_perf device, so one device per process.
        os.environ.setdefault(
            "NEURON_PJRT_PROCESSES_NUM_DEVICES", ",".join(["1"] * world_size)
        )

    @staticmethod
    @contextlib.contextmanager
    def _nrt_init_lock():
        """Serialise NeuronCore reservation across the worker processes.

        A plain file lock in a fixed location: the racing processes are
        siblings on one host, so they only need to agree on a path.

        Set XPU_PERF_NEURON_INIT_LOCK=off to disable. That is needed if the
        plugin ever rendezvouses across ranks during initialisation, since
        holding a lock through it would deadlock.
        """
        lock_path = os.environ.get(
            "XPU_PERF_NEURON_INIT_LOCK", "/tmp/xpu_perf_neuron_init.lock"
        )
        if lock_path.lower() in ("off", "0", "none", ""):
            yield
            return
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _assert_running_on_neuron(self):
        """Fail loudly if the runtime did not actually bind to a NeuronCore."""
        if self._runtime == RUNTIME_EAGER:
            return self._assert_running_on_neuron_eager()
        return self._assert_running_on_neuron_xla()

    @staticmethod
    def _assert_running_on_neuron_xla():
        """The PJRT plugin lives in libneuronxla, pulled in by torch-neuronx.

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

    @staticmethod
    def _assert_running_on_neuron_eager():
        """Confirm the native device exists and actually executes.

        The native stack cannot silently degrade to CPU the way torch_xla can
        -- importing torch_neuronx without a visible device raises outright --
        but it can fall back per-op, so run something and check where the
        result landed rather than trusting the device object.
        """
        import torch_neuronx

        probe = torch.ones(8, 8, device="neuron")
        result = (probe @ probe).sum()
        torch_neuronx.synchronize()
        if result.device.type != "neuron" or result.item() != 512.0:
            raise EnvironmentError(
                f"the native Neuron device produced {result.item()} on "
                f"{result.device} instead of 512.0 on a neuron device, so "
                "measurements would not be Neuron measurements."
            )

    def get_device(self):
        if self._runtime == RUNTIME_EAGER:
            import torch_neuronx
            return torch.device("neuron", torch_neuronx.current_device())
        import torch_xla.core.xla_model as xm
        return xm.xla_device()

    def device_synchronize(self):
        if self._runtime == RUNTIME_EAGER:
            import torch_neuronx
            torch_neuronx.synchronize()
            return
        # mark_step cuts the graph, wait_device_ops blocks until it has run.
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.wait_device_ops()

    def empty_cache(self):
        if self._runtime == RUNTIME_EAGER:
            import torch_neuronx
            torch_neuronx.empty_cache()

    """
    ccl related
    """
    def get_dist_module(self):
        return dist

    def get_dist_backend(self):
        if self._runtime == RUNTIME_EAGER:
            return "neuron"
        return "xla"

    def initialize_ccl(self, rank: int, world_size: int):
        """Record the request; the real init happens in set_device().

        xccl_infer_loop calls initialize_ccl() before set_device(), but neither
        runtime can form a process group this early: the "xla" backend needs
        torch_xla imported, which must not happen before
        NEURON_RT_VISIBLE_CORES is narrowed, and the native "neuron" backend
        assigns cores itself and so must run *after* that narrowing too. The
        init is therefore deferred by one step. Both calls happen on every
        rank, in the same order, so the process group is still formed
        collectively.
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

        if self._runtime == RUNTIME_EAGER:
            # Registers the "neuron" backend with torch.distributed. Importing
            # it also arranges for init_process_group to assign this rank's
            # core, publish NEURON_RT_ROOT_COMM_ID through the store and run an
            # nrt barrier -- all of which it refuses to do if the Neuron
            # runtime is already up, hence the ordering in _set_device_eager.
            import torch_neuronx.distributed.backend  # noqa: F401
        else:
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
        if dist.is_initialized() and group_size > 1:
            dist.all_reduce(
                torch.tensor([1], dtype=torch.int32, device=self.get_torch_device_name()),
                op=dist.ReduceOp.SUM,
                group=op_group
            )
            self.device_synchronize()

    """
    perf related
    """
    def perf(self, op_instance):
        # XLA collectives on a sub-group whose members are only part of the
        # world do not complete on Neuron, so a case narrower than the launched
        # world is reported as skipped instead of being run. Returning a
        # zero-latency summary yields an empty result dict, which the infer
        # loops already treat as "this case produced nothing" -- and the loops
        # exchange that verdict over gloo, so no rank is left waiting.
        # Bench one world_size per launch, e.g. --device 0,1 for world_size=2.
        #
        # The native runtime has no such limitation: a group narrower than the
        # world reduces correctly there, so one launch can cover every
        # world_size up to the device count.
        if (
            self._runtime == RUNTIME_XLA
            and op_instance.group_size > 1
            and dist.is_initialized()
        ):
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
            # many short ones: a 1 s budget capped at 10 iterations. Eager
            # dispatch pays no such per-graph cost, so it can afford the
            # iterations that bring the run-to-run spread down.
            min_test_iters = 2
            max_test_iters = 50 if self._runtime == RUNTIME_EAGER else 10
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
            # on either runtime anyway -- the copies exist to defeat a CPU
            # cache that NeuronCores do not have.
            max_data_cnt = min(max_data_cnt, 4)

            tensor_list = op_instance.create_tensors(max_data_cnt)
            random.shuffle(tensor_list)

            # On XLA, materialise the tensors now so they stop being pending
            # lazy ops. Otherwise the first clone carries "empty -> clone -> op"
            # while later ones carry "empty -> op": different graphs, so every
            # few warmup iterations triggers a fresh compile, and neuronx-cc
            # intermittently fails with "type must be number, but is null".
            # Eager has no pending ops to flush, so this just waits for the
            # copies to land.
            self.device_synchronize()

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
                # device backend hangs. See _deferred_ccl_init().
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
        # Neither runtime offers a kernel-level profiler, so profiling is
        # ignored and the kernel breakdown is always empty (see README).
        if self._runtime == RUNTIME_EAGER:
            return self._core_perf_eager(
                op_instance, warmup_iterations, prefer_iterations, tensor_list
            )
        return self._core_perf_xla(
            op_instance, warmup_iterations, prefer_iterations, tensor_list
        )

    def _core_perf_eager(
        self, op_instance,
        warmup_iterations, prefer_iterations,
        tensor_list
    ):
        """Time an eager op with the wall clock around one synchronize.

        Do not be tempted to use torch_neuronx.Event here. On trn2 with
        torch-neuronx 2.12.3 its elapsed_time() sits at 25-30 us no matter how
        much work the loop submits -- a 1024x4096x4096 bf16 gemm and an
        8192x4096x4096 one both "take" ~24 us, which would be 1,154 and 11,273
        TFLOPS respectively. It is not measuring device execution. Wall clock
        around a single synchronize() scales linearly with the work and lands on
        believable numbers, so that is what is used.

        No keepalive is needed, unlike the XLA path: eager dispatch has already
        executed the op by the time core_run returns, so dropping the result
        cannot delete the work.
        """
        import torch_neuronx

        op_group = op_instance.op_group
        group_size = op_instance.group_size

        self.op_group_barrier(op_group=op_group, group_size=group_size)
        torch_neuronx.synchronize()

        for i in range(warmup_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
        torch_neuronx.synchronize()

        self.op_group_barrier(op_group=op_group, group_size=group_size)

        start_time = time.perf_counter_ns()
        for i in range(prefer_iterations):
            op_instance.core_run(tensor_list[i % len(tensor_list)])
        torch_neuronx.synchronize()
        end_time = time.perf_counter_ns()

        return (end_time - start_time) / 1e3 / prefer_iterations, []

    def _core_perf_xla(
        self, op_instance,
        warmup_iterations, prefer_iterations,
        tensor_list
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
