import time
import copy
import pathlib
import importlib
from functools import partial
from typing import Any, Dict, Optional, Tuple, List

import torch

FILE_DIR = pathlib.Path(__file__).parent.absolute()
MICRO_PERF_DIR = FILE_DIR.parent

from xpu_perf.micro_perf.core.common_utils import logger
from xpu_perf.micro_perf.core.common_utils import discover_plugins, load_all_python_files



def _package_tree_path_to_import(
    file_path: pathlib.Path, base_path: pathlib.Path, prefix: str
) -> Optional[str]:
    """将 base_path 为根的包目录树中的 .py 路径转为 import 名。"""
    try:
        rel_path = pathlib.Path(file_path).relative_to(base_path)
        parts = list(rel_path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        internal = ".".join(parts)
        if not internal:
            return prefix.rstrip(".")
        return f"{prefix.rstrip('.')}.{internal}"
    except ValueError:
        return None


def _vendor_class_with_base(vendor_cls: type, base_cls: type) -> type:
    """
    构造 vendor 实现类：MRO 为 vendor_cls 在前、base_cls 在后，未在 vendor 中定义的方法回落到 base。
    若 vendor_cls 已是 base_cls 子类，则直接返回 vendor_cls。
    """
    if issubclass(vendor_cls, base_cls):
        return vendor_cls
    return type(
        vendor_cls.__name__,
        (vendor_cls, base_cls),
        {
            "__module__": vendor_cls.__module__,
            "__qualname__": getattr(vendor_cls, "__qualname__", vendor_cls.__name__),
            "__doc__": vendor_cls.__doc__,
        },
    )


class ProviderRegistry:
    """
    engine_name_0: {op_name_0, op_name_1, ...}
    engine_name_1: {op_name_2, op_name_3, ...}
    """
    ENGINE_OPS = {}

    """
    "op_name_0": "engine_name_0"
    "op_name_1": "engine_name_1"
    """
    OP_ENGINE_MAPPING = {}


    """
    op_name_0: op_name_cls_0
    op_name_1: op_name_cls_1
    """
    BASE_IMPL_MAPPING = {}
    

    """
    op_name_0: { provider_name_0: op_name_cls_0, provider_name_1: op_name_cls_1, ... }
    op_name_1: { provider_name_2: op_name_cls_2, provider_name_3: op_name_cls_3, ... }
    """
    OP_MAPPING = {}
    OP_PROVIDER_MAPPING = {}

    PROVIDER_INFO = {}

    AVAIL_ENGINES = frozenset({"ComputeEngine", "XCCLEngine"})
    BASE_PROVIDER = "base"


    @classmethod
    def register_base_impl(cls, op_name: str, engine_name: str):
        def decorator(wrapped_class):
            if engine_name not in cls.AVAIL_ENGINES:
                raise ValueError(f"engine name {engine_name} is not supported")

            cls.OP_ENGINE_MAPPING[op_name] = engine_name
            cls.BASE_IMPL_MAPPING[op_name] = wrapped_class
            cls.ENGINE_OPS.setdefault(engine_name, set()).add(op_name)
            return wrapped_class
        return decorator


    @classmethod
    def load_all_base_impls(cls, op_defs: pathlib.Path):
        if op_defs is not None:
            discover_plugins(str(op_defs))


    @classmethod
    def register_vendor_impl(cls, op_name: str, provider_name: str):
        """
        登记 vendor 实现类。若该类尚未继承 ``register_base_impl`` 登记的同名 op 的基类，
        则自动构造 ``(vendor_cls, base_cls)`` 的派生类再登记（模块内绑定也会被替换为派生类）。
        """
        def decorator(wrapped_class: type) -> type:
            if provider_name == cls.BASE_PROVIDER:
                raise ValueError(
                    f'provider name "{cls.BASE_PROVIDER}" is reserved for '
                    "register_base_impl; use another name for register_vendor_impl."
                )
            base_cls = cls.BASE_IMPL_MAPPING.get(op_name)
            if base_cls is None:
                raise ValueError(
                    f'no register_base_impl for op_name="{op_name}" '
                    f"after load_all_base_impls(); check op_name or core op registration."
                )
            impl_cls = _vendor_class_with_base(wrapped_class, base_cls)
            cls.OP_MAPPING.setdefault(op_name, {})[provider_name] = impl_cls
            return impl_cls
        return decorator


    @classmethod
    def register_provider_info(cls, provider_name: str, info: dict):
        cls.PROVIDER_INFO[provider_name] = info


    @classmethod
    def load_all_vendor_impls(cls, op_defs: pathlib.Path, vendor_ops: List[pathlib.Path]):
        cls.load_all_base_impls(op_defs)

        cls.OP_MAPPING = {}
        cls.OP_PROVIDER_MAPPING = {}
        cls.PROVIDER_INFO = {}

        for vendor_op in vendor_ops:
            if not vendor_op.exists():
                continue
            discover_plugins(vendor_op)

    
        for op_name in cls.OP_MAPPING:
            for provider_name, op_cls in cls.OP_MAPPING[op_name].items():
                cls.OP_PROVIDER_MAPPING.setdefault(provider_name, {})[op_name] = op_cls

        for op_name in cls.BASE_IMPL_MAPPING.keys():
            if op_name not in cls.OP_MAPPING:
                cls.OP_MAPPING[op_name] = {cls.BASE_PROVIDER: cls.BASE_IMPL_MAPPING[op_name]}



class BasicOp:
    def __init__(self, args_dict, backend, *args, **kwargs):
        # store args
        self.args_dict = args_dict

        # get backend instance
        self.backend = backend
        
        # store distributed info
        self.op_group = kwargs.get("op_group", None)
        self.group_size = kwargs.get("group_size", 1)

        # default create input and output tensors based on given tensor shapes
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=True
        )

        # default function call, empty run
        self._run_func = self._empty_run


        """
        输入输出的tensor信息, 在算子实例初始化的时候就需要确认好
        """
        self.input_tensor_info = {}
        self.output_tensor_info = {}


        # tensor size for allocate memory
        self.input_tensor_size = 0
        self.output_tensor_size = 0
        self.tensor_size = 0

        # read / write bytes for calc memory bandwidth
        self.read_bytes = 0
        self.write_bytes = 0
        self.io_bytes = 0

        # communication size through links (such as pcie or nvlink)
        self.algo_size = 0
        self.bus_size = 0

        # flops for calc flops power
        self.calc_flops = 0

        # preset info
        self.is_concurrent = False
        self.require_profiling = False
        self.extra_providers = []

        # 1. parse arguments
        # 2. prepare op and compile op if needed
        # 3. define input and output tensor info
        # 4. calculate tensor size
        # 5. calculate read / write bytes
        # 6. calculate algo / bus size
        # 7. calculate flops
        # 8. specify run function, custom run or obey standard test process
        self.prepare()


    """
    默认执行函数, 默认会报错
    self._run_func = self._empty_run
    """
    def _empty_run(self):
        raise NotImplementedError("run func not implemented.")

    # 默认创建输入输出tensor的模板函数

    """
    默认创建输入输出tensors的模板函数
    self._create_tensors_func = partial(
        self._create_in_out_tensors, 
        create_inputs=True, 
        create_outputs=True
    )
    """
    def _create_in_out_tensors(
        self, 
        instance_num, 
        create_inputs=True, 
        create_outputs=True,
    ):
        all_tensor_list = []

        # create first instance
        first_tensor_mapping = {}
        if create_inputs:
            for key, value in self.input_tensor_info.items():
                first_tensor_mapping[key] = value.creator(
                    size=value.shape,
                    dtype=value.dtype,
                    device=value.device
                )
                if value.device == "cpu":
                    first_tensor_mapping[key] = first_tensor_mapping[key].pin_memory()
        if create_outputs:
            for key, value in self.output_tensor_info.items():
                first_tensor_mapping[key] = value.creator(
                    size=value.shape,
                    dtype=value.dtype,
                    device=value.device
                )
                if value.device == "cpu":
                    first_tensor_mapping[key] = first_tensor_mapping[key].pin_memory()
        all_tensor_list.append(first_tensor_mapping)


        # clone following instances
        for _ in range(instance_num - 1):
            tensor_mapping = {}
            for key, value in first_tensor_mapping.items():
                tensor_mapping[key] = value.clone() if value.device != "cpu" else value.clone().pin_memory()
            all_tensor_list.append(tensor_mapping)

        return all_tensor_list





    def core_run(self, *args, **kwargs):
        return self._run_func(*args, **kwargs)


    def prepare(self):
        self.prepare_args()
        self.vendor_parser()
        self.vendor_impl()

    def prepare_args(self):
        pass

    def vendor_parser(self):
        pass

    def vendor_impl(self):
        self._run_func = self.vendor_impl_run

    def vendor_impl_run(self, *args, **kwargs):
        raise NotImplementedError("vendor_impl_run not implemented.")


    def create_tensors(self, instance_num):
        return self._create_tensors_func(instance_num)


    def mfu_dtype(self):
        """Which dtype the matmul engine runs the math in.

        MFU is achieved TFLOPS over the nominal peak *for that dtype*, so a
        quantised op has to be scored against its compute dtype and not its
        storage dtype: a w8a8 matmul stores int8 but the peak that matters is
        the one for the accumulating datapath. Ops that declare a compute dtype
        (quant_matmul, moe_*_gemm) expose it as ``compute_dtype``;
        flash_attention splits it into a q*k and a p*v dtype and the two are
        equal in every shipped workload, so the q*k one stands for the pair.
        Everything else computes in its operand dtype.
        """
        for attr in ("compute_dtype", "qk_compute_dtype", "dtype"):
            dtype = getattr(self, attr, None)
            if isinstance(dtype, str) and dtype:
                return dtype
        return None

    def _peak_tflops(self):
        """Nominal peak for this op's compute dtype, or None if unavailable.

        Never lets a missing spec table break a measurement that already
        succeeded -- summary() runs after the timing.
        """
        dtype = self.mfu_dtype()
        if not dtype or self.backend is None:
            return None
        try:
            return self.backend.get_peak_tflops(dtype)
        except Exception:
            return None

    def summary(self, latency_us, kernel_mapping={}):
        target_dict = {}
        if latency_us > 0:
            target_dict["latency(us)"] = round(latency_us, 3)
            if self.is_concurrent:
                target_dict["algo_size(B)"] = self.algo_size
                target_dict["bus_size(B)"] = self.bus_size
                target_dict["algo_bw(GB/s)"] = round(self.algo_size / latency_us / 1e3, 3)
                target_dict["bus_bw(GB/s)"] = round(self.bus_size / latency_us / 1e3, 3)
            else:
                target_dict["read_bytes(B)"] = self.read_bytes
                target_dict["write_bytes(B)"] = self.write_bytes
                target_dict["io_bytes(B)"] = self.io_bytes
                target_dict["mem_bw(GB/s)"] = round(self.io_bytes / latency_us / 1e3, 3)
                target_dict["calc_flops"] = self.calc_flops
                tflops = round(self.calc_flops / latency_us / 1e6, 3)
                target_dict["calc_flops_power(tflops)"] = tflops
                target_dict["calc_mem_ratio"] = round(self.calc_flops / self.io_bytes, 3) if self.io_bytes!= 0 else 0

                # MFU, only when the backend publishes a nominal peak for this
                # compute dtype. Absent rather than zero otherwise, so a missing
                # spec entry cannot be mistaken for an op that achieved nothing.
                # Skipped for ops that report no arithmetic at all (copy, cast,
                # reorder), where 0.0 would be a property of the op rather than a
                # result. Ops with a little arithmetic and a lot of traffic --
                # add, softmax -- do get an MFU, and it is correctly tiny; read
                # mem_bw for those.
                peak_tflops = self._peak_tflops() if self.calc_flops > 0 else None
                if peak_tflops:
                    target_dict["peak_tflops"] = round(peak_tflops, 3)
                    target_dict["mfu"] = round(tflops / peak_tflops, 4)
            target_dict["kernels"] = kernel_mapping
        return target_dict


    def merge_summary(self, target_dict_list):
        if len(target_dict_list) == 0:
            return {}
        if not self.is_concurrent:
            return target_dict_list[0]
        
        latency_list = [target_dict["latency(us)"] for target_dict in target_dict_list]
        algo_bw_list = [target_dict["algo_bw(GB/s)"] for target_dict in target_dict_list]
        bus_bw_list = [target_dict["bus_bw(GB/s)"] for target_dict in target_dict_list]

        target_dict = copy.deepcopy(target_dict_list[0])
        target_dict["latency(us)"] = round(sum(latency_list) / len(latency_list), 3)
        target_dict["algo_bw(GB/s)"] = round(sum(algo_bw_list) / len(algo_bw_list), 3)
        target_dict["bus_bw(GB/s)"] = round(sum(bus_bw_list) / len(bus_bw_list), 3)
        target_dict["algo_bw_sum(GB/s)"] = round(sum(algo_bw_list), 3)
        target_dict["bus_bw_sum(GB/s)"] = round(sum(bus_bw_list), 3)
        target_dict["latency_list(us)"] = latency_list
        target_dict["algo_bw_list(GB/s)"] = algo_bw_list
        target_dict["bus_bw_list(GB/s)"] = bus_bw_list

        return target_dict