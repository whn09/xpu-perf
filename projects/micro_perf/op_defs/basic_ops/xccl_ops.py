import pathlib
import torch
from functools import partial

from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype
from xpu_perf.micro_perf.core.op import BasicOp, ProviderRegistry


@ProviderRegistry.register_base_impl("all_reduce", "XCCLEngine")
class AllReduceOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict.get("arg_type", "default")
        if not self.arg_type in ["default", "llm"]:
            raise ValueError(f"arg_type must be default or llm, but got {self.arg_type}")

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"dtype must be float32, float16 or bfloat16, but got {self.dtype}")
        self.torch_dtype = get_torch_dtype(self.dtype)

        # [world_size,  batch_size * dim_size // world_size] --> 
        # [world_size,  batch_size * dim_size // world_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = 2 * (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = self.batch_size * self.dim_size

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]

        dist_module = self.backend.get_dist_module()
        dist_module.all_reduce(
            src, 
            op=dist_module.ReduceOp.SUM, 
            group=self.op_group
        )
        return src


@ProviderRegistry.register_base_impl("reduce_scatter", "XCCLEngine")
class ReduceScatterOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict.get("arg_type", "default")
        if not self.arg_type in ["default", "llm"]:
            raise ValueError(f"arg_type must be default or llm, but got {self.arg_type}")

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"dtype must be float32, float16 or bfloat16, but got {self.dtype}")
        self.torch_dtype = get_torch_dtype(self.dtype)

        # [world_size,  batch_size * dim_size // world_size] --> 
        # [1,           batch_size * dim_size // world_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size // self.world_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = self.batch_size * self.dim_size

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True
        )

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]

        dist_module = self.backend.get_dist_module()
        dist_module.reduce_scatter_tensor(
            dst, src, 
            op=dist_module.ReduceOp.SUM, 
            group=self.op_group
        )
        return dst


@ProviderRegistry.register_base_impl("all_gather", "XCCLEngine")
class AllGatherOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict.get("arg_type", "default")
        if not self.arg_type in ["default", "llm"]:
            raise ValueError(f"arg_type must be default or llm, but got {self.arg_type}")

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int32", "int8"]:
            raise ValueError(f"dtype must be float32, float16, bfloat16, int32 or int8, but got {self.dtype}")
        self.torch_dtype = get_torch_dtype(self.dtype)
        
        # [1,           batch_size * dim_size // world_size] --> 
        # [world_size,  batch_size * dim_size // world_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size // self.world_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype, 
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.output_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True
        )

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]

        dist_module = self.backend.get_dist_module()
        dist_module.all_gather_into_tensor(
            dst, src, 
            group=self.op_group
        )
        return dst


@ProviderRegistry.register_base_impl("all_to_all", "XCCLEngine")
class AlltoAllOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict.get("arg_type", "default")
        if not self.arg_type in ["default", "llm"]:
            raise ValueError(f"arg_type must be default or llm, but got {self.arg_type}")

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int32", "int8"]:
            raise ValueError(f"dtype must be float32, float16, bfloat16, int32 or int8, but got {self.dtype}")
        self.torch_dtype = get_torch_dtype(self.dtype)

        # [world_size, batch_size * dim_size // world_size] --> 
        # [world_size, batch_size * dim_size // world_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.output_tensor_size
        self.bus_size = (self.world_size - 1) * self.algo_size / self.world_size

        self.calc_flops = 0

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True
        )

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        
        dist_module = self.backend.get_dist_module()
        dist_module.all_to_all_single(
            dst, src, 
            group=self.op_group
        )
        return dst

        
@ProviderRegistry.register_base_impl("broadcast", "XCCLEngine")
class BroadcastOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default", "llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]
        elif self.arg_type == "llm":
            self.batch_size = self.args_dict["num_tokens"]
            self.dim_size = self.args_dict["hidden_size"]

        # [batch_size, dim_size] --> 
        # [batch_size, dim_size]
        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.world_size, self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.world_size, self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        
        self.read_bytes = self.input_tensor_size / self.world_size
        self.write_bytes = self.input_tensor_size / self.world_size * (self.world_size - 1)
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.tensor_size
        self.bus_size = self.algo_size
        self.calc_flops = 0


        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False
        )

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dist_module = self.backend.get_dist_module()
        for i in range(self.world_size):
            dist_module.broadcast(src[i], src=i, group=self.op_group)
        return src

    # This override had drifted twice from BasicOp.summary and so broadcast could
    # not produce a result on any backend:
    #   * it took (self, latency_us) while Backend.perf calls
    #     summary(latency_us, kernel_mapping) (core/backend.py:378) ->
    #     "TypeError: BroadcastOp.summary() takes 2 positional arguments but 3
    #     were given";
    #   * and it unpacked the return into (target_dict, env_dict) while
    #     BasicOp.summary returns one dict -> "ValueError: too many values to
    #     unpack (expected 2)".
    # Neither was noticed because no workload file referenced broadcast until
    # workloads/xccl_ops/broadcast.json, so the op had never once run.
    #
    # The division is still correct after the fix: vendor_impl_run does one
    # broadcast per rank, so latency covers world_size of them and the sizes are
    # for the whole loop. Scaling latency and both sizes by the same factor leaves
    # algo_bw/bus_bw -- already computed by super() from the undivided pair --
    # unchanged, which is what makes them per-broadcast figures.
    def summary(self, latency_us, kernel_mapping={}):
        target_dict = super().summary(latency_us, kernel_mapping)
        if target_dict:
            target_dict["latency(us)"] = round(target_dict["latency(us)"] / self.world_size, 3)
            target_dict["algo_size(B)"] = round(target_dict["algo_size(B)"] / self.world_size, 3)
            target_dict["bus_size(B)"] = round(target_dict["bus_size(B)"] / self.world_size, 3)

        return target_dict


@ProviderRegistry.register_base_impl("p2p", "XCCLEngine")
class P2POp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.dtype = self.args_dict["dtype"]
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.src_device = self.args_dict["src_device"]
        self.dst_device = self.args_dict["dst_device"]

        dist_module = self.backend.get_dist_module()
        self.world_size = self.group_size
        self.local_rank = dist_module.get_rank(group=self.op_group)


        # self.next_device = (self.local_rank + 1) % self.world_size
        # self.last_device = (self.local_rank - 1 + self.world_size) % self.world_size


        self.input_tensor_info = {
            "send": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "recv": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }

        self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.input_tensor_size
        self.bus_size = self.algo_size

        self.calc_flops = 0

    def vendor_impl_run(self, tensor_mapping):
        send_tensor = tensor_mapping["send"]
        recv_tensor = tensor_mapping["recv"]
        dist_module = self.backend.get_dist_module()

        if self.src_device != self.dst_device:
            reqs = []
            if self.local_rank == self.src_device:
                reqs.append(dist_module.isend(send_tensor, self.dst_device, group=self.op_group))
            if self.local_rank == self.dst_device:
                reqs.append(dist_module.irecv(recv_tensor, self.src_device, group=self.op_group))
            for req in reqs:
                req.wait()
        else:
            recv_tensor.copy_(send_tensor)

        # # 0 --> 1
        # # 0 --> 1 --> 2 --> 3
        # # 0 --> 1 --> 2 --> 3 --> 4 --> 5 --> 6 --> 7 --> 8
        # reqs = []
        # if self.local_rank != self.world_size - 1:
        #     reqs.append(dist_module.isend(send_tensor, self.next_device, group=self.op_group))
        # if self.local_rank != 0:
        #     reqs.append(dist_module.irecv(recv_tensor, self.last_device, group=self.op_group))
        # for req in reqs:
        #     req.wait()


@ProviderRegistry.register_base_impl("all_reduce_h2d", "XCCLEngine")
class AllReduce_H2D_Op(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError
        
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        if self.arg_type == "default":
            self.batch_size = self.args_dict["batch_size"]
            self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "data": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            ), 
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device="cpu", 
                creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(), 
                creator=torch.empty
            )
        }

        self.all_reduce_input_tensor_size = calc_tensor_size(self.input_tensor_info["data"])
        self.all_reduce_output_tensor_size = self.all_reduce_input_tensor_size
        self.h2d_input_tensor_size = calc_tensor_size(self.input_tensor_info["src"])
        self.h2d_output_tensor_size = calc_tensor_size(self.output_tensor_info["dst"])

        self.tensor_size = \
            self.all_reduce_input_tensor_size + \
            self.all_reduce_output_tensor_size + \
            self.h2d_input_tensor_size + \
            self.h2d_output_tensor_size
        
        self.read_bytes = \
            self.all_reduce_input_tensor_size
        self.write_bytes = \
            self.all_reduce_input_tensor_size + \
            self.h2d_output_tensor_size
        
        self.io_bytes = self.read_bytes + self.write_bytes

        self.all_reduce_algo_size = self.all_reduce_input_tensor_size
        self.all_reduce_bus_size = 2 * (self.world_size - 1) * self.all_reduce_algo_size / self.world_size

        self.h2d_algo_size = self.h2d_output_tensor_size
        self.h2d_bus_size = self.h2d_algo_size

        self.algo_size = self.all_reduce_algo_size + self.h2d_algo_size
        self.bus_size = self.all_reduce_bus_size + self.h2d_bus_size

        self.calc_flops = 0

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True
        )

    def vendor_impl_run(self, tensor_mapping):
        data = tensor_mapping["data"]
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dist_module = self.backend.get_dist_module()
        dist_module.all_reduce(data, async_op=True)
        dst.copy_(src)
        return dst
        

@ProviderRegistry.register_base_impl("host2device", "XCCLEngine")
class Host2DeviceOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)
        
        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device="cpu", 
                creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }

        self.input_tensor_size = 0
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = 0
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.tensor_size
        self.bus_size = self.tensor_size

        self.calc_flops = 0

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dst.copy_(src)
        return dst


@ProviderRegistry.register_base_impl("device2host", "XCCLEngine")
class Device2HostOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.world_size = self.args_dict["world_size"]
        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device="cpu", 
                creator=torch.empty
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = 0
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = 0
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = self.tensor_size
        self.bus_size = self.tensor_size

        self.calc_flops = 0

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dst.copy_(src)
        return dst


@ProviderRegistry.register_base_impl("device2device", "ComputeEngine")
class Device2DeviceOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare_args(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["default"]:
            raise NotImplementedError

        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["float32", "float16", "bfloat16", "int8"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)

        self.batch_size = self.args_dict["batch_size"]
        self.dim_size = self.args_dict["dim_size"]

        self.input_tensor_info = {
            "src": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }
        self.output_tensor_info = {
            "dst": OpTensorInfo(
                shape=[self.batch_size * self.dim_size],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty
            )
        }    

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.algo_size = 0
        self.bus_size = 0

        self.calc_flops = 0

    def vendor_impl_run(self, tensor_mapping):
        src = tensor_mapping["src"]
        dst = tensor_mapping["dst"]
        dst.copy_(src)
        return dst