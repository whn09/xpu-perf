"""`quant_matmul` on Trainium3's MX (OCP microscaling) datapath, `nisa.nc_matmul_mx`.

What it measures: a GEMM whose operands are *already* MX-quantized -- the same
contract as the int8 `quant_matmul` (`hidden_states` arrives int8 with its
`per_token_scale`). Activation quantization is a separate op in this suite
(`scale_dynamic_quant` and family), so it is not folded in here. The dtype strings
are the ones `op_defs/llm_ops/quant_matmul.md` defines:

    dtype        (activation)  mxfp8_e4m3fn | mxfp8_e5m2 | mxfp4_e2m1fn
    w_dtype      (weight)      mxfp8_e4m3fn | mxfp8_e5m2 | mxfp4_e2m1fn
    compute_dtype              the activation's dtype (what MFU is scored against;
                               every MX combination runs at the same 2517 TF/chip)
    dst_dtype                  bfloat16

All nine activation x weight combinations are accepted by the NKI simulator at
target trn3 and match an exact host reference to bf16 output rounding (rel 1.66e-3).

**NeuronCore-v4 only.** `nc_matmul_mx` does not exist on Trn2. neuronx-cc will
happily compile this kernel for trn3 on a Trn2 host if the target is overridden,
and the 2.35 runtime does *not* refuse the trn3 NEFF there: it executes it, hits
`SEQUENCER_ERROR_SUBTYPE_BAD_OPCODE`, and leaves the core FATAL until the next
process (measured). So vendor_parser refuses every case off Trn3, by instance
type, rather than trusting anything downstream to.

Layout, chosen by this provider as the op def allows ("厂商可以在自己的impl自行选择"):
K-major and x4-packed along K, which is what `nc_matmul_mx` consumes, so the timed
region is load + matmul and nothing else.

    a_q [H/4, T]   int32 (4x fp8) | int16 (4x fp4)   element (kp, t) packs hidden[t, 4kp:4kp+4]
    a_s [H/32, T]  uint8 E8M0, one scale per 32 consecutive K values
    w_q [H/4, N]   int32 | int16                     element (kp, n) packs weight[n, 4kp:4kp+4]
    w_s [H/32, N]  uint8
    y   [T, N]     bf16

The x4 packing is little-endian (value j in byte j, or nibble j for fp4), and the
scales are stored dense in HBM and spread into the hardware quadrant layout on load
(SBUF partition 32q+i holds dense row 4q+i). Both conventions were pinned down in the
simulator with negative controls: doubling every weight scale doubles the output
exactly, and swapping fp4 nibbles or permuting the weight moves rel err to ~1.4.

This is a straightforward first kernel, not a tuned one: one 512-wide weight slab
stays resident in SBUF across all token tiles, and the activation tile is re-read
per slab. It has not run on Trn3 hardware -- the first Trn3 run should go through
tools/verify_providers.py (which checks it bit-for-bit against the host reference)
before its numbers are used, and nkilib's tuned `experimental/matmul_mxfp8` is the
obvious next step if the MFU comes in low.

Opt-in with `XPU_PERF_NEURON_MX=1`, because registering any vendor provider for
`quant_matmul` *replaces* its base implementation (core/op.py) -- ops/torch/
quant_matmul.py re-registers that base under the same flag, so int8 rows keep running.
Run it against vendor_ops/NEURON/workloads/quant_matmul_mx.json.
"""
import os

import torch

from xpu_perf.micro_perf.core.op import ProviderRegistry
from xpu_perf.micro_perf.core.utils import OpTensorInfo, calc_tensor_size
from xpu_perf.micro_perf.backends.NEURON.backend_neuron import (
    RUNTIME_EAGER,
    detect_neuron_runtime,
)

PROVIDER_NAME = "nkilib"
ENV_FLAG = "XPU_PERF_NEURON_MX"

MX_FORMATS = {"mxfp8_e4m3fn": "e4m3", "mxfp8_e5m2": "e5m2", "mxfp4_e2m1fn": "fp4"}

P = 128      # partitions: one K tile = 128 x4 rows = 512 K values
TT = 128     # tokens per stationary tile (stationary free dim <= 128)
NT = 512     # moving free dim per matmul: one fp32 PSUM bank

FP4_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
_FP8_TORCH = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def enabled():
    return os.environ.get(ENV_FLAG, "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- host helpers
# Module-level so tools/verify_providers.py can build a reference from the packed
# device tensors without re-deriving the layout.

def packed_torch_dtype(fmt):
    return torch.int16 if fmt == "fp4" else torch.int32


def random_packed(kp, cols, fmt, device):
    """Valid finite MX element codes, packed x4 along K, as int32/int16 [kp, cols]."""
    if fmt == "fp4":
        codes = torch.randint(0, 16, (kp, cols, 4), dtype=torch.int32)
        packed = codes[..., 0] | (codes[..., 1] << 4) | (codes[..., 2] << 8) | (codes[..., 3] << 12)
        # values above 0x7FFF wrap into int16's negative range; the bits are what matter
        packed = ((packed + 0x8000) % 0x10000 - 0x8000).to(torch.int16)
    else:
        v = torch.randn(kp, cols, 4).to(_FP8_TORCH[fmt])
        packed = v.view(torch.uint8).reshape(kp, cols * 4).view(torch.int32).reshape(kp, cols)
    return packed.to(device)


def random_scales(groups, cols, device):
    """E8M0 exponents around 2^0, so products stay well inside bf16 range."""
    return torch.randint(124, 131, (groups, cols), dtype=torch.int32).to(torch.uint8).to(device)


def dequant(packed, scale, fmt):
    """Packed [KP, cols] + dense scales [KP/8, cols] -> fp32 [K, cols], k = 4*kp + j."""
    packed = packed.cpu()
    kp, cols = packed.shape
    if fmt == "fp4":
        raw = packed.view(torch.int16).to(torch.int32) & 0xFFFF
        codes = torch.stack([(raw >> (4 * j)) & 0xF for j in range(4)], dim=-1)
        vals = FP4_LUT[codes]
    else:
        vals = packed.view(torch.int32).contiguous().view(torch.uint8).reshape(kp, cols, 4) \
            .view(_FP8_TORCH[fmt]).float()
    sc = torch.pow(2.0, scale.cpu().float() - 127.0).repeat_interleave(8, dim=0)
    return (vals * sc[:, :, None]).permute(0, 2, 1).reshape(kp * 4, cols)


def reference(tensor_mapping, a_fmt, w_fmt):
    """Exact (fp64-accumulated) result the kernel should match to bf16 rounding."""
    a = dequant(tensor_mapping["a_q"], tensor_mapping["a_s"], a_fmt)
    w = dequant(tensor_mapping["w_q"], tensor_mapping["w_s"], w_fmt)
    return (a.double().t() @ w.double()).float()


# ---------------------------------------------------------------- kernel

_KERNELS = {}


def build_kernel(a_fmt, w_fmt):
    key = (a_fmt, w_fmt)
    if key in _KERNELS:
        return _KERNELS[key]

    import nki
    import nki.isa as nisa
    import nki.language as nl

    X4 = {"e4m3": nl.float8_e4m3fn_x4, "e5m2": nl.float8_e5m2_x4, "fp4": nl.float4_e2m1fn_x4}
    A_X4, W_X4 = X4[a_fmt], X4[w_fmt]

    @nki.jit
    def mx_quant_matmul(a_q, a_s, w_q, w_s):
        KP, T = a_q.shape
        _, N = w_q.shape
        n_k = KP // P
        out = nl.ndarray((T, N), dtype=nl.bfloat16, buffer=nl.shared_hbm)

        # LNC programs split N
        n_per_prg = N // nl.num_programs(0)
        pid = nl.program_id(0)

        for ni in nl.affine_range(n_per_prg // NT):
            n0 = pid * n_per_prg + ni * NT
            # the weight slab for this n tile stays resident across all token tiles
            w_sb = nl.ndarray((P, n_k, NT), dtype=w_q.dtype, buffer=nl.sbuf)
            ws_sb = nl.ndarray((P, n_k, NT), dtype=nl.uint8, buffer=nl.sbuf)
            for ki in nl.affine_range(n_k):
                nisa.dma_copy(dst=w_sb[:, ki, :], src=w_q[nl.ds(ki * P, P), nl.ds(n0, NT)])
                for q in nl.affine_range(4):
                    # hardware scale layout: partition 32q+i <- dense row 4q+i
                    nisa.dma_copy(dst=ws_sb[nl.ds(32 * q, 4), ki, :],
                                  src=w_s[nl.ds(ki * 16 + 4 * q, 4), nl.ds(n0, NT)])
            for ti in nl.affine_range(T // TT):
                t0 = ti * TT
                ps = nl.ndarray((TT, NT), dtype=nl.float32, buffer=nl.psum)
                for ki in nl.affine_range(n_k):
                    a_sb = nl.ndarray((P, TT), dtype=a_q.dtype, buffer=nl.sbuf)
                    as_sb = nl.ndarray((P, TT), dtype=nl.uint8, buffer=nl.sbuf)
                    nisa.dma_copy(dst=a_sb, src=a_q[nl.ds(ki * P, P), nl.ds(t0, TT)])
                    for q in nl.affine_range(4):
                        nisa.dma_copy(dst=as_sb[nl.ds(32 * q, 4), :],
                                      src=a_s[nl.ds(ki * 16 + 4 * q, 4), nl.ds(t0, TT)])
                    nisa.nc_matmul_mx(
                        dst=ps,
                        stationary=a_sb.view(A_X4),
                        moving=w_sb[:, ki, :].view(W_X4),
                        stationary_scale=as_sb,
                        moving_scale=ws_sb[:, ki, :],
                    )
                res = nl.ndarray((TT, NT), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_copy(dst=res, src=ps)
                nl.store(out[nl.ds(t0, TT), nl.ds(n0, NT)], value=res)
        return out

    _KERNELS[key] = mx_quant_matmul
    return mx_quant_matmul


# SBUF the resident weight slab needs per partition: data + one scale byte per x4 row
_SLAB_BUDGET = 160 * 1024


class NkiMXQuantMatmulOp:
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_parser(self):
        if self.dtype not in MX_FORMATS or self.w_dtype not in MX_FORMATS:
            raise NotImplementedError(
                f"{PROVIDER_NAME} quant_matmul covers MX formats only "
                f"({sorted(MX_FORMATS)}); got dtype={self.dtype}, w_dtype={self.w_dtype}"
            )
        if self.compute_dtype != self.dtype:
            raise ValueError(
                f"compute_dtype must equal the activation dtype ({self.dtype}) for MX: "
                "nc_matmul_mx dequantizes both operands inside the matmul"
            )
        if self.dst_dtype != "bfloat16":
            raise ValueError("MX quant_matmul writes bfloat16")
        family = self.backend._chip_family()
        if family != "trn3":
            raise NotImplementedError(
                f"nc_matmul_mx needs NeuronCore-v4 (Trainium3); this is {family or 'unknown'}. "
                "Not attempted: on Trn2 a trn3 NEFF executes and faults the core."
            )
        if self.backend.neuron_runtime != RUNTIME_EAGER:
            raise NotImplementedError("MX quant_matmul is wired for the eager runtime")
        if self.has_bias or self.sp_size != 1 or self.transpose_o:
            raise ValueError("has_bias / sp_size / transpose_o are not modelled")

        self.lnc = int(self.backend.backend_info.get("logical_neuroncore_config", 1) or 1)
        T, H, N = self.num_tokens, self.hidden_size, self.new_hidden_size
        if H % (4 * P) != 0:
            raise ValueError(f"hidden_size {H} must be a multiple of {4 * P}")
        if T % TT != 0:
            raise ValueError(f"num_tokens {T} must be a multiple of {TT} (no tail tile)")
        if N % (NT * self.lnc) != 0:
            raise ValueError(f"new_hidden_size {N} must be a multiple of {NT * self.lnc}")
        item = 2 if MX_FORMATS[self.w_dtype] == "fp4" else 4
        if (H // (4 * P)) * NT * (item + 1) > _SLAB_BUDGET:
            raise ValueError(f"hidden_size {H} makes the resident weight slab exceed SBUF")

    def vendor_impl(self):
        self.a_fmt = MX_FORMATS[self.dtype]
        self.w_fmt = MX_FORMATS[self.w_dtype]
        T, H, N = self.num_tokens, self.hidden_size, self.new_hidden_size
        dev = self.backend.get_torch_device_name()
        a_fmt, w_fmt = self.a_fmt, self.w_fmt

        self.input_tensor_info = {
            "a_q": OpTensorInfo(shape=[H // 4, T], dtype=packed_torch_dtype(a_fmt), device=dev,
                                creator=lambda size, dtype, device: random_packed(size[0], size[1], a_fmt, device)),
            "a_s": OpTensorInfo(shape=[H // 32, T], dtype=torch.uint8, device=dev,
                                creator=lambda size, dtype, device: random_scales(size[0], size[1], device)),
            "w_q": OpTensorInfo(shape=[H // 4, N], dtype=packed_torch_dtype(w_fmt), device=dev,
                                creator=lambda size, dtype, device: random_packed(size[0], size[1], w_fmt, device)),
            "w_s": OpTensorInfo(shape=[H // 32, N], dtype=torch.uint8, device=dev,
                                creator=lambda size, dtype, device: random_scales(size[0], size[1], device)),
        }
        self.output_tensor_info = {
            "y": OpTensorInfo(shape=[T, N], dtype=torch.bfloat16, device=dev),
        }
        self.input_tensor_size = sum(calc_tensor_size(i) for i in self.input_tensor_info.values())
        self.output_tensor_size = sum(calc_tensor_size(i) for i in self.output_tensor_info.values())
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        self.calc_flops = 2 * T * H * N

        from functools import partial
        self._create_tensors_func = partial(
            self._create_in_out_tensors, create_inputs=True, create_outputs=False)
        self._kernel = build_kernel(a_fmt, w_fmt)[self.lnc]
        self._run_func = self.mx_run

    def mx_run(self, tensor_mapping):
        return self._kernel(tensor_mapping["a_q"], tensor_mapping["a_s"],
                            tensor_mapping["w_q"], tensor_mapping["w_s"])


try:
    if not enabled():
        raise ImportError(f"{ENV_FLAG} not set")
    if detect_neuron_runtime() != RUNTIME_EAGER:
        raise ImportError("MX quant_matmul requires the eager runtime")
    import nki  # noqa: F401  -- fail registration, not every case, if nki is absent
    NkiMXQuantMatmulOp = ProviderRegistry.register_vendor_impl(
        "quant_matmul", PROVIDER_NAME)(NkiMXQuantMatmulOp)
except Exception:
    pass
