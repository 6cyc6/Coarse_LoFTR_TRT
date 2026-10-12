"""SLiM's selective scan as an op that exports to the TensorRT plugin `SelectiveScan` (slim/plugin)."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_LIBRARY = ROOT / 'build' / 'slim_plugin' / 'libslim_trt_plugins.so'


class SelectiveScan(torch.autograd.Function):
    """Forward selective scan with a float output, as upstream SLiM calls it (csms6s.selective_scan_fn with oflex).

    In torch it runs the upstream CUDA kernel (selective_scan_cuda_oflex, pixi run build-slim-kernels); the ONNX
    export is a `slim::SelectiveScan` node that TensorRT's ONNX parser maps to the plugin of the same name, which
    runs the same kernel.
    """

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias):
        import selective_scan_cuda_oflex
        out, *_ = selective_scan_cuda_oflex.fwd(u.contiguous(), delta.contiguous(), A.contiguous(), B.contiguous(),
                                                 C.contiguous(), D.contiguous(), delta_bias.contiguous(), True, 1, True)
        return out

    @staticmethod
    def symbolic(g, u, delta, A, B, C, D, delta_bias):
        out = g.op('slim::SelectiveScan', u, delta, A, B, C, D, delta_bias, delta_softplus_i=1, plugin_version_s='1',
                   plugin_namespace_s='')
        return out.setType(u.type().with_dtype(torch.float32))


def selective_scan(u, delta, A, B, C, D, delta_bias):
    """u, delta [B, D, L] (float or half), A [D, N], B, C [B, G, N, L] (type of u), D, delta_bias [D] -> [B, D, L] float"""
    return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias)
