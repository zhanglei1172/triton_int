import torch

from ..functional.fused import dq_add_layernorm_q_cpp
from ..kernels.fused import fast_geluQ


class LayerNormQ(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.input_scale = 1.0
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        x = x.to(self.weight.dtype)
        ln_output_fp = torch.nn.functional.layer_norm(
            x, x.shape[-1:], self.weight, self.bias, self.eps
        )
        ln_output_int8 = ln_output_fp.round().clamp(-128, 127).to(torch.int8)
        return ln_output_int8

    @staticmethod
    def from_float(module: torch.nn.LayerNorm, output_scale: float):
        assert module.normalized_shape[0] == module.weight.numel()
        assert module.normalized_shape[0] == module.bias.numel()
        q_module = LayerNormQ(module.normalized_shape[0], module.eps)
        q_module.weight = module.weight / output_scale
        q_module.bias = module.bias / output_scale
        q_module.eps = module.eps
        return q_module


class DQ_Add_LayerNorm_Q(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.input_scale = 1.0
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(dim, dtype=torch.float32))

    def forward(self, residual_input_fp, input_int32):
        # input_int32: [B, L, H] int32
        # residual_input_fp: [B, L, H] fp
        # return residual_output_fp, ln_output_int8
        return dq_add_layernorm_q_cpp(
            input_int32,
            self.input_scale,
            residual_input_fp,
            self.weight,
            self.bias,
            self.eps,
        )


class GeLu_Q(torch.nn.Module):
    def __init__(self, alpha=1):
        super().__init__()
        self.register_buffer("a", torch.tensor(alpha))

    def forward(self, x):
        return fast_geluQ(x.to(torch.float32), self.a.item()).to(x.dtype)
