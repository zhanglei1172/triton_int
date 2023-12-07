import torch

from ..functional.fused import dq_add_layernorm_q_cpp
from ..functional.quantization import quantize_per_tensor_absmax
from ..kernels.fused import fast_geluQ, linear_a8_w8_bfp32_ofp32_GeLu_Q, layer_norm_fwd_fused_single_pass_q


class _LayerNormQ(torch.nn.Module):
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
        q_module = _LayerNormQ(module.normalized_shape[0], module.eps)
        q_module.weight = module.weight / output_scale
        q_module.bias = module.bias / output_scale
        q_module.eps = module.eps
        return q_module

class LayerNormQ(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.input_scale = 1.0
        self.eps = eps
        self.register_buffer("weight", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("bias", torch.zeros(dim, dtype=torch.float32))

    def forward(self, x):
        x = x.to(self.weight.dtype)
        ln_output_int8 = layer_norm_fwd_fused_single_pass_q(
            x, self.weight, self.bias, self.eps
        )
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
        return fast_geluQ(x.to(torch.float32), self.a).to(x.dtype)


class W8A8BFP32OFP32_GeLu_Q(torch.nn.Module):
    # For fc2 and out_proj
    def __init__(self, in_features, out_features, alpha=1.0, beta=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer(
            "weight",
            torch.randint(
                -127,
                127,
                (self.out_features, self.in_features),
                dtype=torch.int8,
                requires_grad=False,
            ),
        )
        self.register_buffer(
            "bias",
            torch.zeros((self.out_features), dtype=torch.float32, requires_grad=False),
        )
        self.register_buffer("a", torch.tensor(alpha))
        self.register_buffer("b", torch.tensor(beta))

    def _apply(self, fn):
        # prevent the bias from being converted to half
        super()._apply(fn)
        self.bias = self.bias
        return self

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.weight = self.weight.to(*args, **kwargs)
        self.bias = self.bias.to(*args, **kwargs)
        self.bias = self.bias
        return self

    @torch.no_grad()
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        self.bias = self.bias
        y = linear_a8_w8_bfp32_ofp32_GeLu_Q(
            x, self.weight, self.bias, self.a, self.b
        )
        y = y.view(*x_shape[:-1], -1)
        return y

    @staticmethod
    def from_float(module: torch.nn.Linear, input_scale, out_scale):
        int8_module = W8A8BFP32OFP32_GeLu_Q(module.in_features, module.out_features)
        int8_weight, weight_scale = quantize_per_tensor_absmax(module.weight)
        alpha = input_scale * weight_scale
        int8_module.weight = int8_weight
        int8_module.bias = module.bias.to(torch.float32)
        int8_module.a = alpha
        int8_module.input_scale = input_scale
        int8_module.weight_scale = weight_scale
        int8_module.b = torch.tensor(
            1 / out_scale, dtype=int8_module.b.dtype, device=int8_module.b.device
        )
        return int8_module
