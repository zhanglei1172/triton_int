import torch
from triton_int.kernels.bmm import (
    bmm_s8t_s8n_f32t,
    bmm_s8t_s8n_s8t,
    bmm_s8t_s8n_s32t,
    bmm_s8t_s8t_s8t,
)


class BMM_S8T_S8N_S8T(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer("a", torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int8
        return bmm_s8t_s8n_s8t(a, b, self.a.item())

    @staticmethod
    def from_scale(a_scale, b_scale, output_scale):
        bmm_module = BMM_S8T_S8N_S8T(1.0)
        alpha = a_scale * b_scale / output_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module


class BMM_S8T_S8T_S8T(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer("a", torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int8
        return bmm_s8t_s8t_s8t(a, b, self.a.item())

    @staticmethod
    def from_scale(a_scale, b_scale, output_scale):
        bmm_module = BMM_S8T_S8T_S8T(1.0)
        alpha = a_scale * b_scale / output_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module


class BMM_S8T_S8N_F32T(torch.nn.Module):
    def __init__(self, alpha, dtype=torch.float32):
        super().__init__()
        self.register_buffer("a", torch.tensor(alpha))
        self.dtype = dtype

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int32
        return bmm_s8t_s8n_f32t(a, b, self.a.item(), dtype=self.dtype)

    @staticmethod
    def from_scale(a_scale, b_scale, dtype=torch.float32):
        bmm_module = BMM_S8T_S8N_F32T(1.0, dtype=dtype)
        alpha = a_scale * b_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module


class BMM_S8T_S8N_F16T(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer("a", torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int32
        return bmm_s8t_s8n_f32t(a, b, self.a.item(), dtype=torch.float16)

    @staticmethod
    def from_scale(a_scale, b_scale):
        bmm_module = BMM_S8T_S8N_F16T(1.0)
        alpha = a_scale * b_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module


class BMM_S8T_S8N_BF16T(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.register_buffer("a", torch.tensor(alpha))

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int32
        return bmm_s8t_s8n_f32t(a, b, self.a.item(), dtype=torch.bfloat16)

    @staticmethod
    def from_scale(a_scale, b_scale):
        bmm_module = BMM_S8T_S8N_BF16T(1.0)
        alpha = a_scale * b_scale
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(alpha)
        bmm_module.a = alpha
        return bmm_module


class BMM_S8T_S8N_S32T(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, a, b):
        # a: [B, M, K] int8
        # b: [B, N, K] int8
        # return: [B, M, N] int32
        return bmm_s8t_s8n_s32t(a, b)
