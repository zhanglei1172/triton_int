import math
import time
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from triton_int.functional.quantization import quantize_per_tensor_absmax
from triton_int.kernels import register_torch_op
from triton_int.kernels.utils import gemm_autotune

sqrt2pi = math.sqrt(2.0 / math.pi)
sqrt2 = math.sqrt(2.0)


@triton.jit
def kernel_dq_add_layernorm_q(
    X,  # pointer to the input
    Y_,  # pointer to the output
    LN_IN,
    W,  # pointer to the weights
    B,  # pointer to the biases
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


@triton.jit
def tanh(x):
    """Tanh activation function"""
    return tl.math.tanh(x)


@triton.jit
def fast_gelu(x):
    """Fast approximation of the gelu function. May slightly decrease accuracy."""
    return 0.5 * x * (1 + tanh(sqrt2pi * (x + 0.044715 * x * x * x)))


@triton.jit
def fast_geluQ(x, scale):
    """Fast approximation of the gelu function. May slightly decrease accuracy."""
    return 0.5 * x * (1 + tanh(sqrt2pi * (x + 0.044715 * x * x * x))) * scale


@triton.jit
def gelu(x):
    """Gaussian Error Linear Unit (GELU)"""
    return x * 0.5 * (1.0 + tl.math.erf(x / sqrt2))


@gemm_autotune()
@triton.jit
def kernel_linear_a8_w8_bfp32_ofp32_GeLu_Q(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    scale_a,
    scale_b,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    pid_sp_k = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = tl.max_contiguous(
        tl.multiple_of(
            (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M, BLOCK_SIZE_M
        ),
        BLOCK_SIZE_M,
    )
    offs_bn = tl.max_contiguous(
        tl.multiple_of(
            (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N, BLOCK_SIZE_M
        ),
        BLOCK_SIZE_M,
    )
    offs_k = pid_sp_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bias_ptrs = bias_ptr + offs_bn[None, :]
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(
            a_ptrs,
            mask=(offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # accumulator += tl.dot(a.to(tl.float32), b.to(tl.float32)).to(tl.int32)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    bias = tl.load(bias_ptrs)
    c = fast_geluQ(
        accumulator.to(tl.float32) * tl.load(scale_a) + bias, tl.load(scale_b)
    ).to(c_ptr.dtype.element_ty)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@register_torch_op
def linear_a8_w8_bfp32_ofp32_GeLu_Q(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out: torch.Tensor=None) -> torch.Tensor:
    # Check constraints.
    tmp_shape = a.shape[:-1]
    assert len(b.shape) == 2
    a = a.view(-1, a.shape[-1])
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    if out == None:
        c = torch.zeros((M, N), device=a.device, dtype=torch.int8)
    else:
        c = out.fill_(0)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel_linear_a8_w8_bfp32_ofp32_GeLu_Q[grid](
        a,
        b,
        c,
        bias,
        scale_a,
        scale_b,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c.view(*tmp_shape, b.shape[0])


# modified from https://github.com/ELS-RD/kernl/blob/main/src/kernl/implementations/layer_norm.py
@triton.jit
def kernel_layer_norm_fwd_fused_single_pass_q(
    output_ptr,
    a_ptr,
    weight_ptr,
    bias_ptr,
    output_row_stride,
    output_col_stride,
    a_row_stride,
    a_col_stride,
    N_SIZE,
    eps,
    HAS_BIAS: tl.constexpr,
    IS_RMSNORM: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    """
    Layernorm based on Welford's variance computation algorithm.
    https://changyaochen.github.io/welford/
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    :param output_ptr: output tensor
    :param a_ptr: input tensor
    :param weight_ptr: weights applied to the normalized input
    :param bias_ptr: bias added to the normalized input
    :param mean_ptr: save mean tensor for backward
    :param rstd_ptr: save standard deviation tensor for backward
    :param a_row_stride: stride of the input tensor
    :param N_SIZE: number of elements per row in the input tensor
    :param eps: epsilon value to avoid division by zero
    :param HAS_BIAS: whether the bias is provided
    :param IS_RMSNORM: whether the normalization is rmsnorm (False == layernorm)
    :param BLOCK_N_SIZE: number of threads per block
    :return: None
    """
    # position of elements processed by this program
    row_idx = tl.program_id(0)

    a_row_off = row_idx * a_row_stride
    block_range_offs = tl.arange(0, BLOCK_N_SIZE)
    # compute mean
    mean = 0.0
    var = 0.0
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        n_end_off = min((block_n_start_idx + BLOCK_N_SIZE), N_SIZE)
        block_cols_count = n_end_off - block_n_start_idx
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        # eviction policy below have little impact now because of new implementation. Kept as is.
        # float32 is used to avoid overflow because of the square operation
        a = tl.load(
            a_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        if IS_RMSNORM:
            var += tl.sum(a * a, axis=0)
        else:
            block_mean = tl.sum(a, axis=0) / block_cols_count
            # mean is 0 or has a mask applied to it, no need to mask delta_mean!
            delta_mean = block_mean - mean
            delta_mean_sqr = delta_mean * delta_mean

            block_delta = tl.sum((a - block_mean) * a, axis=0)
            # mean has a mask
            mean += tl.sum((a - mean) * a_ptr_mask, axis=0) / n_end_off
            var += (
                block_delta
                + delta_mean_sqr * (block_n_start_idx * block_cols_count) / n_end_off
            )

    var /= N_SIZE
    rstd = 1 / tl.sqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        weight = tl.load(weight_ptr + col_offs, mask=a_ptr_mask)

        # eviction policy helps to keep weights in cache (reused by other threads)
        a = tl.load(
            a_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight
        if HAS_BIAS:
            bias = tl.load(bias_ptr + col_offs, mask=a_ptr_mask)
            out = out + bias
        # write-back
        tl.store(
            output_ptr + row_idx * output_row_stride + col_offs * output_col_stride,
            out.to(output_ptr.dtype.element_ty),
            mask=a_ptr_mask,
        )


@register_torch_op
def layer_norm_fwd_fused_single_pass_q(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    use_rms_norm: bool = False,
):
    assert (
        x.dtype == weight.dtype
    ), f"input and weight bias must have the same dtype: {x.dtype}, {weight.dtype}"
    if bias is not None:
        assert (
            x.dtype == bias.dtype
        ), f"input and bias must have the same dtype: {x.dtype}, {bias.dtype}"
    # catch eps being too small if the tensors are fp16
    if x.dtype == torch.float16:
        eps = max(eps, 1.6e-5)
    # allocate output
    out = torch.empty_like(x, dtype=torch.int8)
    # reshape input data into 2D tensor
    a_arg = x.view(-1, x.shape[-1])
    M, N = a_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    kernel_layer_norm_fwd_fused_single_pass_q[(M,)](
        output_ptr=out,
        a_ptr=a_arg,
        weight_ptr=weight,
        bias_ptr=bias if bias is not None else a_arg,
        output_row_stride=out.stride(-2),
        output_col_stride=out.stride(-1),
        a_row_stride=a_arg.stride(0),
        a_col_stride=a_arg.stride(1),
        N_SIZE=N,
        eps=eps,
        HAS_BIAS=bias is not None,
        IS_RMSNORM=use_rms_norm,
        BLOCK_N_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


@triton.jit
def kernel_layer_norm_fwd_fused_single_pass(
    out_ptr,
    a_ptr,
    weight_ptr,
    bias_ptr,
    output_row_stride,
    output_col_stride,
    a_row_stride,
    a_col_stride,
    N_SIZE,
    eps,
    HAS_BIAS: tl.constexpr,
    IS_RMSNORM: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    """
    Layernorm based on Welford's variance computation algorithm.
    https://changyaochen.github.io/welford/
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    :param output_ptr: output tensor
    :param a_ptr: input tensor
    :param weight_ptr: weights applied to the normalized input
    :param bias_ptr: bias added to the normalized input
    :param mean_ptr: save mean tensor for backward
    :param rstd_ptr: save standard deviation tensor for backward
    :param a_row_stride: stride of the input tensor
    :param N_SIZE: number of elements per row in the input tensor
    :param eps: epsilon value to avoid division by zero
    :param HAS_BIAS: whether the bias is provided
    :param IS_RMSNORM: whether the normalization is rmsnorm (False == layernorm)
    :param BLOCK_N_SIZE: number of threads per block
    :return: None
    """
    # position of elements processed by this program
    row_idx = tl.program_id(0)

    a_row_off = row_idx * a_row_stride
    block_range_offs = tl.arange(0, BLOCK_N_SIZE)
    # compute mean
    mean = 0.0
    var = 0.0
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        n_end_off = min((block_n_start_idx + BLOCK_N_SIZE), N_SIZE)
        block_cols_count = n_end_off - block_n_start_idx
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        # eviction policy below have little impact now because of new implementation. Kept as is.
        # float32 is used to avoid overflow because of the square operation
        a = tl.load(
            a_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        if IS_RMSNORM:
            var += tl.sum(a * a, axis=0)
        else:
            block_mean = tl.sum(a, axis=0) / block_cols_count
            # mean is 0 or has a mask applied to it, no need to mask delta_mean!
            delta_mean = block_mean - mean
            delta_mean_sqr = delta_mean * delta_mean

            block_delta = tl.sum((a - block_mean) * a, axis=0)
            # mean has a mask
            mean += tl.sum((a - mean) * a_ptr_mask, axis=0) / n_end_off
            var += (
                block_delta
                + delta_mean_sqr * (block_n_start_idx * block_cols_count) / n_end_off
            )

    var /= N_SIZE
    rstd = 1 / tl.sqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        weight = tl.load(weight_ptr + col_offs, mask=a_ptr_mask)

        # eviction policy helps to keep weights in cache (reused by other threads)
        a = tl.load(
            a_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight
        if HAS_BIAS:
            bias = tl.load(bias_ptr + col_offs, mask=a_ptr_mask)
            out = out + bias
        # write-back
        tl.store(
            out_ptr + row_idx * output_row_stride + col_offs * output_col_stride,
            out.to(out_ptr.dtype.element_ty),
            mask=a_ptr_mask,
        )


@register_torch_op
def layer_norm_fwd_fused_single_pass(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    use_rms_norm: bool = False,
):
    assert (
        x.dtype == weight.dtype
    ), f"input and weight bias must have the same dtype: {x.dtype}, {weight.dtype}"
    if bias is not None:
        assert (
            x.dtype == bias.dtype
        ), f"input and bias must have the same dtype: {x.dtype}, {bias.dtype}"
    # catch eps being too small if the tensors are fp16
    if x.dtype == torch.float16:
        eps = max(eps, 1.6e-5)
    # allocate output
    out = torch.empty_like(x, dtype=x.dtype)
    # reshape input data into 2D tensor
    a_arg = x.view(-1, x.shape[-1])
    M, N = a_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    kernel_layer_norm_fwd_fused_single_pass[(M,)](
        out_ptr=out,
        a_ptr=a_arg,
        weight_ptr=weight,
        bias_ptr=bias if bias is not None else a_arg,
        output_row_stride=out.stride(-2),
        output_col_stride=out.stride(-1),
        a_row_stride=a_arg.stride(0),
        a_col_stride=a_arg.stride(1),
        N_SIZE=N,
        eps=eps,
        HAS_BIAS=bias is not None,
        IS_RMSNORM=use_rms_norm,
        BLOCK_N_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


@triton.jit
def kernel_skip_layer_norm_fwd_fused_single_pass(
    skip_out_ptr,
    a_ptr,
    b_ptr,
    weight_ptr,
    bias_ptr,
    output_row_stride,
    output_col_stride,
    a_row_stride,
    a_col_stride,
    N_SIZE,
    eps,
    HAS_BIAS: tl.constexpr,
    IS_RMSNORM: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    """
    Layernorm based on Welford's variance computation algorithm.
    https://changyaochen.github.io/welford/
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    :param output_ptr: output tensor
    :param a_ptr: input tensor
    :param weight_ptr: weights applied to the normalized input
    :param bias_ptr: bias added to the normalized input
    :param mean_ptr: save mean tensor for backward
    :param rstd_ptr: save standard deviation tensor for backward
    :param a_row_stride: stride of the input tensor
    :param N_SIZE: number of elements per row in the input tensor
    :param eps: epsilon value to avoid division by zero
    :param HAS_BIAS: whether the bias is provided
    :param IS_RMSNORM: whether the normalization is rmsnorm (False == layernorm)
    :param BLOCK_N_SIZE: number of threads per block
    :return: None
    """
    # position of elements processed by this program
    row_idx = tl.program_id(0)

    a_row_off = row_idx * a_row_stride
    block_range_offs = tl.arange(0, BLOCK_N_SIZE)
    # compute mean
    mean = 0.0
    var = 0.0
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        n_end_off = min((block_n_start_idx + BLOCK_N_SIZE), N_SIZE)
        block_cols_count = n_end_off - block_n_start_idx
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        # eviction policy below have little impact now because of new implementation. Kept as is.
        # float32 is used to avoid overflow because of the square operation
        a = tl.load(
            a_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        if IS_RMSNORM:
            var += tl.sum(a * a, axis=0)
        else:
            block_mean = tl.sum(a, axis=0) / block_cols_count
            # mean is 0 or has a mask applied to it, no need to mask delta_mean!
            delta_mean = block_mean - mean
            delta_mean_sqr = delta_mean * delta_mean

            block_delta = tl.sum((a - block_mean) * a, axis=0)
            # mean has a mask
            mean += tl.sum((a - mean) * a_ptr_mask, axis=0) / n_end_off
            var += (
                block_delta
                + delta_mean_sqr * (block_n_start_idx * block_cols_count) / n_end_off
            )

    var /= N_SIZE
    rstd = 1 / tl.sqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        weight = tl.load(weight_ptr + col_offs, mask=a_ptr_mask)

        # eviction policy helps to keep weights in cache (reused by other threads)
        a = tl.load(
            a_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight
        if HAS_BIAS:
            bias = tl.load(bias_ptr + col_offs, mask=a_ptr_mask)
            out = out + bias
        # write-back
        out_ = out.to(skip_out_ptr.dtype.element_ty)
        b = tl.load(
            b_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        tl.store(
            skip_out_ptr + row_idx * output_row_stride + col_offs * output_col_stride,
            (out_ + b),
            mask=a_ptr_mask,
        )


@register_torch_op
def skip_layer_norm_fwd_fused_single_pass(
    x: torch.Tensor,
    skip: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    use_rms_norm: bool = False,
):
    assert (
        x.dtype == weight.dtype
    ), f"input and weight bias must have the same dtype: {x.dtype}, {weight.dtype}"
    if bias is not None:
        assert (
            x.dtype == bias.dtype
        ), f"input and bias must have the same dtype: {x.dtype}, {bias.dtype}"
    # catch eps being too small if the tensors are fp16
    if x.dtype == torch.float16:
        eps = max(eps, 1.6e-5)
    # allocate output
    out = torch.empty_like(x, dtype=x.dtype)
    # reshape input data into 2D tensor
    a_arg = x.view(-1, x.shape[-1])
    b_arg = skip.view(-1, skip.shape[-1])
    M, N = a_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    kernel_skip_layer_norm_fwd_fused_single_pass[(M,)](
        skip_out_ptr=out,
        a_ptr=a_arg,
        b_ptr=b_arg,
        weight_ptr=weight,
        bias_ptr=bias if bias is not None else a_arg,
        output_row_stride=out.stride(-2),
        output_col_stride=out.stride(-1),
        a_row_stride=a_arg.stride(0),
        a_col_stride=a_arg.stride(1),
        N_SIZE=N,
        eps=eps,
        HAS_BIAS=bias is not None,
        IS_RMSNORM=use_rms_norm,
        BLOCK_N_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out


@triton.jit
def kernel_skip_layer_norm_fwd_fused_single_pass_2(
    output_ptr,
    skip_out_ptr,
    a_ptr,
    b_ptr,
    weight_ptr,
    bias_ptr,
    output_row_stride,
    output_col_stride,
    a_row_stride,
    a_col_stride,
    N_SIZE,
    eps,
    HAS_BIAS: tl.constexpr,
    IS_RMSNORM: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    """
    Layernorm based on Welford's variance computation algorithm.
    https://changyaochen.github.io/welford/
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

    :param output_ptr: output tensor
    :param a_ptr: input tensor
    :param weight_ptr: weights applied to the normalized input
    :param bias_ptr: bias added to the normalized input
    :param mean_ptr: save mean tensor for backward
    :param rstd_ptr: save standard deviation tensor for backward
    :param a_row_stride: stride of the input tensor
    :param N_SIZE: number of elements per row in the input tensor
    :param eps: epsilon value to avoid division by zero
    :param HAS_BIAS: whether the bias is provided
    :param IS_RMSNORM: whether the normalization is rmsnorm (False == layernorm)
    :param BLOCK_N_SIZE: number of threads per block
    :return: None
    """
    # position of elements processed by this program
    row_idx = tl.program_id(0)

    a_row_off = row_idx * a_row_stride
    block_range_offs = tl.arange(0, BLOCK_N_SIZE)
    # compute mean
    mean = 0.0
    var = 0.0
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        n_end_off = min((block_n_start_idx + BLOCK_N_SIZE), N_SIZE)
        block_cols_count = n_end_off - block_n_start_idx
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        # eviction policy below have little impact now because of new implementation. Kept as is.
        # float32 is used to avoid overflow because of the square operation
        a = tl.load(
            a_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        if IS_RMSNORM:
            var += tl.sum(a * a, axis=0)
        else:
            block_mean = tl.sum(a, axis=0) / block_cols_count
            # mean is 0 or has a mask applied to it, no need to mask delta_mean!
            delta_mean = block_mean - mean
            delta_mean_sqr = delta_mean * delta_mean

            block_delta = tl.sum((a - block_mean) * a, axis=0)
            # mean has a mask
            mean += tl.sum((a - mean) * a_ptr_mask, axis=0) / n_end_off
            var += (
                block_delta
                + delta_mean_sqr * (block_n_start_idx * block_cols_count) / n_end_off
            )

    var /= N_SIZE
    rstd = 1 / tl.sqrt(var + eps)

    # multiply by weight and add bias
    for block_n_start_idx in range(0, N_SIZE, BLOCK_N_SIZE):
        col_offs = block_n_start_idx + block_range_offs
        a_ptr_mask = col_offs < N_SIZE
        weight = tl.load(weight_ptr + col_offs, mask=a_ptr_mask)

        # eviction policy helps to keep weights in cache (reused by other threads)
        a = tl.load(
            a_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        a_hat = (a - mean) * rstd
        out = a_hat * weight
        if HAS_BIAS:
            bias = tl.load(bias_ptr + col_offs, mask=a_ptr_mask)
            out = out + bias
        # write-back
        out_ = out.to(output_ptr.dtype.element_ty)
        tl.store(
            output_ptr + row_idx * output_row_stride + col_offs * output_col_stride,
            out_,
            mask=a_ptr_mask,
        )
        b = tl.load(
            b_ptr + a_row_off + col_offs * a_col_stride,
            mask=a_ptr_mask,
            other=0.0,
            eviction_policy="evict_last",
        )
        tl.store(
            skip_out_ptr + row_idx * output_row_stride + col_offs * output_col_stride,
            (out_ + b),
            mask=a_ptr_mask,
        )


@register_torch_op
def skip_layer_norm_fwd_fused_single_pass2(
    x: torch.Tensor,
    skip: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    use_rms_norm: bool = False,
) -> tuple:
    assert (
        x.dtype == weight.dtype
    ), f"input and weight bias must have the same dtype: {x.dtype}, {weight.dtype}"
    if bias is not None:
        assert (
            x.dtype == bias.dtype
        ), f"input and bias must have the same dtype: {x.dtype}, {bias.dtype}"
    # catch eps being too small if the tensors are fp16
    if x.dtype == torch.float16:
        eps = max(eps, 1.6e-5)
    # allocate output
    out = torch.empty_like(x, dtype=x.dtype)
    skip_out = torch.empty_like(x, dtype=x.dtype)
    # reshape input data into 2D tensor
    a_arg = x.view(-1, x.shape[-1])
    b_arg = skip.view(-1, skip.shape[-1])
    M, N = a_arg.shape
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    BLOCK_SIZE = max(BLOCK_SIZE, 128)
    BLOCK_SIZE = min(BLOCK_SIZE, 4096)
    # heuristics for number of warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    kernel_skip_layer_norm_fwd_fused_single_pass_2[(M,)](
        output_ptr=out,
        skip_out_ptr=skip_out,
        a_ptr=a_arg,
        b_ptr=b_arg,
        weight_ptr=weight,
        bias_ptr=bias if bias is not None else a_arg,
        output_row_stride=out.stride(-2),
        output_col_stride=out.stride(-1),
        a_row_stride=a_arg.stride(0),
        a_col_stride=a_arg.stride(1),
        N_SIZE=N,
        eps=eps,
        HAS_BIAS=bias is not None,
        IS_RMSNORM=use_rms_norm,
        BLOCK_N_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return out, skip_out


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    a = torch.randn(100, 10, 256, dtype=torch.float16, device="cuda")
    b = torch.randn(100, 10, 256, dtype=torch.float16, device="cuda")
    layer = nn.LayerNorm(256).cuda().to(torch.float16)
    torch_res_0 = layer(a)
    torch_res_1 = torch_res_0 + b
    triton_res_0, triton_res_1 = skip_layer_norm_fwd_fused_single_pass2(
        a, b, layer.weight, layer.bias, 1e-5, False
    )
    print((torch_res_0 - triton_res_0).abs().max())
    print((torch_res_1 - triton_res_1).abs().max())

    triton_res_0 = torch.ops.triton_op.layer_norm_fwd_fused_single_pass(
        a, layer.weight, layer.bias, 1e-5, False
    )
    print((torch_res_0 - triton_res_0).abs().max())
