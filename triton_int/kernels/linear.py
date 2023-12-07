import torch
import triton
import triton.language as tl

try:
    from torch_int._CUDA import linear_a8_w8_b8_o8 as linear_a8_w8_b8_o8_cuda
    from torch_int._CUDA import linear_a8_w8_b32_o32 as linear_a8_w8_b32_o32_cuda
    from torch_int._CUDA import (
        linear_a8_w8_b32_o32_with_scaling as linear_a8_w8_b32_o32_with_scaling_cuda,
    )
    from torch_int._CUDA import (
        linear_a8_w8_bfp32_ofp32 as linear_a8_w8_bfp32_ofp32_cuda,
    )
    from torch_int._CUDA import linear_relu_a8_w8_b8_o8 as linear_relu_a8_w8_b8_o8_cuda
except:
    pass

from triton_int.functional.quantization import quantize_per_tensor_absmax
from triton_int.kernels.utils import bmm_autotune, gemm_autotune


@bmm_autotune()
@triton.jit
def kernel_linear_a8_w8_ofp32(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    scale,
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
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    c = (tl.load(scale) * accumulator).to(c_ptr.dtype.element_ty)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def linear_a8_w8_ofp32(a, b, scale, out=None, dtype=torch.float):
    # Check constraints.
    tmp_shape = a.shape[:-1]
    assert len(b.shape) == 2
    a = a.view(-1, a.shape[-1])
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    if out == None:
        c = torch.empty((M, N), device=a.device, dtype=dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel_linear_a8_w8_ofp32[grid](
        a,
        b,
        c,
        scale,
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
    return c.view(*tmp_shape, N)


@gemm_autotune()
@triton.jit
def kernel_linear_a8_w8_b8_o8(
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
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    bias = tl.load(bias_ptrs)
    c = (
        accumulator.to(tl.float32) * tl.load(scale_a)
        + bias.to(tl.float32) * tl.load(scale_b)
    ).to(tl.int8)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def linear_a8_w8_b8_o8(a, b, bias, scale_a: float, scale_b: float, out=None):
    # Check constraints.
    tmp_shape = a.shape[:-1]
    a = a.view(-1, a.shape[-1])
    assert len(b.shape) == 2
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    if out == None:
        c = torch.empty((M, N), device=a.device, dtype=torch.int8)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel_linear_a8_w8_b8_o8[grid](
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


@gemm_autotune()
@triton.jit
def kernel_linear_relu_a8_w8_b8_o8(
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
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    bias = tl.load(bias_ptrs)
    c = tl.maximum(
        (
            accumulator.to(tl.float32) * tl.load(scale_a)
            + bias.to(tl.float32) * tl.load(scale_b)
        ).to(tl.int8),
        0,
    )
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def linear_relu_a8_w8_b8_o8(a, b, bias, scale_a: float, scale_b: float, out=None):
    # Check constraints.
    tmp_shape = a.shape[:-1]
    a = a.view(-1, a.shape[-1])
    assert len(b.shape) == 2
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    if out == None:
        c = torch.empty((M, N), device=a.device, dtype=torch.int8)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel_linear_relu_a8_w8_b8_o8[grid](
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


@gemm_autotune()
@triton.jit
def kernel_linear_a8_w8_b32_o32(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
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
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    bias = tl.load(bias_ptrs)
    accumulator += bias
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def linear_a8_w8_b32_o32(a, b, bias, out=None):
    # Check constraints.
    tmp_shape = a.shape[:-1]
    assert len(b.shape) == 2
    a = a.view(-1, a.shape[-1])
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    if out == None:
        c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel_linear_a8_w8_b32_o32[grid](
        a,
        b,
        c,
        bias,
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


@gemm_autotune()
@triton.jit
def kernel_linear_a8_w8_b32_o32_with_scaling(
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
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    bias = tl.load(bias_ptrs)
    c = (
        accumulator.to(tl.float32) * tl.load(scale_a)
        + bias.to(tl.float32) * tl.load(scale_b)
    ).to(tl.int32)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def linear_a8_w8_b32_o32_with_scaling(a, b, bias, scale_a, scale_b, out=None):
    # Check constraints.
    tmp_shape = a.shape[:-1]
    assert len(b.shape) == 2
    a = a.view(-1, a.shape[-1])
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    if out == None:
        c = torch.empty((M, N), device=a.device, dtype=torch.int32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel_linear_a8_w8_b32_o32_with_scaling[grid](
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


@gemm_autotune()
@triton.jit
def kernel_linear_a8_w8_bfp32_ofp32(
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
    c = accumulator.to(tl.float32) * tl.load(scale_a) + bias * tl.load(scale_b)
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c.to(c_ptr.dtype.element_ty), mask=c_mask)


def linear_a8_w8_bfp32_ofp32(a, b, bias, scale_a, scale_b, out=None, dtype=None):
    # Check constraints.
    tmp_shape = a.shape[:-1]
    assert len(b.shape) == 2
    a = a.view(-1, a.shape[-1])
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    M, K = a.shape
    N, K = b.shape
    # Allocates output.
    if out == None:
        if dtype is None:
            dtype = bias.dtype
        c = torch.empty((M, N), device=a.device, dtype=dtype)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    kernel_linear_a8_w8_bfp32_ofp32[grid](
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


@torch.no_grad()
def test_correct_int8(M=32, N=4096, K=4096):
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((N, K), device="cuda", dtype=torch.float32)
    bias = torch.randn((N), device="cuda", dtype=torch.float32)
    int_a, scale_a = quantize_per_tensor_absmax(a.clone())
    int_bias, scale_bias = quantize_per_tensor_absmax(bias.clone())
    cos = torch.nn.CosineSimilarity(0)
    print(
        "Quantization cos",
        cos(
            (int_a * scale_a).flatten().to(torch.float32),
            a.flatten().to(torch.float32),
        ),
    )
    int_b, scale_b = quantize_per_tensor_absmax(b.clone())
    torch_output = torch.matmul(a, b.transpose(0, 1))
    scale_out = quantize_per_tensor_absmax(torch_output.clone())[1]
    triton_output = linear_a8_w8_b8_o8(
        int_a,
        int_b,
        int_bias,
        (scale_a * scale_b / scale_out).item(),
        (scale_bias / scale_out).item(),
    ).float() * (scale_out)
    cuda_output = linear_a8_w8_b8_o8_cuda(
        int_a,
        int_b,
        int_bias,
        (scale_a * scale_b / scale_out).item(),
        (scale_bias / scale_out).item(),
    ).float() * (scale_out)
    print(f"triton_output={triton_output}")
    print(f"torch_output={torch_output}")
    print(f"cuda_output={cuda_output}")
    cos = torch.nn.CosineSimilarity(0)
    print(
        "Output cos",
        cos(
            triton_output.flatten().to(torch.float32),
            torch_output.flatten().to(torch.float32),
        ),
    )
    int_bias32 = (
        (bias / (scale_a * scale_b))
        .clamp(torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max)
        .to(torch.int32)
    )
    triton_output = linear_a8_w8_b32_o32(int_a, int_b, int_bias32)
    cuda_output = linear_a8_w8_b32_o32_cuda(int_a, int_b, int_bias32)
    print(
        "Output abs_max", (triton_output.flatten() - cuda_output.flatten()).abs().max()
    )

    bias_scale = bias.abs().max() / (2**31 - 1)
    int32_bias = (bias / bias_scale).round().to(torch.int32)
    triton_output = linear_a8_w8_b32_o32_with_scaling(
        int_a,
        int_b,
        int32_bias,
        (scale_a * scale_b / scale_out).item(),
        (bias_scale / scale_out).item(),
    )
    cuda_output = linear_a8_w8_b32_o32_with_scaling_cuda(
        int_a,
        int_b,
        int32_bias,
        (scale_a * scale_b / scale_out).item(),
        (bias_scale / scale_out).item(),
    )
    print(
        "Output abs_max", (triton_output.flatten() - cuda_output.flatten()).abs().max()
    )

    triton_output = linear_a8_w8_bfp32_ofp32(
        int_a, int_b, bias, (scale_a * scale_b).item(), 1
    )
    cuda_output = linear_a8_w8_bfp32_ofp32_cuda(
        int_a, int_b, bias, (scale_a * scale_b).item(), 1
    )
    print(
        "Output abs_max", (triton_output.flatten() - cuda_output.flatten()).abs().max()
    )

    relu_scale_out = quantize_per_tensor_absmax(torch_output.clip(0).clone())[1]
    triton_output = linear_relu_a8_w8_b8_o8(
        int_a,
        int_b,
        int_bias,
        (scale_a * scale_b / relu_scale_out).item(),
        (scale_bias / relu_scale_out).item(),
    ).float() * (relu_scale_out)
    cuda_output = linear_relu_a8_w8_b8_o8_cuda(
        int_a,
        int_b,
        int_bias,
        (scale_a * scale_b / relu_scale_out).item(),
        (scale_bias / relu_scale_out).item(),
    ).float() * (relu_scale_out)
    print(
        "Output abs_max", (triton_output.flatten() - cuda_output.flatten()).abs().max()
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],  # Argument names to use as an x-axis for the plot
        x_vals=[32, 64, 128, 256]
        + [512 * i * 2 for i in range(1, 17)],  # Different possible values for `x_name`
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=["cutlass", "triton-i8", "triton-i8-fp16", "triton-i8-bf16"],
        # Label name for the lines
        line_names=["cutlass", "triton-i8", "triton-i8-fp16", "triton-i8-bf16"],
        # Line styles
        styles=[("green", "-"), ("blue", "-"), ("red", "-"), ("purple", "-")],
        ylabel="ms",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, provider):
    K = 10240
    N = 27392 * 2 // 8
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cutlass":
        a = (
            torch.randn((M, K), device="cuda", dtype=torch.float16)
            .to(torch.int8)
            .contiguous()
        )
        b = (
            torch.randn((N, K), device="cuda", dtype=torch.float16)
            .to(torch.int8)
            .contiguous()
        )
        bias = torch.randn((N), device="cuda", dtype=torch.float32).contiguous()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: linear_a8_w8_bfp32_ofp32_cuda(a, b, bias, 1, 1), quantiles=quantiles
        )
    if provider == "triton-i8":
        a = (
            torch.randn((M, K), device="cuda", dtype=torch.float16)
            .to(torch.int8)
            .contiguous()
        )
        b = (
            torch.randn((N, K), device="cuda", dtype=torch.float16)
            .to(torch.int8)
            .contiguous()
        )
        bias = torch.randn((N), device="cuda", dtype=torch.float32).contiguous()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: linear_a8_w8_bfp32_ofp32(a, b, bias, 1, 1), quantiles=quantiles
        )
    if provider == "triton-i8-fp16":
        a = (
            torch.randn((M, K), device="cuda", dtype=torch.float16)
            .to(torch.int8)
            .contiguous()
        )
        b = (
            torch.randn((N, K), device="cuda", dtype=torch.float16)
            .to(torch.int8)
            .contiguous()
        )
        bias = torch.randn((N), device="cuda", dtype=torch.float16).contiguous()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: linear_a8_w8_bfp32_ofp32(a, b, bias, 1, 1), quantiles=quantiles
        )

    if provider == "triton-i8-bf16":
        a = (
            torch.randn((M, K), device="cuda", dtype=torch.float16)
            .to(torch.int8)
            .contiguous()
        )
        b = (
            torch.randn((N, K), device="cuda", dtype=torch.float16)
            .to(torch.int8)
            .contiguous()
        )
        bias = torch.randn((N), device="cuda", dtype=torch.bfloat16).contiguous()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: linear_a8_w8_bfp32_ofp32(a, b, bias, 1, 1), quantiles=quantiles
        )
    return ms, min_ms, max_ms


if __name__ == "__main__":
    test_correct_int8()
    benchmark.run(show_plots=False, print_data=True)
