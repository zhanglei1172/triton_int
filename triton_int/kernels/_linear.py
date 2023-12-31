import torch
import triton
import triton.language as tl
from triton_int.kernels import register_torch_op

from triton_int.kernels.utils import gemm_autotune_full


@gemm_autotune_full()
@triton.jit
def kernel_linear_afp32_wfp32_bfp32_ofp32(
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
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    bias_ptrs = bias_ptr + offs_bn
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    bias = tl.load(bias_ptrs)
    accumulator += bias[None, :].to(tl.float32)
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
    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)

@register_torch_op
def linear_afp32_wfp32_bfp32_ofp32(a, b, bias, out=None, dtype=None):
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
    kernel_linear_afp32_wfp32_bfp32_ofp32[grid](
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


if __name__ == "__main__":
    import time

    import torch
    import triton

    torch.manual_seed(0)
    torch.set_printoptions(precision=10)

    M, N, K = 1024, 1024, 1024
    
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    layer = torch.nn.Linear(K, N, bias=True, dtype=torch.float16).cuda()
    
    torch_res = layer(a)
    triton_res = linear_afp32_wfp32_bfp32_ofp32(a, layer.weight, layer.bias)
    
    triton_res.contiguous()
    print((torch_res-triton_res).abs().max())