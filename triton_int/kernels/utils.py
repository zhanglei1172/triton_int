from functools import partial

import triton


def bmm_autotune():
    return triton.autotune(
        configs=[
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 256,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 128,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 16,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 16,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 16,
                },
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 64,
                    "GROUP_SIZE_M": 16,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 16,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 256,
                    "GROUP_SIZE_M": 16,
                },
                num_stages=2,
                num_warps=4,
            ),
        ],
        key=["M", "N", "K"],
        reset_to_zero=["c_ptr"],
    )


def gemm_autotune():
    return triton.autotune(
        configs=[
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 16,
                },
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 256,
                    "GROUP_SIZE_M": 16,
                },
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 32,
                    "BLOCK_SIZE_N": 32,
                    "BLOCK_SIZE_K": 128,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=2,
                num_warps=4,
            ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 64,
            #         "BLOCK_SIZE_N": 16,
            #         "BLOCK_SIZE_K": 64,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 64,
            #         "BLOCK_SIZE_N": 32,
            #         "BLOCK_SIZE_K": 64,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 16,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 32,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 256,
            #         "BLOCK_SIZE_N": 64,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 64,
            #         "BLOCK_SIZE_N": 256,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 128,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 128,
            #         "BLOCK_SIZE_N": 64,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
            # triton.Config(
            #     {
            #         "BLOCK_SIZE_M": 64,
            #         "BLOCK_SIZE_N": 128,
            #         "BLOCK_SIZE_K": 32,
            #         "GROUP_SIZE_M": 8,
            #     },
            #     num_stages=4,
            #     num_warps=4,
            # ),
        ],
        key=["M", "N", "K"],
        reset_to_zero=["c_ptr"],
    )
