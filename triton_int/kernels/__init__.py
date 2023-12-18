import inspect

import torch

lib = torch.library.Library("triton_op", "FRAGMENT")

def register_torch_op(func):
    # get func name:
    func_name = func.__name__
    # get func args types:
    func_args = inspect.getfullargspec(func)
    

    lib.define(f"{func_name}({', '.join([ty.__name__ + ' ' + arg for arg, ty in func_args.annotations.items() if arg != 'return'])}) -> {func_args.annotations.get('return', torch.Tensor).__name__}")


    # # All that's needed for torch.compile support
    # @torch.library.impl(lib, func_name, "Meta")
    # def skip_layer_norm_fwd_fused_single_pass2(q, k, v, rel_h_w, sm_scale):
    #     return q.contiguous()

    torch.library.impl(lib, func_name, "CUDA")(func)
    
    return func