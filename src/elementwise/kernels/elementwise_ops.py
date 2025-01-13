from typing import Optional
import torch
import time
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

"""
* O3 Flag, standing for level 3 optimization,
is used to let the compiler optimize the kernel
as much as possible.

* Undefining some macros makes sure that we use 
half, half2 and bfloat in the kernel with no problems.

* Using the '--expt-relaxed-constexpr' flag relaxes the 
restriction of calling a device function on the host and vice versa.

Refrences:
 - [constexpr][https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-functions]
 - [constexpr][https://forums.developer.nvidia.com/t/check-for-expt-relaxed-constexpr/62425]
 - [o3][https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html]
"""

elementwise = load(name="elementwise",
                   sources=["elemenetwise_ops.cu"],
                   extra_cuda_cflags=[
                     "-O3",
                     "-U__CUDA_NO_HALF_CONVERSIONS__",
                     "-U__CUDA_NO_HALF_OPERATORS__",
                     "-U__CUDA_NO_HALF2_OPERATORS__",
                     "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                     "--expt-relaxed-constexpr",
                     "--expt-extend-lambda",
                     "--use_fast_math",
                   ], extra_cflags=['-std=c++17'])

def run_bench():
