import torch
import time
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

histogram = load(name="histogram_kernel",
                   sources=["histogram.cu"],
                   extra_cuda_cflags=[
                     "-O3",
                     "-U__CUDA_NO_HALF_CONVERSIONS__",
                     "-U__CUDA_NO_HALF_OPERATORS__",
                     "-U__CUDA_NO_HALF2_OPERATORS__",
                     "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                     "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                     "--expt-relaxed-constexpr",
                     "--expt-extended-lambda",
                     "--use_fast_math",
                   ], extra_cflags=['-std=c++17'])

