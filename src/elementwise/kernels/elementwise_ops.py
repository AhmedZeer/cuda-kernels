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

def run_bench(kernel, a:torch.Tensor, b:torch.Tensor,
              kernel_name:str, c:Optional[torch.Tensor]=None,
              warmup:int=10, itr:int=1000, print_res:bool=False):

    if c is not None:
        c.zero_();
        for _ in range(warmup):
            kernel(a,b,c)
    else:
        for _ in range(warmup):
            c = kernel(a,b)

    torch.cuda.synchronize()

    start = time.time()

    if c is not None:
        for _ in range(itr):
            kernel(a,b,c)
    else:
        for _ in range(itr):
            c = kernel(a,b)

    torch.cuda.synchronize()
    end = time.time()
    duration = (end - start) * 1e3
    vals = c.flatten().detach().numpy().tolist()[-2:]
    vals = [round(val, 6) for val in vals]
    print(f"out_{kernel_name}:", vals, "duration:", duration)
    return c

def main():
    samples_nums = [1024, 2048, 4096]
    features_nums = [1024, 2048, 4096]

    print("-=" * 80)
    for sample_num, feature_num in zip(samples_nums, features_nums):
        print("Sample Num:", sample_num, "Feature Num:", feature_num)

        a = torch.randn([sample_num, feature_num]).cuda().float()
        b = torch.randn([sample_num, feature_num]).cuda().float()
        c = torch.randn([sample_num, feature_num]).cuda().float()

        run_bench(elementwise.element_wise_add_f32, a, b, "f32", c)
        run_bench(elementwise.element_wise_add_f32x4, a, b, "f32x4", c)

        print("=-" * 80)
        a = torch.randn([sample_num, feature_num]).cuda().half()
        b = torch.randn([sample_num, feature_num]).cuda().half()
        c = torch.randn([sample_num, feature_num]).cuda().half()

        run_bench(elementwise.element_wise_add_f16, a, b, "f16", c)
        run_bench(elementwise.element_wise_add_f16x2, a, b, "f16x2", c)
        run_bench(elementwise.element_wise_add_f16x8, a, b, "f16x8", c)
        run_bench(elementwise.element_wise_add_f16x8_pack, a, b, "f16x8_pack", c)

if __name__ == "__main__":
    main()
