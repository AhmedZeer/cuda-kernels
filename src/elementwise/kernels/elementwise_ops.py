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
                   sources=["elementwise_ops.cu"],
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
    vals = c.flatten().detach().cpu().numpy().tolist()[-2:]
    vals = [round(val, 6) for val in vals]
    print(f"{vals}, duration: {duration:.2f}ms, out_{kernel_name}")
    return c

def main():
    samples_nums = [1024, 2048, 4096]
    features_nums = [1024, 2048, 4096]

    for sample_num, feature_num in zip(samples_nums, features_nums):
        print("-=" * 20, f"({sample_num}, {feature_num})", "=-" * 20 )

        print("\n", "-" * 20, "FP32", "-" * 20)
        a = torch.randn([sample_num, feature_num]).cuda().float()
        b = torch.randn([sample_num, feature_num]).cuda().float()
        c = torch.randn([sample_num, feature_num]).cuda().float()

        print()
        run_bench(elementwise.element_wise_add_f32, a, b, "f32", c)
        run_bench(elementwise.element_wise_add_f32x4, a, b, "f32x4", c)

        print("\n", "-" * 20, "FP16", "-" * 20)
        a_16f = a.half().contiguous()
        b_16f = b.half().contiguous()
        c_16f = c.half().contiguous()

        print()
        run_bench(elementwise.element_wise_add_f16, a_16f, b_16f, "f16", c_16f)
        run_bench(elementwise.element_wise_add_f16x2, a_16f, b_16f, "f16x2", c_16f)
        run_bench(elementwise.element_wise_add_f16x8, a_16f, b_16f, "f16x8", c_16f)
        run_bench(elementwise.element_wise_add_f16x8_packed, a_16f, b_16f, "f16x8_packed", c_16f)
        print()

if __name__ == "__main__":
    main()

