import torch
import time
from torch.utils.cpp_extension import load

# Disable gradient calculations for efficiency
torch.set_grad_enabled(False)

# Load the CUDA extension (ensure 'histogram.cu' contains the corrected C++/CUDA code)
relu = load(
    name="relu",
    sources=["relu.cu"],
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
    ],
    extra_cflags=['-std=c++17']
)


def run_bench(kernel, a:torch.Tensor, kernel_name:str,
              warmup:int=10, itr:int=1000, print_res:bool=False):

  for _ in range(warmup):
    c = kernel(a)

  torch.cuda.synchronize()

  start = time.time()

  for _ in range(itr):
    c = kernel(a)

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

  for sample_num in samples_nums:
    for feature_num in features_nums:
      print("-=" * 20, f"({sample_num}, {feature_num})", "=-" * 20 )

      print("\n", "-" * 20, "FP32", "-" * 20)
      a = torch.randn([sample_num, feature_num]).cuda().float()

      print()
      run_bench(relu.relu_fp32_launcher, a,"f32")
      run_bench(relu.relu_fp32x4_launcher, a, "f32x4")

      print("\n", "-" * 20, "FP16", "-" * 20)
      a_16f = a.half().contiguous()

      print()
      run_bench(relu.relu_fp16_launcher, a_16f, "f16")
      run_bench(relu.relu_fp16x2o_launcher, a_16f, "f16x2o")
      run_bench(relu.relu_fp16x8_launcher, a_16f, "f16x8")
      print()


if __name__ == "__main__":
  main()
