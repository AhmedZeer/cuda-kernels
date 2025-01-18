#include <torch/torch.h>
#include<cuda_runtime.h>
#include<cuda_fp16.h>

// Remember to compile with '--use_fast_math' flag.

#define LDST128BITS(val) (reinterpret_cast<float4 *>(&(val))[0])
#define FLOAT4(val) (reinterpret_cast<float4 *>(&(val))[0])
#define HALF2(val) (reinterpret_cast<half2 *>(&(val))[0])

// To ensure numerical stability and bypass overflow,
// we calculate the maximum number to be exponentiated,
// and make sure that we never pass it.
// Calculating that number is quite straightforward:
// e^x = MAX_FLOAT -> x = ln(MAX_FLOAT).

#define MAX_EXP_F32  88.0f
#define MIN_EXP_F32 -88.0f
#define MAX_EXP_F16  11.0f
#define MIN_EXP_F16 -11.0f

#define MAX_EXP_H16 __float2half(11.0f)
#define MIN_EXP_H16 __float2half(-11.0f)

#define SIGMOID(val) (1.0f / (1.0f + expf(-(val))))
#define HSIGMOID(val) (__float2half(1.0f) / (__float2half(1.0f) + hexp(-(val))))

#define BOUNDRIES(val) (fmaxf(fminf((val), (MAX_EXP_F32)), (MIN_EXP_F32)))
#define HBOUNDRIES(val) (__hmax(__hmin((val), (MAX_EXP_H16)), (MIN_EXP_H16)))

__global__ void sigmoid_fp32_kernel(float *a, float *b, int N){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < N){
    b[idx] = SIGMOID(BOUNDRIES(a[idx]));
  }
}

__global__ void sigmoid_fp32x4_kernel(float *a, float *b, int N){
  int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
  float4 register_a = FLOAT4(a[idx]);

  register_a.x = BOUNDRIES(register_a.x);
  register_a.y = BOUNDRIES(register_a.y);
  register_a.z = BOUNDRIES(register_a.z);
  register_a.w = BOUNDRIES(register_a.w);

  if(idx < N){
    b[idx + 0] = SIGMOID(register_a.x);
    b[idx + 1] = SIGMOID(register_a.y);
    b[idx + 2] = SIGMOID(register_a.z);
    b[idx + 3] = SIGMOID(register_a.w);
  }
}

__global__ void sigmoid_fp32x4o_kernel(float *a, float *b, int N){
  int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);
  float4 register_a = FLOAT4(a[idx]);

  register_a.x = SIGMOID(BOUNDRIES(register_a.x));
  register_a.y = SIGMOID(BOUNDRIES(register_a.y));
  register_a.z = SIGMOID(BOUNDRIES(register_a.z));
  register_a.w = SIGMOID(BOUNDRIES(register_a.w));

  if(idx < N){
    FLOAT4(b[idx]) = register_a;
  }
}

__global__ void sigmoid_fp16_kernel(half *a, half *b, int N){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < N){
    b[idx] = HSIGMOID(HBOUNDRIES(a[idx]));
  }
}

__global__ void sigmoid_fp16x2o_kernel(half *a, half *b, int N){
  int idx = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
  half2 register_a = HALF2(a[idx]);
  register_a.x = HSIGMOID(HBOUNDRIES(register_a.x));
  register_a.y = HSIGMOID(HBOUNDRIES(register_a.y));
  if(idx < N){
    HALF2(b[idx]) = register_a;
  }
}

__global__ void sigmoid_fp16x8_kernel(half *a, half *b, int N){
  int idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);

  half2 register_a_0 = HALF2(a[idx + 0]);
  half2 register_a_1 = HALF2(a[idx + 2]);
  half2 register_a_2 = HALF2(a[idx + 4]);
  half2 register_a_3 = HALF2(a[idx + 6]);

  if(idx < N){
    b[idx + 0] = HSIGMOID(HBOUNDRIES(register_a_0.x));
    b[idx + 1] = HSIGMOID(HBOUNDRIES(register_a_0.y));
  }

  if(idx + 2 < N){
    b[idx + 2] = HSIGMOID(HBOUNDRIES(register_a_1.x));
    b[idx + 3] = HSIGMOID(HBOUNDRIES(register_a_1.y));
  }

  if(idx + 4 < N){
    b[idx + 4] = HSIGMOID(HBOUNDRIES(register_a_2.x));
    b[idx + 5] = HSIGMOID(HBOUNDRIES(register_a_2.y));
  }

  if(idx + 6 < N){
    b[idx + 6] = HSIGMOID(HBOUNDRIES(register_a_3.x));
    b[idx + 7] = HSIGMOID(HBOUNDRIES(register_a_3.y));
  }
}

// WARNING: To use this kernel, tensor elements must be
// contiguos in memory.
__global__ void sigmoid_fp16x8_pack_kernel(half *a, half *b, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  half pack_a[8], pack_b[8];
  // pack_a[0] = LDST128BITS(a[idx]);
  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);

  #pragma unroll
  for(int i = 0; i < 8; i++){
    pack_b[i] = HSIGMOID(HBOUNDRIES(pack_a[i]));
  }

  if(idx + 7 < N){
    LDST128BITS(b[idx]) = LDST128BITS(pack_b[0]);
  }
}

#define TENSOR_CHECK_TYPE(T, t_dtype) \
  if(T.options().dtype() != t_dtype){ \
    throw std::runtime_error("Tensor dtype doesn't match"); \
  }

// kernel_name
// elements_per_thread
// tensor_type
// element_type

#define LAUNCHER(kernel_name, elements_per_thread, tensor_type, element_type) \
  torch::Tensor sigmoid_##kernel_name##_launcher(torch::Tensor a){ \
    TENSOR_CHECK_TYPE(a, tensor_type) \
    int N = 1; \
    int dim = a.dim(); \
    auto b = torch::empty_like(a); \
    if(dim != 2){ \
      for(int i = 0; i < a.dim(); i++) N *= a.size(i); \
      dim3 blockDim(256/elements_per_thread); \
      dim3 gridDim((256 + N - 1) / 256); \
      sigmoid_##kernel_name##_kernel<<<gridDim, blockDim>>>( \
          reinterpret_cast<element_type*>(a.data_ptr()), \
          reinterpret_cast<element_type*>(b.data_ptr()), \
          N); \
      return b; \
    } else { \
      int feature_size = a.size(1); \
      int batch_size = a.size(0); \
      int N = feature_size * batch_size; \
      if(feature_size/elements_per_thread < 1024){ \
        dim3 blockDim(feature_size/elements_per_thread); \
        dim3 gridDim(batch_size); \
        sigmoid_##kernel_name##_kernel<<<gridDim, blockDim>>>( \
            reinterpret_cast<element_type*>(a.data_ptr()), \
            reinterpret_cast<element_type*>(b.data_ptr()), \
            N); \
      } else { \
        dim3 blockDim(256/elements_per_thread); \
        dim3 gridDim((N + 256 - 1)/ 256); \
        sigmoid_##kernel_name##_kernel<<<gridDim, blockDim>>>( \
            reinterpret_cast<element_type*>(a.data_ptr()), \
            reinterpret_cast<element_type*>(b.data_ptr()), \
            N); \
      } \
    } \
  }

#define STRINGFY(str) #str
#define BIND(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

LAUNCHER(fp32, 1, torch::kFloat, float);
LAUNCHER(fp32x4, 4, torch::kFloat, float);
LAUNCHER(fp32x4o, 4, torch::kFloat, float);

LAUNCHER(fp16, 1, torch::kHalf, half);
LAUNCHER(fp16x2o, 2, torch::kHalf, half);
LAUNCHER(fp16x8, 8, torch::kHalf, half);
LAUNCHER(fp16x8_pack, 8, torch::kHalf, half);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  BIND(sigmoid_fp32_launcher)
  BIND(sigmoid_fp32x4_launcher)
  BIND(sigmoid_fp32x4o_launcher)
  BIND(sigmoid_fp16_launcher)
  BIND(sigmoid_fp16x2o_launcher)
  BIND(sigmoid_fp16x8_launcher)
  BIND(sigmoid_fp16x8_pack_launcher)
}
