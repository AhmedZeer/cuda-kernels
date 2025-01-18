#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

#define HALF(val) reinterpret_cast<half *>(&(val))[0]
#define HALF2(val) reinterpret_cast<half2 *>(&(val))[0]
#define FLOAT4(val) reinterpret_cast<float4 *>(&(val))[0]

#define ELU(x,alpha) (((x) < (0.0f)) ? ((alpha)*(expf(x)-1.0f)) : (0.0f))
#define HELU(x,alpha) (((x) < (__float2half(0.0f))) ? ((alpha)*(expf(x)-(__float2half(1.0f)))) : (__float2half(0.0f)))

__global__ void elu_fp32_kernel(float *a, float *b, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float alpha = 1.0f
    b[idx] = ELU(a[idx], alpha);
  }
}

__global__ void elu_fp32x4_kernel(float *a, float *b, int N) {
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);

  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b;
    float alpha = 1.0f;

    reg_b.x = ELU(reg_a.x, alpha);
    reg_b.y = ELU(reg_a.y, alpha);
    reg_b.z = ELU(reg_a.z, alpha);
    reg_b.w = ELU(reg_a.w, alpha);

    FLOAT4(b[idx]) = reg_b;
  }
}

__global__ void elu_fp16_kernel(half *a, half *b, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    b[idx] = HELU(a[idx]);
  }
}

__global__ void elu_fp16x2_kernel(half *a, half *b, int N) {
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 a_reg = HALF2(a[idx]);
  half2 b_reg;

  if (idx < N) {
    half alpha = 1.0f;
    b_reg.x = HELU(a_reg.x, alpha);
    b_reg.y = HELU(a_reg.y, alpha);
    HALF2(b[idx]) = b_reg;
  }
}

__global__ void relu_fp16x8_kernel(half *a, half *b, int N) {
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 reg_a[8], reg_b[8];
  FLOAT4(reg_a[0]) = FLOAT4(a[idx]);

  if (idx < N) {
    float alpha = 1.0f;

    #pragma unroll
    for (int i = 0; i < 8; i++) {
      reg_b[i + 0] = HELU(reg_a[i].x, alpha);
      reg_b[i + 1] = HELU(reg_a[i].y, alpha);
    }

    FLOAT4(b[idx]) = FLOAT4(reg_b[0]);
  }
}

#define LAUNCHER(kernel_name, element_type, tensor_type, elements_per_thread)  \
  torch::Tensor elu_##kernel_name##_launcher(torch::Tensor a) {               \
    int ndim = a.dim();                                                        \
    int N = 1;                                                                 \
    for (int i = 0; i < ndim; i++) {                                           \
      N *= a.size(i);                                                          \
    }                                                                          \
    auto b = torch::empty_like(a);                                             \
    if (ndim != 2) {                                                           \
      dim3 blockDim(256 / elements_per_thread);                                \
      dim3 gridDim((256 + N - 1) / 256);                                       \
      elu_##kernel_name##_kernel<<<gridDim, blockDim>>>(                      \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()), N);                  \
    } else {                                                                   \
      int features_num = a.size(1);                                            \
      int batch_size = a.size(0);                                              \
      if (features_num / elements_per_thread <= 1024) {                        \
        dim3 blockDim(features_num / elements_per_thread);                     \
        dim3 gridDim(batch_size);                                              \
        elu_##kernel_name##_kernel<<<gridDim, blockDim>>>(                    \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()), N);                \
      } else {                                                                 \
        dim3 blockDim(256 / elements_per_thread);                              \
        dim3 gridDim((256 + N - 1) / 256);                                     \
        elu_##kernel_name##_kernel<<<gridDim, blockDim>>>(                    \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
    return b;                                                                  \
  }

#define STRINGFY(str) #str
#define BIND(func) m.def(STRINGFY(func), &func, STRINGFY(func));

LAUNCHER(fp32, float, torch::kFloat, 1)
LAUNCHER(fp32x4, float, torch::kFloat, 4)

LAUNCHER(fp16, half, torch::kHalf, 1)
LAUNCHER(fp16x2, half, torch::kHalf, 2)
LAUNCHER(fp16x2o, half, torch::kHalf, 2)
LAUNCHER(fp16x8, half, torch::kHalf, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  BIND(elu_fp32_launcher)
  BIND(elu_fp32x4_launcher)
  BIND(elu_fp16_launcher)
  BIND(elu_fp16x2_launcher)
  BIND(elu_fp16x2o_launcher)
  BIND(elu_fp16x8_launcher)
}

