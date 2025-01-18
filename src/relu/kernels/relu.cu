#include<cuda_runtime.h>
#include<cuda_fp16.h>

#define HALF(val) reinterpret_cast<half*>(&(val))[0]
#define HALF2(val) reinterpret_cast<half2*>(&(val))[0]
#define FLOAT4(val) reinterpret_cast<float4*>(&(val))[0]

#define RELU(x) fmaxf(0.0f, (x))
#define HRELU(x) __max(__float2half(0.0f), (x))
#define H2RELU(x) __hmax2(__float2half2_rn(0.0f), (x))

__global__ void relu_fp32_kernel(float *a, float *b, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N){
    b[idx] = RELU(a[idx]);
  }
}

__global__ void relu_fp32x4_kernel(float *a, float *b, int N){
  int idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  float4 reg_a = FLOAT4(a[idx]);

  reg_a.x = RELU(reg_a.x);
  reg_a.y = RELU(reg_a.y);
  reg_a.z = RELU(reg_a.z);
  reg_a.w = RELU(reg_a.w);

  if(idx < N){
    FLOAT4(b[idx]) = reg_a;
  }
}

__global__ void relu_fp16_kernel(half *a, half *b, int N){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < N){
    b[idx] = HRELU(a[idx]);
  }
}

__global__ void relu_fp16x2_kernel(half *a, half *b, int N){
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 a_reg = HALF2(a[idx]);
  if(idx < N){
    HALF2(b[idx]) = H2RELU(a_reg);
  }
}

__global__ void relu_fp16x2o_kernel(half *a, half *b, int N){
  int idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 a_reg = HALF2(a[idx]);
  reg_a.x = HRELU(reg_a.x);
  reg_a.y = HRELU(reg_a.y);
  if(idx < N){
    HALF2(b[idx]) = reg_a;
  }
}

__global__ void relu_fp16x8_kernel(half *a, half *b, int N){
  int idx = 8 * (blockIdx.x * blockDim.x + threadIdx.x);
  half2 reg_a[8], reg_b[8];
  reg_a[0] = FLOAT4(a[idx]);

  #pragma unroll
  for(int i = 0; i < 8; i++){
    reg_b[i] = H2RELU(reg_a[i]);
  }

  if(idx < N){
    FLOAT4(b[idx]) = FLOAT4(reg_b[0]);
  }
}

#define LAUNCHER(kernel_name, element_type, tensor_type, elements_per_thread) \
  torch::Tensor relu_##kernel_name##_launcher(torch::Tensor a){ \
    int ndim = a.dim(); \
    int N = 1; \
    for(int i = 0; i < ndim; i++){ \
      N *= a.size(i); \
    } \
    auto b = torch::emptylike(a); \
    if(ndim != 2){  \
        dim3 blockDim(256/elements_per_thread); \
        dim3 gridDim((256 + N - 1)/256); \
        relu_##kernel_name##_kernel<<<gridDim, blockDim>>> \
        (reinterpret_cast<element_type*>(a.data_ptr(), \
         reinterpret_cast<element_type*>(b.data_ptr(), \
         N); \
    } else { \
      int features_num = a.size(1); \
      int batch_size = a.size(0); \
      if(features_num/elements_per_thread <= 1024){ \
        dim3 blockDim(features_num/elements_per_thread); \
        dim3 gridDim(batch_size); \
        relu_##kernel_name##_kernel<<<gridDim, blockDim>>> \
        (reinterpret_cast<element_type*>(a.data_ptr(), \
         reinterpret_cast<element_type*>(b.data_ptr(), \
         N); \
      } else { \
        dim3 blockDim(256/elements_per_thread); \
        dim3 gridDim((256 + N - 1)/256); \
        relu_##kernel_name##_kernel<<<gridDim, blockDim>>> \
        (reinterpret_cast<element_type*>(a.data_ptr(), \
         reinterpret_cast<element_type*>(b.data_ptr(), \
         N); \
      } \
    } \
    return b; \
  }

#define STRINGFY(str) #str
#define BIND(func) \
  m.def(STRINGFY(func), &func, STRINGFY(func));

LAUNCHER(fp32, float, torch::kFloat, 1)
LAUNCHER(fp32x4, float, torch::kFloat, 4)

LAUNCHER(fp16, half, torch::kHalf, 1)
LAUNCHER(fp16x2, half, torch::kHalf, 2)
LAUNCHER(fp16x2o, half, torch::kHalf, 2)
LAUNCHER(fp16x8, half, torch::kHalf, 8)

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  BIND(relu_fp32_launcher)
  BIND(relu_fp32x4_launcher)
  BIND(relu_fp16_launcher)
  BIND(relu_fp16x2_launcher)
  BIND(relu_fp16x2o_launcher)
  BIND(sigmoid_fp16x8_launcher)
}
