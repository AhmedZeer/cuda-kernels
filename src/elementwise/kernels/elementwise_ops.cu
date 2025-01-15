#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>
#include <stdexcept>
using namespace std;


// Macros for casting.
// We get the addres of the 'vector',
// cast it, and derefrence it using [0]
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])

// Ensuring warp-wide memory alignment:
// 4 * Single Precision Floats (4 * 4 bytes = 16 bytes per thread),
// 2 * Half Precision Floats (2 * 2 bytes = 4 bytes per thread).
// Proper alignment ensures memory coalescing by enabling efficient
// 32-thread warp access to aligned memory blocks (e.g., 32 * 16 bytes = 512
// bytes). Single Instruction Multiple (SIMD).

// -> FP32 Kernels <- //

// Element Wise Add Operation.
// gridDim(N/256), blockDim(256)
// a: Nx1, b: Nx1, c: Nx1, c = add(a,b)
__global__ void element_wise_add_f32_kernel(float *a, float *b, float *c, int N) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

// Element Wise Add Operation.
// gridDim(N/256), blockDim(256 / 4).
// a: Nx1, b: Nx1, c: Nx1, c = add(a,b).
// This kernel leverages float4 datatype, which
// allows processing 4 floats at the same time.
__global__ void element_wise_add_f32x4_kernel(float *a, float *b, float *c, uint N) {
  uint idx = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    float4 reg_a = FLOAT4(a[idx]);
    float4 reg_b = FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FLOAT4(c[idx]) = reg_c;
  }
}

// -> FP16 Kernels <- //

// Element Wise Add Operation.
// gridDim(N/256), blockDim(256)
// a: Nx1, b: Nx1, c: Nx1, c = add(a,b)
__global__ void element_wise_add_f16_kernel(half *a, half *b, half *c, uint N) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    c[idx] = __hadd(a[idx], b[idx]);
}

// Element Wise Add Operation.
// gridDim(N/256), blockDim(256/2)
// a: Nx1, b: Nx1, c: Nx1, c = add(a,b)
__global__ void element_wise_add_f16x2_kernel(half *a, half *b, half *c, uint N) {
  uint idx = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < N) {
    half2 reg_a = HALF2(a[idx]);
    half2 reg_b = HALF2(b[idx]);
    half2 reg_c;
    reg_c.x = __hadd(reg_a.x, reg_b.x);
    reg_c.y = __hadd(reg_a.y, reg_b.y);
    HALF2(c[idx]) = reg_c;
  }
}

// Element Wise Add Operation.
// gridDim(N/256), blockDim(256/8)
// a: Nx1, b: Nx1, c: Nx1, c = add(a,b)
// This kernel leverages manual unrolling for
// better compiler based optimizations,
// L2 caching for future calls,
// reduced overhead from controls.
__global__ void element_wise_add_f16x8_kernel(half *a, half *b, half *c, uint N) {
  uint idx = 8 * (blockDim.x * blockIdx.x + threadIdx.x);
  half2 reg_a_0 = HALF2(a[idx + 0]);
  half2 reg_a_1 = HALF2(a[idx + 2]);
  half2 reg_a_2 = HALF2(a[idx + 4]);
  half2 reg_a_3 = HALF2(a[idx + 6]);

  half2 reg_b_0 = HALF2(b[idx + 0]);
  half2 reg_b_1 = HALF2(b[idx + 2]);
  half2 reg_b_2 = HALF2(b[idx + 4]);
  half2 reg_b_3 = HALF2(b[idx + 6]);
  half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;

  reg_c_0.x = __hadd(reg_a_0.x, reg_b_0.x);
  reg_c_0.y = __hadd(reg_a_0.y, reg_b_0.y);

  reg_c_1.x = __hadd(reg_a_1.x, reg_b_1.x);
  reg_c_1.y = __hadd(reg_a_1.y, reg_b_1.y);

  reg_c_2.x = __hadd(reg_a_2.x, reg_b_2.x);
  reg_c_2.y = __hadd(reg_a_2.y, reg_b_2.y);

  reg_c_3.x = __hadd(reg_a_3.x, reg_b_3.x);
  reg_c_3.y = __hadd(reg_a_3.y, reg_b_3.y);

  if ((idx + 0) < N)
    HALF2(c[idx + 0]) = reg_c_0;
  if ((idx + 2) < N)
    HALF2(c[idx + 2]) = reg_c_1;
  if ((idx + 4) < N)
    HALF2(c[idx + 4]) = reg_c_2;
  if ((idx + 6) < N)
    HALF2(c[idx + 6]) = reg_c_3;
}

// Element Wise Add Operation.
// gridDim(N/256), blockDim(256/8)
// a: Nx1, b: Nx1, c: Nx1, c = add(a,b)
// Unlike the previous one, this kernel
// uses #pargma unroll which unrolls
// the loop in an automatic way. Furthermore,
// We leverage Load Store 128Bit in a single
// instruction.
__global__ void element_wise_add_f16x8_packed_kernel(half *a, half *b, half *c,
                                             int N) {
  int idx = 8 * (threadIdx.x + blockDim.x * blockIdx.x);
  half2 pack_a[8], pack_b[8], pack_c[8];

  LDST128BITS(pack_a[0]) = LDST128BITS(a[idx]);
  LDST128BITS(pack_b[0]) = LDST128BITS(b[idx]);

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    HALF2(pack_c[i]) = __hadd2(HALF2(a[i]), HALF2(b[i]));
  }

  if ((idx + 7) < N) {
    LDST128BITS(c[idx]) = LDST128BITS(pack_c[0]);
  }
}

// -> Bindings <- //

#define STRINGIFY(str) #str

#define PYTORCH_MODULE(func) \
  m.def(STRINGIFY(func), &func, STRINGIFY(func));

#define CHECK_TENSOR(T, th_dtype)                                              \
  if(((T).options().dtype() != (th_dtype))) {                                   \
    std::cout << "Tensor type: " << (T).options().dtype() << std::endl;        \
    throw runtime_error("Must be: "  #th_dtype);                           \
  }

// Here, n_element  should not be confused with N.
// N is the total number of elements in the tensor.
// On the other hand, n_element is the number of
// elements that each thread is responsible of computing.
#define BIND_ELM_ADD(packed_type, th_dtype, element_type, n_element)            \
  void element_wise_add_##packed_type(torch::Tensor a, torch::Tensor b,        \
                                      torch::Tensor c) {                       \
    CHECK_TENSOR(a, (th_dtype));                                               \
    CHECK_TENSOR(b, (th_dtype));                                               \
    CHECK_TENSOR(c, (th_dtype));                                               \
    int ndim = a.dim();                                                        \
    if (ndim != 2) {                                                           \
      int N = 1;                                                               \
      for (int i = 0; i < ndim; i++) {                                         \
        N *= a.size(i);                                                        \
      }                                                                        \
      dim3 blockDim(256 / n_element);                                          \
      dim3 gridDim((256 - 1 + N) / 256);                                       \
      element_wise_add_##packed_type##_kernel<<<gridDim, blockDim>>>(                   \
          reinterpret_cast<element_type *>(a.data_ptr()),                      \
          reinterpret_cast<element_type *>(b.data_ptr()),                      \
          reinterpret_cast<element_type *>(c.data_ptr()), N);                  \
    } else {                                                                   \
      int sample_n = a.size(0);                                                \
      int features_n = a.size(1);                                              \
      int N = sample_n * features_n;                                           \
      if ((features_n / n_element) <= 1024) {                                  \
        dim3 blockDim(features_n / n_element);                                      \
        dim3 gridDim(sample_n);                                                     \
        element_wise_add_##packed_type##_kernel<<<gridDim, blockDim>>>(                 \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      } else {                                                                 \
        dim3 blockDim(256 / n_element);                                             \
        dim3 gridDim((256 + N - 1) / 256);                                          \
        element_wise_add_##packed_type##_kernel<<<gridDim, blockDim>>>(                 \
            reinterpret_cast<element_type *>(a.data_ptr()),                    \
            reinterpret_cast<element_type *>(b.data_ptr()),                    \
            reinterpret_cast<element_type *>(c.data_ptr()), N);                \
      }                                                                        \
    }                                                                          \
  }

BIND_ELM_ADD(f32,          torch::kFloat,   float, 1);
BIND_ELM_ADD(f32x4,        torch::kFloat,   float, 4);
BIND_ELM_ADD(f16,          torch::kHalf,    half,  1);
BIND_ELM_ADD(f16x2,        torch::kHalf,    half,  2);
BIND_ELM_ADD(f16x8,        torch::kHalf,    half,  8);
BIND_ELM_ADD(f16x8_packed, torch::kHalf,    half,  8);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  PYTORCH_MODULE(element_wise_add_f32)
  PYTORCH_MODULE(element_wise_add_f32x4)
  PYTORCH_MODULE(element_wise_add_f16)
  PYTORCH_MODULE(element_wise_add_f16x2)
  PYTORCH_MODULE(element_wise_add_f16x8)
  PYTORCH_MODULE(element_wise_add_f16x8_packed)
}
