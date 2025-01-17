#include<iostream>
#include<cuda_runtime.h>

// Binding Macroes.
#define STRINGIFY(str) #str
#define PYTORCH_MODULE(func) m.def(STRINGIFY(func), &func, STRINGIFY(func));

// Casting Macroes.
#define INT4(a) (reinterpert_cast<int4 *>(&(a))[0]);

// The items in array 'a' represents
// indices in 'b' array. We accumulate
// how many times an index in 'b' occurs
// in 'a' array, which is basically the definition
// of a histogram.

// Each thread is responsible of 1 element in 'a', and
// it will increment 'b' by +1. However, a bunch of threads
// could try to incerement 'b' at the same time, because
// different thread ids could correspond to the same index 
// in 'b' but NOT in 'a'.

// Histogram.
// 32-Bits
// blockDim(256), gridDim((N+256-1)/256)
// a: Nx1, b: histogramSize:1 
__global__ void histogram_i32_kernel(float *a, float *b, int N){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if(idx < N){
    atmoicAdd(&b[a[idx]], 1);
  }
}

// Histogram.
// 32-Bits * 4 -> 128-Bits
// blockDim(256/4), gridDim((N+64-1)/64)
// a: Nx1, b: histogramSize:1 

// When we cast an arbitrary pointer at idx(i) to INT4,
// we get a[i:4] elements which we can directly use to
// construct 'int4' type element.
__global__ void histogram_i32x4_kernel(float *a, float *b, int N){
  int idx = 4 * (blockDim.x * blockIdx.x + threadIdx.x);

  if(idx < N){
    int4 int4_a = INT4(a[idx]);
    atmoicAdd(&b[a[int4_a.x]], 1);
    atmoicAdd(&b[a[int4_a.y]], 1);
    atmoicAdd(&b[a[int4_a.z]], 1);
    atmoicAdd(&b[a[int4_a.w]], 1);
  }
}

#define LAUNCHER(kernel_name, elm_per_thread, cast_type, tensor_type) \
  void histogram_##kernel_name##_kernel(torch::Tensor a, torch::Tensor b){ \
    int BLOCKSIZE=256/elm_per_thread; \
    int N = 1; \
    for(int i = 0; i < a.dim(); i++){ \
      N *= a.size(i); \
    } \
    dim3 blockDim(BLOCKSIZE); \
    dim3 gridDim((BLOCKSIZE + N - 1) / BLOCKSIZE); \
    histogram_##kernel_name<<<gridDim, blockDim>>> \
                      (interpert_cast<cast_type*>(a.data_ptr()), \
                      interpert_cast<cast_type*>(b.data_ptr()), \
                      N); \
  }

#define STRING(val)  #str
#define BINDER(func) m.def(STRING(func), &func, STRING(func));

// Declare functions:
LAUNCHER(i32,   1, float, torch::kFloat)
LAUNCHER(i32x4, 4, float, torch::kFloat)

PYBIND11_MODULE(TORCH_EXTENTION_NAME, m){
  BINDER(historgram_i32)
  BINDER(histogram_i32x4)
}
