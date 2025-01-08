#include <cuda_runtime.h>
__global__ void naiveGEMM(float *A, float *B, float *C, uint m, uint n, uint k,
                          float alpha, float beta);
