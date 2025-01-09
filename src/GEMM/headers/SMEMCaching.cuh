// SMEMCaching.cuh
#ifndef SMEMCACHING_CUH
#define SMEMCACHING_CUH

#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void SMEMCaching(float *A, float *B, float *C, unsigned int m,
                            unsigned int n, unsigned int k, float alpha,
                            float beta);

#include "../kernels/SMEMCaching.cu" // Include the implementation

#endif // SMEMCACHING_CUH
