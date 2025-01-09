#include "../headers/common.cuh" // Adjust the path as needed
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void SMEMCaching(float *A, float *B, float *C, uint m, uint n,
                            uint k, float alpha, float beta) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint threadRow = threadIdx.x / BLOCKSIZE;
  const uint threadCol = threadIdx.x % BLOCKSIZE;

  float sum = 0.0f;
  A += cRow * BLOCKSIZE * k;
  B += cCol * BLOCKSIZE;
  C += cCol * BLOCKSIZE + cRow * BLOCKSIZE * n;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
  if(cRow < m && cCol < n){
    for (int blkIdx = 0; blkIdx < k; blkIdx += BLOCKSIZE) {

      // Populating Shared Memory.
      As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * k + threadCol];
      Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * n + threadCol];
      __syncthreads();

      // Shift the tile.
      A += BLOCKSIZE;
      B += BLOCKSIZE * n;

      for (int i = 0; i < BLOCKSIZE; i++) {
        sum += As[threadRow * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + threadCol];
      }
      __syncthreads();
    }

    C[threadRow * n + threadCol] =
    alpha * sum + beta * C[threadRow * n + threadCol];
  }
}
