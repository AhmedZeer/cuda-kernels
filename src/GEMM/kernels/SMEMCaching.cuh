#include "../headers/common.cuh" // Ensure 'uint' is defined as 'unsigned int'

template <const uint BLOCKSIZE>
__global__ void SMEMCaching(float *A, float *B, float *C, uint m, uint n,
                            uint k, float alpha, float beta) {

  uint cRow = blockIdx.y * BLOCKSIZE + threadIdx.y;
  uint cCol = blockIdx.x * BLOCKSIZE + threadIdx.x;

  float sum = 0.0f;

  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  for (int blkIdx = 0; blkIdx < (k + BLOCKSIZE - 1) / BLOCKSIZE; blkIdx++) {

      As[threadIdx.y][threadIdx.x] =
          A[cRow * k + blkIdx * BLOCKSIZE + threadIdx.x];
      Bs[threadIdx.y][threadIdx.x] =
          B[(blkIdx * BLOCKSIZE + threadIdx.y) * n + cCol];

    __syncthreads();

    // Perform the multiplication for this tile
    for (int i = 0; i < BLOCKSIZE; i++) {
      sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
    }

    __syncthreads();
  }

  // Write the result to C with boundary checks
  if (cRow < m && cCol < n) {
    C[cRow * n + cCol] = alpha * sum + beta * C[cRow * n + cCol];
  }
}
