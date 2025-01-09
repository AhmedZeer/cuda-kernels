#include "../headers/common.cuh" // Ensure this file defines 'uint' as 'unsigned int'
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void SMEMCaching(float *A, float *B, float *C, uint m, uint n,
                            uint k, float alpha, float beta) {

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint threadRow = threadIdx.x / BLOCKSIZE;
  const uint threadCol = threadIdx.x % BLOCKSIZE;

  // Calculate global row and column indices
  const uint globalRow = cRow * BLOCKSIZE + threadRow;
  const uint globalCol = cCol * BLOCKSIZE + threadCol;

  float sum = 0.0f;

  // Adjust A, B, and C pointers to the start of the current tile
  float *A_tile = A + cRow * BLOCKSIZE * k + threadRow * k;
  float *B_tile = B + cCol * BLOCKSIZE + threadCol;
  float *C_tile = C + globalRow * n + globalCol;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  if (cRow < (m + BLOCKSIZE - 1) / BLOCKSIZE &&
      cCol < (n + BLOCKSIZE - 1) / BLOCKSIZE) {
    for (int blkIdx = 0; blkIdx < k; blkIdx += BLOCKSIZE) {

      // Load A tile into shared memory with boundary check
      if (globalRow < m && (blkIdx + threadCol) < k) {
        As[threadRow * BLOCKSIZE + threadCol] =
            A_tile + blkIdx * k < A_tile + k * k ? A_tile[blkIdx + threadCol]
                                                 : 0.0f;
      } else {
        As[threadRow * BLOCKSIZE + threadCol] = 0.0f;
      }

      // Load B tile into shared memory with boundary check
      if ((blkIdx + threadRow) < k && globalCol < n) {
        Bs[threadRow * BLOCKSIZE + threadCol] = B_tile[blkIdx * n + threadRow];
      } else {
        Bs[threadRow * BLOCKSIZE + threadCol] = 0.0f;
      }

      __syncthreads();

      // Perform multiplication
      for (int i = 0; i < BLOCKSIZE; ++i) {
        sum += As[threadRow * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + threadCol];
      }

      __syncthreads();
    }

    // Write the result to C with boundary check
    if (globalRow < m && globalCol < n) {
      C_tile[0] = alpha * sum + beta * C_tile[0];
    }
  }
}
