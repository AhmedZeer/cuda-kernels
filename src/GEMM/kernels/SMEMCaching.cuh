#include "../headers/common.cuh" // Ensure 'uint' is defined as 'unsigned int'

template <const uint BLOCKSIZE>
__global__ void SMEMCaching(float *A, float *B, float *C, uint m, uint n,
                            uint k, float alpha, float beta) {

  // Calculate the row and column index of the C element to work on
  uint cRow = blockIdx.y * BLOCKSIZE + threadIdx.y;
  uint cCol = blockIdx.x * BLOCKSIZE + threadIdx.x;

  float sum = 0.0f;

  // Allocate shared memory for A and B tiles
  __shared__ float As[BLOCKSIZE][BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

  // Loop over all tiles in the K dimension
  for (int blkIdx = 0; blkIdx < (k + BLOCKSIZE - 1) / BLOCKSIZE; blkIdx++) {

    // Load data into shared memory with boundary checks
    if ((cRow < m) && (blkIdx * BLOCKSIZE + threadIdx.x) < k) {
      As[threadIdx.y][threadIdx.x] =
          A[cRow * k + blkIdx * BLOCKSIZE + threadIdx.x];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    if ((blkIdx * BLOCKSIZE + threadIdx.y) < k && (cCol < n)) {
      Bs[threadIdx.y][threadIdx.x] =
          B[(blkIdx * BLOCKSIZE + threadIdx.y) * n + cCol];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

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
