#include "../headers/common.cuh" // Ensure 'uint' is defined as 'unsigned int'

template <const uint BLOCKSIZE>
__global__ void SMEMCaching(float *A, float *B, float *C, uint m, uint n,
                            uint k, float alpha, float beta) {

  uint cRow = blockIdx.y * BLOCKSIZE + threadIdx.y;
  uint cCol = blockIdx.x * BLOCKSIZE + threadIdx.x;

  uint threadRow = threadIdx.x;
  uint threadCol = threadIdx.y;

  float sum = 0.0f;

  __shared__ As[BLOCKSIZE * BLOCKSIZE];
  __shared__ Bs[BLOCKSIZE * BLOCKSIZE];

  for (int blkIdx = 0; blkIdx < (k + BLOCKSIZE - 1) / BLOCKSIZE; blkIdx++) {

    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * k + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * k + threadCol];

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
