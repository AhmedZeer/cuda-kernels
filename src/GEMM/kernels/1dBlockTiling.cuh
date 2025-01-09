template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void blockTiling1d(float *A, float *B, float *C, int m, int n, int k,
                              float alpha, float beta) {

  // Calculate block row and column
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  // Calculate thread row and column within the block
  const int threadRow = threadIdx.x / BN;
  const int threadCol = threadIdx.x % BN;

  // Calculate inner row and column for loading A
  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;

  // Compute starting pointers with bounds checking
  // Ensure cRow and cCol are within the valid range
  if (cRow * BM >= m || cCol * BN >= n) {
    // Out of bounds, exit the kernel
    return;
  }

  // Adjust A, B, C pointers based on block position
  A += cRow * BM * k;
  B += cCol * BN;
  C += cCol * BN + cRow * BM * n;

  // Declare shared memory with size checks
  // Ensure BM, BK, BN do not exceed shared memory limits
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  // Initialize thread-local results
  float threadResults[TM] = {0.0f};

  // Loop over all blocks in the K dimension
  for (int blkIdx = 0; blkIdx < k; blkIdx += BK) {

    // Load a tile of A into shared memory with bounds checking
    if (innerRowA < BM && (blkIdx + innerColA) < k && (cRow * BM + innerRowA) < m) {
      As[innerRowA][innerColA] = A[innerRowA * k + blkIdx + innerColA];
    } else {
      As[innerRowA][innerColA] = 0.0f; // Padding with zero if out of bounds
    }

    // Load a tile of B into shared memory with bounds checking
    if (threadRow < BK && threadCol < BN && (blkIdx + threadRow) < k && (cCol * BN + threadCol) < n) {
      Bs[threadRow][threadCol] = B[(blkIdx + threadRow) * n + threadCol];
    } else {
      Bs[threadRow][threadCol] = 0.0f; // Padding with zero if out of bounds
    }

    // Synchronize to ensure all data is loaded
    __syncthreads();

    // Perform the matrix multiplication for the current tile
    for (int i = 0; i < BK; i++) {
      float tmpB = Bs[i][threadCol];
      for (int resIdx = 0; resIdx < TM; resIdx++) {
        // Ensure the index for As does not exceed BM
        int aRow = threadRow * TM + resIdx;
        if (aRow < BM) {
          threadResults[resIdx] += As[aRow][i] * tmpB;
        }
      }
    }

    // Synchronize before loading the next tile
    __syncthreads();
  }

  // Write the computed results back to C with bounds checking
  for (int i = 0; i < TM; i++) {
    int cRowIdx = threadRow * TM + i;
    if (cRowIdx < BM && (cRow * BM + cRowIdx) < m && (cCol * BN + threadCol) < n) {
      C[cRowIdx * n + threadCol] = alpha * threadResults[i] + beta * C[cRowIdx * n + threadCol];
    }
  }
}

