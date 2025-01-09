template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void blockTiling1d(float *A, float *B, float *C, int m, int n,
                              int k) {

  int cRow = blockIdx.y;
  int cCol = blockIdx.x;

  int threadRow = threadIdx.x / BN;
  int threadCol = threadIdx.x % BN;

  int innerRowA = threadIdx.x / BK;
  int innerColA = threadIdx.x % BK;

  A += cRow * BM * k;
  B += cCol * BN;
  C += cCol * BN + cRow * BM * n;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float threadResults[TM] = {0.0f};

  for (int blkIdx = 0; blkIdx < k; blkIdx++) {
    As[innerRowA * BK + innerColA] = A[innerRowA * k + innerColA];
    Bs[threadRow * BN + threadCol] = B[threadRow * n + threadCol];

    __syncthreads();

    As += BK;
    Bs += BN * n;

    for (int i = 0; i < BK; i++) {
      for (int resIdx = 0; resIdx < TM; resIdx++) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + i] * Bs[i * BN + threadCol];
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; i++) {
    C[(threadRow * TM + i) * n + threadCol] = threadResults[i];
  }
}
