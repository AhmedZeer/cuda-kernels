template <const uint BM, const uint BN, const uint BK, const uint TM>
__global__ void blockTiling1d(float *A, float *B, float *C, int m, int n, int k,
                              float alpha, float beta) {

  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  const int threadRow = threadIdx.x / BN;
  const int threadCol = threadIdx.x % BN;

  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;

  A += cRow * BM * k;
  B += cCol * BN;
  C += cCol * BN + cRow * BM * n;

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  float threadResults[TM] = {0.0f};

  for (int blkIdx = 0; blkIdx < k; blkIdx++) {
    As[innerRowA][innerColA] = A[innerRowA * k + innerColA];
    Bs[threadRow][threadCol] = B[threadRow * n + threadCol];

    __syncthreads();

    A += BK;
    B += BN * n;

    for (int i = 0; i < BK; i++) {
      float tmpB = Bs[i][threadCol];
      for (int resIdx = 0; resIdx < TM; resIdx++) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx)][i] * tmpB;
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < TM; i++) {
    C[(threadRow * TM + i) * n + threadCol] = threadResults[i];
  }
}
