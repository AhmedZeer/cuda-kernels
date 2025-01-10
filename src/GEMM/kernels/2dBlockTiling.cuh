template<int BM, int BN, int BK, int TM, int TN>
__global__ void blockTiling2d(float* A, float *B, float *C, int m, int n, int k, float alpha, float beta){

  int totalTiles = BM * BN;
  int threadsPerTile = totalTiles / (TM*TN);

  int cCol = blockIdx.x;
  int cRow = blockIdx.y;

  int threadCol = threadIdx.x % (BN/TN);
  int threadRow = threadIdx.x / (BN/TN);

  int innerColA = threadIdx.x % BK;
  int innerRowA = threadIdx.x / BK;
  
  int innerColB = threadIdx.x % BN;
  int innerRowB = threadIdx.x / BN;

  int strideA = threadsPerTile / BK;
  int strideB = threadsPerTile / BN;

  A += cRow * BM * k;
  B += cCol * BN;
  C += cCol * BN + cRow * BM * n;

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  
  float threadResults[TM][TN];
  float regM[TM];
  float regN[TN];

  for(int blkIdx = 0; blkIdx < k; blkIdx+=BK){
    
    for(int i = 0; i < BM; i+=strideA){
      As[(innerRowA + i)][innerColA] = A[(innerRowA + i) * k + innerColA];
    }   

    for(int i = 0; i < BN; i+=strideB){
      Bs[(innerRowB + i)][innerColB] = B[(innerRowB + i) * n + innerColB];
    }   

    __syncthreads();
    
    A += BK;
    B += BK * n;
    
    for(int i = 0; i < BK; i++){
      
      for(int regIdx = 0; regIdx < TM; regIdx++){
        regM[regIdx] = As[threadRow * TM + regIdx][i];
      }

      for(int regIdx = 0; regIdx < TN; regIdx++){
        regN[regIdx] = Bs[i][threadCol * TN + regIdx];
      }

      for(int resIdxM = 0; resIdxM < TM ; resIdxM++){
        for(int resIdxN = 0; resIdxN < TN; resIdxN++){
          threadResults[resIdxM][resIdxN] += regM[resIdxM] * regN[resIdxN];
        }
      }
      __syncthreads();
    }
  }
  
  for(int resIdxM = 0; resIdxM < TM ; resIdxM++){
    for(int resIdxN = 0; resIdxN < TN; resIdxN++){
      C[(threadRow * TM + resIdxM) * n +threadCol * TN + resIdxN] = threadResults[resIdxM][resIdxN];
    }
  }
}
