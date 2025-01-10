// blockTiling2d.cuh

template<int BM, int BN, int BK, int TM, int TN>
__global__ void blockTiling2d(const float* __restrict__ A, const float* __restrict__ B, float* C, int m, int n, int k, float alpha, float beta){

    
    int tileSize = BM * BN;
    int threadPerTile = tileSize / (TM*TN);
    int strideA = threadPerTile / BK;
    int strideB = threadPerTile / BN;
 
    // Determine the block's position in the grid
    int cCol = blockIdx.x;
    int cRow = blockIdx.y;

    // Map thread index to thread row and column within the tile
    int threadCol = threadIdx.x % (BN / TN); // 0..7 for BN=64, TN=8
    int threadRow = threadIdx.x / (BN / TN); // 0..7 for blockDim.x=64

    // Calculate indices for loading A and B
    int innerColA = threadIdx.x % BK; // 0..7 for BK=8
    int innerRowA = threadIdx.x / BK; // 0..7 for blockDim.x=64, BK=8
    
    int innerColB = threadIdx.x % BN;
    int innerRowB = threadIdx.x / BN;

    // Offset the pointers to the starting positions for this block
    const float* A_block = A + cRow * BM * k;
    const float* B_block = B + cCol * BN;
    float* C_block = C + cRow * BM * n + cCol * BN;

    // Declare shared memory for tiles of A and B
    __shared__ float As[BM][BK]; // [64][8]
    __shared__ float Bs[BK][BN]; // [8][64]

    // Declare arrays for thread-local computations
    float threadResults[TM][TN];
    float regM[TM];
    float regN[TN];


    // Iterate over all tiles in the K dimension
    for(int blkIdx = 0; blkIdx < k; blkIdx += BK){

        for(int i = 0; i < BM; i += strideA){
            int rowA = innerRowA + i;
            As[rowA][innerColA] = A_block[rowA * k + blkIdx + innerColA];
        }

        for(int i = 0; i < BK; i += strideB){
            int rowB = innerRowB + i;
            Bs[rowB][innerColB] = B_block[(blkIdx + rowB) * n + innerColB];
        }

        __syncthreads(); // Ensure all data is loaded into shared memory

        // Perform the multiplication for the loaded tiles
        for(int i = 0; i < BK; i++){
            // Load registers from shared memory
            for(int regIdx = 0; regIdx < TM; regIdx++){
                regM[regIdx] = As[threadRow * TM + regIdx][i];
            }

            for(int regIdx = 0; regIdx < TN; regIdx++){
                regN[regIdx] = Bs[i][threadCol * TN + regIdx];
            }

            // Accumulate the results
            for(int resIdxM = 0; resIdxM < TM ; resIdxM++){
                for(int resIdxN = 0; resIdxN < TN; resIdxN++){
                    threadResults[resIdxM][resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }

        __syncthreads(); // Ensure all threads have finished using shared memory
    }

    for(int resIdxM = 0; resIdxM < TM ; resIdxM++){
        for(int resIdxN = 0; resIdxN < TN; resIdxN++){
            int localRow = threadRow * TM + resIdxM;
            int localCol = threadCol * TN + resIdxN;

            C_block[localRow * n + localCol] = alpha * threadResults[resIdxM][resIdxN] + beta * C_block[localRow * n + localCol];
        }
    }
}

