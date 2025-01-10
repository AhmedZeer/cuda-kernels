template<int BM, int BN, int BK, int TM, int TN>
__global__ void blockTiling2d(float* A, float *B, float *C, int m, int n, int k, float alpha, float beta){

    // Calculate total tiles and threads per tile
    int totalTiles = BM * BN;
    int threadsPerTile = totalTiles / (TM * TN);

    // Determine the block's position in the grid
    int cCol = blockIdx.x;
    int cRow = blockIdx.y;

    // Map thread index to thread row and column within the tile
    int threadCol = threadIdx.x % (BN / TN); // 0..7 for BN=64, TN=8
    int threadRow = threadIdx.x / (BN / TN); // 0..7 for blockDim.x=64

    // Calculate indices for loading A and B
    int innerColA = threadIdx.x % BK; // 0..7 for BK=8
    int innerRowA = threadIdx.x / BK; // 0..7 for blockDim.x=64, BK=8

    // For B, since Bs has BK=8 rows, we adjust the loop to prevent out-of-bounds
    // No need for innerRowB as it would always be 0 with blockDim.x=64 and BN=64
    // Thus, innerRowB is effectively 0
    // Remove innerRowB and adjust the loading loop accordingly

    int strideA = threadsPerTile / BK; // 64 / 8 = 8
    int strideB = threadsPerTile / BK; // Adjusted to 8 for correct loading

    // Offset the pointers to the starting positions for this block
    A += cRow * BM * k;
    B += cCol * BN;
    C += cCol * BN + cRow * BM * n;

    // Declare shared memory for tiles of A and B
    __shared__ float As[BM][BK]; // [64][8]
    __shared__ float Bs[BK][BN]; // [8][64]

    // Declare arrays for thread-local computations
    float threadResults[TM][TN];
    float regM[TM];
    float regN[TN];

    // Initialize threadResults to zero
    for(int resIdxM = 0; resIdxM < TM; resIdxM++){
        for(int resIdxN = 0; resIdxN < TN; resIdxN++){
            threadResults[resIdxM][resIdxN] = 0.0f;
        }
    }

    // Iterate over all tiles in the K dimension
    for(int blkIdx = 0; blkIdx < k; blkIdx += BK){

        // Load a tile of A into shared memory
        for(int i = 0; i < BM; i += strideA){
            int rowA = innerRowA + i;
            if(rowA < BM && (blkIdx + innerColA) < k){
                As[rowA][innerColA] = A[rowA * k + blkIdx + innerColA];
            }
            else{
                As[rowA][innerColA] = 0.0f; // Handle out-of-bounds by setting to zero
            }
        }

        // Load a tile of B into shared memory
        for(int i = 0; i < BK; i += strideB){
            if(i < BK && (blkIdx + i) < k){
                Bs[i][threadCol * TN + 0] = B[(blkIdx + i) * n + threadCol * TN + 0];
                // If TN > 1, you may need to load multiple elements per thread
                // For simplicity, assuming TN=8 and each thread loads TN elements
                for(int tn = 1; tn < TN; tn++){
                    if((threadCol * TN + tn) < BN){
                        Bs[i][threadCol * TN + tn] = B[(blkIdx + i) * n + threadCol * TN + tn];
                    }
                    else{
                        Bs[i][threadCol * TN + tn] = 0.0f; // Handle out-of-bounds
                    }
                }
            }
            else{
                // Handle out-of-bounds by setting to zero
                Bs[i][threadCol * TN + 0] = 0.0f;
                for(int tn = 1; tn < TN; tn++){
                    Bs[i][threadCol * TN + tn] = 0.0f;
                }
            }
        }

        __syncthreads(); // Ensure all data is loaded into shared memory

        // Move pointers to the next tile in K dimension
        A += BK;
        B += BK * n;

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

    // Write the accumulated results back to matrix C with boundary checks
    for(int resIdxM = 0; resIdxM < TM ; resIdxM++){
        for(int resIdxN = 0; resIdxN < TN; resIdxN++){
            int globalRow = cRow * BM + threadRow * TM + resIdxM;
            int globalCol = cCol * BN + threadCol * TN + resIdxN;

            // Boundary check to prevent out-of-bounds access
            if(globalRow < m && globalCol < n){
                // Incorporate alpha and beta
                C[globalRow * n + globalCol] = alpha * threadResults[resIdxM][resIdxN] + beta * C[globalRow * n + globalCol];
            }
        }
    }
}

