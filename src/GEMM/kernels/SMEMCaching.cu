__global__ void SMEMCaching(float* A, float *B, float *C, uint m, uint n, uint k, float alpha, float beta, uint BLOCKSIZE){

  uint cRow = blockIdx.y;
  uint cCol = blockIdx.x;

  uint threadRow = blockDim.x / BLOCKSIZE;
  uint threadCol = blockDim.x % BLOCKSIZE;
  
  if(cRow < m && cCol < n){
    A += cRow * BLOCKSIZE * k;
    B += cCol * BLOCKSIZE;
    C += cCol * BLOCKSIZE + cRow * BLOCKSIZE * n;
    
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
    float sum = 0.0f;
    for(int blkIdx = 0; blkIdx < k; blkIdx += BLOCKSIZE){
      
      // Populating Shared Memory.
      for(int i = 0; i < BLOCKSIZE; i++){
        for(int j = 0; j < BLOCKSIZE; j++){
          As[i * BLOCKSIZE + j] = A[i * k + j];
          Bs[i * BLOCKSIZE + j] = B[i * n + j];
        }
      }
      __syncthreads();
      
      // Shift the tile.
      A += BLOCKSIZE;
      B += BLOCKSIZE * n;
      
      for(int i = 0; i < BLOCKSIZE ; i++){
        sum += As[threadRow * BLOCKSIZE + i] * Bs[i * BLOCKSIZE + threadCol];
      }
      __syncthreads();
    }
  } 
  C[threadRow * n + threadCol] = alpha * sum + beta * C[threadRow * n + threadCol];
}
