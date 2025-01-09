#include "../headers/common.cuh" // Ensure 'uint' is defined as 'unsigned int'
template <const uint BLOCKSIZE>
__global__ void naiveCoalescingGEMM(float *A, float *B, float *C, uint m,
                                    uint n, uint k, float alpha, float beta) {
  // C = alpha * A @ beta - beta * C
  int threadRow = blockIdx.y * blockDim.x + (threadIdx.x / blockDim.x);
  int threadCol = blockIdx.x * blockDim.x + (threadIdx.x % blockDim.x);

  if (threadRow < m && threadCol < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
      sum += A[threadRow * k + i] * B[i * n + threadCol];
    }
    C[threadRow * n + threadCol] = sum;
    /*
    C[threadRow * n + threadCol] =
        alpha * sum + beta * C[threadRow * n + threadCol];
    */
  }
}
