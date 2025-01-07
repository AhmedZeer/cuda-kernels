#include "common.cuh"
__global__ void naiveGEMM(float *A, float *B, float *C, uint m, uint n, uint k,
                          float alpha, float beta) {
  int threadRow = blockDim.x * blockIdx.y + threadIdx.y;
  int threadCol = blockDim.x * blockIdx.x + threadIdx.x;

  if (threadRow < m && threadCol < n) {
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
      sum += alpha * A[threadRow * k + i] * B[i * n + threadCol] +
             beta * C[threadRow * n + threadCol];
    }
    C[threadRow * n + threadCol] = sum;
  }
}
