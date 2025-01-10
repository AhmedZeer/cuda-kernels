#include "utils/util.cuh"
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

// Declare function prototypes for each kernel runner
void runNaiveGEMM(float *A, float *B, float *C, uint m, uint n, uint k);
void runNaiveCoalescingGEMM(float *A, float *B, float *C, uint m, uint n,
                            uint k);
void runSMEMCaching(float *A, float *B, float *C, uint m, uint n, uint k);
void runblockTiling1d(float *A, float *B, float *C, uint m, uint n, uint k);
void runblockTiling2d(float *A, float *B, float *C, uint m, uint n, uint k);

int main() {
  printf("=== GEMM Benchmark ===\n");

  uint m = 1024 * 1; // Number of rows in A and C
  uint n = 1024 * 1; // Number of columns in B and C
  uint k = 1024 * 1; // Number of columns in A and rows in B
  float *A = (float *)malloc(sizeof(float) * m * k);
  float *B = (float *)malloc(sizeof(float) * k * n);
  float *C = (float *)malloc(sizeof(float) * m * n);

  // Initialize matrices
  initRandMatrix(A, m, k);
  initRandMatrix(B, k, n);
  initRandMatrix(C, m, n);

  // cuBLAS setup
  cublasHandle_t handle;
  cublasCreate(&handle);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, sizeof(float) * m * k);
  cudaMalloc((void **)&d_B, sizeof(float) * k * n);
  cudaMalloc((void **)&d_C, sizeof(float) * m * n);

  // Copy data from host to device
  cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, sizeof(float) * k * n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, sizeof(float) * m * n, cudaMemcpyHostToDevice);

  // Perform cuBLAS matrix multiplication: C = alpha * A * B + beta * C
  float alpha = 1.0f;
  float beta = 0.0f;

  // cuBLAS SGEMM: CUBLAS_OP_N (no transpose)
  cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                                      &alpha, d_B, n, d_A, k, &beta, d_C, n);

  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("cuBLAS SGEMM failed!\n");
    return -1;
  }

  // Copy result from device to host
  cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

  printf("Benchmarking Naive GEMM...\n");
  runNaiveGEMM(A, B, C, m, n, k);

  printf("\nBenchmarking Naive Coalescing GEMM...\n");
  runNaiveCoalescingGEMM(A, B, C, m, n, k);

  printf("\nBenchmarking SMEM Caching GEMM...\n");
  runSMEMCaching(A, B, C, m, n, k);

  printf("\nBenchmarking 1D Block Tiling Caching GEMM...\n");
  runblockTiling1d(A, B, C, m, n, k);

  printf("\nBenchmarking 2D Block Tiling Caching GEMM...\n");
  runblockTiling2d(A, B, C, m, n, k);

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cublasDestroy(handle);
  free(A);
  free(B);
  free(C);

  return 0;
}
