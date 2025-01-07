#include "../../utils/util.h"
#include "../headers/common.cuh"
#include <cuda_runtime.h>
#include <stdio.h>
#define BLOCK_SIZE 256

// Kernel declaration from naive.cu
extern __global__ void naiveGEMM(float *A, float *B, float *C, int m, int n,
                                 int k);

void runNaiveGEMM(int m, int n, int k) {
  // Host matrices
  float *h_A, *h_B, *h_C, *h_C_ref;
  size_t size_A = m * k * sizeof(float);
  size_t size_B = k * n * sizeof(float);
  size_t size_C = m * n * sizeof(float);

  // Allocate host memory
  h_A = (float *)malloc(size_A);
  h_B = (float *)malloc(size_B);
  h_C = (float *)malloc(size_C);
  h_C_ref = (float *)malloc(size_C);

  // Initialize matrices
  initRandMatrix(h_A, m, k);
  initRandMatrix(h_B, k, n);

  // Perform CPU matrix multiplication for reference
  cpuMatmul(h_A, h_B, h_C_ref, m, n, k);

  // Device matrices
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, size_A);
  cudaMalloc((void **)&d_B, size_B);
  cudaMalloc((void **)&d_C, size_C);

  // Copy data to device
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

  // Define grid and block dimensions
  int blockDim(BLOCK_SIZE); // 16x16 threads per block
  dim3 gridDim((n + blockDim - 1) / blockDim,
              (m + blockDim - 1) / blockDim);

  // Benchmark the kernel
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  naiveGEMM<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);
  cudaEventRecord(stop);

  // Wait for the kernel to finish and measure time
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Copy result back to host
  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  // Validate the result
  bool isValid = validateMatrices(h_C, h_C_ref, m, n, 1e-4);
  printf("Validation: %s\n", isValid ? "SUCCESS" : "FAILURE");

  // Print performance metrics
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Clean up
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
