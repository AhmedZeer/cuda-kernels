#include "../../utils/util.cuh"
#include "../headers/common.cuh"
#include <stdio.h>
#define BLOCK_SIZE 256

// Kernel declaration from naive.cu
extern __global__ void naiveCoalescingGEMM(float *A, float *B, float *C, uint m,
                                           uint n, uint k, float alpha,
                                           float beta);

void runNaiveCoalescingGEMM(uint m, uint n, uint k) {
  // Host matrices
  float *h_A, *h_B, *h_C, *h_C_ref;
  size_t size_A = m * k * sizeof(float);
  size_t size_B = k * n * sizeof(float);
  size_t size_C = m * n * sizeof(float);
  float alpha = 1, beta = 0;

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
  dim3 gridDim((n + blockDim - 1) / blockDim, (m + blockDim - 1) / blockDim);

  // Warmup loop
  for (int i = 0; i < 10; ++i) {
    naiveCoalescingGEMM<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k, alpha,
                                               beta);
  }
  cudaDeviceSynchronize(); // Ensure all operations are finished

  // Benchmark loop
  int numRuns = 100;
  float totalMilliseconds = 0;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < numRuns; ++i) {
    cudaEventRecord(start);
    naiveCoalescingGEMM<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k, alpha,
                                               beta);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    totalMilliseconds += milliseconds;
  }

  // Compute average execution time
  float averageMilliseconds = totalMilliseconds / numRuns;

  // Copy result back to host
  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  // Validate the result
  bool isValid = validateMatrices(h_C, h_C_ref, m, n, 1e-4);
  printf("Validation: %s\n", isValid ? "SUCCESS" : "FAILURE");

  // Print performance metrics
  float seconds = averageMilliseconds / 1000.0; // Convert to seconds
  float flop = 2.0 * m * n * k;           // FLOP for matrix multiplication
  float tflops = flop / (seconds * 1e12); // TFLOPS
  float bandwidth = (size_A + size_B + size_C) / 1e9 / seconds; // GB/s

  printf("Kernel average execution time (ms): %f \n", averageMilliseconds);
  printf("Effective Bandwidth (GB/s): %f \n", bandwidth);
  printf("Performance (TFLOPS): %f \n", tflops);

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
