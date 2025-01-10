#include "../../utils/util.cuh"
#include "../headers/common.cuh"
#include "../kernels/SMEMCaching.cuh"
#include <stdio.h>

// Define 'uint' if not defined in common.cuh
#ifndef UINT_DEFINED
typedef unsigned int uint;
#define UINT_DEFINED
#endif

void runSMEMCaching(float *h_A, float *h_B, float *h_C_ref, uint m, uint n,
                    uint k) {
  // Host matrices
  float *h_C;
  float alpha = 1.0f, beta = 0.0f;
  const uint BLOCK_SIZE = 32; // Adjusted to 16 for better occupancy

  size_t size_A = m * k * sizeof(float);
  size_t size_B = k * n * sizeof(float);
  size_t size_C = m * n * sizeof(float);

  h_C = (float *)malloc(size_C);

  // Device matrices
  float *d_A, *d_B, *d_C;
  cudaMalloc((void **)&d_A, size_A);
  cudaMalloc((void **)&d_B, size_B);
  cudaMalloc((void **)&d_C, size_C);

  // Copy data to device
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0, size_C); // Initialize device C to zero

  // Define grid and block dimensions
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads per block
  dim3 gridDim((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Warmup loop
  for (int i = 0; i < 2; ++i) {
    SMEMCaching<BLOCK_SIZE>
        <<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k, alpha, beta);
  }
  cudaDeviceSynchronize(); // Ensure all operations are finished

  // Benchmark loop
  int numRuns = 3;
  float totalMilliseconds = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < numRuns; ++i) {
    cudaEventRecord(start);
    SMEMCaching<BLOCK_SIZE>
        <<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k, alpha, beta);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    totalMilliseconds += milliseconds;
  }

  // Compute average execution time
  float averageMilliseconds = totalMilliseconds / numRuns;

  // Copy result back to host
  cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

  // Validate the result
  bool isValid = validateMatrices(h_C, h_C_ref, m, n, 1e-4f);
  printf("Validation: %s\n", isValid ? "SUCCESS" : "FAILURE");

  float maxDiff = maxDifferenceBetweenMatrices(h_C, h_C_ref, m, n);
  printf("Max Diff: %f\n", maxDiff);

  // Print performance metrics
  float seconds = averageMilliseconds / 1000.0f; // Convert to seconds
  float flop = 2.0f * m * n * k;           // FLOP for matrix multiplication
  float tflops = flop / (seconds * 1e12f); // TFLOPS
  float bandwidth = (size_A + size_B + size_C) / 1e9f / seconds; // GB/s

  printf("Kernel average execution time (ms): %f\n", averageMilliseconds);
  printf("Effective Bandwidth (GB/s): %f\n", bandwidth);
  printf("Performance (TFLOPS): %f\n", tflops);

  cudaDeviceReset();
  // Clean up
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}
