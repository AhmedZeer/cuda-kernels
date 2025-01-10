#include "../../utils/util.cuh"
#include "../headers/common.cuh"
#include "../kernels/2dBlockTiling.cuh"
#include <stdio.h>

// Define 'uint' if not defined in common.cuh
#ifndef UINT_DEFINED
typedef unsigned int uint;
#define UINT_DEFINED
#endif

void runblockTiling2d(float *h_A, float *h_B, float *h_C_ref, uint m, uint n,
                      uint k) {
  // Host matrices
  float *h_C;
  float alpha = 1.0f, beta = 0.0f;

  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;

  // const uint BLOCK_SIZE = 32;
  const uint BLOCK_SIZE = (BM * BN) / (TM * TN);

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
  dim3 blockDim(BLOCK_SIZE); 
  dim3 gridDim((n + BN - 1) / BN, (m + BM - 1) / BM);

  // Warmup loop
  for (int i = 0; i < 10; ++i) {
    blockTiling2d<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k, alpha, beta);
  }
  cudaDeviceSynchronize(); // Ensure all operations are finished
  printf("\nWarmup. Done.\n");

  // Benchmark loop
  int numRuns = 10;
  float totalMilliseconds = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < numRuns; ++i) {
    cudaEventRecord(start);

    printf("\nBefire kernel Runs (%d)\n", i);

    blockTiling2d<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k, alpha, beta);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    printf("\nAfter kernel Runs (%d)\n", i);

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

  float minDiff = minDifferenceBetweenMatrices(h_C, h_C_ref, m, n);
  printf("Min Diff: %f\n", minDiff);

  // Print performance metrics
  float seconds = averageMilliseconds / 1000.0f; // Convert to seconds
  float flop = 2.0f * m * n * k;           // FLOP for matrix multiplication
  float tflops = flop / (seconds * 1e12f); // TFLOPS
  float bandwidth = (size_A + size_B + size_C) / 1e9f / seconds; // GB/s

  printf("Kernel average execution time (ms): %f\n", averageMilliseconds);
  printf("Effective Bandwidth (GB/s): %f\n", bandwidth);
  printf("Performance (TFLOPS): %f\n", tflops);

  // Clean up
  free(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}