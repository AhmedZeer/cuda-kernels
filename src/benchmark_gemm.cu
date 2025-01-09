#include "GEMM/headers/common.cuh"
#include "utils/util.cuh"
#include <cstdlib>
#include <stdio.h>

// Declare function prototypes for each kernel runner
void runNaiveGEMM(float *A, float *B, float *C, uint m, uint n, uint k);
void runNaiveCoalescingGEMM(float *A, float *B, float *C, uint m, uint n,
                            uint k);
void runSMEMCaching(float *A, float *B, float *C, uint m, uint n, uint k);

int main() {
  uint m = 512 * 4; // Number of rows in A and C
  uint n = 512 * 4; // Number of columns in B and C
  uint k = 512 * 4; // Number of columns in A and rows in B
  float *A = (float *)malloc(sizeof(float) * m * k);
  float *B = (float *)malloc(sizeof(float) * k * n);
  float *C = (float *)malloc(sizeof(float) * m * n);
  initRandMatrix(A, m, k);
  initRandMatrix(B, k, n);
  initRandMatrix(C, m, n);
  cpuMatmul(A, B, C, m, n, k);

  printf("=== GEMM Benchmark ===\n");
  /*
  printf("Benchmarking Naive GEMM...\n");
  runNaiveGEMM(A, B, C, m, n, k);

  printf("\nBenchmarking Naive Coalescing GEMM...\n");
  runNaiveCoalescingGEMM(A, B, C, m, n, k);
  */

  printf("\nBenchmarking SMEM Caching GEMM...\n");
  runSMEMCaching(A, B, C, m, n, k);

  return 0;
}
