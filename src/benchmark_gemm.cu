#include "GEMM/headers/common.cuh"
#include <stdio.h>

// Declare function prototypes for each kernel runner
void runNaiveGEMM(uint m, uint n, uint k);
void runNaiveCoalescingGEMM(uint m, uint n, uint k);
void runSMEMCaching(uint m, uint n, uint k);

int main() {
  uint m = 512; // Number of rows in A and C
  uint n = 512; // Number of columns in B and C
  uint k = 512; // Number of columns in A and rows in B

  printf("=== GEMM Benchmark ===\n");

  /*
  */
  printf("Benchmarking Naive GEMM...\n");
  runNaiveGEMM(m, n, k);

  printf("\nBenchmarking Naive Coalescing GEMM...\n");
  runNaiveCoalescingGEMM(m, n, k);

  printf("\n Benchmarking SMEM Caching GEMM...\n");
  runSMEMCaching(m, n, k);

  return 0;
}
