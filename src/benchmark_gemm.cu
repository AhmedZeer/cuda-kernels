#include "GEMM/headers/common.cuh"
#include <stdio.h>

// Declare function prototypes for each kernel runner
void runNaiveGEMM(int m, int n, int k);

int main() {
  int m = 512; // Number of rows in A and C
  int n = 512; // Number of columns in B and C
  int k = 512; // Number of columns in A and rows in B

  printf("=== GEMM Benchmark ===\n");

  // Run the naive kernel benchmark
  printf("Benchmarking Naive GEMM...\n");
  runNaiveGEMM(m, n, k);

  // Add calls to other kernel runners as needed
  // e.g., runOptimizedMatmulBenchmark(m, n, k);

  return 0;
}
