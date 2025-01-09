#include "util.cuh"
#include <cstdlib>

srand(0);

void initRandMatrix(float *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      A[i * n + j] = (float)rand() / RAND_MAX;
    }
  }
}

void printMatrix(float *A, int m, int n) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      printf("%0.2f ", A[i * n + j]); // Print each value with 2 decimal places
    }
    printf("\n"); // Newline after each row
  }
}

void cpuMatmul(float *A, float *B, float *C, int m, int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int l = 0; l < k; l++) {
        sum += A[i * k + l] * B[l * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

bool validateMatrices(float *A, float *B, int m, int n, float epsilon) {
  int i = 0, j = 0;
  bool eq = true;

  while (i < m && eq) {
    j = 0;
    while (j < n && eq) {
      if (A[i * n + j] - B[i * n + j] > epsilon)
        eq = false;
      j++;
    }
    i++;
  }
  return eq;
}
