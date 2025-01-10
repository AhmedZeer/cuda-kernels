#include "util.cuh"
#include <cstdlib>
#include <math.h>
#include <iostream>

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

float maxDifferenceBetweenMatrices(float *A, float *B, int m, int n) {
  float maxDiff = 0.0f;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float diff = fabs(A[i * n + j] - B[i * n + j]);
      if (diff > maxDiff) {
        maxDiff = diff;
      }
    }
  }

  return maxDiff;
}

bool validateMatrices(float *A, float *B, int m, int n, float epsilon) {
  // debugging hardcode
  epsilon = 0.001;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (fabs(A[i * n + j] - B[i * n + j]) > epsilon)
        return false;
    }
  }
  return true;
}

float minDifferenceBetweenMatrices(float *A, float *B, int m, int n) {
  float maxDiff = +99999.0f;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float diff = fabs(A[i * n + j] - B[i * n + j]);
      if (diff < maxDiff) {
        maxDiff = diff;
      }
    }
  }

  return maxDiff;
}

void cppVerifier(float *h_C, float *h_C_ref, int m, int n) {
  const float epsilon = 1e-3f;
  int wrong = 0;
  for(int i = 0; i < m * n; ++i){
      if(abs(h_C[i] - h_C_ref[i]) > epsilon){
          // std::cerr << "Mismatch at index " << i << ": GPU " << h_C[i] << " vs CPU " << h_C_ref[i] << std::endl;
          wrong++;
      }
  }
  std::cerr << "Total Mismatched: " << wrong << std::endl;
}
