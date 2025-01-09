#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void initRandMatrix(float *A, int m, int n);
void printMatrix(float *A, int m, int n);
void cpuMatmul(float *A, float *B, float *C, int m, int n, int k);
bool validateMatrices(float *A, float *B, int m, int n, float epsilon);
float maxDifferenceBetweenMatrices(float *A, float *B, int m, int n);
float minDifferenceBetweenMatrices(float *A, float *B, int m, int n);
