// runner.cu

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
// #include "blockTiling2d.cuh"
// #include "tilingWorking.cuh"
#include "isItWorking.cuh"

// Define tiling and thread block parameters
constexpr int BM = 64; // Block size in M dimension
constexpr int BN = 64; // Block size in N dimension
constexpr int BK = 8;  // Block size in K dimension
constexpr int TM = 8;  // Threads per tile in M dimension
constexpr int TN = 8;  // Threads per tile in N dimension

// CUDA error checking macro
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__        \
                      << ": " << cudaGetErrorString(err) << std::endl;          \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Function to initialize a matrix with random values
void initializeMatrix(float* mat, int rows, int cols) {
    for(int i = 0; i < rows * cols; ++i){
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// CPU implementation for verification
void cpuMatrixMultiply(const float* A, const float* B, float* C, int m, int n, int k, float alpha, float beta){
    for(int row = 0; row < m; ++row){
        for(int col = 0; col < n; ++col){
            float sum = 0.0f;
            for(int p = 0; p < k; ++p){
                sum += A[row * k + p] * B[p * n + col];
            }
            C[row * n + col] = alpha * sum + beta * C[row * n + col];
        }
    }
}

int main(int argc, char* argv[]){
    // Matrix dimensions
    int m = 1024; // Number of rows in A and C
    int n = 1024; // Number of columns in B and C
    int k = 1024; // Number of columns in A and rows in B

    // Scalars
    float alpha = 1.0f;
    float beta = 0.0f;

    // Allocate host memory
    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);

    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC); // For CPU reference

    if(!h_A || !h_B || !h_C || !h_C_ref){
        std::cerr << "Failed to allocate host matrices." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Initialize matrices
    srand(0); // For reproducibility
    initializeMatrix(h_A, m, k);
    initializeMatrix(h_B, k, n);
    initializeMatrix(h_C, m, n); // Initialize C (for beta scaling)

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    // Each block computes a BM x BN tile of C
    int gridDimX = (n + BN - 1) / BN;
    int gridDimY = (m + BM - 1) / BM;
    dim3 grid(gridDimX, gridDimY);

    // Calculate block dimensions
    dim3 block((BM * BN) / (TM*TN)); 

    // Launch the kernel
    // Instantiate the kernel with template parameters
    // blockTiling2d<64, 64, 8, 8, 8><<<grid, block>>>(d_A, d_B, d_C, m, n, k, alpha, beta);
    // To handle CUDA errors after kernel launch, use cudaGetLastError and cudaDeviceSynchronize

    // Start timing
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure all previous operations are done
    auto start = std::chrono::high_resolution_clock::now();

    // Launch the kernel
    blockTiling2d<BM, BN, BK, TM, TN><<<grid, block>>>(d_A, d_B, d_C, m, n, k, alpha, beta);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for kernel to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_duration = end - start;
    std::cout << "GPU kernel execution time: " << gpu_duration.count() << " seconds." << std::endl;

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // (Optional) Compute reference result on CPU and verify
    std::cout << "Computing reference result on CPU..." << std::endl;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuMatrixMultiply(h_A, h_B, h_C_ref, m, n, k, alpha, beta);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU computation time: " << cpu_duration.count() << " seconds." << std::endl;

    // Verify the result
    bool correct = true;
    const float epsilon = 1e-3f;
    for(int i = 0; i < m * n; ++i){
        if(abs(h_C[i] - h_C_ref[i]) > epsilon){
            std::cerr << "Mismatch at index " << i << ": GPU " << h_C[i] << " vs CPU " << h_C_ref[i] << std::endl;
            correct = false;
            break;
        }
    }

    if(correct){
        std::cout << "Result verification: SUCCESS" << std::endl;
    }
    else{
        std::cout << "Result verification: FAILURE" << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    // Reset CUDA device
    CUDA_CHECK(cudaDeviceReset());

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

