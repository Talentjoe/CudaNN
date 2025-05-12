//
// Created by lenovo on 2025/5/11.
//

#ifndef CUDAFUNCTIONS_CUH
#define CUDAFUNCTIONS_CUH

#include <curand_kernel.h>
#include "Matrix.cuh"
#include "Vector.cuh"

namespace NN{
    __global__ void setup_kernel(curandState *state, unsigned long seed, int total_size);

    __global__ void rand_fill_kernel(float *arr, curandState *state, float min_n, float max_n, int total_size);

    __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C);

    __global__ void MatrixMulVectorKernel(Vector A, Matrix B, Vector C);

    static void GetRand(Matrix A, float max_n = 1, float min_n = -1){
        int size = A.width * A.height;

        curandState *devStates;
        cudaMalloc(&devStates, sizeof(curandState) * size);

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        setup_kernel<<<numBlocks, blockSize>>>(devStates, time(NULL), size);
        cudaDeviceSynchronize();

        rand_fill_kernel<<<numBlocks, blockSize>>>(A.d_elements, devStates, min_n, max_n, size);
        cudaDeviceSynchronize();

        cudaFree(devStates);
    }

    static void GetRand(Vector A, float max_n = 1, float min_n = -1){
        int size = A.size;

        curandState *devStates;
        cudaMalloc(&devStates, sizeof(curandState) * size);

        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;

        setup_kernel<<<numBlocks, blockSize>>>(devStates, time(NULL), size);
        cudaDeviceSynchronize();

        rand_fill_kernel<<<numBlocks, blockSize>>>(A.d_elements, devStates, min_n, max_n, size);
        cudaDeviceSynchronize();

        cudaFree(devStates);
    }
}
#endif //CUDAFUNCTIONS_CUH
