#include "CudaFunctions.cuh"

namespace NN {
    __global__ void setup_kernel(curandState *state, unsigned long seed, int total_size) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < total_size) {
            curand_init(seed, id, 0, &state[id]); // (seed, subsequence, offset, &state)
        }
    }

    __global__ void rand_fill_kernel(float *arr, curandState *state, float min_n, float max_n, int total_size) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < total_size) {
            float rand_0_1 = curand_uniform(&state[id]); // [0,1)
            arr[id] = min_n + rand_0_1 * (max_n - min_n);
        }
    }

    __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
        // Each thread computes one element of C
        // by accumulating results into Cvalue
        float Cvalue = 0;
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        for (int e = 0; e < A.width; ++e)
            Cvalue += A.d_elements[row * A.width + e]
                    * B.d_elements[e * B.width + col];
        C.d_elements[row * C.width + col] = Cvalue;
    }

    __global__ void MatrixMulVectorKernel(Vector A, Matrix B, Vector C) {
        // Each thread computes one element of C
        // by accumulating results into Cvalue
        float Cvalue = 0;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        for (int e = 0; e < A.size; ++e)
            Cvalue += A.d_elements[e] * B.d_elements[e * B.width + col];
        C.d_elements[col] = Cvalue;
    }


}