#include "CudaFunctions.cuh"

namespace NN {
    // __global__ void reset_layer(Vector layer, float* value) {
    //     int id = blockIdx.x * blockDim.x + threadIdx.x;
    //     if (id < layer.size) {
    //         layer.d_elements[id] = value[id];
    //     }
    // }

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

    __global__ void mat_mul_kernel(Matrix A, Matrix B, Matrix C) {
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

    __global__ void matrix_mul_vector_kernel(Vector A, Matrix B, Vector C) {
        // Each thread computes one element of C
        // by accumulating results into Cvalue
        float Cvalue = 0;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        for (int e = 0; e < A.size; ++e)
            Cvalue += A.d_elements[e] * B.d_elements[e * B.width + col];
        C.d_elements[col] = Cvalue;
    }

    __global__ void push_forward_kernel(Vector layer, Vector layerZ, Vector b, Matrix w) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < layerZ.size) {
            layerZ.d_elements[id] = b.d_elements[id];
            for (int i = 0; i < layer.size; ++i) {
                layerZ.d_elements[id] += w.d_elements[id * w.width + i] * layer.d_elements[i];
            }
        }
    }

    __global__ void update_weights_kernel(Matrix w, Vector b,Vector delta, Vector layer, float studyRate) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < w.width) {
            for (int i = 0; i < w.width; ++i) {
                w.d_elements[idx * w.width + i] -= studyRate * delta.d_elements[idx] * layer.d_elements[i];
            }
            b.d_elements[idx] -= studyRate * delta.d_elements[idx];
        }
    }

}
