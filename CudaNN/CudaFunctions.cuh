//
// Created by lenovo on 2025/5/11.
//

#ifndef CUDAFUNCTIONS_CUH
#define CUDAFUNCTIONS_CUH

#include <curand_kernel.h>
#include "Matrix.cuh"
#include "Vector.cuh"

namespace NN{
    //__global__ void reset_layer(Vector layer, float* value);

    __global__ void setup_kernel(curandState *state, unsigned long seed, int total_size);

    __global__ void rand_fill_kernel(float *arr, curandState *state, float min_n, float max_n, int total_size);

    __global__ void mat_mul_kernel(Matrix A, Matrix B, Matrix C);

    __global__ void matrix_mul_vector_kernel(Vector A, Matrix B, Vector C);

    __global__ void push_forward_kernel(Vector layer, Vector layerZ, Vector b, Matrix w);

    __global__ void update_weights_kernel(Matrix w, Vector b,Vector delta, Vector layer, float studyRate);

    template<typename ActFunP>
    __global__ void back_propagate_delta_kernel(Vector deltaPreLayer, Vector delta, Matrix w, Vector layerZ, ActFunP af_p) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < deltaPreLayer.size) {
            float val = 0;
            for (int i = 0; i < delta.size; ++i) {
                val += delta.d_elements[i] * w.d_elements[ i * w.width + id];
            }
            deltaPreLayer.d_elements[id] = af_p(layerZ.d_elements[id]) * val;
        }
    }

    template<typename ActFunP>
    __global__ void get_last_layer_delta_kernel(Vector layer, Vector layerZ,float* correctAns, Vector delta, ActFunP af_p) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < layer.size) {
            delta.d_elements[id] = af_p(layerZ.d_elements[id]) * (layer.d_elements[id] - correctAns[id]);
        }
    }

    template<typename ActFun>
    __global__ void activate_kernel(Vector layerZ, Vector layer, ActFun af) {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < layerZ.size) {
            layer.d_elements[id] = af(layerZ.d_elements[id]);
        }
    }

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
