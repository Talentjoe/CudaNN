#include "Matrix.cuh"

#include <iostream>
#include "CudaFunctions.cuh"

namespace NN {
    void Matrix::initRand() const {
        GetRand(*this);
    }

    void Matrix::resize(int w, int h) {
        width = w;
        height = h;
        elements = new float[w * h];
        cudaMalloc(&d_elements, sizeof(float) * w * h);
    }

    void Matrix::cpDtoH() const {
        cudaMemcpy(elements, d_elements, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    }

    void Matrix::cpDtoHAsync() const {
        cudaMemcpyAsync(elements, d_elements, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    }

    void Matrix::cpHoD() const {
        cudaMemcpy(d_elements, elements, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    }

    void Matrix::cpHoDAsync() const {
        cudaMemcpyAsync(d_elements, elements, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    }

    void Matrix::free() const {
        delete[] elements;
        cudaFree(d_elements);
    }

    void Matrix::printMat() const {
            for (int j = 0; j < width; j++) {
        for (int i = 0; i < height; i++) {
                std::cout << elements[i * width + j] << " ";
            }
            std::cout << std::endl;
        }
    }
}
