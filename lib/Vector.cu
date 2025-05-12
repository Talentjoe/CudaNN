#include "Vector.cuh"

#include <iostream>
#include "CudaFunctions.cuh"

namespace NN {
    void Vector::initRandom() {
        GetRand(*this);
    }

    void Vector::resize(int s) {
        size = s;
        elements = new float[s];
        cudaMalloc(&d_elements, sizeof(float) * s);
    }

    void Vector::cpDtoH() const {
        cudaMemcpy(elements, d_elements, sizeof(float) * size, cudaMemcpyDeviceToHost);
    }

    void Vector::cpHoD() const {
        cudaMemcpy(d_elements, elements, sizeof(float) * size, cudaMemcpyHostToDevice);
    }

    void Vector::free() const {
        delete[] elements;
        cudaFree(d_elements);
    }

    void Vector::printVec() const {
        for (int i = 0; i < size; i++) {
            std::cout << elements[i] << " ";
        }
        std::cout << std::endl;
    }
}
