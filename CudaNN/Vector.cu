#include "Vector.cuh"

#include <iostream>
#include "CudaFunctions.cuh"

namespace NN {
    void Vector::initRandom(float max, float min) {
        GetRand(*this,max,min);
    }

    void Vector::resize(int s) {
        if (elements != nullptr) {
            delete[] elements;
        }
        size = s;
        elements = new float[s];
        cudaMalloc(&d_elements, sizeof(float) * s);
    }

    void Vector::cpDtoH() const {
        cudaMemcpy(elements, d_elements, sizeof(float) * size, cudaMemcpyDeviceToHost);
    }

    void Vector::cpDtoHAsync() const {
        cudaMemcpyAsync(elements, d_elements, sizeof(float) * size, cudaMemcpyDeviceToHost);
    }

    void Vector::cpHoD() const {
        cudaMemcpy(d_elements, elements, sizeof(float) * size, cudaMemcpyHostToDevice);
    }

    void Vector::cpHoDAsync() const {
        cudaMemcpyAsync(d_elements, elements, sizeof(float) * size, cudaMemcpyHostToDevice);
    }

    void Vector::free() const {
        cudaFree(d_elements);
    }

    void Vector::printVec() const {
        for (int i = 0; i < size; i++) {
            std::cout << elements[i] << " ";
        }
        std::cout << std::endl;
    }
}
