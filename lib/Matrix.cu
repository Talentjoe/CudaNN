#include "Matrix.cuh"
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

    void Matrix::cpHoD() const {
        cudaMemcpy(d_elements, elements, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    }

    void Matrix::free() const {
        delete[] elements;
        cudaFree(d_elements);
    }
}
