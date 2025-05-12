//
// Created by lenovo on 2025/5/11.
//

#ifndef VECTOR_CUH
#define VECTOR_CUH
#include <string>

namespace NN {
    struct Vector {
        int size;
        float *elements = nullptr;
        float *d_elements;

        void initRandom();

        void resize(int s);

        void cpDtoH() const;
        void cpDtoHAsync() const;

        void cpHoD() const;
        void cpHoDAsync() const;

        void free() const;

        void printVec() const;
    };
}

#endif //VECTOR_CUH
