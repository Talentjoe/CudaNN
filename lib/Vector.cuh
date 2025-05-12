//
// Created by lenovo on 2025/5/11.
//

#ifndef VECTOR_CUH
#define VECTOR_CUH

namespace NN {
    struct Vector {
        int size;
        float *elements;
        float *d_elements;

        void initRandom();

        void resize(int s);

        void cpDtoH() const;

        void cpHoD() const;

        void free() const;
    };
}

#endif //VECTOR_CUH
