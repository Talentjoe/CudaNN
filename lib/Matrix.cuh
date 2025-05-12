//
// Created by lenovo on 2025/5/11.
//

#ifndef MATRIX_H
#define MATRIX_H

namespace NN {
    struct Matrix {
        int width;
        int height;
        float *elements; // HostPointer
        float *d_elements; // DevicePointer

        void initRand() const;

        void resize(int w, int h);

        void cpDtoH() const;

        void cpHoD() const;

        void free() const;
    };
}

#endif //MATRIX_H
