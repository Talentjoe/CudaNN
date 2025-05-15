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
        //[i][j] -> i * w.width + j     width previous layer

        void initRand(float max = 1, float min = -1) const;

        void resize(int w, int h);

        void cpDtoH() const;

        void cpDtoHAsync() const;

        void cpHoD() const;

        void cpHoDAsync() const;

        void free() const;

        void printMat() const;
    };
}

#endif //MATRIX_H
