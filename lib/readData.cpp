//
// Created by lenovo on 25-1-7.
//

#include "readData.h"
#include <iostream>
#include <fstream>
#include <cstdint>

using namespace std;

namespace readData {
    unsigned int reverseint(unsigned int A) {
        return ((((uint32_t) (A) & 0xff000000) >> 24) |
                (((uint32_t) (A) & 0x00ff0000) >> 8) |
                (((uint32_t) (A) & 0x0000ff00) << 8) |
                (((uint32_t) (A) & 0x000000ff) << 24));
    }

    unsigned char reverse(unsigned char A) {
        return ((((A) & (unsigned char) 0xff00) >> 8) |
                (((A) & (unsigned char) 0x00ff) << 8));
    }

    vector<vector<double> > readData::readImageData(std::string path, int sizeE) {
        ifstream inFile(path, ios::in | ios::binary);
        if (!inFile) {
            cout << "error" << endl;
            return vector<vector<double> >();
        }

        unsigned int n, a;

        int size;
        int height;
        int width;

        inFile.read((char *) &a, sizeof(unsigned int));
        inFile.read((char *) &n, sizeof(unsigned int));
        size = reverseint(n);
        size = sizeE == -1 ? size : sizeE > size ? size : sizeE;
        inFile.read((char *) &a, sizeof(unsigned int));
        height = reverseint(a);
        inFile.read((char *) &a, sizeof(unsigned int));
        width = reverseint(a);

        vector<vector<double> > imageList(size, vector<double>(height * width));

        for (int k = 0; k < size; k++) {
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    unsigned char temp;
                    inFile.read((char *) &temp, sizeof(unsigned char));
                    imageList[k][i * width + j] = ((double) temp / 255);
                }
            }
        }

        return imageList;
    }

    std::vector<int> readData::readTagData(std::string path, int sizeE) {
        ifstream inFileLable(path, ios::in | ios::binary);
        if (!inFileLable) {
            cout << "error" << endl;
            return vector<int>();
        }

        int size;
        unsigned int n, a;
        inFileLable.read((char *) &a, sizeof(unsigned int));

        inFileLable.read((char *) &n, sizeof(unsigned int));
        size = reverseint(n);
        size = sizeE == -1 ? size : sizeE > size ? size : sizeE;

        vector<int> tagList(size);

        for (int k = 0; k < size; k++) {
            unsigned char temp;
            inFileLable.read((char *) &temp, sizeof(unsigned char));
            tagList[k] = temp;
        }

        return tagList;
    }
}
