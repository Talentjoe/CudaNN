#include <cstdint>
#include <ctime>
#include <iostream>

#include "./lib/NNCore.cuh"
#include "./lib/readData.h"

using namespace std;


int main() {
    NN::Vector inNums;

    inNums.resize(100);

    inNums.initRandom();
    inNums.cpDtoH();

    inNums.printVec();

    NN::Matrix inMat;
    
    inMat.resize(10, 10);

    inMat.initRand();
    inMat.cpDtoH();

    inMat.printMat();

    return 0;
}
