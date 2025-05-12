#include "./lib/NNCore.cuh"
#include "./lib/readData.h"

using namespace std;

void genMatWithRand() {
    NN::Matrix A;

    A.resize(10, 10);

    A.initRand();
    A.cpDtoH();

    A.printMat();
}

void genVecWithRand() {
    NN::Vector A;

    A.resize(100);

    A.initRandom();
    A.cpDtoH();

    A.printVec();
}


int main() {

    return 0;
}
