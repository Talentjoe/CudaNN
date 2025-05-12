#include "./lib/NNCore.cuh"
#include "./lib/readData.h"
#include <chrono>
#include <iostream>

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

void testForward() {
    vector<vector<float> > testInData;
    vector<int> testOutData;
    testInData = readData::readData::readImageData("../Data/t10k-images.idx3-ubyte");
    testOutData = readData::readData::readTagData("../Data/t10k-labels.idx1-ubyte");

    auto start = std::chrono::high_resolution_clock::now();
    NN::NNCore *nn = new NN::NNCore("./test.mod", 0.01);

    nn->test(testInData, testOutData);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "Runtime: " << duration.count() << " Seconds" << std::endl;

    delete nn;
}

int main() {
    testForward();
    return 0;
}
