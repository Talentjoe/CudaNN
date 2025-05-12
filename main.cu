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
    NN::NNCore *nn = new NN::NNCore("./test.mod", 0.01);

    //nn->w[1].cpDtoHAsync();
    //nn->w[1].printMat();

    vector<vector<float> > testInData;
    vector<int> testOutData;
    testInData = readData::readData::readImageData("../Data/t10k-images.idx3-ubyte");
    testOutData = readData::readData::readTagData("../Data/t10k-labels.idx1-ubyte");

    nn->test(testInData, testOutData);

    return 0;
}
