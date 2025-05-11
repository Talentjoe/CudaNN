#include <cstdint>
#include <ctime>

#include "./lib/NNCore.cuh"
#include "./lib/readData.h"

using namespace std;


int main() {
    const int termsOfTrain = 5;
    float Srate = 1;

    srand(time(nullptr) * 3049);

    auto *nn = new NN::NNCore();
    nn->init(vector{28 * 28, 128, 10}, Srate);

    vector<vector<float> > inData;
    vector<int> outData;
    inData = readData::readData::readImageData("../Data/train-images.idx3-ubyte", 10000);
    outData = readData::readData::readTagData("../Data/train-labels.idx1-ubyte", 10000);

    for (int j = 0; j < termsOfTrain; j++) {
        nn->train(inData, outData, true);
        //Srate *= 0.05;
        nn->changeStudyRate(Srate);
    }

    vector<vector<float> > testInData;
    vector<int> testOutData;
    testInData = readData::readData::readImageData("../Data/t10k-images.idx3-ubyte");
    testOutData = readData::readData::readTagData("../Data/t10k-labels.idx1-ubyte");

    nn->test(testInData, testOutData);

    NN::NNCore::save(*nn, "test.mod");

    delete nn;

    return 0;
}
