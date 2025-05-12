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

    NN::NNCore *nn = new NN::NNCore("./test.mod", 0.01);

    auto start = std::chrono::high_resolution_clock::now();

    nn->test(testInData, testOutData);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "Runtime: " << duration.count() << " Seconds" << std::endl;

    delete nn;
}

void testTrain() {
    const int termsOfTrain = 10;
    float Srate = 0.1;

    vector<NN::NNCore::LayerStructure> layerStructure = {
        {784, ""},
        {300, "ReLU"},
        {100, "ReLU"},
        {10, "sigmoid"}
    };

    auto *nn = new NN::NNCore(layerStructure, Srate);

    vector<vector<float> > inData;
    vector<int> outData;
    inData = readData::readData::readImageData("../Data/train-images.idx3-ubyte");
    outData = readData::readData::readTagData("../Data/train-labels.idx1-ubyte");

    vector<vector<float> > testInData;
    vector<int> testOutData;
    testInData = readData::readData::readImageData("../Data/t10k-images.idx3-ubyte");
    testOutData = readData::readData::readTagData("../Data/t10k-labels.idx1-ubyte");

    for (int j = 0; j < termsOfTrain; j++) {
        nn->train(inData, outData, true);
        nn->test(testInData, testOutData);
        //Srate *= 0.05;
        //nn->changeStudyRate(Srate);
    }


    delete nn;
}

int main() {
    testForward();

    // float Srate = 0.05;
    // NN::NNCore *nn = new NN::NNCore("./test.mod", 1);
    //
    // // vector<NN::NNCore::LayerStructure> layerStructure = {
    // //     {784, "sigmoid"},
    // //     {100, "sigmoid"},
    // //     {10, "sigmoid"}
    // // };
    // //     auto *nn = new NN::NNCore(layerStructure, Srate);
    //
    // auto testInData = readData::readData::readImageData("../Data/t10k-images.idx3-ubyte");
    // auto testOutData = readData::readData::readTagData("../Data/t10k-labels.idx1-ubyte");
    //
    // nn->forward(testInData[0]);
    // //nn->layers[2].printVec();
    //
    // nn->backpropagation({0,0,0,0,0,0,0,1,0,0});
    // nn->forward(testInData[0]);
    // nn->w[1].cpDtoH();
    // nn->w[1].printMat();
    // nn->layersZ[1].cpDtoH();
    // nn->layersZ[1].printVec();
    // nn->delta[1].cpDtoH();
    // nn->delta[1].printVec();

    return 0;
}
