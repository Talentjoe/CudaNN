#include "./CudaNN/NNCore.cuh"
#include "./lib/readData.h"
#include <chrono>
#include <iostream>

using namespace std;

void printPic(vector<float> pic) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            if (pic[i * 28 + j] > 0.5) {
                cout << "#";
            } else {
                cout << " ";
            }
        }
        cout << endl;
    }
}

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
        {512, "ReLU"},
        {128, "ReLU"},
        {64, "ReLU"},
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


    auto start = std::chrono::high_resolution_clock::now();

    for (int j = 0; j < termsOfTrain; j++) {
        std::cout << "Epoch: " << j << std::endl;
        vector<vector<float>> wrongData;
        vector<int> correctData;
        nn->train_with_retrain(inData, outData,wrongData, correctData,true);
        if (j > -1) {
            cout<<"Change Study Rate to 0.01 to train "<< wrongData.size()<<" of wrong pic" << endl;
            nn->changeStudyRate(0.01);
            vector<vector<float>> wrongData1;
            vector<int> correctData1;
            nn->train_with_retrain(wrongData, correctData,wrongData1, correctData1,true);

        }
        float crate = nn->test(testInData, testOutData);
        nn->save("Model_Epoch" + std::to_string(j) +"_With_Rate_" +std::to_string(crate*100)+ "%.module");
        Srate = Srate * 0.75;
        nn->changeStudyRate(Srate);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "Runtime: " << duration.count() << " Seconds" << std::endl;
    //nn->save("test.modle");

    delete nn;
}

void test() {
    vector<vector<float> > testInData;
    vector<int> testOutData;
    testInData = readData::readData::readImageData("../Data/t10k-images.idx3-ubyte");
    testOutData = readData::readData::readTagData("../Data/t10k-labels.idx1-ubyte");

    NN::NNCore *nn = new NN::NNCore("./Model_Epoch8_With_Rate_98.089996%.mod", 0.01);

    auto start = std::chrono::high_resolution_clock::now();

    vector<vector<float>> wrongData;
    vector<int> correctData;
    nn->test_with_wrong(testInData, testOutData,wrongData, correctData);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "Runtime: " << duration.count() << " Seconds" << std::endl;

    cout<< "The number of wrong data is: " << wrongData.size() << endl;
    while (true) {
        int index;
        cout << "Please input the index of the pic you want to see: ";
        cin >> index;
        if (index < 0 || index >= wrongData.size()) {
            cout << "Index out of range!" << endl;
            break;
        }
        printPic(wrongData[index]);
        cout << "The correct answer is: " << correctData[index] << endl;
        nn->forward(wrongData[index]);
        cout << "The NN's answer is: " << nn->choice() << endl;
    }

    delete nn;
}

#include <QApplication>
#include "./UI/mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWindow w;
    w.resize(300, 400);
    w.setWindowTitle("Qt Handwriting Board");
    w.show();
    return a.exec();
}