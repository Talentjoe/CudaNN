//
// Created by lenovo on 25-1-6.
//

#include "NNCore.cuh"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

namespace NN {
#define uint unsigned int

    using namespace std;


    NNCore::NNCore(const vector<LayerStructure> &Layers, const float studyR) {
        size = Layers.size();

        ranges::transform(Layers, back_inserter(ActivationFunction),
                          [](const LayerStructure &layer) { return layer.activationFunction; });

        ranges::transform(Layers, back_inserter(layerSize),
                          [](const LayerStructure &layer) { return layer.layerSize; });

        studyRate = studyR;

        layers = new Vector[size];
        layersZ = new Vector[size];
        b = new Vector[size];
        w = new Matrix[size - 1];

        for (int i = 0; i < size; ++i) {
            layers[i].resize(layerSize[i]);
            layers[i].initRandom();
            layersZ[i].resize(layerSize[i]);
            layersZ[i].initRandom();
            b[i].resize(layerSize[i]);
            b[i].initRandom();
            if (i<size-1) {
                w[i].resize(layerSize[i], layerSize[i + 1]);
                w[i].initRand();
            }
        }

        cout << "RESIZED And Inited" << endl;
    }

    NNCore::~NNCore() {
        for (int i = 0; i < size; ++i) {
            layers[i].free();
            layersZ[i].free();
            b[i].free();
            if (i < size - 1) {
                w[i].free();
            }
        }

        delete[] layers;
        delete[] layersZ;
        delete[] b;
        delete[] w;
    }

    vector<float> NNCore::forward(vector<float> inNums, bool printRes) {
        if (inNums.size() != layerSize[0]) {
            cout << "Size Not Mathch !! " << endl;
            return {};
        }

        layers[0].elements = inNums.data();
        layers[0].cpHoD();

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        for (int i = 0; i < size - 1; ++i) {


            if (ActivationFunction[i] == "sigmoid") {

            } else if (ActivationFunction[i] == "ReLU") {

            }


            layers[i + 1].cpDtoHAsync();
            layersZ[i + 1].cpDtoHAsync();
            w[i + 1].cpDtoHAsync();
            b[i + 1].cpDtoHAsync();
        }

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return {};
    }

    float NNCore::train(vector<vector<float> > inNums, vector<int> correctOut, bool getAcc) {
        if (inNums.size() != correctOut.size()) {
            cout << "Size Not Match !! " << endl;
            return -1;
        }

        int corrctCnt = 0;
        int wrongCnt = 0;
        vector answer(10, 0.0f);

        for (int i = 0; i < inNums.size(); i++) {
            if (inNums[i].size() != layerSize[0] || correctOut[i] > layerSize[size - 1]) {
                cout << "Size Not Match !! " << endl;
                return -1;
            }

            forward(inNums[i]);
            answer[correctOut[i]] = 1;
            backpropagation(answer);
            answer[correctOut[i]] = 0;

            if (getAcc) {
                if (choice() == correctOut[i]) {
                    corrctCnt++;
                } else {
                    wrongCnt++;
                }
            }

            if (i % 1000 == 0) {
                cout << "\rProgress: " << i / (float) inNums.size() * 100 << "%";
                if (getAcc) {
                    cout << " Correct Percentage: " << corrctCnt / (float) (corrctCnt + wrongCnt) * 100 << "%";
                }
                cout << "                    " << flush;
            }
        }
        cout << endl;
        cout << "Finish Training " << inNums.size() << " Data" << endl;

        if (!getAcc) return 0;

        cout << "With Accuracy: " << corrctCnt / (float) (corrctCnt + wrongCnt) * 100 << "%" << endl;
        return corrctCnt / (float) (corrctCnt + wrongCnt);
    }

    float NNCore::test(vector<vector<float> > inNums, vector<int> correctOut) {
        if (inNums.size() != correctOut.size()) {
            cout << "Size Not Match !! " << endl;
            return -1;
        }

        int corrctCnt = 0;
        int wrongCnt = 0;

        for (int i = 0; i < inNums.size(); i++) {
            if (inNums[i].size() != layerSize[0] || correctOut[i] > layerSize[size - 1]) {
                cout << "Size Not Match !! " << endl;
                return -1;
            }

            forward(inNums[i]);

            if (choice() == correctOut[i]) {
                corrctCnt++;
            } else {
                wrongCnt++;
            }

            if (i % 1000 == 0) {
                cout << "\rProgress: " << i / (float) inNums.size() * 100 << "%";
                cout << " Correct Percentage: " << corrctCnt / (float) (corrctCnt + wrongCnt) * 100 << "%";
                cout << "                    " << flush;
            }
        }
        cout << endl;
        cout << "Finish Testing " << inNums.size() << " Data" << endl;

        cout << "With Accuracy: " << corrctCnt / (float) (corrctCnt + wrongCnt) * 100 << "%" << endl;
        return corrctCnt / (corrctCnt + wrongCnt);
    }


    float NNCore::CalCost(vector<float> correctOut) {
        return 0;
    }

    float NNCore::backpropagation(vector<float> correctOut) {
        return 0;
    }

    void NNCore::printLayers() {
    }

    void NNCore::printLayers(const NNCore &nn) {
    }

    void NNCore::printW(int layerNumberToPrint) {
    }

    void NNCore::printW(const NNCore &nn, int layerNumberToPrint) {
    }

    int NNCore::choice() {
        return 0;
    }

    void NNCore::changeStudyRate(const float rate) {
    }

    void NNCore::changeDropOutRate(const float rate) {
    }

    void NNCore::dropSome() {
    }

    void NNCore::save(const NNCore &nn, string path) {
    }
} // NN
