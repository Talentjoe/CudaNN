//
// Created by lenovo on 25-1-6.
//

#include "NNCore.cuh"

#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace NN {
#define uint unsigned int

    using namespace std;

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

    vector<float> NNCore::forward(vector<float> inNums, bool printRes) {
        return{};
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

    NNCore::NNCore(const vector<int> &LayerS, const float studyR,
                      std::vector<std::function<float(float)> > activationFunction,
                      std::vector<std::function<float(float)> > activationFunctionP) {
        ActivationFunction = activationFunction;
        ActivationFunctionP = activationFunctionP;

        size = LayerS.size();
        layerSize = LayerS;
        studyRate = studyR;

        h_layers = new Vector[size];
        h_layersZ = new Vector[size];
        h_b = new Vector[size];

        for (int i = 0; i < size; ++i) {
            h_layers[i].resize(layerSize[i]);
            h_layersZ[i].resize(layerSize[i]);
            h_b[i].resize(layerSize[i]);
        }

        h_w = new Matrix[size - 1];
        for (int i = 0; i < size - 1; ++i) {
            h_w[i].resize(layerSize[i], layerSize[i + 1]);
        }

        cout << "RESIZED" << endl;


    }
} // NN
