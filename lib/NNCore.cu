//
// Created by lenovo on 25-1-6.
//

#include "NNCore.cuh"

#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>

namespace NN {
    using namespace std;


    float NNcore::train(vector<vector<float> > inNums, vector<int> correctOut, bool getAcc) {
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
            if (addDropout)
                dropSome();
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

    float NNcore::test(vector<vector<float> > inNums, vector<int> correctOut) {
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

    vector<float> NNcore::forward(vector<float> inNums, bool printRes) {
        if (inNums.size() != layerSize[0]) {
            cout << "Size Not Mathch !! " << endl;
            return {};
        }
        layers[0] = inNums;

        for (auto &row: layersZ) // 将内容归零
        {
            fill(row.begin(), row.end(), 0.0);
        }

        for (int i = 1; i < size; i++) {
            for (int k = 0; k < layerSize[i]; k++) {
                for (int j = 0; j < layerSize[i - 1]; j++)
                    layersZ[i][k] += layers[i - 1][j] * w[i - 1][j][k];
                layersZ[i][k] += b[i][k];
                layers[i][k] = ActivationFunction(layersZ[i][k]);
            }
        }
        if (printRes) {
            for (auto i: layers[size - 1]) cout << setw(10) << i;
            cout << endl;
        }
        return layers[size - 1];
    }

    float NNcore::CalCost(vector<float> correctOut) {
        float cost = 0;
        if (correctOut.size() != layerSize[size - 1]) {
            cout << "Size Error" << endl;
            return 0;
        }
        for (int i = 0; i < layerSize[size - 1]; i++) {
            cost += (correctOut[i] * log(layers[size - 1][i]) + (1 - correctOut[i]) * log(1 - layers[size - 1][i])) /
                    2.0;
        }
        cost /= -layerSize[size - 1];
        return cost;
    }

    float NNcore::backpropagation(vector<float> correctOut) {
        vector<vector<float> > delta(size);
        for (int i = 1; i < size; i++) {
            delta[i].resize(layerSize[i]);
        }

        for (int i = 0; i < layerSize[size - 1]; i++) {
            delta[size - 1][i] = (layers[size - 1][i] - correctOut[i]) * ActivationFunctionP(layersZ[size - 1][i]); //BP1
        }
        for (int i = size - 2; i > 0; i--) {
            for (int j = 0; j < layerSize[i]; j++) {
                float value = 0;
                for (int k = 0; k < layerSize[i + 1]; k++) {
                    value += w[i][j][k] * delta[i + 1][k];
                }
                delta[i][j] = ActivationFunctionP(layersZ[i][j]) * value; // BP2
            }
        }
        for (int i = 1; i < size; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                b[i][j] -= delta[i][j] * studyRate;

                for (int k = 0; k < layerSize[i - 1]; k++) {
                    w[i - 1][k][j] -= layers[i - 1][k] * delta[i][j] * studyRate +
                            ( 0.00001*w[i - 1][k][j] );
                }
            }
        }

        float pre = CalCost(correctOut);
        forward(layers[0]);
        float after = CalCost(correctOut);

        return after - pre;
    }

    void NNcore::printLayers() {
        for (int i = 0; i < size; i++) {
            cout << "Layer " << setw(2) << i << ": ";
            for (int j = 0; j < layerSize[i]; j++)
                cout << setw(10) << layers[i][j] << " ";

            cout << endl << endl;
        }
    }

    void NNcore::printLayers(const NNcore &nn) {
        for (int i = 0; i < nn.size; i++) {
            cout << "Layer " << setw(2) << i << ": ";
            for (int j = 0; j < nn.layerSize[i]; j++)
                cout << setw(10) << nn.layers[i][j] << " ";

            cout << endl << endl;
        }
    }

    void NNcore::printW(int layerNumberToPrint) {
        if (layerNumberToPrint >= size - 1) {
            cout << "Layer Number Error" << endl;
            return;
        }

        cout << "W from layer " << layerNumberToPrint << " to " << layerNumberToPrint + 1 << ": " << endl;
        cout << "From↓   To-> ";
        for (int i = 0; i < layerSize[layerNumberToPrint + 1]; i++) {
            cout << setw(10) << i << " ";
        }
        cout << endl << endl;

        for (int i = 0; i < layerSize[layerNumberToPrint]; i++) {
            cout << "From " << setw(2) << i << " note:";
            for (int j = 0; j < layerSize[layerNumberToPrint + 1]; j++)
                cout << setw(10) << w[layerNumberToPrint][i][j] << " ";

            cout << endl << endl;
        }
    }

    void NNcore::printW(const NNcore &nn, int layerNumberToPrint) {
        if (layerNumberToPrint >= nn.size - 1) {
            cout << "Layer Number Error" << endl;
            return;
        }

        cout << "W from layer " << layerNumberToPrint << " to " << layerNumberToPrint + 1 << ": " << endl;
        cout << "From↓   To-> ";
        for (int i = 0; i < nn.layerSize[layerNumberToPrint + 1]; i++) {
            cout << setw(10) << i << " ";
        }
        cout << endl << endl;

        for (int i = 0; i < nn.layerSize[layerNumberToPrint]; i++) {
            cout << "From " << setw(2) << i << " note:";
            for (int j = 0; j < nn.layerSize[layerNumberToPrint + 1]; j++)
                cout << setw(10) << nn.w[layerNumberToPrint][i][j] << " ";

            cout << endl << endl;
        }
    }

    int NNcore::choice() {
        float max = 0;
        int res = 0;
        for (int i = 0; i < layerSize[size - 1]; i++) {
            if (layers[size - 1][i] > max) {
                max = layers[size - 1][i];
                res = i;
            }
        }
        return res;
    }



    void NNcore::changeStudyRate(const float rate) {
        studyRate = rate;
    }

    void NNcore::changeDropOutRate(const float rate) {
        if (dropOutRate < 0) {
            addDropout = false;
        } else {
            addDropout = true;
            dropOutRate = rate;
        }
    }

    void NNcore::dropSome() {
        if (!addDropout)
            return;


        for (int i = 1; i < size; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                if (getRandomfloatNumber(1,0)< dropOutRate) //So Slow Here Improve Later
                    b[i][j] = 0;
            }
        }

        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                for (int k = 0; k < layerSize[i + 1]; k++) {
                    if (getRandomfloatNumber(1,0)< dropOutRate) //So Slow Here Improve Later
                        w[i][j][k] = 0;
                }
            }
        }




    }

    void NNcore::save(const NNcore &nn, string path) {
        ofstream outFile(path);
        if (!outFile.is_open()) {
            cout << "error" << endl;
            return;
        }
        outFile << nn.size << endl;

        for (int i = 0; i < nn.size; i++) {
            outFile << nn.layerSize[i] << " ";
        }
        outFile << endl;

        for (const auto &layer: nn.b) {
            for (const auto &bias: layer) {
                outFile << bias << " ";
            }
            outFile << "\n";
        }

        for (const auto &layer: nn.w) {
            for (const auto &neuron: layer) {
                for (const auto &weight: neuron) {
                    outFile << weight << " ";
                }
                outFile << "\n";
            }
            outFile << "\n";
        }
    }

    void NNcore::init(const vector<int>& LayerS, const float studyR, const float drRate,
        function<float(float)> activateFunction, function<float(float)> activateFunctionP) {

        ActivationFunction = activateFunction;
        ActivationFunctionP = activateFunctionP;

        size = LayerS.size();
        layerSize = LayerS;
        studyRate = studyR;

        if (dropOutRate < 0) {
            addDropout = false;
        } else {
            addDropout = true;
            dropOutRate = drRate;
        }

        layers = vector<vector<float> >(size);
        for (int i = 0; i < size; ++i) {
            layers[i].resize(layerSize[i]);
        }

        layersZ = vector<vector<float> >(size);
        b = vector<vector<float> >(size);

        for (int i = 1; i < size; ++i) {
            layersZ[i] = vector<float>(layerSize[i]);
            b[i] = vector<float>(layerSize[i]);
        }

        w = vector<vector<vector<float> > >(size - 1);
        for (int i = 0; i < size - 1; ++i) {
            w[i] = vector(
                layerSize[i],
                vector<float>(layerSize[i + 1])
            );
        }

        cout << "RESIZED" << endl;

        for (int i = 0; i < size; i++) {
            bool firstLayer = i == 0;
            bool lastLayer = i == size - 1;

            for (int j = 0; j < layerSize[i]; j++) {
                layers[i][j] = getRandomfloatNumber();
                if (!firstLayer) {
                    layersZ[i][j] = getRandomfloatNumber();
                    b[i][j] = getRandomfloatNumber();
                }
                if (!lastLayer) {
                    for (int k = 0; k < layerSize[i + 1]; k++) {
                        w[i][j][k] = getRandomfloatNumber();
                    }
                }
            }
        }
    }

    void NNcore::init(const string &path, float studyR, float drRate,
        function<float(float)> activateFunction, function<float(float)> activateFunctionP) {

        ActivationFunction = activateFunction;
        ActivationFunctionP = activateFunctionP;

        ifstream inFile(path);

        if (!inFile.is_open()) {
            cout << "error" << endl;
            return;
        }
        studyRate = studyR;

        if (dropOutRate < 0) {
            addDropout = false;
        } else {
            addDropout = true;
            dropOutRate = drRate;
        }

        inFile >> size;
        layerSize = vector<int>(size);
        for (int i = 0; i < size; i++) {
            inFile >> layerSize[i];
        }

        layers = vector<vector<float> >(size);
        for (int i = 0; i < size; ++i) {
            layers[i].resize(layerSize[i]);
        }

        layersZ = vector<vector<float> >(size);
        b = vector<vector<float> >(size);
        for (int i = 1; i < size; ++i) {
            layersZ[i] = vector<float>(layerSize[i]);
            b[i] = vector<float>(layerSize[i]);
        }

        w = vector<vector<vector<float> > >(size - 1);
        for (int i = 0; i < size - 1; ++i) {
            w[i] = vector<vector<float> >(
                layerSize[i],
                vector<float>(layerSize[i + 1])
            );
        }

        for (int i = 1; i < size; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                inFile >> b[i][j];
            }
        }

        for (int i = 0; i < size - 1; i++) {
            for (int j = 0; j < layerSize[i]; j++) {
                for (int k = 0; k < layerSize[i + 1]; k++) {
                    inFile >> w[i][j][k];
                }
            }
        }
    }
} // NN
