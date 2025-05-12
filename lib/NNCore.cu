//
// Created by lenovo on 25-1-6.
//

#include "NNCore.cuh"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <assert.h>

#include "CudaFunctions.cuh"

namespace NN {
#define uint unsigned int
#define SIGMOID_NAME "sigmoid"
#define RULE_NAME "ReLU"

    using namespace std;


    NNCore::NNCore(const std::string &path, float studyRate) {
        ifstream inFile(path);
        ActivationFunction.clear();

        if (!inFile.is_open()) {
            cout << "error" << endl;
            return;
        }

        this->studyRate = studyRate;

        inFile >> size;
        layerSize = vector<int>(size);
        for (int i = 0; i < size; i++) {
            inFile >> layerSize[i];
        }

        layers = new Vector[size];
        layersZ = new Vector[size];
        b = new Vector[size];
        w = new Matrix[size - 1];
        delta = new Vector[size];

        for (int i = 0; i < size; ++i) {
            ActivationFunction.emplace_back(SIGMOID_NAME);
            layers[i].resize(layerSize[i]);
            layersZ[i].resize(layerSize[i]);
            b[i].resize(layerSize[i]);
            delta[i].resize(layerSize[i]);
            if (i < size - 1) {
                w[i].resize(layerSize[i], layerSize[i + 1]);
            }
        }

        for (int i = 1; i < size; i++) {
            //need to be changed based on the activation function
            for (int j = 0; j < layerSize[i]; j++) {
                inFile >> b[i].elements[j];
            }
            b[i].cpHoDAsync();
        }

        for (int i = 0; i < size - 1; i++) {
            for (int k = 0; k < w[i].width; k++) {
                for (int j = 0; j < w[i].height; j++) {
                    inFile >> w[i].elements[j * w[i].width + k];
                }
            }
            w[i].cpHoDAsync();
        }
        cudaDeviceSynchronize();
    }

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
        delta = new Vector[size];

        for (int i = 0; i < size; ++i) {
            layers[i].resize(layerSize[i]);
            layers[i].initRandom();
            layersZ[i].resize(layerSize[i]);
            layersZ[i].initRandom();
            b[i].resize(layerSize[i]);
            b[i].initRandom();
            delta[i].resize(layerSize[i]);
            delta[i].initRandom();
            if (i < size - 1) {
                float lim = heLimit(layerSize[i]);
                w[i].resize(layerSize[i], layerSize[i + 1]);
                w[i].initRand(lim,-lim);
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
    }

    vector<float> NNCore::forward(vector<float> inNums, bool printRes) {
        if (inNums.size() != layerSize[0]) {
            cout << "Size Does Not Match !! " << endl;
            return {};
        }

        layers[0].elements = inNums.data();
        layers[0].cpHoD();

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        for (int i = 0; i < size - 1; ++i) {
            int blockSize = layerSize[i + 1];
            int gridSize = 1;
            if (layerSize[i + 1] > MAX_BLOCK_SIZE) {
                blockSize = MAX_BLOCK_SIZE;
                gridSize = (layerSize[i + 1] + blockSize - 1) / blockSize;
            }
            push_forward_kernel<<<gridSize,blockSize,0,stream>>>(layers[i], layersZ[i + 1], b[i + 1], w[i]);

            if (ActivationFunction[i] == SIGMOID_NAME) {
                activate_kernel<<<gridSize,blockSize,0,stream>>>(layersZ[i + 1], layers[i + 1], sigmoid());
            } else if (ActivationFunction[i] == RULE_NAME) {
                activate_kernel<<<gridSize,blockSize,0,stream>>>(layersZ[i + 1], layers[i + 1], ReLU());
            }

            layers[i + 1].cpDtoHAsync();
            layersZ[i + 1].cpDtoHAsync();
            //w[i + 1].cpDtoHAsync();
            //b[i + 1].cpDtoHAsync();
        }

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        return vector(layers[size - 1].elements, layers[size - 1].elements + layerSize[size - 1]);
    }

    float NNCore::backpropagation(const vector<float> &correctOut) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        float *correctOutD;
        cudaMalloc(&correctOutD, sizeof(float) * correctOut.size());
        cudaMemcpy(correctOutD, correctOut.data(), sizeof(float) * correctOut.size(), cudaMemcpyHostToDevice);

        int blockSize = layerSize[size - 1];
        int gridSize = 1;
        if (layerSize[size - 1] > MAX_BLOCK_SIZE) {
            blockSize = MAX_BLOCK_SIZE;
            gridSize = (layerSize[size - 1] + blockSize - 1) / blockSize;
        }

        if (ActivationFunction[size - 1] == SIGMOID_NAME) {
            get_last_layer_delta_kernel<<<gridSize,blockSize,0,stream>>>(layers[size - 1], layersZ[size - 1], correctOutD,delta[size-1],sigmoidP());
        } else if (ActivationFunction[size - 1] == RULE_NAME) {
            get_last_layer_delta_kernel<<<gridSize,blockSize,0,stream>>>(layers[size - 1], layersZ[size - 1], correctOutD,delta[size-1],ReLUP());
        }

        for (int i = size - 2; i > 0; i--) {
            blockSize = layerSize[i];
            gridSize = 1;
            if (layerSize[i] > MAX_BLOCK_SIZE) {
                blockSize = MAX_BLOCK_SIZE;
                gridSize = (layerSize[i] + blockSize - 1) / blockSize;
            }
            if (ActivationFunction[i] == SIGMOID_NAME) {
                back_propagate_delta_kernel<<<gridSize,blockSize,0,stream>>>(
                    delta[i], delta[i + 1],w[i], layersZ[i], sigmoidP());
            } else if (ActivationFunction[i] == RULE_NAME) {
                back_propagate_delta_kernel<<<gridSize,blockSize,0,stream>>>(
                    delta[i], delta[i + 1],w[i], layersZ[i], ReLUP());
            }
        }

        for (int i = 0; i < size -1;i++) {
            assert(layerSize[i] == w[i].width);
            assert(layerSize[i + 1] == w[i].height);
            blockSize = layerSize[i+1];
            gridSize = 1;
            if (layerSize[i+1] > MAX_BLOCK_SIZE) {
                blockSize = MAX_BLOCK_SIZE;
                gridSize = (layerSize[i+1] + blockSize - 1) / blockSize;
            }
            update_weights_kernel<<<blockSize,gridSize>>>(w[i], b[i+1], delta[i+1], layers[i], studyRate);
        }

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(correctOutD);

        return 0;
    }

    float NNCore::train(const vector<vector<float> > &inNums, const vector<int> &correctOut, bool getAcc) {
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

    float NNCore::test(const vector<vector<float> > &inNums, const vector<int> &correctOut) {
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

    void NNCore::printLayers() {
    }

    void NNCore::printLayers(const NNCore &nn) {
    }

    void NNCore::printW(int layerNumberToPrint) {
    }

    void NNCore::printW(const NNCore &nn, int layerNumberToPrint) {
    }

    int NNCore::choice() {
        double max = 0;
        int res = 0;
        for (int i = 0; i < layerSize[size - 1]; i++) {
            if (layers[size - 1].elements[i] > max) {
                max = layers[size - 1].elements[i];
                res = i;
            }
        }
        return res;
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
