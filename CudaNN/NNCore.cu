//
// Created by lenovo on 25-1-6.
//

#include "NNCore.cuh"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <algorithm>

#include "CudaFunctions.cuh"

namespace NN {
#define uint unsigned int
#define SIGMOID_NAME "sigmoid"
#define RULE_NAME "ReLU"

    using namespace std;



    struct sigmoid {
        __device__ float operator()(float x) const {
            return 1 / (1 + expf(-x));
        }
    };

    struct sigmoidP {
        __device__ float operator()(float x) const {
            float t = 1 / (1 + expf(-x));
            return t * (1 - t);
        }
    };

    struct ReLU {
        __device__ float operator()(float x) const {
            return max(x, 0.0f);
        }
    };

    struct ReLUP {
        __device__ float operator()(float x) const {
            return x > 0 ? 1 : 0.01;
        }
    };

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
        ActivationFunction = vector<string>(size);
        for (int i = 0; i < size; i++) {
            inFile >> layerSize[i];
        }
        for (int i = 1; i < size; i++) {
            inFile >> ActivationFunction[i];
        }

        layers = new Vector[size];
        layersZ = new Vector[size];
        b = new Vector[size];
        w = new Matrix[size - 1];
        delta = new Vector[size];

        for (int i = 0; i < size; ++i) {
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
                w[i].initRand(lim, -lim);
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

            if (ActivationFunction[i + 1] == SIGMOID_NAME) {
                activate_kernel<<<gridSize,blockSize,0,stream>>>(layersZ[i + 1], layers[i + 1], sigmoid());
            } else if (ActivationFunction[i + 1] == RULE_NAME) {
                activate_kernel<<<gridSize,blockSize,0,stream>>>(layersZ[i + 1], layers[i + 1], ReLU());
            } else
                throw std::runtime_error("Activation function not supported");

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
            get_last_layer_delta_kernel<<<gridSize,blockSize,0,stream>>>(
                layers[size - 1], layersZ[size - 1], correctOutD, delta[size - 1], sigmoidP());
        } else if (ActivationFunction[size - 1] == RULE_NAME) {
            get_last_layer_delta_kernel<<<gridSize,blockSize,0,stream>>>(
                layers[size - 1], layersZ[size - 1], correctOutD, delta[size - 1], ReLUP());
        } else
            throw std::runtime_error("Activation function not supported");

        for (int i = size - 2; i > 0; i--) {
            blockSize = layerSize[i];
            gridSize = 1;
            if (layerSize[i] > MAX_BLOCK_SIZE) {
                blockSize = MAX_BLOCK_SIZE;
                gridSize = (layerSize[i] + blockSize - 1) / blockSize;
            }
            if (ActivationFunction[i] == SIGMOID_NAME) {
                back_propagate_delta_kernel<<<gridSize,blockSize,0,stream>>>(
                    delta[i], delta[i + 1], w[i], layersZ[i], sigmoidP());
            } else if (ActivationFunction[i] == RULE_NAME) {
                back_propagate_delta_kernel<<<gridSize,blockSize,0,stream>>>(
                    delta[i], delta[i + 1], w[i], layersZ[i], ReLUP());
            } else
                throw std::runtime_error("Activation function not supported");
        }

        for (int i = 0; i < size - 1; i++) {
            blockSize = layerSize[i + 1];
            gridSize = 1;
            if (layerSize[i + 1] > MAX_BLOCK_SIZE) {
                blockSize = MAX_BLOCK_SIZE;
                gridSize = (layerSize[i + 1] + blockSize - 1) / blockSize;
            }
            update_weights_kernel<<<blockSize,gridSize>>>(w[i], b[i + 1], delta[i + 1], layers[i], studyRate);
        }

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        cudaFree(correctOutD);

        return 0;
    }

    float NNCore::train_with_retrain(const vector<vector<float> > &inNums, const vector<int> &correctOut, std::vector<std::vector<float>> &wrongAns,std::vector<int> &correctAns, bool getAcc) {
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
                    wrongAns.push_back(inNums[i]);
                    correctAns.push_back(correctOut[i]);
                    wrongCnt++;
                }
            }

            if (i % 1000 == 0) {
                cout << "\rProgress: " << setw(7)<< i / (float) inNums.size() * 100 << "%";
                if (getAcc) {
                    cout << " Correct Percentage: "<< setw(7) << corrctCnt / (float) (corrctCnt + wrongCnt) * 100 << "%";
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
                cout << "\rProgress: " << setw(7)<< i / (float) inNums.size() * 100 << "%";
                if (getAcc) {
                    cout << " Correct Percentage: "<< setw(7) << corrctCnt / (float) (corrctCnt + wrongCnt) * 100 << "%";
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
                cout << "\rProgress: " << setw(7)<< i / (float) inNums.size() * 100 << "%";
                cout << " Correct Percentage: " << setw(7)<< corrctCnt / (float) (corrctCnt + wrongCnt) * 100 << "%";
                cout << "                    " << flush;
            }
        }
        cout << endl;
        cout << "Finish Testing " << inNums.size() << " Data" << endl;

        cout << "With Accuracy: " << corrctCnt / (float) (corrctCnt + wrongCnt) * 100 << "%" << endl;
        return corrctCnt / (float)(corrctCnt + wrongCnt);
    }

    float NNCore::test_with_wrong(const vector<vector<float> > &inNums, const vector<int> &correctOut, std::vector<std::vector<float>> &wrongAns,std::vector<int> &correctAns) {
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
                correctAns.push_back(correctOut[i]);
                wrongAns.push_back(inNums[i]);
                wrongCnt++;
            }

            if (i % 1000 == 0) {
                cout << "\rProgress: " << setw(7)<< i / (float) inNums.size() * 100 << "%";
                cout << " Correct Percentage: " << setw(7)<< corrctCnt / (float) (corrctCnt + wrongCnt) * 100 << "%";
                cout << "                    " << flush;
            }
        }
        cout << endl;
        cout << "Finish Testing " << inNums.size() << " Data" << endl;

        cout << "With Accuracy: " << corrctCnt / (float) (corrctCnt + wrongCnt) * 100 << "%" << endl;
        return corrctCnt / (float)(corrctCnt + wrongCnt);
    }


    float NNCore::CalCost(vector<float> correctOut) {
        return 0;
    }

    void NNCore::printLayers() {
        for (int i = 0; i < size - 1; i++) {
            cout << "Layer Value" << i << ": " << endl;
            layers[i].cpDtoH();
            layers[i].printVec();
        }
    }

    void NNCore::printLayers(const NNCore &nn) {
        for (int i = 0; i < nn.size - 1; i++) {
            cout << "Layer Value" << i << ": " << endl;
            nn.layers[i].cpDtoH();
            nn.layers[i].printVec();
        }
    }

    void NNCore::printW(int layerNumberToPrint) {
        w[layerNumberToPrint].cpDtoH();
        w[layerNumberToPrint].printMat();
    }

    void NNCore::printW(const NNCore &nn, int layerNumberToPrint) {
        nn.w[layerNumberToPrint].cpDtoH();
        nn.w[layerNumberToPrint].printMat();
    }

    int NNCore::choice() {
        layers[size - 1].cpDtoH();
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
        studyRate = rate;
    }

    void NNCore::save(string path) {
        ofstream outFile(path);
        if (!outFile.is_open()) {
            cout << "error" << endl;
            return;
        }

        outFile << size << endl;

        for (int i = 0; i < size; i++) {
            outFile << layerSize[i] << " ";
        }
        outFile << endl;
        for (int i = 1; i < size; i++) {
            outFile << ActivationFunction[i] << " ";
        }
        outFile << endl;

        for (int i = 1; i < size; i++) {
            b[i].cpDtoH();
            //need to be changed based on the activation function
            for (int j = 0; j < layerSize[i]; j++) {
                outFile << b[i].elements[j] << " ";
            }
            outFile << endl;
        }

        for (int i = 0; i < size - 1; i++) {
            w[i].cpDtoH();
            for (int k = 0; k < w[i].width; k++) {
                for (int j = 0; j < w[i].height; j++) {
                    outFile << w[i].elements[j * w[i].width + k] << " ";
                }
                outFile << endl;
            }
            outFile << endl;
        }
    }
} // NN
