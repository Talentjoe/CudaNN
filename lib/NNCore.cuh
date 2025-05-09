//
// Created by lenovo on 2025/4/20.
//

#ifndef NNCORE_CUH
#define NNCORE_CUH


#include <vector>
#include <string>
#include <functional>

namespace NN {
    class NNcore {
        int size; // size of layers
        float studyRate; // study rate
        bool addDropout; // add dropout or not
        float dropOutRate; // dropout rate

        std::vector<std::vector<float> > layers; //value of each layer
        std::vector<std::vector<float> > layersZ; //value of each layer before sigmoid
        std::vector<int> layerSize; // size of each layer

        std::vector<std::vector<float> > b; // bias
        std::vector<std::vector<std::vector<float> > > w; // weight

    public:

        inline static float getRandomfloatNumber(float max = 1, float min = -1) {
            return min + static_cast<float>(rand()) / (RAND_MAX / (max - min));
        }

        inline static float sigmoid(float x) {
            return 1 / (1 + exp(-x));
        }

        inline static float sigmoidP(float x) {
            return sigmoid(x) * (1 - sigmoid(x));
        }

        std::function<float(float)> ActivationFunction;
        std::function<float(float)> ActivationFunctionP;

        /**
         * Start training process, modify current NN function
         * @param inNums input data, each element in the outer vector is a piece of data,
         * and each element in the inner vector is a feature, the size of the feature
         * should be equal to the size of the first layer
         * @param correctOut correct output data, each element in the vector is a piece of
         * data, and the value should be less than the size of the last layer
         * @param getAcc a bool which indicate whether to get the accuracy of the training
         * @return if getAcc is true, return the accuracy of the training, otherwise return -1
         */
        float train(std::vector<std::vector<float> > inNums, std::vector<int> correctOut, bool getAcc = false);

        /**
         * Test the NN with the given data
         * @param inNums input data, each element in the outer vector is a piece of data,
         * and each element in the inner vector is a feature, the size of the feature
         * should be equal to the size of the first layer
         * @param correctOut correct output data, each element in the vector is a piece of
         * data, and the value should be less than the size of the last layer
         * @return the accuracy of the test
         */
        float test(std::vector<std::vector<float> > inNums, std::vector<int> correctOut);

        /**
         * Forward propagation
         * @param inNums input data, each element in the vector is a feature, the size of the feature
         * @param printRes if true, print the running process
         * @return the output of the last layer
         */
        std::vector<float> forward(std::vector<float> inNums, bool printRes = false);

        /**
         * Back propagation, modify the current NN function. Should first do forward propagation which
         * gives the output of current framework and modify based on that
         * @param correctOut Expect output of the last layer, the size should be equal to the size of the last layer
         * @return the cost after modify framework
         */
        float backpropagation(std::vector<float> correctOut);

        /**
         * Calculate the cost of the current framework, with current data.
         * @param correctOut Expected output of the last layer, the size should be equal to the size of the last layer
         * @return the cost.
         */
        float CalCost(std::vector<float> correctOut);

        /**
         * Change the study rate of the NN
         * @param rate the new study rate
         */
        void changeStudyRate(float rate);

        /**
         * Change the dropout rate of the NN
         * @param rate the new dropout rate
         */
        void changeDropOutRate(float rate);

        /**
         * process the dropout, modify the current NN function
         */
        void dropSome();

        /**
         * Get the choice of the NN, which is the index of the output layer with the largest value
         * @return the choice number
         */
        int choice();

        /**
         * Print the weight of the given layer
         * @param layerNumberToPrint the layer number to print
         */
        void printLayers();

        /**
         * Print the weight of the given layer
         * @param nn the NN to print
         * @param layerNumberToPrint the layer number to print
         */
        static void printLayers(const NNcore &nn);

        /**
         * Print the weight of the given layer
         * @param layerNumberToPrint the layer number to print
         */
        void printW(int layerNumberToPrint);

        /**
         * Print the weight of the given layer
         * @param nn the NN to print
         * @param layerNumberToPrint the layer number to print
         */
        static void printW(const NNcore &nn, int layerNumberToPrint);

        /**
         * Save the nn framework to the given path
         * @param nn the framework to save
         * @param path the path to save
         */
        static void save(const NNcore &nn, std::string path);

        /**
         * Initialize the NN with the given path, load the framework from the path
         * @param path the target framework path
         * @param studyRate the study rate
         * @param drRate the dropout rate
         */
        void init(const std::string &path, float studyRate, float drRate = -1,
                  std::function<float(float)> activationFunction = sigmoid,
                  std::function<float(float)> activationFunctionP = sigmoidP);

        /**
         * Initialize the NN with the given layer size, study rate and dropout rate
         * @param LayerS the size of each layer, each number indicates the size of the layer
         * @param studyR the study rate
         * @param drRate the dropout rate
         */
        void init(const std::vector<int> &LayerS, float studyR, float drRate = -1,
                  std::function<float(float)> activationFunction = sigmoid,
                  std::function<float(float)> activationFunctionP = sigmoidP);
    };
}


#endif //NNCORE_CUH
