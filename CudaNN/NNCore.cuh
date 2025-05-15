#ifndef NNCORE_CUH
#define NNCORE_CUH


#include <vector>
#include <string>
#include "Matrix.cuh"
#include "Vector.cuh"


namespace NN {
#define BLOCK_SIZE_MATRIX 32
#define BLOCK_SIZE_VECTOR 1024
#define MAX_BLOCK_SIZE 1024
#define RAND_SEED 1234

    [[deprecated("This function is deprecated, please use the new version")]]
    inline static float getRandomFloatNumber(float max = 1, float min = -1) {
        return min + static_cast<float>(rand()) / (RAND_MAX / (max - min));
    }

    class NNCore {
        unsigned long long size; // size of layers
        float studyRate; // study rate

        std::vector<int> layerSize; // size of each layer
    public:
        Vector *layersZ; // value of each layer before activation function
        Vector *layers; // value of each layer
        Vector *b; // bias
        Matrix *w; // weight
        Vector *delta; // delta of each layer


    public:
        struct LayerStructure {
            int layerSize;
            std::string activationFunction;
        };

        float heLimit(int fan_in) {
            return sqrt(6.0 / fan_in);
        }

        std::vector<std::string> ActivationFunction;


        /**
         * Initialize the NN with the given path, load the framework from the path
         * @param path the target framework path
         * @param studyRate the study rate
         */
        NNCore(const std::string &path, float studyRate);

        /**
         * Initialize the NN with the given layer size, study rate and dropout rate
         * @param LayerS the size of each layer, each number indicates the size of the layer
         * @param studyR the study rate
         */
        NNCore(const std::vector<LayerStructure> &LayerS, float studyR);

        ~NNCore();

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
        float train(const std::vector<std::vector<float> > &inNums, const std::vector<int> &correctOut, bool getAcc);
        float train_with_retrain(const std::vector<std::vector<float> > &inNums, const std::vector<int> &correctOut, std::vector<std::vector<float>> &wrongAns,std::vector<int> &correctAns, bool getAcc = false);

        /**
         * Test the NN with the given data
         * @param inNums input data, each element in the outer vector is a piece of data,
         * and each element in the inner vector is a feature, the size of the feature
         * should be equal to the size of the first layer
         * @param correctOut correct output data, each element in the vector is a piece of
         * data, and the value should be less than the size of the last layer
         * @return the accuracy of the test
         */
        float test(const std::vector<std::vector<float> > &inNums, const std::vector<int> &correctOut);
        float test_with_wrong(const std::vector<std::vector<float> > &inNums, const std::vector<int> &correctOut, std::vector<std::vector<float>> &wrongAns,std::vector<int> &correctAns);

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
        float backpropagation(const std::vector<float>& correctOut);

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
        static void printLayers(const NNCore &nn);

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
        static void printW(const NNCore &nn, int layerNumberToPrint);

        /**
         * Save the nn framework to the given path
         * @param nn the framework to save
         * @param path the path to save
         */
        void save(std::string path);
    };
}


#endif //NNCORE_CUH
