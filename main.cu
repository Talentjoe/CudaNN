#include <cstdint>
#include <ctime>
#include <iostream>

#include "./lib/NNCore.cuh"
#include "./lib/readData.h"

using namespace std;


int main() {
    NN::Vector inNums;

    inNums.resize(100);

    inNums.initRandom();
    inNums.cpDtoH();

    for (int i = 0 ;i < 100; i++) {
        cout << i <<": "<< inNums.elements[i] << endl;
    }

    return 0;
}
