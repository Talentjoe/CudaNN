cmake_minimum_required(VERSION 3.30)
project(cudaNN LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(cudaNN main.cu ./lib/readData.h ./lib/readData.cpp
        lib/NNCore.cu
        lib/NNCore.cuh
        lib/Vector.cu
        lib/Matrix.cu
        lib/CudaFunctions.cu
        lib/Vector.cuh
        lib/Matrix.cuh
        lib/CudaFunctions.cuh)

set_target_properties(cudaNN PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

file(COPY ${CMAKE_SOURCE_DIR}/Data DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/test.mod DESTINATION ${CMAKE_BINARY_DIR})