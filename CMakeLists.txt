cmake_minimum_required(VERSION 3.30)
project(cudaNN LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_PREFIX_PATH "C:/Qt/6.9.0/msvc2022_64/")

find_package(Qt6 REQUIRED COMPONENTS Widgets)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

add_executable(cudaNN main.cpp
        lib/readData.cpp
        CudaNN/NNCore.cu
        CudaNN/Vector.cu
        CudaNN/Matrix.cu
        CudaNN/CudaFunctions.cu
        UI/mainwindow.cpp
        UI/mainwindow.h)
target_include_directories(cudaNN PRIVATE ${CMAKE_SOURCE_DIR}/lib)
target_include_directories(cudaNN PRIVATE ${CMAKE_SOURCE_DIR}/CudaNN)



set_target_properties(cudaNN PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

file(COPY ${CMAKE_SOURCE_DIR}/Data DESTINATION ${CMAKE_BINARY_DIR})
file(COPY ${CMAKE_SOURCE_DIR}/test.mod DESTINATION ${CMAKE_BINARY_DIR})
target_link_libraries(cudaNN PRIVATE Qt6::Widgets)