cmake_minimum_required(VERSION 3.9)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(neuralnetwork utilities.h neuralnetwork.h main.cpp)