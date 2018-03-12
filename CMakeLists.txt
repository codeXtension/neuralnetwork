cmake_minimum_required(VERSION 3.9)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(neuralnetwork utilities.h neural_network.h main.cpp)