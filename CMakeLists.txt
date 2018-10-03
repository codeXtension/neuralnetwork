cmake_minimum_required(VERSION 3.9)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")

add_executable(neuralnetwork utilities.h neural_network.h main.cpp synapse.h neuron.h brain.h neurals.h Log.h)