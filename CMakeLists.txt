cmake_minimum_required(VERSION 3.9)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 17)

add_executable(neuralnetwork utilities.h neural_network.h main.cpp neuron.h brain.h cifar10_reader.h synapse.cpp synapse.h neuron.cpp brain.cpp neural_network.cpp utilities.cpp data_holder.cpp data_holder.h)