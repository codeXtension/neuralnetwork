cmake_minimum_required(VERSION 3.12)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(neuralnetwork
        brain.h
        cifar10_reader.h
        main.cpp
        neural_network.h
        neurals.h
        neuron.h
        synapse.h
        utilities.h)
