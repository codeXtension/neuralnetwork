cmake_minimum_required(VERSION 3.12)
project(neuralnetwork)

set(CMAKE_CXX_STANDARD 17)

include_directories(.)

add_executable(neuralnetwork
        brain.cpp
        brain.h
        cifar10_reader.h
        data_holder.cpp
        data_holder.h
        main.cpp
        neural_network.cpp
        neural_network.h
        neuron.cpp
        neuron.h
        synapse.cpp
        synapse.h
        utilities.cpp
        utilities.h
        config_reader.cpp
        config_reader.h)
