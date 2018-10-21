//
// Created by elie on 21.10.18.
//

#ifndef NEURALNETWORK_CONFIGREADER_H
#define NEURALNETWORK_CONFIGREADER_H

#include <string>
#include <iostream>
#include <vector>
#include "utilities.h"
#include <list>
#include <map>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

// Short alias for this namespace
namespace pt = boost::property_tree;
using namespace std;

namespace cx {
    class config_reader {
    public:
        explicit config_reader(const string &properties_file);

        method_type method;
        vector<int> hidden_layers_data;
        int batch_size;
        string test_file;
        string training_file;
        float accuracy;
        int max_nb_iterations;
        int output_size;
        float learning_rate;
        int input_size;
        bool with_bias;
    private:
        template<typename T>
        std::vector<T> as_vector(pt::ptree const &pt, pt::ptree::key_type const &key);
    };
}

#endif //NEURALNETWORK_CONFIGREADER_H
