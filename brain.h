//
// Created by elie on 12.03.18.
//

#ifndef NEURALNETWORK_BRAIN_H
#define NEURALNETWORK_BRAIN_H

#include <string>
#include <iostream>
#include <vector>
#include "utilities.h"
#include <list>
#include <map>
#include <sstream>
#include <random>
#include "neuron.h"
#include "synapse.h"
#include <typeinfo>

using namespace std;

namespace cx {
    class brain {
    private:
        void create_synapses();

    public:
        vector<int> expected_output_values;

        brain(int in_size, int out_size, int nb_hidden_layers, int hidden_layer_size, bool with_bias);

        map<int, vector<neuron>> layers;

        void load(data_holder &test_data_holder, bool ignore_weights);

        data_holder unload();

        map<string, double> actualWeights();

        void update_value(const string &neuron_id, double val);
    };
}

#endif //NEURALNETWORK_BRAIN_H
