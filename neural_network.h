//
// Created by elie on 07.03.18.
//

#ifndef NEURALNETWORK_NEURAL_NETWORK_H
#define NEURALNETWORK_NEURAL_NETWORK_H

#include <string>
#include <iostream>
#include <vector>
#include "utilities.h"
#include <list>
#include <map>
#include <sstream>
#include <climits>
#include "brain.h"
#include "data_holder.h"

using namespace std;

namespace cx {
    class neural_network {
    public:
        neural_network();

        neural_network(bool with_bias, double learning_rate, method_type meth_type, int input_size, int output_size,
                       vector<int> hidden_layers_data, float accuracy);

        void initialize_data(vector<map<value_type, vector<float>>> data);

        long think();

        long think(long max_nb_iterations);

        void guess();

        int batch_size = 1;

    private:
        brain current_brain = brain(0, 0, vector<int>(), false);

        bool break_on_epoc = false;

        thinking_result think_batch(long max_nb_iterations);

        thinking_result think_sgd(long max_nb_iterations);

        thinking_result think_minibatch(long max_nb_iterations);

        bool not_all_true(vector<bool> states);

        bool values_matching(vector<neuron> neurons, vector<float> expected_values);

        void eval_fwd_propagation();

        map<string, double> delta_weights(map<string, vector<double>> gradients);

        void update_weights(map<string, double> deltas);

        map<string, vector<double>> eval_gradients();

        float match_range;
        double learning_rate;
        int current_iteration;
        method_type meth_type;
        vector<data_holder> training_data;
    };
}

#endif //NEURALNETWORK_NEURAL_NETWORK_H
