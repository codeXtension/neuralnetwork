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
#include "brain.h"
#include "synapse.h"
#include "neuron.h"

using namespace std;

namespace cx {
    class neural_network {
    public:
        void log_weights(brain value);
        neural_network(bool with_bias, double learning_rate, method_type meth_type, int input_size, int output_size, int nb_hidden_layers, int size_hidden_layer);
        void initialize_data(vector<map<value_type, vector<int>>> data);
    private:
        brain current_brain = brain(0, 0, 0, 0, false);
        double match_range;
        bool with_bias;
        double learning_rate;
        int current_iteration;
        method_type meth_type;
        vector<data_holder> training_data;
        int nb_hidden_layers;
        int size_hidden_layer;
    };

    void neural_network::log_weights(brain value) {
        for (int i = 0; i < value.getLayers().size() - 1; i++) {
            list<neuron> sources = value.getLayers().at(i);
            cout << "BRAIN - Synapses from layer " << (i + 1) << " --> " << (i + 2) << endl;
            for (neuron source : sources) {
                for (synapse synapse : source.getOutgoing_synapse()) {
                    neuron *target = synapse.getTarget();
                    cout << "BRAIN - " << source.getId() << " ---" << synapse.getWeight() << "---> " << target->getId()
                         << " a(" << target->activationValue() << ")" << endl;
                }
            }
        }
    }

    neural_network::neural_network(bool with_bias, double learning_rate, method_type meth_type, int input_size, int output_size, int nb_hidden_layers, int size_hidden_layer)
    {
        this->training_data = {};
        this->match_range =0.1;
        this->meth_type=meth_type;
        this->with_bias=with_bias;
        this->nb_hidden_layers=nb_hidden_layers;
        this->size_hidden_layer=size_hidden_layer;
        this->learning_rate=learning_rate;
        current_brain = brain(input_size, output_size, nb_hidden_layers, size_hidden_layer, with_bias);

        log_weights(current_brain);
    }

    void neural_network::initialize_data(vector<map<value_type, vector<int>>> data) {
        for (map<value_type , vector<int>> instance : data) {
            data_holder dataHolder;
            dataHolder.add_input(instance.at(INPUT));
            dataHolder.setExpected_outputs(instance.at(OUTPUT));
            dataHolder.setWeights(this->current_brain.actual_weights());
            training_data.push_back(dataHolder);
        }
    }
}

#endif //NEURALNETWORK_NEURAL_NETWORK_H
