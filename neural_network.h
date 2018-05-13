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
#include "synapse.h"
#include "neuron.h"

using namespace std;

namespace cx {
    class neural_network {
    public:
        void log_weights(brain value);

        neural_network(bool with_bias, double learning_rate, method_type meth_type, int input_size, int output_size,
                       int nb_hidden_layers, int size_hidden_layer);

        void initialize_data(vector<map<value_type, vector<int>>> data);

        int think();

        int think(long max_nb_iterations);

        void guess();

    private:
        brain current_brain = brain(0, 0, 0, 0, false);

        int think_batch(long max_nb_iterations);

        int think_sgd(long max_nb_iterations);

        bool not_all_true(map<int, bool> states);

        bool values_matching(vector<neuron> neurons, vector<double> expected_values);

        void eval_fwd_propagation(brain &brain_instance);

        map<string, double> delta_weights(brain brain_instance, map<string, double[]> gradients);

        void update_weights(brain &brain_instance, map<string, double> deltas);

        map<string, vector<double>> gradients(brain &brain_instance);

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
            vector<neuron> sources = value.getLayers().at(i);
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

    neural_network::neural_network(bool with_bias, double learning_rate, method_type meth_type, int input_size,
                                   int output_size, int nb_hidden_layers, int size_hidden_layer) {
        this->current_iteration = 0;
        this->training_data = {};
        this->match_range = 0.1;
        this->meth_type = meth_type;
        this->with_bias = with_bias;
        this->nb_hidden_layers = nb_hidden_layers;
        this->size_hidden_layer = size_hidden_layer;
        this->learning_rate = learning_rate;
        current_brain = brain(input_size, output_size, nb_hidden_layers, size_hidden_layer, with_bias);

        log_weights(current_brain);
    }

    void neural_network::initialize_data(vector<map<value_type, vector<int>>> data) {
        training_data.clear();
        for (map<value_type, vector<int>> instance : data) {
            data_holder dataHolder;
            dataHolder.add_input(instance.at(INPUT));
            dataHolder.setExpected_outputs(instance.at(OUTPUT));
            dataHolder.setWeights(this->current_brain.actual_weights());
            training_data.push_back(dataHolder);
        }
    }

    int neural_network::think() {
        return think(LONG_MAX);
    }

    int neural_network::think(long max_nb_iterations) {
        switch (meth_type) {
            case SGD:
                return think_sgd(LONG_MAX);
            case BATCH:
                return think_batch(LONG_MAX);
            default:
                return think_sgd(LONG_MAX);
        }
    }

    void neural_network::guess() {
        eval_fwd_propagation(current_brain);
    }

    bool neural_network::not_all_true(map<int, bool> states) {
        bool result = true;

        for (int i = 0; i < states.size(); i++) {
            result &= states.at(i);
        }

        return result;
    }

    bool neural_network::values_matching(vector<neuron> neurons, vector<double> expected_values) {
        bool result = true;

        for (unsigned long i = 0; i < neurons.size(); i++) {
            neuron n = neurons.at(i);
            result &= (abs(expected_values.at(i) - n.activationValue())) <= match_range;
        }

        return result;
    }

    int neural_network::think_batch(long max_nb_iterations) {
        return 0;
    }

    int neural_network::think_sgd(long max_nb_iterations) {
        return 0;
    }

    void neural_network::eval_fwd_propagation(brain &brain_instance) {
        for (int i = 1; i < current_brain.getLayers().size(); i++) {
            for (unsigned long j = 0; j < current_brain.get_layer(i).size(); j++) {
                neuron hidden_neuron = current_brain.get_layer(i).at(j);
                double value = 0.0;
                if (hidden_neuron.getId().find("BN") == string::npos) {
                    for (synapse synapse_instance : hidden_neuron.getIncoming_synapse()) {
                        value += synapse_instance.getWeight() * synapse_instance.getSource()->activationValue();
                    }
                    hidden_neuron.setValue(value);
                }
            }
        }
    }

    map<string, double> neural_network::delta_weights(brain brain_instance, map<string, double[]> gradients) {
        return map<string, double>();
    }

    void neural_network::update_weights(brain &brain_instance, map<string, double> deltas) {

    }

    map<string, vector<double>> neural_network::gradients(brain &brain_instance) {
        return map<string, vector<double>>();
    }
}

#endif //NEURALNETWORK_NEURAL_NETWORK_H
