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

        bool not_all_true(vector<bool> states);

        bool values_matching(vector<neuron> neurons, vector<int> expected_values);

        void eval_fwd_propagation();

        map<string, double> delta_weights(map<string, vector<double>> gradients);

        void update_weights(map<string, double> deltas);

        map<string, vector<double>> eval_gradients();

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
                for (synapse synapse : source.getOutgoingSynapse()) {
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
        this->match_range = 0.2;
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
            dataHolder.setWeights(this->current_brain.actualWeights());
            training_data.push_back(dataHolder);
        }
    }

    int neural_network::think() {
        return think(LONG_MAX);
    }

    int neural_network::think(long max_nb_iterations) {
        switch (meth_type) {
            case SGD:
                return think_sgd(max_nb_iterations);
            case BATCH:
                return think_batch(max_nb_iterations);
            default:
                return think_sgd(max_nb_iterations);
        }
    }

    void neural_network::guess() {
        eval_fwd_propagation();
    }

    bool neural_network::not_all_true(vector<bool> states) {
        bool result = true;

        for (int i = 0; i < states.size(); i++) {
            result &= states[i];
        }

        return !result;
    }

    bool neural_network::values_matching(vector<neuron> neurons, vector<int> expected_values) {
        bool result = true;

        for (int i = 0; i < neurons.size(); i++) {
            neuron n = neurons.at(i);
            result &= (abs(expected_values.at(i) - n.activationValue())) <= match_range;
        }

        return result;
    }

    int neural_network::think_batch(long max_nb_iterations) {
        //TODO: still needs to be migrated
        return 0;
    }

    int neural_network::think_sgd(long max_nb_iterations) {
        vector<bool> instanceState;
        for (int u = 0; u < training_data.size(); u++) {
            instanceState.push_back(false);
        }
        while (not_all_true(instanceState) && current_iteration < max_nb_iterations) {
            current_iteration++;
            for (int u = 0; u < training_data.size(); u++) {
                current_brain.load(training_data.at(u), true);
                eval_fwd_propagation();
                map<string, vector<double>> gradients = eval_gradients();
                auto d_weights = delta_weights(gradients);
                update_weights(d_weights);
                instanceState[u] = values_matching(current_brain.getOutputs(),
                                                   current_brain.getExpectedOutputValues());
            }
        }
        return current_iteration;
    }

    void neural_network::eval_fwd_propagation() {
        for (int i = 1; i < current_brain.getLayers().size(); i++) {
            for (unsigned long j = 0; j < current_brain.getLayerAt(i).size(); j++) {
                neuron &hidden_neuron = current_brain.getLayerAt(i).at(j);
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

    map<string, double> neural_network::delta_weights(map<string, vector<double>> gradients) {
        map<string, double> deltasWeight;
        for (int i = current_brain.getLayers().size() - 2; i >= 0; i--) {
            for (int j = 0; j < current_brain.getLayerAt(i).size(); j++) {
                neuron neuron_instance = current_brain.getLayerAt(i).at(j);
                vector<double> values = gradients.at("hidden_" + to_string(i + 1));
                for (int s = 0; s < neuron_instance.getOutgoingSynapse().size(); s++) {
                    synapse synapse_instance = neuron_instance.getOutgoingSynapse().at(s);
                    double delta_weight = values[s] * neuron_instance.activationValue();
                    deltasWeight.insert(pair<string, double>(synapse_instance.getId(), delta_weight));
                }
            }
        }
        return deltasWeight;
    }

    void neural_network::update_weights(map<string, double> deltas) {
        for (int i = current_brain.getLayers().size() - 2; i >= 0; i--) {
            for (int j = 0; j < current_brain.getLayerAt(i).size(); j++) {
                neuron &neuron_instance = current_brain.getLayerAt(i).at(j);
                for (int s = 0; s < neuron_instance.getOutgoingSynapse().size(); s++) {
                    synapse &synapse_instance = neuron_instance.getOutgoingSynapse().at(s);
                    double weight =
                            synapse_instance.getWeight() - (learning_rate * deltas.at(synapse_instance.getId()));
                    synapse_instance.setWeight(weight);
                }
            }
        }
    }

    map<string, vector<double>> neural_network::eval_gradients() {
        map<string, vector<double>> deltas = map<string, vector<double>>();
        for (int i = current_brain.getLayers().size() - 1; i > 0; i--) {
            vector<double> deltaHiddenSum = vector<double>();
            for (int z = 0; z < current_brain.getLayerAt(i).size(); z++) {
                neuron neuron_instance = current_brain.getLayerAt(i).at(z);

                if (i == (current_brain.getLayers().size() - 1)) {
                    deltaHiddenSum.push_back(
                            (neuron_instance.activationValue() - current_brain.getExpectedOutputValues().at(z)) *
                            neuron_instance.activationPrimeValue());
                } else {
                    double dhs = 0;
                    for (int j = 0; j < neuron_instance.getOutgoingSynapse().size(); j++) {
                        synapse synapse_instance = neuron_instance.getOutgoingSynapse().at(j);
                        dhs += deltas.at("hidden_" + to_string(i + 1))[j] * synapse_instance.getWeight();
                    }
                    deltaHiddenSum.push_back(dhs * neuron_instance.activationPrimeValue());
                }
            }
            string label = "hidden_" + to_string(i);
            deltas.insert(pair<string, vector<double>>(label, deltaHiddenSum));
        }
        return deltas;
    }
}

#endif //NEURALNETWORK_NEURAL_NETWORK_H
