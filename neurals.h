//
// Created by elie on 17.06.18.
//

#ifndef NEURALNETWORK_NEURALS_H
#define NEURALNETWORK_NEURALS_H

#include <string>
#include <iostream>
#include <vector>
#include "neural_network.h"
#include <list>
#include <map>
#include <sstream>

using namespace std;

namespace cx {

    neuron::neuron() {
    }

    bool neuron::operator==(neuron rhs) {
        return id == rhs.id;
    }

    bool neuron::operator!=(neuron rhs) {
        return !(rhs == *this);
    }

    double neuron::activationPrimeValue() {
        if (!this->incoming_synapse.empty()) {
            return derivativeSigmoid(value);
        } else {
            return value;
        }
    }

    double neuron::activationValue() {
        if (!this->incoming_synapse.empty()) {
            return sigmoid(value);
        } else {
            return value;
        }
    }

    neuron::neuron(string id) {
        this->id = id;
        cout << "Neuron created with id " << id << endl;
    }

    void neuron::addOutgoingSynapse(synapse *pSynapse) {
        this->outgoing_synapse.push_back(*pSynapse);
    }

    void neuron::addIncomingSynapse(synapse *pSynapse) {
        this->incoming_synapse.push_back(*pSynapse);
    }

    bool synapse::operator==(synapse rhs) {
        return id == rhs.id;
    }

    bool synapse::operator!=(synapse rhs) {
        return rhs.id != this->id;
    }

    synapse::synapse(double weight, neuron *source, neuron *target) {
        this->id = source->id + "-" + target->id;
        this->weight = weight;
        this->source = source;
        this->target = target;
    }

    void neural_network::log_weights(brain value) {
        for (int i = 0; i < value.layers.size() - 1; i++) {
            vector<neuron> sources = value.layers[i];
            cout << "BRAIN - Synapses from layer " << (i + 1) << " --> " << (i + 2) << endl;
            for (neuron source : sources) {
                for (synapse synapse : source.outgoing_synapse) {
                    neuron *target = synapse.target;
                    cout << "BRAIN - " << source.id << " [" << source.value << "] ---" << synapse.weight
                         << "---> " << target->id << " [" << target->value << "] a("
                         << target->activationValue() << ")"
                         << endl;
                }
            }
        }
        cout << endl;
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

        //log_weights(current_brain);
    }

    void neural_network::initialize_data(vector<map<value_type, vector<int>>> data) {
        training_data.clear();
        for (map<value_type, vector<int>> instance : data) {
            data_holder dataHolder;
            dataHolder.add_input(instance.at(INPUT));
            dataHolder.expected_outputs = instance.at(OUTPUT);
            dataHolder.weights = this->current_brain.actualWeights();
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

        cout << "Current states: ";
        for (auto &&state : states) {
            result &= state;
            cout << state;
        }
        cout << endl;

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
            cout << "running iteration " << current_iteration << endl;
            for (int u = 0; u < training_data.size(); u++) {
                cout << "running iteration " << current_iteration << " - training data " << u + 1 << endl;
                current_brain.load(training_data.at(u), true);
                eval_fwd_propagation();
                map<string, vector<double>> gradients = eval_gradients();
                map<string, double> d_weights = delta_weights(gradients);
                update_weights(d_weights);
                this->log_weights(current_brain);
                instanceState[u] = values_matching(current_brain.layers[current_brain.layers.size() - 1],
                                                   current_brain.expected_output_values);
            }
        }
        return current_iteration;
    }

    void neural_network::eval_fwd_propagation() {
        for (int i = 1; i < current_brain.layers.size(); i++) {
            cout << "FWD - Reading layer " << i + 1 << endl;
            for (neuron &hidden_neuron : current_brain.layers[i]) {
                double value = 0.0;
                if (hidden_neuron.id.find("BN") == string::npos) {
                    cout << "FWD - Reading " << hidden_neuron.id << " in layer " << i + 1 << endl;
                    for (synapse synapse_instance : hidden_neuron.incoming_synapse) {
                        cout << "FWD - Incrementing " << hidden_neuron.id << " value [v=v_old+w("
                             << synapse_instance.id << ")*a(" << synapse_instance.source->id
                             << ")] --> v=" << value << "+" << synapse_instance.weight << "*"
                             << synapse_instance.source->activationValue() << endl;
                        value += synapse_instance.weight * synapse_instance.source->activationValue();
                    }
                    hidden_neuron.value = value;
                    current_brain.update_value(hidden_neuron.id, value);
                    cout << "FWD - Final " << hidden_neuron.id << " value is " << value << " (a="
                         << hidden_neuron.activationValue() << ")" << endl;
                }
            }
        }
    }

    map<string, double> neural_network::delta_weights(map<string, vector<double>> gradients) {
        map<string, double> deltasWeight;
        for (int i = current_brain.layers.size() - 2; i >= 0; i--) {
            for (int j = 0; j < current_brain.layers[i].size(); j++) {
                neuron neuron_instance = current_brain.layers[i][j];
                vector<double> values = gradients.at("hidden_" + to_string(i + 1));
                for (int s = 0; s < neuron_instance.outgoing_synapse.size(); s++) {
                    synapse synapse_instance = neuron_instance.outgoing_synapse[s];
                    double delta_weight = values[s] * neuron_instance.activationValue();
                    deltasWeight.insert(pair<string, double>(synapse_instance.id, delta_weight));
                }
            }
        }
        return deltasWeight;
    }

    void neural_network::update_weights(map<string, double> deltas) {
        for (int i = current_brain.layers.size() - 2; i >= 0; i--) {
            cout << "BACK - WEIGHT - Updating weights for layer " << i + 1 << endl;
            for (int j = 0; j < current_brain.layers[i].size(); j++) {
                neuron neuron_instance = current_brain.layers[i][j];
                for (int s = 0; s < neuron_instance.outgoing_synapse.size(); s++) {
                    synapse synapse_instance = neuron_instance.outgoing_synapse[s];
                    double weight =
                            synapse_instance.weight - (learning_rate * deltas.at(synapse_instance.id));
                    synapse_instance.weight = weight;
                    current_brain.layers[i][j].outgoing_synapse[s] = synapse_instance;

                    cout << "BACK - DELTA_WEIGHT - DW[" << synapse_instance.id << "] = Delta["
                         << deltas.at(synapse_instance.id) << "] * Act" << neuron_instance.id << "["
                         << neuron_instance.activationValue() << "]" << endl;
                    cout << "BACK - WEIGHT - Synapse " << synapse_instance.id << " new weight is OLDW("
                         << synapse_instance.weight << ")-(LR(" << learning_rate << ")*DW("
                         << deltas.at(synapse_instance.id) << ")) => " << weight << endl;
                }
            }
        }
    }

    map<string, vector<double>> neural_network::eval_gradients() {
        map<string, vector<double>> deltas = map<string, vector<double>>();
        for (int i = current_brain.layers.size() - 1; i > 0; i--) {
            vector<double> deltaHiddenSum = vector<double>();
            for (int z = 0; z < current_brain.layers[i].size(); z++) {
                neuron neuron_instance = current_brain.layers[i].at(z);

                if (i == (current_brain.layers.size() - 1)) {
                    deltaHiddenSum.push_back(
                            (neuron_instance.activationValue() - current_brain.expected_output_values[z]) *
                            neuron_instance.activationPrimeValue());
                } else {
                    double dhs = 0;
                    for (int j = 0; j < neuron_instance.outgoing_synapse.size(); j++) {
                        synapse synapse_instance = neuron_instance.outgoing_synapse.at(j);
                        dhs += deltas.at("hidden_" + to_string(i + 1))[j] * synapse_instance.weight;
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
#endif //NEURALNETWORK_NEURALS_H
