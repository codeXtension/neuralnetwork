//
// Created by elie on 17.06.18.
//

#ifndef NEURALNETWORK_NEURALS_H
#define NEURALNETWORK_NEURALS_H

#include <string>
#include <iostream>
#include <vector>
#include "neural_network.h"
#include "brain.h"
#include <limits>
#include <list>
#include <map>
#include <sstream>

using namespace std;

namespace cx {

    const float MATCH_RANGE = 0.1;

    bool neuron::operator==(neuron rhs) {
        return id == rhs.id;
    }

    bool neuron::operator!=(neuron rhs) {
        return !(rhs == *this);
    }

    double neuron::activationPrimeValue() {
        string::size_type bn = id.find("BN");
        string::size_type n1 = id.find("N1.");
        if (bn != string::npos || n1 != string::npos) {
            return value;
        } else {
            return derivativeSigmoid(value);
        }
    }

    double neuron::activationValue() {
        string::size_type bn = id.find("BN");
        string::size_type n1 = id.find("N1.");
        if (bn != string::npos || n1 != string::npos) {
            return value;
        } else {
            return sigmoid(value);
        }
    }

    neuron::neuron(const string &id) {
        this->id = id;
    }

    bool synapse::operator==(synapse rhs) {
        return id == rhs.id;
    }

    bool synapse::operator!=(synapse rhs) {
        return rhs.id != this->id;
    }

    synapse::synapse(const double &weight, const string &source_neuron_id, const string &target_neuron_id) {
        this->id = source_neuron_id + "-" + target_neuron_id;
        this->weight = weight;
        this->source_neuron_id = source_neuron_id;
        this->target_neuron_id = target_neuron_id;
    }

    neural_network::neural_network(bool with_bias, double learning_rate, method_type meth_type, int input_size,
                                   int output_size, int nb_hidden_layers, int size_hidden_layer) {
        this->current_iteration = 0;
        this->training_data = {};
        this->match_range = MATCH_RANGE;
        this->meth_type = meth_type;
        this->with_bias = with_bias;
        this->nb_hidden_layers = nb_hidden_layers;
        this->size_hidden_layer = size_hidden_layer;
        this->learning_rate = learning_rate;
        current_brain = brain(input_size, output_size, nb_hidden_layers, size_hidden_layer, with_bias);
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
        for (auto &&state : states) {
            result &= state;
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

        vector<bool> instanceState;
        for (int u = 0; u < training_data.size(); u++) {
            instanceState.push_back(false);
        }

        while (not_all_true(instanceState) && current_iteration < max_nb_iterations) {
            current_iteration++;
            for (int u = 0; u < training_data.size(); u++) {
                current_brain.load(training_data.at(u), true);
                eval_fwd_propagation();
                training_data[u] = current_brain.unload();
            }

            map<string, double> all_deltas;

            for (int u = 0; u < training_data.size(); u++) {
                current_brain.load(training_data.at(u), true);
                map<string, vector<double>> gradients = eval_gradients();
                map<string, double> d_weights = delta_weights(gradients);
                if (all_deltas.size() == 0) {
                    all_deltas.insert(d_weights.begin(), d_weights.end());
                } else {
                    map<string, double>::iterator it;
                    for (it = all_deltas.begin(); it != all_deltas.end(); it++) {
                        it->second += d_weights.at(it->first);
                    }
                }
            }

            map<string, double>::iterator it;
            for (it = all_deltas.begin(); it != all_deltas.end(); it++) {
                it->second /= training_data.size();
            }

            for (int u = 0; u < training_data.size(); u++) {
                current_brain.load(training_data.at(u), true);
                update_weights(all_deltas);
                training_data[u] = current_brain.unload();
                instanceState[u] = values_matching(current_brain.layers[current_brain.layers.size() - 1],
                                                   current_brain.expected_output_values);
            }

        }

        return current_iteration;
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
                map<string, double> d_weights = delta_weights(gradients);
                update_weights(d_weights);
                instanceState[u] = values_matching(current_brain.layers[current_brain.layers.size() - 1],
                                                   current_brain.expected_output_values);
            }

            if (break_on_epoc) {
                cout << "Press Enter to Continue" << endl;
                cin.ignore(std::numeric_limits<streamsize>::max(), '\n');
            }
        }
        return current_iteration;
    }

    void neural_network::eval_fwd_propagation() {

        for (int i = 1; i < current_brain.layers.size(); i++) {
            for (neuron hidden_neuron : current_brain.layers[i]) {
                double value = 0.0;
                if (hidden_neuron.id.find("BN") == string::npos) {
                    for (neuron prev_neuro : current_brain.layers[i - 1]) {
                        vector<synapse> synapses = current_brain.find_by_neuron_id(prev_neuro.id, false, i - 1);
                        for (synapse synapse_instance : synapses) {
                            if (synapse_instance.id.find(hidden_neuron.id) != string::npos) {
                                value += synapse_instance.weight *
                                         current_brain.find_by_id(synapse_instance.source_neuron_id).activationValue();
                            }
                        }
                    }
                    current_brain.update_value(hidden_neuron.id, value);
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
                vector<synapse> outgoing_synapses = current_brain.find_by_neuron_id(neuron_instance.id, false, i);
                for (int s = 0; s < outgoing_synapses.size(); s++) {
                    synapse synapse_instance = outgoing_synapses[s];
                    double delta_weight = values[s] * neuron_instance.activationValue();
                    deltasWeight.insert(pair<string, double>(synapse_instance.id, delta_weight));
                }
            }
        }
        return deltasWeight;
    }

    void neural_network::update_weights(map<string, double> deltas) {
        for (int i = current_brain.layers.size() - 2; i >= 0; i--) {
            for (int j = 0; j < current_brain.layers[i].size(); j++) {
                neuron neuron_instance = current_brain.layers[i][j];
                vector<synapse> outgoing_synapses = current_brain.find_by_neuron_id(neuron_instance.id, false, i);

                for (int s = 0; s < outgoing_synapses.size(); s++) {
                    synapse synapse_instance = outgoing_synapses[s];
                    double weight =
                            synapse_instance.weight - (learning_rate * deltas.at(synapse_instance.id));

                    current_brain.update_synapse(synapse_instance.id, i, weight);
                }
            }
        }
    }

    map<string, vector<double>> neural_network::eval_gradients() {
        map<string, vector<double>> deltas = map<string, vector<double>>();
        for (unsigned long i = current_brain.layers.size() - 1; i > 0; i--) {
            vector<double> deltaHiddenSum = vector<double>();
            for (unsigned long z = 0; z < current_brain.layers[i].size(); z++) {
                neuron neuron_instance = current_brain.layers[i].at(z);

                if (i == (current_brain.layers.size() - 1)) {
                    deltaHiddenSum.push_back(
                            (neuron_instance.activationValue() - current_brain.expected_output_values[z]) *
                            neuron_instance.activationPrimeValue());
                } else {
                    double dhs = 0;
                    vector<synapse> outgoing_synapses = current_brain.find_by_neuron_id(neuron_instance.id, false, i);

                    for (int j = 0; j < outgoing_synapses.size(); j++) {
                        synapse synapse_instance = outgoing_synapses.at(j);
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

    neural_network::neural_network() {

    }

    void neural_network::breakOnEpoc() {
        break_on_epoc = true;
    }

    void brain::create_synapses() {
        random_device rd;
        mt19937 mt(rd());
        uniform_real_distribution<double> dist(0.1, 0.99);
        for (int i = 0; i < this->layers.size() - 1; i++) {
            vector<neuron> sources = layers.at(i);
            vector<neuron> targets = layers.at(i + 1);
            vector<synapse> layered_synapses;

            for (neuron source : sources) {
                for (neuron target : targets) {
                    if (!(target.id.find("BN") != string::npos)) {
                        double value = (dist(mt) * 78 + 20) / 100;
                        synapse _synapse_ = synapse(value, source.id, target.id);
                        layered_synapses.push_back(_synapse_);
                    }
                }
            }
            synapses.insert(pair<int, vector<synapse>>(i, layered_synapses));
        }
    }

    brain::brain(int in_size, int out_size, int nb_hidden_layers, int hidden_layer_size, bool with_bias) {
        layers.clear();

        vector<neuron> neurons;

        for (int j = 0; j < in_size; j++) {
            stringstream id;
            id << "N1." << (j + 1);
            neurons.emplace_back(neuron(id.str()));
        }
        layers.insert(pair<int, vector<neuron>>(0, neurons));

        for (int i = 1; i < nb_hidden_layers + 1; i++) {
            neurons.clear();
            if (with_bias) {
                stringstream id;
                id << "BN" << (i + 1) << ".1";
                neurons.emplace_back(neuron(id.str()));
            }
            for (int j = 0; j < hidden_layer_size; j++) {
                stringstream id;
                int index = j + (with_bias ? 1 : 0) + 1;
                id << "N" << (i + 1) << "." << index;
                neurons.emplace_back(neuron(id.str()));
            }
            layers.insert(pair<int, vector<neuron>>(i, neurons));
        }

        neurons.clear();
        for (int j = 0; j < out_size; j++) {
            stringstream id;
            id << "N" << (nb_hidden_layers + 2) << "." << (j + 1);
            neurons.emplace_back(neuron(id.str()));
        }
        layers.insert(pair<int, vector<neuron>>(nb_hidden_layers + 1, neurons));

        create_synapses();


    }

    void brain::load(const data_holder &test_data_holder, bool ignore_weights) {
        this->expected_output_values = test_data_holder.expected_outputs;

        for (int i = 0; i < layers.size(); i++) {
            vector<neuron> &sources = layers.at(i);
            for (neuron &source : sources) {
                if (test_data_holder.values.count(source.id) > 0) {
                    source.value = test_data_holder.values.at(source.id);
                }
                if (!ignore_weights) {
                    vector<synapse> outgoing_synapses = find_by_neuron_id(source.id, false, i);
                    for (synapse synapse : outgoing_synapses) {
                        if (test_data_holder.weights.count(synapse.id) > 0) {
                            synapse.weight = test_data_holder.weights.at(synapse.id);
                        }
                    }
                }
            }
        }
    }

    data_holder brain::unload() {
        data_holder dataHolder;

        for (int i = 0; i < layers.size(); i++) {
            vector<neuron> sources = layers.at(i);
            for (neuron source : sources) {
                dataHolder.values.insert(pair<string, int>(source.id, source.value));
                vector<synapse> outgoing_synapses = find_by_neuron_id(source.id, false, i);
                for (synapse synapse : outgoing_synapses) {
                    dataHolder.weights.insert(pair<string, int>(synapse.id, synapse.weight));
                }
            }
        }

        dataHolder.expected_outputs = expected_output_values;

        return dataHolder;
    }

    map<string, double> brain::actualWeights() {
        map<string, double> results;
        for (int i = 0; i < layers.size() - 1; i++) {
            vector<neuron> sources = layers.at(i);
            for (neuron source : sources) {
                vector<synapse> outgoing_synapses = find_by_neuron_id(source.id, false, i);
                for (synapse synapse : outgoing_synapses) {
                    results.insert(pair<string, double>(synapse.id, synapse.weight));
                }
            }
        }
        return results;
    }

    void brain::update_value(const string &neuron_id, double val) {
        for (int i = 0; i < layers.size(); i++) {
            vector<neuron> &sources = layers[i];
            for (neuron &source : sources) {
                if (source.id == neuron_id) {
                    source.value = val;
                    return;
                }
            }
        }
    }

    vector<synapse> brain::find_by_neuron_id(const string &neuron_id, bool incoming, int layer_nb) {
        vector<synapse> output;
        try {
            vector<synapse> layered_synapses = synapses.at(layer_nb);
            for (synapse s : layered_synapses) {
                if (s.source_neuron_id == neuron_id && !incoming) {
                    output.push_back(s);
                } else if (s.target_neuron_id == neuron_id && incoming) {
                    output.push_back(s);
                }
            }
        } catch (exception ex) {
            return output;
        }

        return output;
    }

    neuron brain::find_by_id(const string &neuron_id) {
        for (int i = 0; i < layers.size(); i++) {
            vector<neuron> sources = layers.at(i);
            for (neuron source : sources) {
                if (source.id == neuron_id) {
                    return source;
                }
            }
        }
        return neuron("");
    }

    void brain::update_synapse(const string &synapse_id, int layer_nb, const double &weight) {
        vector<synapse> &layered_synapses = synapses[layer_nb];
        for (int i = 0; i < layered_synapses.size(); i++) {
            if (layered_synapses[i].id == synapse_id) {
                layered_synapses[i].weight = weight;
                return;
            }
        }
    }
}
#endif //NEURALNETWORK_NEURALS_H
