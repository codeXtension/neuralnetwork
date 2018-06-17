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

    void brain::create_synapses() {
        random_device rd;
        mt19937 mt(rd());
        uniform_real_distribution<double> dist(0.1, 0.95);

        for (int i = 0; i < this->layers.size() - 1; i++) {
            vector<neuron> &sources = layers.at(i);
            vector<neuron> &targets = layers.at(i + 1);
            for (neuron &source : sources) {
                for (neuron &target : targets) {
                    if (target.id.find("BN") == string::npos) {
                        double value = (dist(mt) * 78 + 20) / 100;
                        synapse _synapse_ = synapse(value, &source, &target);
                        source.addOutgoingSynapse(&_synapse_);
                        target.addIncomingSynapse(&_synapse_);
                    }
                }
            }
        }
        cout << endl;
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

    void brain::load(data_holder &test_data_holder, bool ignore_weights) {
        this->expected_output_values = test_data_holder.expected_outputs;

        for (int i = 0; i < layers.size(); i++) {
            vector<neuron> &sources = layers.at(i);
            for (neuron &source : sources) {
                if (test_data_holder.values.count(source.id) > 0) {
                    source.value = test_data_holder.values.at(source.id);
                }
                if (!ignore_weights) {
                    for (synapse synapse : source.outgoing_synapse) {
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
                for (synapse synapse : source.outgoing_synapse) {
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
                for (synapse synapse : source.outgoing_synapse) {
                    results.insert(pair<string, double>(synapse.id, synapse.weight));
                }
            }
        }
        return results;
    }

    void brain::update_value(const string &neuron_id, double val) {
        for (int i = 0; i < layers.size(); i++) {
            vector<neuron> &sources = layers.at(i);
            for (neuron &source : sources) {
                if (source.id == neuron_id) {
                    source.value = val;
                    return;
                }
            }
        }
    }
}

#endif //NEURALNETWORK_BRAIN_H
