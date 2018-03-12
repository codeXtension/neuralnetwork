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

using namespace std;

namespace cx {
    class brain {
    private:
        map<int, std::list<neuron>> layers;
        vector<double> expected_output_values;

        void create_synapses();

    public:
        brain(int in_size, int out_size, int nb_hidden_layers, int hidden_layer_size, bool with_bias);

        void load(data_holder &test_data_holder, bool ignore_weights);

        data_holder unload();

        list<neuron> &get_layer(int layer_nb);

        map<string, double> actual_weights();

        map<int, std::list<neuron>> &getLayers();

        void setLayers(map<int, std::list<neuron>> &layers);

        vector<double> &getExpected_output_values();

        void setExpected_output_values(vector<double> &expected_output_values);
    };

    map<int, std::list<neuron>> &brain::getLayers() {
        return layers;
    }

    void brain::setLayers(map<int, std::list<neuron>> &layers) {
        brain::layers = layers;
    }

    vector<double> &brain::getExpected_output_values() {
        return expected_output_values;
    }

    void brain::setExpected_output_values(vector<double> &expected_output_values) {
        brain::expected_output_values = expected_output_values;
    }

    void brain::create_synapses() {
        random_device rd;
        mt19937 mt(rd());
        uniform_real_distribution<double> dist(0.1, 0.95);

        for (int i = 0; i < this->layers.size() - 1; i++) {
            list<neuron> sources = layers.at(i);
            list<neuron> targets = layers.at(i + 1);
            for (neuron &source : sources) {
                for (neuron target : targets) {
                    double value = (dist(mt) * 78 + 20) / 100;
                    if (neuron *v = dynamic_cast<neuron *>(&target)) {
                        synapse(value, source, target);
                    }
                }
            }
        }
    }

    brain::brain(int in_size, int out_size, int nb_hidden_layers, int hidden_layer_size, bool with_bias) {
        layers.clear();

        list<neuron> neurons;

        for (int j = 0; j < in_size; j++) {
            neurons.emplace_back(neuron("N1." + (j + 1)));
        }
        layers.insert(pair<int, std::list<neuron>>(0, neurons));

        for (int i = 1; i < nb_hidden_layers + 1; i++) {
            neurons.clear();
            if (with_bias) {
                stringstream id;
                id << "BN" << (i + 1) << ".1";
                neurons.emplace_back(bias_neuron(id.str()));
            }
            for (int j = 0; j < hidden_layer_size; j++) {
                stringstream id;
                id << "N" << (i + 1) << "." << (j + (with_bias ? 1 : 0) + 1);
                neurons.emplace_back(neuron(id.str()));
            }
            layers.insert(pair<int, std::list<neuron>>(i, neurons));
        }

        neurons.clear();
        for (int j = 0; j < out_size; j++) {
            stringstream id;
            id << "N" << (nb_hidden_layers + 2) << "." << (j + 1);
            neurons.emplace_back(neuron(id.str()));
        }
        layers.insert(pair<int, std::list<neuron>>(nb_hidden_layers + 1, neurons));

        create_synapses();
    }

    void brain::load(data_holder &test_data_holder, bool ignore_weights) {
        this->expected_output_values = test_data_holder.getExpected_outputs();

        for (int i = 0; i < layers.size(); i++) {
            list<neuron> sources = layers.at(i);
            for (neuron source : sources) {
                if (test_data_holder.getValues().count(source.getId()) > 0) {
                    source.setValue(test_data_holder.getValues().at(source.getId()));
                }
                if (!ignore_weights) {
                    for (synapse synapse : source.getOutgoing_synapse()) {
                        if (test_data_holder.getWeights().count(synapse.getId()) > 0) {
                            synapse.setWeight(test_data_holder.getWeights().at(synapse.getId()));
                        }
                    }
                }
            }
        }
    }

    data_holder brain::unload() {
        data_holder dataHolder;

        for (int i = 0; i < layers.size(); i++) {
            list<neuron> sources = layers.at(i);
            for (neuron source : sources) {
                dataHolder.getValues().insert(pair<string, double>(source.getId(), source.getValue()));
                for (synapse synapse : source.getOutgoing_synapse()) {
                    dataHolder.getWeights().insert(pair<string, double>(synapse.getId(), synapse.getWeight()));
                }
            }
        }

        dataHolder.setExpected_outputs(expected_output_values);

        return dataHolder;
    }

    list<neuron> &brain::get_layer(int layer_nb) {
        return this->layers.at(layer_nb);
    }

    map<string, double> brain::actual_weights() {
        map<string, double> results;
        for (int i = 0; i < layers.size() - 1; i++) {
            list<neuron> sources = layers.at(i);
            for (neuron source : sources) {
                for (synapse synapse : source.getOutgoing_synapse()) {
                    results.insert(pair<string, double>(synapse.getId(), synapse.getWeight()));
                }
            }
        }
        return results;
    }
}

#endif //NEURALNETWORK_BRAIN_H
