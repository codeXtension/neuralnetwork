//
// Created by elie on 20.10.18.
//

#include "brain.h"

namespace cx {
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
                        synapse _synapse = synapse(value, source.id, target.id);
                        layered_synapses.push_back(_synapse);
                    }
                }
            }
            synapses.insert(pair<int, vector<synapse>>(i, layered_synapses));
        }
    }

    brain::brain(int in_size, int out_size, const vector<int> & hidden_layers_data, bool with_bias) {
        layers.clear();

        vector<neuron> neurons;
        int nb_hidden_layers = hidden_layers_data.size();

        for (int j = 0; j < in_size; j++) {
            stringstream id;
            id << "N1." << (j + 1);
            neurons.emplace_back(neuron(id.str()));
        }
        layers.insert(pair<int, vector<neuron>>(0, neurons));

        for (int i = 1; i < nb_hidden_layers + 1; i++) {
            int hidden_layer_size = hidden_layers_data[i - 1];
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

    void brain::load(const data_holder &test_data_holder) {
        this->expected_output_values = test_data_holder.expected_outputs;

        for (int i = 0; i < layers.size(); i++) {
            vector<neuron> &sources = layers.at(i);
            for (neuron &source : sources) {
                if (test_data_holder.values.count(source.id) > 0) {
                    source.value = test_data_holder.values.at(source.id);
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
        } catch (const exception &ex) {
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

    void brain::update_synapse(const string &synapseid, int layer_nb, const double &weight) {
        vector<synapse> &layered_synapses = synapses[layer_nb];
        for (int i = 0; i < layered_synapses.size(); i++) {
            if (layered_synapses[i].id == synapseid) {
                layered_synapses[i].weight = weight;
                return;
            }
        }
    }
}