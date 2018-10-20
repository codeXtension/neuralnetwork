//
// Created by elie on 20.10.18.
//

#include "neural_network.h"

namespace cx {
    const float MATCH_RANGE = 0.1;

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

    void neural_network::initialize_data(vector<map<value_type, vector<float>>> data) {
        training_data.clear();
        for (map<value_type, vector<float>> instance : data) {
            data_holder dataHolder;
            dataHolder.add_input(instance.at(INPUT));
            dataHolder.expected_outputs = instance.at(OUTPUT);
            dataHolder.weights = this->current_brain.actualWeights();
            training_data.push_back(dataHolder);
        }
    }

    long neural_network::think() {
        return think(LONG_MAX);
    }

    long neural_network::think(long max_nb_iterations) {
        switch (meth_type) {
            case SGD:
                return think_sgd(max_nb_iterations);
            case BATCH:
                return think_batch(max_nb_iterations);
            case MINI_BATCH:
                return think_minibatch(max_nb_iterations);
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

    bool neural_network::values_matching(vector<neuron> neurons, vector<float> expected_values) {
        bool result = true;

        for (int i = 0; i < neurons.size(); i++) {
            neuron n = neurons.at(i);
            result &= (abs(expected_values.at(i) - n.activationValue())) <= match_range;
        }

        return result;
    }

    long neural_network::think_batch(long max_nb_iterations) {

        vector<bool> instanceState;
        for (int u = 0; u < training_data.size(); u++) {
            instanceState.push_back(false);
        }

        while (not_all_true(instanceState) && current_iteration < max_nb_iterations) {
            current_iteration++;
            map<string, double> all_deltas;

            for (int u = 0; u < training_data.size(); u++) {
                current_brain.load(training_data.at(u));
                eval_fwd_propagation();
                training_data[u] = current_brain.unload();
                map<string, vector<double>> gradients = eval_gradients();
                map<string, double> d_weights = delta_weights(gradients);
                if (all_deltas.size() == 0) {
                    all_deltas.insert(d_weights.begin(), d_weights.end());
                    map<string, double>::iterator it;
                    for (it = all_deltas.begin(); it != all_deltas.end(); it++) {
                        it->second = it->second / training_data.size();
                    }
                } else {
                    map<string, double>::iterator it;
                    for (it = all_deltas.begin(); it != all_deltas.end(); it++) {
                        it->second += (d_weights.at(it->first) / training_data.size());
                    }
                }
            }

            for (int u = 0; u < training_data.size(); u++) {
                current_brain.load(training_data.at(u));
                update_weights(all_deltas);
//                training_data[u] = current_brain.unload();
                instanceState[u] = values_matching(current_brain.layers[current_brain.layers.size() - 1],
                                                   current_brain.expected_output_values);
            }

        }

        return current_iteration;
    }

    long neural_network::think_minibatch(long max_nb_iterations) {
        vector<bool> instanceState;
        for (int u = 0; u < training_data.size(); u++) {
            instanceState.push_back(false);
        }

        while (not_all_true(instanceState) && current_iteration < max_nb_iterations) {
            current_iteration++;

            int counter = 0;

            while (counter < training_data.size()) {

                map<string, double> all_deltas;
                int new_batch_size = batch_size;
                int upper_limit = counter + new_batch_size;

                if ((training_data.size() - upper_limit < new_batch_size)) {
                    new_batch_size += (training_data.size() - upper_limit);
                    upper_limit += (training_data.size() - upper_limit);
                }

                for (int u = counter; u < upper_limit; u++) {
                    current_brain.load(training_data.at(u));
                    eval_fwd_propagation();
                    training_data[u] = current_brain.unload();
                    map<string, vector<double>> gradients = eval_gradients();
                    map<string, double> d_weights = delta_weights(gradients);
                    if (all_deltas.size() == 0) {
                        all_deltas.insert(d_weights.begin(), d_weights.end());
                        map<string, double>::iterator it;
                        for (it = all_deltas.begin(); it != all_deltas.end(); it++) {
                            it->second = it->second / new_batch_size;
                        }
                    } else {
                        map<string, double>::iterator it;
                        for (it = all_deltas.begin(); it != all_deltas.end(); it++) {
                            it->second += (d_weights.at(it->first) / new_batch_size);
                        }
                    }
                }

                for (int u = counter; u < upper_limit; u++) {
                    current_brain.load(training_data.at(u));
                    update_weights(all_deltas);
                    instanceState[u] = values_matching(current_brain.layers[current_brain.layers.size() - 1],
                                                       current_brain.expected_output_values);
                }
                counter += upper_limit;
            }
        }

        return current_iteration;
    }

    long neural_network::think_sgd(long max_nb_iterations) {
        vector<bool> instanceState;
        for (int u = 0; u < training_data.size(); u++) {
            instanceState.push_back(false);
        }
        while (not_all_true(instanceState) && current_iteration < max_nb_iterations) {
            current_iteration++;
            for (int u = 0; u < training_data.size(); u++) {
                current_brain.load(training_data.at(u));
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
                        for (synapse synapseinstance : synapses) {
                            if (synapseinstance.id.find(hidden_neuron.id) != string::npos) {
                                value += synapseinstance.weight *
                                         current_brain.find_by_id(synapseinstance.source_neuron_id).activationValue();
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
                    synapse synapseinstance = outgoing_synapses[s];
                    double delta_weight = values[s] * neuron_instance.activationValue();
                    deltasWeight.insert(pair<string, double>(synapseinstance.id, delta_weight));
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
                    synapse synapseinstance = outgoing_synapses[s];
                    double weight = synapseinstance.weight - (learning_rate * deltas.at(synapseinstance.id));

                    current_brain.update_synapse(synapseinstance.id, i, weight);
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
                        synapse synapseinstance = outgoing_synapses.at(j);
                        dhs += deltas.at("hidden_" + to_string(i + 1))[j] * synapseinstance.weight;
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
}