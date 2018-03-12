//
// Created by elie on 07.03.18.
//

#ifndef NEURALNETWORK_SYNAPSE_H
#define NEURALNETWORK_SYNAPSE_H

#include <string>
#include <iostream>
#include <vector>
#include "utilities.h"
#include <list>
#include <map>
#include <sstream>


namespace cx {
    using namespace std;

    class neural_network;

    class neuron {
    protected:
        double value;
        string id;
    private:
        vector<neural_network> incoming_synapse;
        vector<neural_network> outgoing_synapse;
    public:
        neuron();

        neuron(string id);

        virtual vector<neural_network> &getIncoming_synapse();

        bool operator==(neuron &rhs);

        bool operator!=(neuron &rhs);

        virtual void setIncoming_synapse(vector<neural_network> &incoming_synapse);

        vector<neural_network> &getOutgoing_synapse();

        void setOutgoing_synapse(vector<neural_network> &outgoing_synapse);

        double getValue();

        virtual void setValue(double value);

        string &getId();

        void setId(string &id);

        virtual double activationValue();

        virtual double activationPrimeValue();
    };

    class bias_neuron : public neuron {
    public:
        bias_neuron();

        explicit bias_neuron(string id);

        double activationValue() override;

        double activationPrimeValue() override;

        void setValue(double value) override;

        void setIncoming_synapse(vector<neural_network> &incoming_synapse) override;
    };

    class neural_network {

    private:
        string id;
        double weight;
        neuron *source;
        neuron *target;

    public:
        neural_network();

        neural_network(double weight, neuron &source, neuron &target);

        neuron *getSource();

        void setSource(neuron *source);

        neuron *getTarget();

        void setTarget(neuron *target);

        string &getId();

        void setId(string &id);

        double getWeight();

        void setWeight(double weight);

        bool operator==(neural_network &rhs);

        bool operator!=(neural_network &rhs);
    };

    class brain {
    private:
        map<int, std::list<neuron>> layers;
        vector<double> expected_output_values;

        void create_synapses();

    public:
        brain(int in_size, int out_size, int nb_hidden_layers, int hidden_layer_size, bool with_bias);

        void load(data_holder &test_data_holder, bool ignore_weights);

        data_holder &unload();

        list <neuron> &get_layer(int layer_nb);

        map<string, double> &actual_weights();

        map<int, std::list<neuron>> &getLayers();

        void setLayers(map<int, std::list<neuron>> &layers);

        vector<double> &getExpected_output_values();

        void setExpected_output_values(vector<double> &expected_output_values);
    };

    bias_neuron::bias_neuron() {
        this->value = 1;
    }

    void bias_neuron::setIncoming_synapse(vector<neural_network> &incoming_synapse) {

    }

    void bias_neuron::setValue(double value) {

    }

    double bias_neuron::activationPrimeValue() {
        return neuron::getValue();
    }

    double bias_neuron::activationValue() {
        return neuron::getValue();
    }

    bias_neuron::bias_neuron(string id) : neuron(id) {
        this->id = id;
    }

    neuron::neuron() {
    }

    vector<neural_network> &neuron::getIncoming_synapse() {
        return incoming_synapse;
    }

    void neuron::setIncoming_synapse(vector<neural_network> &incoming_synapse) {
        neuron::incoming_synapse = incoming_synapse;
    }

    vector<neural_network> &neuron::getOutgoing_synapse() {
        return outgoing_synapse;
    }

    void neuron::setOutgoing_synapse(vector<neural_network> &outgoing_synapse) {
        neuron::outgoing_synapse = outgoing_synapse;
    }

    double neuron::getValue() {
        return value;
    }

    void neuron::setValue(double value) {
        neuron::value = value;
    }

    string &neuron::getId() {
        return id;
    }

    void neuron::setId(string &id) {
        neuron::id = id;
    }

    bool neuron::operator==(neuron &rhs) {
        return id == rhs.id;
    }

    bool neuron::operator!=(neuron &rhs) {
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
    }


    neural_network::neural_network() {
        this->id = "-";
        this->weight = 0;
        this->source = new neuron("");
        this->target = new neuron("");
    }

    bool neural_network::operator==(neural_network &rhs) {
        return id == rhs.id;
    }

    bool neural_network::operator!=(neural_network &rhs) {
        return rhs.id != this->id;
    }

    string &neural_network::getId() {
        return id;
    }

    void neural_network::setId(string &id) {
        neural_network::id = id;
    }

    double neural_network::getWeight() {
        return weight;
    }

    void neural_network::setWeight(double weight) {
        neural_network::weight = weight;
    }

    neuron *neural_network::getSource() {
        return source;
    }

    void neural_network::setSource(neuron *source) {
        neural_network::source = source;
    }

    neuron *neural_network::getTarget() {
        return target;
    }

    void neural_network::setTarget(neuron *target) {
        neural_network::target = target;
    }

    neural_network::neural_network(double weight, neuron &source, neuron &target) {
        this->weight = weight;
        this->source = &source;
        this->target = &target;
        this->source->getOutgoing_synapse().push_back(*this);
        this->target->getIncoming_synapse().push_back(*this);
    }


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
        for (int i = 0; i < this->layers.size() - 1; i++) {
            list <neuron> sources = layers.at(i);
            list <neuron> targets = layers.at(i + 1);
            for (neuron &source : sources) {
                for (neuron target : targets) {
                    double value = (rand() * 78 + 20) / 100;
                    if (neuron *v = dynamic_cast<neuron *>(&target)) {
                        new neural_network(value, source, target);
                    }
                }
            }
        }
    }

    brain::brain(int in_size, int out_size, int nb_hidden_layers, int hidden_layer_size, bool with_bias) {
        layers.clear();

        list <neuron> neurons;

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
            list <neuron> sources = layers.at(i);
            for (neuron source : sources) {
                if (test_data_holder.getValues().count(source.getId()) > 0) {
                    source.setValue(test_data_holder.getValues().at(source.getId()));
                }
                if (!ignore_weights) {
                    for (neural_network synapse : source.getOutgoing_synapse()) {
                        if (test_data_holder.getWeights().count(synapse.getId()) > 0) {
                            synapse.setWeight(test_data_holder.getWeights().at(synapse.getId()));
                        }
                    }
                }
            }
        }
    }

    data_holder &brain::unload() {
        data_holder dataHolder;

        for (int i = 0; i < layers.size(); i++) {
            list <neuron> sources = layers.at(i);
            for (neuron source : sources) {
                dataHolder.getValues().insert(pair<string, double>(source.getId(), source.getValue()));
                for (neural_network synapse : source.getOutgoing_synapse()) {
                    dataHolder.getWeights().insert(pair<string, double>(synapse.getId(), synapse.getWeight()));
                }
            }
        }

        dataHolder.setExpected_outputs(getExpected_output_values());

        return dataHolder;
    }

    list <neuron> &brain::get_layer(int layer_nb) {
        return this->layers.at(layer_nb);
    }

    map<string, double> &brain::actual_weights() {
        map<string, double> results;
        for (int i = 0; i < layers.size() - 1; i++) {
            list <neuron> sources = layers.at(i);
            for (neuron source : sources) {
                for (neural_network synapse : source.getOutgoing_synapse()) {
                    results.insert(pair<string, double>(synapse.getId(), synapse.getWeight()));
                }
            }
        }
        return results;
    }

}

#endif //NEURALNETWORK_SYNAPSE_H
