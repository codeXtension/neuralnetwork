//
// Created by elie on 12.03.18.
//

#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

#include <string>
#include <iostream>
#include <vector>
#include "utilities.h"
#include <list>
#include <map>
#include <sstream>

using namespace std;

namespace cx {

    class synapse;

    class neuron {
    protected:
        double value = 0;
        string id;
    private:
        vector<synapse> incoming_synapse;
        vector<synapse> outgoing_synapse;
    public:
        neuron();

        explicit neuron(string id);

        virtual vector<synapse> getIncoming_synapse();

        bool operator==(neuron rhs);

        bool operator!=(neuron rhs);

        virtual void setIncoming_synapse(vector<synapse> incoming_synapse);

        vector<synapse> &getOutgoing_synapse();

        void setOutgoing_synapse(vector<synapse> outgoing_synapse);

        double getValue();

        virtual void setValue(double value);

        string getId();

        void setId(string id);

        virtual double activationValue();

        virtual double activationPrimeValue();

        void add_outgoing_synapse(synapse *pSynapse);

        virtual void add_incoming_synapse(synapse *pSynapse);
    };

    class bias_neuron : public neuron {
    public:
        bias_neuron();

        explicit bias_neuron(string id);

        double activationValue() override;

        double activationPrimeValue() override;

        void setValue(double value) override;

        void setIncoming_synapse(vector<synapse> incoming_synapse) override;

        void add_incoming_synapse(synapse *pSynapse) override;

    };

    bias_neuron::bias_neuron() {
        this->value = 1;
    }

    void bias_neuron::setIncoming_synapse(vector<synapse> incoming_synapse) {

    }

    void bias_neuron::setValue(double value) {

    }

    double bias_neuron::activationPrimeValue() {
        return neuron::getValue();
    }

    double bias_neuron::activationValue() {
        return neuron::getValue();
    }

    bias_neuron::bias_neuron(string id) {
        this->id = id;
        cout << "BiasNeuron created with id " << id << endl;
    }

    void bias_neuron::add_incoming_synapse(synapse *pSynapse) {
    }

    neuron::neuron() {
    }

    vector<synapse> neuron::getIncoming_synapse() {
        return incoming_synapse;
    }

    void neuron::setIncoming_synapse(vector<synapse> incoming_synapse) {
        neuron::incoming_synapse = incoming_synapse;
    }

    vector<synapse> &neuron::getOutgoing_synapse() {
        return outgoing_synapse;
    }

    void neuron::setOutgoing_synapse(vector<synapse> outgoing_synapse) {
        neuron::outgoing_synapse = outgoing_synapse;
    }

    double neuron::getValue() {
        return value;
    }

    void neuron::setValue(double value) {
        neuron::value = value;
    }

    string neuron::getId() {
        return id;
    }

    void neuron::setId(string id) {
        neuron::id = id;
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

    void neuron::add_outgoing_synapse(synapse *pSynapse) {
        this->outgoing_synapse.push_back(*pSynapse);
    }

    void neuron::add_incoming_synapse(synapse *pSynapse) {
        this->incoming_synapse.push_back(*pSynapse);
    }
}
#endif //NEURALNETWORK_NEURON_H
