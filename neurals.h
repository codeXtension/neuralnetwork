//
// Created by elie on 17.06.18.
//

#ifndef NEURALNETWORK_NEURALS_H
#define NEURALNETWORK_NEURALS_H

#include <string>
#include <iostream>
#include <vector>
#include "neuron.h"
#include "synapse.h"
#include <list>
#include <map>
#include <sstream>

using namespace std;

namespace cx {

    bias_neuron::bias_neuron() {
        this->value = 1;
    }

    void bias_neuron::setIncomingSynapse(vector<synapse> incoming_synapse) {

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

    void bias_neuron::addIncomingSynapse(synapse *pSynapse) {
    }

    neuron::neuron() {
    }

    vector<synapse> neuron::getIncoming_synapse() {
        return incoming_synapse;
    }

    void neuron::setIncomingSynapse(vector<synapse> incoming_synapse) {
        neuron::incoming_synapse = incoming_synapse;
    }

    vector<synapse> neuron::getOutgoing_synapse() {
        return outgoing_synapse;
    }

    void neuron::setOutgoingSynapse(vector<synapse> outgoing_synapse) {
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

    string synapse::getId() {
        return id;
    }

    void synapse::setId(string id) {
        synapse::id = id;
    }

    double synapse::getWeight() {
        return weight;
    }

    void synapse::setWeight(double weight) {
        this->weight = weight;
    }

    neuron *synapse::getSource() const {
        return source;
    }

    void synapse::setSource(neuron *source) {
        synapse::source = source;
    }

    neuron *synapse::getTarget() const {
        return target;
    }

    void synapse::setTarget(neuron *target) {
        synapse::target = target;
    }


    synapse::synapse(double weight, neuron *source, neuron *target) {
        this->id = source->getId() + "-" + target->getId();
        this->weight = weight;
        this->source = source;
        this->target = target;
    }
}
#endif //NEURALNETWORK_NEURALS_H
