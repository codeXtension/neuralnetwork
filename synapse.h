//
// Created by elie on 12.03.18.
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

using namespace std;
namespace cx {
    class synapse {

    private:
        string id;
        double weight;
        neuron *source;
        neuron *target;

    public:
        synapse();

        synapse(double weight, neuron &source, neuron &target);

        neuron *getSource();

        void setSource(neuron *source);

        neuron *getTarget();

        void setTarget(neuron *target);

        string &getId();

        void setId(string &id);

        double getWeight();

        void setWeight(double weight);

        bool operator==(synapse &rhs);

        bool operator!=(synapse &rhs);
    };

    synapse::synapse() {
        this->id = "-";
        this->weight = 0;
        this->source = new neuron("");
        this->target = new neuron("");
    }

    bool synapse::operator==(synapse &rhs) {
        return id == rhs.id;
    }

    bool synapse::operator!=(synapse &rhs) {
        return rhs.id != this->id;
    }

    string &synapse::getId() {
        return id;
    }

    void synapse::setId(string &id) {
        synapse::id = id;
    }

    double synapse::getWeight() {
        return weight;
    }

    void synapse::setWeight(double weight) {
        synapse::weight = weight;
    }

    neuron *synapse::getSource() {
        return source;
    }

    void synapse::setSource(neuron *source) {
        synapse::source = source;
    }

    neuron *synapse::getTarget() {
        return target;
    }

    void synapse::setTarget(neuron *target) {
        synapse::target = target;
    }

    synapse::synapse(double weight, neuron &source, neuron &target) {
        this->weight = weight;
        this->source = &source;
        this->target = &target;
        this->source->getOutgoing_synapse().push_back(*this);
        this->target->getIncoming_synapse().push_back(*this);
    }
}
#endif //NEURALNETWORK_SYNAPSE_H
