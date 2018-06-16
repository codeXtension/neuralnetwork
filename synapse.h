//
// Created by elie on 12.03.18.
//

#ifndef NEURALNETWORK_SYNAPSE_H
#define NEURALNETWORK_SYNAPSE_H

#include <string>
#include <iostream>
#include <vector>
#include "utilities.h"
#include "neuron.h"
#include <list>
#include <map>
#include <sstream>

using namespace std;
namespace cx {
    class synapse {

    private:
        string id;
        double weight = 0.0;
        neuron *source;
        neuron *target;

    public:
        synapse(double weight, neuron *source, neuron *target);

        string getId();

        void setId(string id);

        double getWeight();

        void setWeight(double weight);

        bool operator==(synapse rhs);

        bool operator!=(synapse rhs);

        neuron *getSource() const;

        void setSource(neuron *source);

        neuron *getTarget() const;

        void setTarget(neuron *target);
    };

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
        synapse::weight = weight;
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
#endif //NEURALNETWORK_SYNAPSE_H
