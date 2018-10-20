//
// Created by elie on 20.10.18.
//

#include "neuron.h"

namespace cx {
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
}