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
}
#endif //NEURALNETWORK_SYNAPSE_H
