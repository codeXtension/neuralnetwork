//
// Created by elie on 20.10.18.
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

    public:
        string id;
        double weight = 0.0;
        string source_neuron_id;
        string target_neuron_id;

        synapse(const double &weight, const string &source_neuron_id, const string &target_neuron_id);

        bool operator==(synapse rhs);

        bool operator!=(synapse rhs);
    };
}
#endif //NEURALNETWORK_SYNAPSE_H
