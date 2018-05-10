//
// Created by elie on 07.03.18.
//

#ifndef NEURALNETWORK_NEURAL_NETWORK_H
#define NEURALNETWORK_NEURAL_NETWORK_H

#include <string>
#include <iostream>
#include <vector>
#include "utilities.h"
#include <list>
#include <map>
#include <sstream>
#include "brain.h"
#include "synapse.h"
#include "neuron.h"

using namespace std;

namespace cx {
    class neural_network {
    public:
        void log_weights(brain value);
    };

    void neural_network::log_weights(brain value) {
        for (int i = 0; i < value.getLayers().size() - 1; i++) {
            list<neuron> sources = value.getLayers().at(i);
            cout << "BRAIN - Synapses from layer " << (i + 1) << " --> " << (i + 2) << endl;
            for (neuron source : sources) {
                for (synapse synapse : source.getOutgoing_synapse()) {
                    neuron *target = synapse.getTarget();
                    cout << "BRAIN - " << source.getId() << " ---" << synapse.getWeight() << "---> " << target->getId()
                         << " a(" << target->activationValue() << ")" << endl;
                }
            }
        }
    }
}

#endif //NEURALNETWORK_NEURAL_NETWORK_H
