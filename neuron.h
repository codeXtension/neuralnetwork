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
    private:
    public:
        double value = 0.0;
        string id;
        vector<synapse> incoming_synapse;
        vector<synapse> outgoing_synapse;

        neuron();

        explicit neuron(string id);

        bool operator==(neuron rhs);

        bool operator!=(neuron rhs);

        double activationValue();

        double activationPrimeValue();

        void addOutgoingSynapse(synapse *pSynapse);

        void addIncomingSynapse(synapse *pSynapse);
    };
}
#endif //NEURALNETWORK_NEURON_H
