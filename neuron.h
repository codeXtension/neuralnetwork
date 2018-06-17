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
        double value = 0.0;
        string id;
    private:
        vector<synapse> incoming_synapse;
    public:
        neuron();

        vector<synapse> outgoing_synapse;

        explicit neuron(string id);

        virtual vector<synapse> getIncoming_synapse();

        bool operator==(neuron rhs);

        bool operator!=(neuron rhs);

        virtual void setIncomingSynapse(vector<synapse> incoming_synapse);

        vector<synapse> getOutgoing_synapse();

        void setOutgoingSynapse(vector<synapse> outgoing_synapse);

        double getValue();

        virtual void setValue(double value);

        string getId();

        void setId(string id);

        virtual double activationValue();

        virtual double activationPrimeValue();

        void addOutgoingSynapse(synapse *pSynapse);

        virtual void addIncomingSynapse(synapse *pSynapse);
    };

    class bias_neuron : public neuron {
    public:
        bias_neuron();

        explicit bias_neuron(string id);

        double activationValue() override;

        double activationPrimeValue() override;

        void setValue(double value) override;

        void setIncomingSynapse(vector<synapse> incoming_synapse) override;

        void addIncomingSynapse(synapse *pSynapse) override;

    };
}
#endif //NEURALNETWORK_NEURON_H
