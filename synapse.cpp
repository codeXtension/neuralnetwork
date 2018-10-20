//
// Created by elie on 20.10.18.
//

#include "synapse.h"


namespace cx {
    bool synapse::operator==(synapse rhs) {
        return id == rhs.id;
    }

    bool synapse::operator!=(synapse rhs) {
        return rhs.id != this->id;
    }

    synapse::synapse(const double &weight, const string &source_neuron_id, const string &target_neuron_id) {
        this->id = source_neuron_id + "-" + target_neuron_id;
        this->weight = weight;
        this->source_neuron_id = source_neuron_id;
        this->target_neuron_id = target_neuron_id;
    }
}