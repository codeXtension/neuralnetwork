//
// Created by elie on 20.10.18.
//

#include "data_holder.h"

namespace cx {
    void data_holder::add_input(vector<float> &inputs) {
        for (int i = 1; i <= inputs.size(); i++) {
            int input = inputs.at(static_cast<unsigned long>(i - 1));
            string node_name = "N1." + to_string(i);
            this->values.insert(pair<string, int>(node_name, input));
        }
    }
}