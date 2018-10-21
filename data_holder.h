//
// Created by elie on 20.10.18.
//

#ifndef NEURALNETWORK_DATA_HOLDER_H
#define NEURALNETWORK_DATA_HOLDER_H

#include <cmath>
#include <map>
#include <string>
#include <vector>

using namespace std;

namespace cx {
    class data_holder {
    public:
        void add_input(vector<float> &inputs);
        map<string, float> values;
        vector<float> expected_outputs;
    };
}
#endif //NEURALNETWORK_DATA_HOLDER_H
