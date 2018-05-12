//
// Created by elie on 07.03.18.
//

#ifndef NEURALNETWORK_UTILITIES_H
#define NEURALNETWORK_UTILITIES_H

#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

namespace cx {
    double sigmoid(double value) {
        return 1.0f / (1 + exp(-value));
    }

    double derivativeSigmoid(double value) {
        double sigmoid = cx::sigmoid(value);
        return sigmoid * (1.0d - sigmoid);
    }

    enum value_type {
        INPUT,
        OUTPUT
    };

    enum method_type {
        SGD,
        BATCH,
        MINI_BATCH
    };

    class data_holder {
    public:
        map<string, double> getWeights();

        void setWeights(map<string, double> weights);

        map<string, double> &getValues();

        void setValues(map<string, double> &values);

        vector<double> &getExpected_outputs();

        void setExpected_outputs(vector<double> &expected_outputs);

        vector<double> &getPrev_outputs();

        void setPrev_outputs(vector<double> &prev_outputs);

        void add_input(vector<double> &inputs);

    private:
        map<string, double> weights;
        map<string, double> values;
        vector<double> expected_outputs;
        vector<double> prev_outputs;
    };

    map<string, double> data_holder::getWeights() {
        return weights;
    }

    void data_holder::setWeights(map<string, double> weights) {
        data_holder::weights = weights;
    }

    map<string, double> &data_holder::getValues() {
        return values;
    }

    void data_holder::setValues(map<string, double> &values) {
        data_holder::values = values;
    }

    vector<double> &data_holder::getExpected_outputs() {
        return expected_outputs;
    }

    void data_holder::setExpected_outputs(vector<double> &expected_outputs) {
        data_holder::expected_outputs = expected_outputs;
    }

    vector<double> &data_holder::getPrev_outputs() {
        return prev_outputs;
    }

    void data_holder::setPrev_outputs(vector<double> &prev_outputs) {
        data_holder::prev_outputs = prev_outputs;
    }

    void data_holder::add_input(vector<double> &inputs) {
        for (int i = 0; i < inputs.size(); i++) {
            double input = inputs.at(i);
            this->values.insert(pair<string, double>("N1." + i, input));
        }
    }
}
#endif //NEURALNETWORK_UTILITIES_H
