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
#include <fstream>
#include <cstring>

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

    vector<map<value_type, vector<int>>> readFile(const string &filepath) {
        ifstream input_file(filepath);
        vector<map<value_type, vector<int>>> output;
        for (string line; getline(input_file, line);) {
            istringstream ss(line);
            map<value_type, vector<int>> results;
            int x=1;
            while (ss.good()) {
                string s;
                getline(ss, s, ';');
                vector<int> input;
                int n = s.length();
                char char_array[n];
                strcpy(char_array, s.c_str());
                for (int i=0; i<n; i++)
                    input.push_back(char_array[i]-'0');

                if(x==1) {
                    results.insert(pair<value_type, vector<int>>(INPUT, input));
                    x++;
                }else{
                    results.insert(pair<value_type, vector<int>>(OUTPUT, input));
                    x--;
                }
                cout<<endl;
            }
            output.push_back(results);
        }
        return output;
    };

    class data_holder {
    public:
        map<string, double> getWeights();

        void setWeights(map<string, double> weights);

        map<string, int> &getValues();

        void setValues(map<string, int> &values);

        vector<int> &getExpected_outputs();

        void setExpected_outputs(vector<int> &expected_outputs);

        vector<int> &getPrev_outputs();

        void setPrev_outputs(vector<int> &prev_outputs);

        void add_input(vector<int> &inputs);

    private:
        map<string, double> weights;
        map<string, int> values;
        vector<int> expected_outputs;
        vector<int> prev_outputs;
    };

    map<string, double> data_holder::getWeights() {
        return weights;
    }

    void data_holder::setWeights(map<string, double> weights) {
        data_holder::weights = weights;
    }

    map<string, int> &data_holder::getValues() {
        return values;
    }

    void data_holder::setValues(map<string, int> &values) {
        data_holder::values = values;
    }

    vector<int> &data_holder::getExpected_outputs() {
        return expected_outputs;
    }

    void data_holder::setExpected_outputs(vector<int> &expected_outputs) {
        data_holder::expected_outputs = expected_outputs;
    }

    vector<int> &data_holder::getPrev_outputs() {
        return prev_outputs;
    }

    void data_holder::setPrev_outputs(vector<int> &prev_outputs) {
        data_holder::prev_outputs = prev_outputs;
    }

    void data_holder::add_input(vector<int> &inputs) {
        for (int i = 0; i < inputs.size(); i++) {
            int input = inputs.at(i);
            string node_name = "N1." + to_string(i);
            this->values.insert(pair<string, int>(node_name, input));
        }
    }
}
#endif //NEURALNETWORK_UTILITIES_H
