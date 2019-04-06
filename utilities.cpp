//
// Created by elie on 20.10.18.
//

#include "utilities.h"

namespace cx {
    double sigmoid(const double &value) {
        return 1.0f / (1 + exp(-value));
    }

    double derivativeSigmoid(const double &value) {
        double sigmoid = cx::sigmoid(value);
        return sigmoid * (1.0 - sigmoid);
    }

    vector<map<value_type, vector<float>>>
    readData(vector<vector<unsigned char>> images, vector<unsigned char> labels) {
        vector<map<value_type, vector<float>>> output;

        for (int j = 0; j < images.size(); j++) {
            vector<unsigned char> value = images.at(j);
            vector<float> inputs;
            vector<float> outputs;
            map<value_type, vector<float>> out;
            for (int i = 0; i < value.size(); i++) {
                inputs.push_back(((int) value.at(i)) / 256.0);
            }

            for (int k = 0; k < 4; ++k) {
                outputs.push_back((labels.at(j) >> k) & 1);
            }

            out.insert(pair<value_type, vector<float>>(INPUT, inputs));
            out.insert(pair<value_type, vector<float>>(OUTPUT, outputs));

            output.push_back(out);
        }

        return output;
    }

    vector<map<value_type, vector<float>>> readFile(const string &file_path) {
        ifstream input_file(file_path);
        vector<map<value_type, vector<float>>> output;
        for (string line; getline(input_file, line);) {
            istringstream ss(line);
            map<value_type, vector<float>> results;
            int x = 1;
            while (ss.good()) {
                string s;
                getline(ss, s, ';');
                vector<float> input;
                unsigned long n = s.length();
                char char_array[n];
                strcpy(char_array, s.c_str());
                for (int i = 0; i < n; i++)
                    input.push_back(char_array[i] - '0');

                if (x == 1) {
                    results.insert(pair<value_type, vector<float>>(OUTPUT, input));
                    x++;
                } else {
                    results.insert(pair<value_type, vector<float>>(INPUT, input));
                    x--;
                }
            }
            output.push_back(results);
        }
        return output;
    }

    string method_name(const cx::method_type &methodType) {

        if (methodType == cx::method_type::BATCH) {
            return "BATCH";
        } else if (methodType == cx::method_type::MINI_BATCH) {
            return "MINI_BATCH";
        } else {
            return "SGD";
        }
    }
}