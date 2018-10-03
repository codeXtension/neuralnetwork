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
#include <boost/algorithm/string.hpp>
#include <iomanip>
#include <algorithm>
#include <cctype>

using namespace std;

enum LogLevel {
    TRACE, DEBUG, INFO, WARNING, ERROR
};

std::ostream &operator<<(std::ostream &out, const LogLevel value) {
    static std::map<LogLevel, std::string> strings;
    if (strings.size() == 0) {
#define INSERT_ELEMENT(p) strings[p] = #p
        INSERT_ELEMENT(TRACE);
        INSERT_ELEMENT(DEBUG);
        INSERT_ELEMENT(INFO);
        INSERT_ELEMENT(WARNING);
        INSERT_ELEMENT(ERROR);
#undef INSERT_ELEMENT
    }

    return out << strings[value];
}

LogLevel DEFAULT_LOG_LEVEL = TRACE;

namespace cx {

    enum value_type {
        INPUT,
        OUTPUT
    };

    enum method_type {
        SGD,
        BATCH,
        MINI_BATCH
    };

    double sigmoid(const double &value) {
        return 1.0f / (1 + exp(-value));
    }

    double derivativeSigmoid(const double &value) {
        double sigmoid = cx::sigmoid(value);
        return sigmoid * (1.0 - sigmoid);
    }

    vector<map<value_type, vector<int>>> readFile(const string &file_path) {
        ifstream input_file(file_path);
        vector<map<value_type, vector<int>>> output;
        for (string line; getline(input_file, line);) {
            istringstream ss(line);
            map<value_type, vector<int>> results;
            int x = 1;
            while (ss.good()) {
                string s;
                getline(ss, s, ';');
                vector<int> input;
                unsigned long n = s.length();
                char char_array[n];
                strcpy(char_array, s.c_str());
                for (int i = 0; i < n; i++)
                    input.push_back(char_array[i] - '0');

                if (x == 1) {
                    results.insert(pair<value_type, vector<int>>(OUTPUT, input));
                    x++;
                } else {
                    results.insert(pair<value_type, vector<int>>(INPUT, input));
                    x--;
                }
            }
            output.push_back(results);
        }
        return output;
    };

    map<string, string> read_startup_attributes(const string &properties_file) {
        ifstream input_file(properties_file);
        map<string, string> output;
        for (string line; getline(input_file, line);) {
            vector<std::string> strs;
            boost::split(strs, line, boost::is_any_of("="));
            output.insert(pair<string, string>(strs[0], strs[1]));
        }

        return output;
    }

    class data_holder {
    public:
        void add_input(vector<int> &inputs);

        map<string, double> weights;
        map<string, int> values;
        vector<int> expected_outputs;
        vector<int> prev_outputs;
    };

    void data_holder::add_input(vector<int> &inputs) {
        for (int i = 1; i <= inputs.size(); i++) {
            int input = inputs.at(i - 1);
            string node_name = "N1." + to_string(i);
            this->values.insert(pair<string, int>(node_name, input));
        }
    }
}
#endif //NEURALNETWORK_UTILITIES_H
