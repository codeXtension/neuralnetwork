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

    double sigmoid(const double &value);

    double derivativeSigmoid(const double &value);
    vector<map<value_type, vector<float>>>
    readData(vector<vector<unsigned char>> images, vector<unsigned char> labels);

    vector<map<value_type, vector<float>>> readFile(const string &file_path);

    map<string, string> read_startup_attributes(const string &properties_file);
}
#endif //NEURALNETWORK_UTILITIES_H
