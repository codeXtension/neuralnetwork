
#include "neural_network.h"

using namespace cx;

int main() {
    neural_network network = neural_network(true, 0.002, SGD, 2,1,1,3);
    auto out = cx::readFile("/home/elie/Workspaces/machine-learning/src/main/resources/nn/test.dat");
    network.initialize_data(out);
    return 0;
}



