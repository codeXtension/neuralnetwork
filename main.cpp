
#include "neurals.h"

using namespace cx;

int main() {
    cout.precision(17);

    neural_network network = neural_network(false, 0.7, SGD, 2, 1, 1, 5);
    auto out = cx::readFile("/home/elie/Workspaces/machine-learning/src/main/resources/nn/training.dat");
    network.initialize_data(out);
    cout << "number of iterations: " << network.think(1000);
    return 0;
}



