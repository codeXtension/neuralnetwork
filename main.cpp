
#include "neural_network.h"

using namespace cx;

int main() {
    neural_network network = neural_network(true, 0.002, SGD, 2,1,1,3);

    return 0;
}

