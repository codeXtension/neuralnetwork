
#include "neural_network.h"

using namespace cx;

int main() {
    brain br = brain(2, 1, 1, 3, true);
    neural_network network;
    network.log_weights(br);
    return 0;
}