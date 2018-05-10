
#include "neural_network.h"

using namespace cx;

int main() {
    char32_t initia = 'a';

    test t = test(&initia);

    initia = 'b';

    cout << *t.getInitials() << ";" << initia << endl;

    brain br = brain(2, 1, 1, 3, true);
    neural_network network;
    network.log_weights(br);
    return 0;
}

