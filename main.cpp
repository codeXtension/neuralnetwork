#include <iostream>
#include "neuralnetwork.h"

using namespace cx;

int main() {
    brain br = brain(2, 1, 1, 3, false);
    double d = br.getLayers().count(1);
    std:cout<<d<<"-"<<&d<<endl;
    return 0;
}