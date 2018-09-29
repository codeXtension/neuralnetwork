
#include "neurals.h"
#include <boost/convert.hpp>
#include <boost/convert/lexical_cast.hpp>

using namespace cx;
using boost::lexical_cast;
using boost::convert;

struct boost::cnv::by_default : public boost::cnv::lexical_cast {
};

int main() {
    cout.precision(17);
    map<string, string> props = read_startup_attributes("/home/elie/neural_data/config.dat");

    neural_network network = neural_network(
            (props.at("with_bias") == "true" ? true : false),
            convert<double>(props.at("learning_rate")).value(),
            (props.at("method") == "SGD" ? SGD : BATCH),
            convert<int>(props.at("input_size")).value(),
            convert<int>(props.at("output_size")).value(),
            convert<int>(props.at("nb_hidden_layers")).value(),
            convert<int>(props.at("size_hidden_layer")).value());

    auto out = cx::readFile(props.at("training_file"));

    network.initialize_data(out);
    cout << "trained after a number of iterations: " << network.think() << endl;
    return 0;
}



