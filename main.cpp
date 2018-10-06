
#include "neurals.h"
#include <boost/convert.hpp>
#include <boost/convert/lexical_cast.hpp>
#include "chrono"

using namespace cx;
using boost::lexical_cast;
using boost::convert;

struct boost::cnv::by_default : public boost::cnv::lexical_cast {
};

int main(int argc, char *argv[]) {
    cout.precision(17);
    if (argc == 1) {
        cout << "Please provide the config file path"
             << endl; // "/home/elie/Workspaces/neuralnetwork/neural_data/config.dat"
        return 0;
    }
    map<string, string> props = read_startup_attributes(argv[1]);

    neural_network network = neural_network(
            props.at("with_bias") == "true",
            convert<double>(props.at("learning_rate")).value(),
            (props.at("method") == "SGD" ? SGD : BATCH),
            convert<int>(props.at("input_size")).value(),
            convert<int>(props.at("output_size")).value(),
            convert<int>(props.at("nb_hidden_layers")).value(),
            convert<int>(props.at("size_hidden_layer")).value());

    auto out = cx::readFile(props.at("training_file"));

    if (props.at("break_on_epoc") == "true")
        network.breakOnEpoc();

    network.initialize_data(out);
    auto started = std::chrono::high_resolution_clock::now();
    long res = network.think(convert<int>(props.at("max_nb_iterations")).value_or(1000000));
    auto done = std::chrono::high_resolution_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
    cout << "trained using " << props.at("method") << " after a number of iterations: " << res << ", and took " << ms
         << "ms"
         << endl;
    return 0;
}



