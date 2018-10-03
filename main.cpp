
#include "neurals.h"
#include <boost/convert.hpp>
#include <boost/convert/lexical_cast.hpp>
#include "log.h"

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

    string imported_log_level = props.at("log_level");

    if (imported_log_level == "TRACE")
        DEFAULT_LOG_LEVEL = TRACE;
    else if (imported_log_level == "DEBUG")
        DEFAULT_LOG_LEVEL = DEBUG;
    else if (imported_log_level == "INFO")
        DEFAULT_LOG_LEVEL = INFO;
    else if (imported_log_level == "WARNING")
        DEFAULT_LOG_LEVEL = WARNING;
    else if (imported_log_level == "ERROR")
        DEFAULT_LOG_LEVEL = ERROR;
    else
        DEFAULT_LOG_LEVEL = INFO;

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
    int res = network.think(convert<int>(props.at("max_nb_iterations")).value_or(1000000));
    cx::log(INFO, "AFTERTHOUGHT") << "trained after a number of iterations: " << res << endl;
    return 0;
}



