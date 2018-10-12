
#include "neurals.h"
#include <boost/convert.hpp>
#include <boost/convert/lexical_cast.hpp>
#include "chrono"
#include "cifar10_reader.h"

using namespace cx;
using boost::lexical_cast;
using boost::convert;

struct boost::cnv::by_default : public boost::cnv::lexical_cast {
};

int main(int argc, char *argv[]) {
    cout.precision(17);

    auto dataset = cifar::read_dataset<std::vector, std::vector, unsigned char, unsigned char>();

    if (argc == 1) {
        cout << "Please provide the configuration file path"
             << endl;
        return 0;
    }
    map<string, string> props = read_startup_attributes(argv[1]);

    neural_network network = neural_network(
            props.at("with_bias") == "true",
            convert<double>(props.at("learning_rate")).value(),
            (props.at("method") == "SGD" ? SGD : (props.at("method") == "BATCH" ? BATCH : MINI_BATCH)),
            convert<int>(props.at("input_size")).value(),
            convert<int>(props.at("output_size")).value(),
            convert<int>(props.at("nb_hidden_layers")).value(),
            convert<int>(props.at("size_hidden_layer")).value());
    network.batch_size=convert<int>(props.at("batch_size")).value_or(1);

    auto out_file = cx::readFile(props.at("training_file"));
    auto out_data = cx::readData(dataset.training_images, dataset.training_labels);

    if (props.at("break_on_epoc") == "true")
        network.breakOnEpoc();

    network.initialize_data(out_file);
    auto started = std::chrono::high_resolution_clock::now();
    long res = network.think(convert<int>(props.at("max_nb_iterations")).value_or(1000000));
    auto done = std::chrono::high_resolution_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
    cout << "trained using " << props.at("method") << " after a number of iterations: " << res << ", and took " << ms
         << "ms"
         << endl;
    return 0;
}



