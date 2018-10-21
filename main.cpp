
#include <boost/convert.hpp>
#include <boost/convert/lexical_cast.hpp>
#include "chrono"
#include "cifar10_reader.h"
#include "neural_network.h"
#include "config_reader.h"

using namespace cx;
using boost::lexical_cast;
using boost::convert;

struct boost::cnv::by_default : public boost::cnv::lexical_cast {
};

int main(int argc, char *argv[]) {
    cout.precision(17);


    if (argc == 1) {
        cout << "Please provide the configuration file path"
             << endl;
        return 0;
    }

    config_reader props = config_reader(argv[1]);

    neural_network network = neural_network(
            props.with_bias,
            props.learning_rate,
            props.method,
            props.input_size,
            props.output_size,
            props.hidden_layers_data,
            props.accuracy);

    network.batch_size = props.batch_size;

    auto out_file = cx::readFile(props.training_file);

//    auto dataset = cifar::read_dataset<std::vector, std::vector, unsigned char, unsigned char>();
//    auto out_data = cx::readData(dataset.training_images, dataset.training_labels);

    network.initialize_data(out_file);
    auto started = std::chrono::high_resolution_clock::now();
    long res;
    if (props.max_nb_iterations == -1) {
        res = network.think();
    } else {
        res = network.think(props.max_nb_iterations);
    }
    auto done = std::chrono::high_resolution_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(done - started).count();
    cout << "trained using " << props.method << " after a number of iterations: " << res << ", and took " << ms
         << "ms"
         << endl;
    return 0;
}



