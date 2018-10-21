//
// Created by elie on 21.10.18.
//

#include "config_reader.h"

namespace cx {
    config_reader::config_reader(const string &properties_file) {
        pt::ptree root;
        pt::read_json(properties_file, root);

        method =
                root.get<string>("method", "SGD") == "SGD" ? SGD : (root.get<string>("method", "SGD") == "BATCH" ? BATCH
                                                                                                                 : MINI_BATCH);
        batch_size = root.get<int>("options.batch_size");
        with_bias = root.get<bool>("options.with_bias", false);
        learning_rate = root.get<float>("options.learning_rate", 1.0);
        input_size = root.get<int>("options.input_size");
        output_size = root.get<int>("options.output_size");
        max_nb_iterations = root.get<int>("options.max_nb_iterations", -1);
        accuracy = root.get<float>("options.accuracy");
        training_file = root.get<string>("training_file");
        test_file = root.get<string>("test_file");
        hidden_layers_data = as_vector<int>(root, "options.hidden_layers");
    }

    template<typename T>
    vector<T> config_reader::as_vector(pt::ptree const &pt, pt::ptree::key_type const &key) {
        std::vector<T> r;
        for (auto &item : pt.get_child(key))
            r.push_back(item.second.get_value<T>());
        return r;
    }
}

