//
// Created by elie on 07.03.18.
//

#ifndef NEURALNETWORK_NEURAL_NETWORK_H
#define NEURALNETWORK_NEURAL_NETWORK_H

#include <string>
#include <iostream>
#include <vector>
#include "utilities.h"
#include <list>
#include <map>
#include <sstream>
#include <climits>
#include "brain.h"
#include "neurals.h"

using namespace std;

namespace cx {
	class neural_network {
		public:
			neural_network();

			neural_network(bool with_bias, double learning_rate, method_type meth_type, int input_size, int output_size, int nb_hidden_layers, int size_hidden_layer);

			void initialize_data(vector<map<value_type, vector<int>>> data);

			long think();

			long think(long max_nb_iterations);

			void guess();

			void breakOnEpoc();

			int batch_size=1;

		private:
			brain current_brain = brain(0, 0, 0, 0, false);

			bool break_on_epoc = false;

			long think_batch(long max_nb_iterations);

			long think_sgd(long max_nb_iterations);

			long think_minibatch(long max_nb_iterations);

			bool not_all_true(vector<bool> states);

			bool values_matching(vector<neuron> neurons, vector<int> expected_values);

			void eval_fwd_propagation();

			map<string, double> delta_weights(map<string, vector<double>> gradients);

			void update_weights(map<string, double> deltas);

			map<string, vector<double>> eval_gradients();

			double match_range;
			bool with_bias;
			double learning_rate;
			int current_iteration;
			method_type meth_type;
			vector<data_holder> training_data;
			int nb_hidden_layers;
			int size_hidden_layer;
	};
}

#endif //NEURALNETWORK_NEURAL_NETWORK_H
