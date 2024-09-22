#pragma once
#include "neural_network_structs.h"
#include <vector>
#include "neural_network_teacher.h"

// Designed to create a neural network structure
class linear_artif_neural_network
{
protected:
    std::vector<unsigned int> layer_neurons_count_;
    // (Layer idx, neuron idx), value
    std::unordered_map<neuron_loc_t, neuron, pair_hash> neurons_;
    // weight position, weight value
    std::unordered_map<weight_loc, double, weight_hash> neuron_connections_;

public:
    linear_artif_neural_network() = default;
    ~linear_artif_neural_network() = default;
    linear_artif_neural_network(const linear_artif_neural_network&) = default;
    linear_artif_neural_network(linear_artif_neural_network&&) noexcept = default;
    linear_artif_neural_network& operator=(const linear_artif_neural_network&) = default;
    linear_artif_neural_network& operator=(linear_artif_neural_network&&) = default;

    linear_artif_neural_network& add_layer_neurons(const unsigned int neuron_num = 0);

    linear_artif_neural_network& set_neuron_connection_weight(const weight_loc& weight_location,
                                                              const double value = 0.0);
    linear_artif_neural_network& add_neuron_connection_weight(const weight_loc& weight_location,
                                                              const double value = 0.0);
    double get_neuron_connection_weight(const weight_loc& weight_location) const;

    linear_artif_neural_network& set_neuron(const unsigned int neural_layer_idx, const unsigned int neuron_idx,
                                            const neuron& value);
    neuron get_neuron(const unsigned int neural_layer_idx, const unsigned int neuron_idx) const;

    static friend void neural_network_teacher::init_neural_network(linear_artif_neural_network& neural_network,
                                                                   double weights_from, double weights_to,
                                                                   double bias_from, double bias_to);
};
