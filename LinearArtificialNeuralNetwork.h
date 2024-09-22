#pragma once
#include "pair_hash.h"

// Designed to create a neural network structure
class linear_artif_neural_network
{
protected:
    // (Layer idx, neuron idx), value
    std::unordered_map<std::pair<unsigned int, unsigned int>, double, pair_hash> neurons_;
    // (previous neuron, next neuron), weight
    std::unordered_map<std::pair<unsigned int, unsigned int>, double, pair_hash> neuron_connections_;

public:
    linear_artif_neural_network() = default;
    ~linear_artif_neural_network() = default;
    linear_artif_neural_network(const linear_artif_neural_network&) = default;
    linear_artif_neural_network(linear_artif_neural_network&&) noexcept = default;
    linear_artif_neural_network& operator=(const linear_artif_neural_network&) = default;
    linear_artif_neural_network& operator=(linear_artif_neural_network&&) = default;

    void set_neuron_connection_weight(const unsigned int f_neuron_idx, const unsigned int s_neuron_idx,
                                      const double value);
    void set_neuron(const unsigned int neural_layer_idx, const unsigned int neuron_idx, const double value);

    double get_neuron_value(const unsigned int neural_layer_idx, const unsigned int neuron_idx) const;
    double get_neuron_connection_weight(const unsigned int f_neuron_idx, const unsigned int s_neuron_idx) const;
};
