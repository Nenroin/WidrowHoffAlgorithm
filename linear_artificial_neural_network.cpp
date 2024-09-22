#include "linear_artificial_neural_network.h"

linear_artif_neural_network& linear_artif_neural_network::set_neuron(const unsigned int neural_layer_idx,
                                                                     const unsigned int neuron_idx,
                                                                     const neuron& value)
{
    neurons_.at(neuron_loc_t(neural_layer_idx, neuron_idx)) = value;
    return *this;
}

neuron linear_artif_neural_network::get_neuron(const unsigned int neural_layer_idx,
                                               const unsigned int neuron_idx) const
{
    return neurons_.at(neuron_loc_t(neural_layer_idx, neuron_idx));
}

linear_artif_neural_network& linear_artif_neural_network::set_neuron_connection_weight(
    const weight_loc& weight_location, const double value)
{
    neuron_connections_.at(weight_location) = value;
    return *this;
}

double linear_artif_neural_network::get_neuron_connection_weight(const weight_loc& weight_location) const
{
    return neuron_connections_.at(weight_location);
}

linear_artif_neural_network& linear_artif_neural_network::add_neuron_connection_weight(
    const weight_loc& weight_location, const double value)
{
    neuron_connections_.emplace(weight_location, value);
    return *this;
}

linear_artif_neural_network& linear_artif_neural_network::add_layer_neurons(const unsigned int neuron_num)
{
    layer_neurons_count_.emplace_back(neuron_num);

    unsigned int layer_idx = static_cast<unsigned int>(layer_neurons_count_.size()) - 1;

    for (unsigned int i{0}; i < neuron_num; i++)
    {
        neurons_.emplace(neuron_loc_t(layer_idx, i), neuron{0.0, 0.0});
    }
    return *this;
}
