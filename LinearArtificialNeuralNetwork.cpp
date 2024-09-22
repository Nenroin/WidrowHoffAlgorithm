#include "LinearArtificialNeuralNetwork.h"

void linear_artif_neural_network::set_neuron(const unsigned int neural_layer_idx, const unsigned int neuron_idx,
                                             const double value)
{
    neurons_[std::pair<unsigned int, unsigned int>(neural_layer_idx, neuron_idx)] = value;
}

double linear_artif_neural_network::get_neuron_value(const unsigned int neural_layer_idx,
                                                     const unsigned int neuron_idx) const
{
    return neurons_.at(std::pair<unsigned int, unsigned int>(neural_layer_idx, neuron_idx));
}

void linear_artif_neural_network::set_neuron_connection_weight(const unsigned int f_neuron_idx,
                                                               const unsigned int s_neuron_idx, const double value)
{
    neuron_connections_[std::pair<unsigned int, unsigned int>(f_neuron_idx, s_neuron_idx)] = value;
}

double linear_artif_neural_network::get_neuron_connection_weight(const unsigned int f_neuron_idx,
                                                                 const unsigned int s_neuron_idx) const
{
    return neuron_connections_.at(std::pair<unsigned int, unsigned int>(f_neuron_idx, s_neuron_idx));
}
