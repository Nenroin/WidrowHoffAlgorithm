#include "neural_network_teacher.h"
#include "linear_artificial_neural_network.h"
#include "neural_network_structs.h"
#include <random>

double neural_network_teacher::get_min_rms_error() const
{
    return min_rms_error_;
}

void neural_network_teacher::set_min_rms_error(const double min_rms_error)
{
    min_rms_error_ = min_rms_error;
}

double neural_network_teacher::get_learning_step() const
{
    return learning_step_;
}

void neural_network_teacher::set_learning_step(const double learning_step)
{
    learning_step_ = learning_step;
}

void neural_network_teacher::init_neural_network(linear_artif_neural_network& neural_network, const double weights_from,
                                                 const double weights_to, const double bias_from, const double bias_to)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib_weight(weights_from, weights_to);
    std::uniform_real_distribution<> distrib_bias(bias_from, bias_to);

    for (unsigned int layer_idx = 0; layer_idx < neural_network.layer_neurons_count_.size(); ++layer_idx)
    {
        for (unsigned int neuron_idx = 0; neuron_idx < neural_network.layer_neurons_count_.at(layer_idx); ++neuron_idx)
        {
            neuron buff_neuron = {distrib_bias(gen), 0.0};
            neural_network.set_neuron(layer_idx, neuron_idx, buff_neuron);
        }
    }
    
    for (auto &neuron_connection : neural_network.neuron_connections_)
    {
        neuron_connection.second = distrib_weight(gen);
    }
}
