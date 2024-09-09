#pragma once
#include <vector>
#include <iostream>

using neuron_layer_t = std::vector<double>;
using neuron_t = double;
using neuron_connect_t =  std::vector<double>;
using neuron_connect_weight_t = double;

class linear_artificial_neural_network
{
    std::vector<neuron_layer_t> neural_network_;
    std::vector<neuron_connect_t> neuron_connections_;
    
public:
    void set_neuron_connection_weight(const int neuron_connect_idx, const int neuron_idx, const neuron_t neuron_value)
    {
        neural_network_.at(neuron_connect_idx).at(neuron_idx) = neuron_value;
    }
    
    neuron_connect_weight_t get_neuron_connection_weight(const int neural_layer_idx, const int neuron_idx) const
    {
        return neural_network_.at(neural_layer_idx).at(neuron_idx);
    }
    
    void set_neuron(const int neural_layer_idx, const int neuron_idx, const neuron_t neuron_value)
    {
        neural_network_.at(neural_layer_idx).at(neuron_idx) = neuron_value;
    }
    
    neuron_t get_neuron(const int neural_layer_idx, const int neuron_idx) const
    {
        return neural_network_.at(neural_layer_idx).at(neuron_idx);
    }
    
    linear_artificial_neural_network& add_neural_layer(const int neurons_number)
    {
        neural_network_.push_back(neuron_layer_t(neurons_number));  // NOLINT(modernize-use-emplace)

        return *this; 
    }
    
    void show_neural_network_structure() const
    {
        int idx{ 1 };
        const int layer_number{ static_cast<int>(neural_network_.size()) } ;
        
        for(const auto &neural_layer : neural_network_)
        {
            const int neurons_number = static_cast<int>(neural_layer.size());
            std::cout << neurons_number << (idx < layer_number ? " <-> " : "");
            
            ++idx;
        }
    }
};
