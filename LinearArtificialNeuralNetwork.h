#pragma once
#include <vector>
#include <iostream>

using neural_layer_t = std::vector<double>;
using neuron_t = double;

class linear_artificial_neural_network
{
    std::vector<neural_layer_t> neural_network_;
public:
    linear_artificial_neural_network& add_neural_layer(const int neurons_number)
    {
        neural_network_.push_back(neural_layer_t(neurons_number));  // NOLINT(modernize-use-emplace)

        return *this; 
    }
    
    void show_neural_network_structure() const
    {
        int neurons_number{ 0 }, idx{ 1 };
        const int layer_number{ static_cast<int>(neural_network_.size()) } ;
        
        for(const auto &neural_layer : neural_network_)
        {
            neurons_number = static_cast<int>(neural_layer.size());
            std::cout << neurons_number << (idx < layer_number ? " <-> " : "");
            
            ++idx;
        }
    }
};
