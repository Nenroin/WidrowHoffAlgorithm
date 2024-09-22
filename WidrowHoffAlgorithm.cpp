#include <iostream>
#include <conio.h>
#include "linear_artificial_neural_network.h"


int main()
{
    linear_artif_neural_network neural_network;

    constexpr weight_loc weights[] = {
        {neuron_loc_t(0, 0), neuron_loc_t(0, 0)},
        {neuron_loc_t(0, 1), neuron_loc_t(0, 0)},
        {neuron_loc_t(0, 2), neuron_loc_t(0, 0)}
    };

    neural_network.add_layer_neurons(3).add_layer_neurons(1).add_neuron_connection_weight(weights[0])
    .add_neuron_connection_weight(weights[1]).add_neuron_connection_weight(weights[2]);
    
    neural_network_teacher::init_neural_network(neural_network);
    
    std::cout << "\nPut any button to exit...";
    _getch();

    return 0;
}
