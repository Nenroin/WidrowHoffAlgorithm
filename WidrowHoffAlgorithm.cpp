#include <iostream>
#include <conio.h>
#include "LinearArtificialNeuralNetwork.h"

int main()
{
    linear_artificial_neural_network neural_network;
    neural_network.add_neural_layer(2).add_neural_layer(5).add_neural_layer(4);
    neural_network.show_neural_network_structure();
    
    std::cout << "\nPut any button to exit...";
    _getch();
    
    return 0;
}
