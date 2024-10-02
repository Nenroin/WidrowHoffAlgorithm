#include <iostream>
#include <conio.h>
#include "NeuralNetwork.h"


int main()
{
    NeuralNetwork neuralNetwork;

    neuralNetwork.AddLayer(3).AddLayer(1).CreateConnection(1, 4)
                 .CreateConnection(2, 4).CreateConnection(3, 4);

    NeuralNetworkTeacher teacher(0.01f, 0.001f);

    NeuralNetworkTeacher::InitNeuralNetwork(neuralNetwork);

    LearningData data();

    std::cout << "\nPut any button to exit...";
    _getch();

    return 0;
}
