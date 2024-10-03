#include <iostream>
#include <conio.h>
#include "NeuralNetwork.h"
#include <cmath>

int main()
{
    NeuralNetwork neuralNetwork(0.1, 0.001);
    neuralNetwork.SetNeuronNumber(4, 1);
    neuralNetwork.InitNeuralNetwork();

    std::vector<double> learnData, testData;
    double x{0.315};
    constexpr double endValue{1.57};
    constexpr int steps{45};

    const double stepSize = (endValue - x) / steps;
    for (int i{0}; i < steps; ++i)
    {
        double val{3.0 * std::sin(5.0 * x) + 0.5};

        if ((i + 1) % 5 == 0)
        {
            testData.emplace_back(val);
        }
        else
        {
            learnData.emplace_back(val);
        }
        
        x += stepSize;
    }

    LearningData data;
    data.AddTrainingVal(learnData, testData);

    neuralNetwork.Teach(data, 100);

    std::cout << "\nPut any button to exit...";
    _getch();

    return 0;
}
