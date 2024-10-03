#include <iostream>
#include <conio.h>
#include "NeuralNetwork.h"
#include <cmath>

int main()
{
    NeuralNetwork neuralNetwork;

    neuralNetwork.AddLayer(4).AddLayer(1).CreateConnection(1, 5)
                 .CreateConnection(2, 5).CreateConnection(3, 5)
                 .CreateConnection(4, 5);
    NeuralNetworkTeacher teacher(0.002f, 0.001f);
    NeuralNetworkTeacher::InitNeuralNetwork(neuralNetwork);

    std::vector<float> learnData, testData;
    float x{0.315f};
    constexpr float endValue{1.57f};
    constexpr int steps{ 45 };

    const float stepSize = (endValue - x) / steps;
    for (int i{0}; i < steps; ++i)
    {
        const float buffFloatX{static_cast<float>(x)};
        float val{3.0f * std::sin(5.0f * buffFloatX) + 0.5f};

        if((i + 1) % 5 == 0)
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

    teacher.Teach(neuralNetwork, data,100);
    
    std::cout << "\nPut any button to exit...";
    _getch();

    return 0;
}
