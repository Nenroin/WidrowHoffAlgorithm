#include <iostream>
#include <conio.h>
#include "NeuralNetwork.h"
#include <cmath>

int main()
{
    NeuralNetwork neuralNetwork(0.08, 0.001);
    neuralNetwork.SetNeuronNumber(4, 1);
    neuralNetwork.InitNeuralNetwork();

    std::vector<double> learnDataLearn, testDataLearn;
    std::vector<double> learnDataTest, testDataTest;
    constexpr int steps{45};
    
    double xLearn{0.315};
    constexpr double endValueLearn{1.57};
    const double stepSizeLearn = (endValueLearn - xLearn) / steps;
    
    double xTest{1.57};
    constexpr double endValueTest{2.83};
    const double stepSizeTest = (endValueTest - xTest) / steps;

    for (int i{0}; i < steps; ++i)
    {
        double learnValue{3.0 * std::sin(5.0 * xLearn) + 0.5};
        double testValue{3.0 * std::sin(5.0 * xTest) + 0.5};
        
        if ((i + 1) % 5 == 0)
        {
            testDataLearn.emplace_back(learnValue);
            testDataTest.emplace_back(testValue);
        }
        else
        {
            learnDataLearn.emplace_back(learnValue);
            learnDataTest.emplace_back(testValue);
        }
        
        xLearn += stepSizeLearn;
        xTest += stepSizeTest;
    }

    LearningData data;
    data.AddTrainingVal(learnDataLearn, testDataLearn);
    data.AddTestVal(learnDataTest, testDataTest);
    
    neuralNetwork.Teach(data, 1000);
    neuralNetwork.Test(data);
    
    std::cout << "\nPut any button to exit...";
    _getch();

    return 0;
}
