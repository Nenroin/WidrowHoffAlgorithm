#pragma once
#include "NeuralNetworkStructs.h"
#include "LearningData.h"
#include <vector>
#include <array>

// Designed to create a neural network structure
class NeuralNetwork
{
protected:
    double learningStep_;
    double minRmsError_;
    std::array<std::vector<Neuron>, 2> neurons_;
    std::vector<double> connections_;

public:
    explicit NeuralNetwork(const double learningStep = 0.01, const double minRmsError = 0.01) :
        learningStep_{learningStep}, minRmsError_{minRmsError}
    {}
    NeuralNetwork& SetNeuronNumber(const unsigned int leftNeuronNum,const unsigned int rightNeuronNum);
    void InitNeuralNetwork(const double weightsFrom = -0.5, const double weightsTo = 0.5,
                           const double biasFrom = -0.5, const double biasTo = 0.5);
    // Teaching -------------------------------------------------------------------------------------------------------
    void Teach(const LearningData& data, const unsigned int epochs);
    void Test(const LearningData& data);
    // ----------------------------------------------------------------------------------------------------------------
    ~NeuralNetwork() = default;
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) noexcept = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(NeuralNetwork&&) = delete;
    // ----------------------------------------------------------------------------------------------------------------
};
