#include "NeuralNetwork.h"
#include <iostream>
#include <random>
#include <cmath>

NeuralNetwork& NeuralNetwork::SetNeuronNumber(const unsigned int leftNeuronNum,const unsigned int rightNeuronNum)
{
    neurons_[0].resize(leftNeuronNum);
    neurons_[1].resize(rightNeuronNum);
    connections_.resize(neurons_[0].size() * neurons_[1].size());
    return *this;
}

void NeuralNetwork::InitNeuralNetwork(const double weightsFrom, const double weightsTo, const double biasFrom,
                                      const double biasTo)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> randomWeight(weightsFrom, weightsTo);
    std::uniform_real_distribution<> randomBias(biasFrom, biasTo);

    for (auto& neuron : neurons_[1])
    {
        neuron.bias = static_cast<double>(randomBias(gen));
    }

    for (auto& neuronConnection : connections_)
    {
        neuronConnection = static_cast<double>(randomWeight(gen));
    }
}

void NeuralNetwork::Teach(const LearningData& data, const unsigned int epochs)
{
    unsigned int learnDataIdx{0};
    unsigned int testDataIdx{0};
    double standardError{0.0};

    for (unsigned int epoch{0}; epoch < epochs; ++epoch)
    {
        for (unsigned int i{0}; i < data.GetTrainingDataSize(); ++i)
        {
            // Assign incoming values to the first layer of neurons ----------------------------------------------------
            for (auto& neuron : neurons_[0])
            {
                neuron.value = data.GetTrainingLearnValAtAnyIdx(learnDataIdx++);
            }
            // Calculate the values of the output neurons --------------------------------------------------------------
            unsigned int connectionIdx{0};
            for (auto& rNeuron : neurons_[1])
            {
                rNeuron.value = 0;
                for (const auto& lNeuron : neurons_[0])
                {
                    rNeuron.value += lNeuron.value * connections_[connectionIdx++];
                }
                rNeuron.value -= rNeuron.bias;
            }
            // Calculate the standard error and calculate the bias of the output neurons -------------------------------
            double buffStandardError{0.0};
            for (auto& rNeuron : neurons_[1])
            {
                buffStandardError += static_cast<double>(pow(
                    rNeuron.value - data.GetTrainingTestValAtAnyIdx(testDataIdx), 2));
                rNeuron.bias += learningStep_ * (rNeuron.value - data.GetTrainingTestValAtAnyIdx(testDataIdx));
                
                connectionIdx = 0;
                for (const auto& lNeuron : neurons_[0])
                {
                    connections_[connectionIdx++] -= learningStep_ * lNeuron.value * (rNeuron.value - data.
                        GetTrainingTestValAtAnyIdx(testDataIdx));
                }
                ++testDataIdx;
            }
            standardError += 0.5 * buffStandardError;
        }
        // Output information about the training parameters
        std::cout << "Epoch: " << epoch + 1 << '\n';
        std::cout << "Standard error: " << standardError << '\n';

        if(minRmsError_ > standardError) break;

        standardError = 0;
    }
}

void NeuralNetwork::Test(const LearningData& data)
{
    unsigned int learnDataIdx{0};
    unsigned int testDataIdx{0};
    double standardError{0.0};
    
    for (unsigned int i{0}; i < data.GetTestDataSize(); ++i)
    {
        // Assign incoming values to the first layer of neurons ----------------------------------------------------
        for (auto& neuron : neurons_[0])
        {
            neuron.value = data.GetTestLearnValAtAnyIdx(learnDataIdx++);
        }
        // Calculate the values of the output neurons --------------------------------------------------------------
        unsigned int connectionIdx{0};
        for (auto& rNeuron : neurons_[1])
        {
            rNeuron.value = 0;
            for (const auto& lNeuron : neurons_[0])
            {
                rNeuron.value += lNeuron.value * connections_[connectionIdx++];
            }
            rNeuron.value -= rNeuron.bias;
        }
        // Calculate the standard error and calculate the bias of the output neurons -------------------------------
        double buffStandardError{0.0};
        for (const auto& rNeuron : neurons_[1])
        {
            buffStandardError += static_cast<double>(pow(
                rNeuron.value - data.GetTestTestValAtAnyIdx(testDataIdx), 2));
            ++testDataIdx;
        }
        standardError += 0.5 * buffStandardError;
    }
    
    std::cout << "Standard error in test: " << standardError << '\n';
}
