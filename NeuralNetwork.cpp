#include "NeuralNetwork.h"
#include <iostream>
#include <random>
#include <cmath>

NeuralNetwork& NeuralNetwork::SetNeuronNumberFirstLayer(const unsigned int neuronNum)
{
    neurons_[0].resize(neuronNum);
    return *this;
}

NeuralNetwork& NeuralNetwork::SetNeuronNumberSecondLayer(const unsigned int neuronNum)
{
    neurons_[1].resize(neuronNum);
    return *this;
}

NeuralNetwork& NeuralNetwork::CreateConnection(const unsigned int lNeuronEdx, const unsigned int rNeuronEdx,
                                               const float value)
{
    connections_[ConnectiontLoc(lNeuronEdx, rNeuronEdx)] = value;
    return *this;
}

NeuralNetwork& NeuralNetwork::SetConnectionValueIfConnectionExists(const unsigned int lNeuronEdx,
                                                                   const unsigned int rNeuronEdx, const float value)
{
    const ConnectiontLoc keyVal{lNeuronEdx, rNeuronEdx};
    const auto connectionIt = connections_.find(keyVal);
    if (connectionIt != connections_.end())
    {
        connectionIt->second = value;
    }
    return *this;
}

NeuralNetwork& NeuralNetwork::CalculateNeuronValueIfConnectionExists(const unsigned int lNeuronEdx,
                                                                     const unsigned int rNeuronEdx)
{
    const ConnectiontLoc keyVal{lNeuronEdx, rNeuronEdx};
    const auto connectionIt = connections_.find(keyVal);
    if (connectionIt != connections_.end())
    {
        neurons_[1][rNeuronEdx].value += neurons_[0][lNeuronEdx].value * connectionIt->second;
    }
    return *this;
}

float NeuralNetwork::GetConnectionValue(const unsigned int lNeuronEdx, const unsigned int rNeuronEdx) const
{
    return connections_.at({lNeuronEdx, rNeuronEdx});
}

void NeuralNetwork::InitNeuralNetwork(const float weightsFrom, const float weightsTo, const float biasFrom,
                                      const float biasTo)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> randomWeight(weightsFrom, weightsTo);
    std::uniform_real_distribution<> randomBias(biasFrom, biasTo);

    for (auto& neuron : neurons_[1])
    {
        neuron.bias = static_cast<float>(randomBias(gen));
    }

    for (auto& neuronConnection : connections_)
    {
        neuronConnection.second = static_cast<float>(randomWeight(gen));
    }
}

void NeuralNetwork::Teach(const LearningData& data, const unsigned int epochs)
{
    unsigned int learnDataIdx{0};
    unsigned int testDataIdx{0};
    float standardError{0.0f};
    
    for (unsigned int epoch{0}; epoch < epochs; ++epoch)
    {
        for (unsigned int i{0}; i < 45; ++i)
        {
            // Assign incoming values to the first layer of neurons ----------------------------------------------------
            for (auto& neuron : neurons_[0])
            {
                neuron.value = data.GetTrainingLearnValAtAnyIdx(learnDataIdx++);
            }
            // Calculate the values of the output neurons --------------------------------------------------------------
            for (unsigned int rNeuronEdx{0}; rNeuronEdx < neurons_[1].size(); ++rNeuronEdx)
            {
                for (unsigned int lNeuronEdx{0}; lNeuronEdx < neurons_[0].size(); ++lNeuronEdx)
                {
                    const ConnectiontLoc keyVal{lNeuronEdx, rNeuronEdx};
                    const auto connectionIt = connections_.find(keyVal);
                    if (connectionIt != connections_.end())
                    {
                        neurons_[1][rNeuronEdx].value += neurons_[0][lNeuronEdx].value * connectionIt->second;
                    }
                }
                neurons_[1][rNeuronEdx].value -= neurons_[1][rNeuronEdx].bias;
            }
            // Calculate the standard error and calculate the bias of the output neurons -------------------------------
            const unsigned int lastTestDataIdx {testDataIdx};
            float buffStandardError{0.0f};
            for (auto& neuron : neurons_[1])
            {
                buffStandardError += static_cast<float>(pow(neuron.value - data.GetTrainingTestValAtAnyIdx(testDataIdx),
                                                        2));
                neuron.bias += learningStep_ * (neuron.value - data.GetTrainingTestValAtAnyIdx(testDataIdx++));
            }
            standardError += 0.5f * buffStandardError;
            testDataIdx = lastTestDataIdx;
            // Adjusting the values of the links of the last layer ----------------------------------------------------
            for (unsigned int rNeuronEdx{0}; rNeuronEdx < neurons_[1].size(); ++rNeuronEdx)
            {
                for (unsigned int lNeuronEdx{0}; lNeuronEdx < neurons_[0].size(); ++lNeuronEdx)
                {
                    ConnectiontLoc keyVal{lNeuronEdx, rNeuronEdx};
                    auto connectionIt = connections_.find(keyVal);
                    if (connectionIt != connections_.end())
                    {
                        connectionIt->second -= learningStep_ * neurons_[0][lNeuronEdx].value
                            * (neurons_[1][rNeuronEdx].value - data.GetTrainingTestValAtAnyIdx(testDataIdx++));
                    }
                }
            }
            
            for(auto &neuron : neurons_[1])
            {
                neuron.value = 0;
            }
        }
        // Output information about the training parameters
        std::cout << "Epoch: " << epoch + 1 << '\n';
        std::cout << "Standard error: " << standardError << '\n';
        for(auto &i : connections_)
        {
            std::cout << i.second << '\n';
        }
        standardError = 0;
    }
}
