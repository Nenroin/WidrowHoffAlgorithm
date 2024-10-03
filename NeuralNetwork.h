#pragma once
#include "NeuralNetworkStructs.h"
#include "LearningData.h"
#include <unordered_map>
#include <vector>
#include <array>

// Designed to create a neural network structure
class NeuralNetwork
{
protected:
    float learningStep_;
    float minRmsError_;
    std::array<std::vector<Neuron>, 2> neurons_;
    std::unordered_map<ConnectiontLoc, float, WeightHash> connections_;
    NeuralNetwork& CalculateNeuronValueIfConnectionExists(const unsigned int lNeuronEdx, const unsigned int rNeuronEdx);

public:
    explicit NeuralNetwork(const float learningStep = 0.01f, const float minRmsError = 0.01f) :
        learningStep_{learningStep}, minRmsError_{minRmsError}
    {
        neurons_[0].emplace_back();
        neurons_[1].emplace_back();
    }

    NeuralNetwork& SetNeuronNumberFirstLayer(const unsigned int neuronNum);
    NeuralNetwork& SetNeuronNumberSecondLayer(const unsigned int neuronNum);
    NeuralNetwork& CreateConnection(const unsigned int lNeuronEdx, const unsigned int rNeuronEdx,
                                    const float value = 0.0);
    NeuralNetwork& SetConnectionValueIfConnectionExists(const unsigned int lNeuronEdx, const unsigned int rNeuronEdx,
                                                        const float value = 0.0);
    float GetConnectionValue(const unsigned int lNeuronEdx, const unsigned int rNeuronEdx) const;

    // Teaching -------------------------------------------------------------------------------------------------------
    void InitNeuralNetwork(const float weightsFrom = -0.5, const float weightsTo = 0.5,
                           const float biasFrom = -0.5, const float biasTo = 0.5);
    void Teach(const LearningData& data, const unsigned int epochs);
    // ----------------------------------------------------------------------------------------------------------------
    ~NeuralNetwork() = default;
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) noexcept = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(NeuralNetwork&&) = delete;
    // ----------------------------------------------------------------------------------------------------------------
};
