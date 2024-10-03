#pragma once
#include "NeuralNetworkStructs.h"
#include "NeuralNetworkTeacher.h"

#include <unordered_map>
#include <vector>

// Designed to create a neural network structure
class NeuralNetwork
{
private:
    static unsigned int staticNeuronId_;

protected:
    std::vector<std::unordered_map<unsigned int, Neuron>> neurons_;
    std::unordered_map<ConnectiontLoc, float, WeightHash> connections_;

public:
    NeuralNetwork() = default;
    ~NeuralNetwork() = default;
    NeuralNetwork(const NeuralNetwork&) = delete;
    NeuralNetwork(NeuralNetwork&&) noexcept = delete;
    NeuralNetwork& operator=(const NeuralNetwork&) = delete;
    NeuralNetwork& operator=(NeuralNetwork&&) = delete;

    NeuralNetwork& AddLayer(const unsigned int neuronNum);
    NeuralNetwork& CreateConnection(const unsigned int lNeuronId, const unsigned int rNeuronId,
                                    const float value = 0.0);
    NeuralNetwork& FoundAndSetConnectionValue(const unsigned int lNeuronId, const unsigned int rNeuronId,
                                              const float value = 0.0);
    float FoundAndGetConnectionValue(const unsigned int lNeuronId, const unsigned int rNeuronId) const;
    Neuron GetNeuronValueById(const unsigned int id) const;

    static unsigned int GetLastNeuronId() { return staticNeuronId_; }

    static friend void NeuralNetworkTeacher::InitNeuralNetwork(NeuralNetwork& neuralNetwork, float weightsFrom,
                                                               float weightsTo, float biasFrom, float biasTo);
    friend void NeuralNetworkTeacher::Teach(NeuralNetwork& neuralNetwork, const LearningData& data,
                                            const unsigned int epochs);
};
