#include "NeuralNetwork.h"

unsigned int NeuralNetwork::staticNeuronId_{ 1 };

NeuralNetwork& NeuralNetwork::AddLayer(const unsigned int neuronNum)
{
    neurons_.emplace_back();

    const unsigned int layerIdx = static_cast<unsigned int>(neurons_.size()) - 1;

    for (unsigned int i{0}; i < neuronNum; i++)
    {
        neurons_[layerIdx][staticNeuronId_++];
    }

    return *this;
}

NeuralNetwork& NeuralNetwork::CreateConnection(const unsigned int lNeuronId, const unsigned int rNeuronId,
                                               const float value)
{
    connections_[ConnectiontLoc(lNeuronId, rNeuronId)] = value;
    return *this;
}

NeuralNetwork& NeuralNetwork::FoundAndSetConnectionValue(const unsigned int lNeuronId, const unsigned int rNeuronId,
                                                         const float value)
{
    connections_.at(ConnectiontLoc(lNeuronId, rNeuronId)) = value;
    return *this;
}

float NeuralNetwork::FoundAndGetConnectionValue(const unsigned int lNeuronId, const unsigned int rNeuronId) const
{
    return connections_.at(ConnectiontLoc(lNeuronId, rNeuronId));
}

Neuron NeuralNetwork::GetNeuronValueById(const unsigned int id) const
{
    for (const auto& neuron : neurons_)
    {
        auto it = neuron.find(id);
        if (it != neuron.end())
        {
            return it->second;
        }
    }
    return {};
}