#pragma once
#include <vector>

struct LData
{
    std::vector<float> learn;
    std::vector<float> test;
};

struct Neuron
{
    float bias;
    float value;
    
    Neuron() : bias{0.0f}, value{0.0f}
    {
    }
};

struct ConnectiontLoc
{
    unsigned int lNeuronId;
    unsigned int rNeuronId;

    ConnectiontLoc(const unsigned int lNeuronId, const unsigned int rNeuronId) : lNeuronId{lNeuronId},
        rNeuronId{rNeuronId}
    {
    }

    bool operator==(const ConnectiontLoc& other) const
    {
        return (lNeuronId == other.lNeuronId) && (rNeuronId == other.rNeuronId);
    }
};

struct WeightHash
{
    std::size_t operator()(const ConnectiontLoc& w) const
    {
        const size_t hash1 = std::hash<unsigned int>{}(w.lNeuronId);
        const size_t hash2 = std::hash<unsigned int>{}(w.rNeuronId);

        return hash1 ^ (hash2 << 1);
    }
};
