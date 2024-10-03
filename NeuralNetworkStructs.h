#pragma once
#include <vector>

struct LData
{
    std::vector<double> learn;
    std::vector<double> test;
};

struct Neuron
{
    double bias;
    double value;
    
    Neuron() : bias{0.0f}, value{0.0f}
    {}
};
