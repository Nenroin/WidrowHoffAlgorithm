#pragma once
#include <unordered_map>
#include <utility>
#include <functional>

//Unordered Map does not contain a hash function for a pair like it has for int, string, etc,
//So if we want to hash a pair then we have to explicitly provide it with a hash function that can hash a pair.

using neuron_loc_t = std::pair<unsigned int, unsigned int>;

struct pair_hash
{
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& pair) const
    {
        const size_t hash1 = std::hash<T1>{}(pair.first);
        const size_t hash2 = std::hash<T2>{}(pair.second);

        return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2));
    }
};

struct neuron
{
    double bias;
    double value;
};

struct weight_loc
{
    neuron_loc_t left_neuron;
    neuron_loc_t right_neuron;

    bool operator==(const weight_loc& other) const
    {
        return (left_neuron == other.left_neuron) && (right_neuron == other.right_neuron);
    }
};

struct weight_hash
{
    std::size_t operator()(const weight_loc& w) const
    {
        const size_t hash1 = std::hash<unsigned int>{}(w.left_neuron.first);
        const size_t hash2 = std::hash<unsigned int>{}(w.left_neuron.second);
        const size_t hash3 = std::hash<unsigned int>{}(w.right_neuron.first);
        const size_t hash4 = std::hash<unsigned int>{}(w.right_neuron.second);

        return hash1 ^ (hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2)) ^
            (hash3 ^ (hash4 + 0x9e3779b9 + (hash3 << 6) + (hash3 >> 2)));
    }
};
