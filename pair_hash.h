#pragma once
#include <unordered_map>
#include <utility>
#include <functional>

//Unordered Map does not contain a hash function for a pair like it has for int, string, etc,
//So if we want to hash a pair then we have to explicitly provide it with a hash function that can hash a pair.
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
