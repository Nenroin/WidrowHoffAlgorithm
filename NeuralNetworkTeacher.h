#pragma once
#include "LearningData.h"

class NeuralNetwork;

// Implements the neural network learning functionality
class NeuralNetworkTeacher
{
protected:
    float learningStep_;
    float minRmsError_;

public:
    explicit NeuralNetworkTeacher(const float learningStep = 0.0, const float minRmsError = 0.0)
        : learningStep_{learningStep}, minRmsError_{minRmsError}
    {
    }
    ~NeuralNetworkTeacher() = default;
    NeuralNetworkTeacher(const NeuralNetworkTeacher&) = delete;
    NeuralNetworkTeacher(NeuralNetworkTeacher&&) = delete;
    NeuralNetworkTeacher& operator=(const NeuralNetworkTeacher&) = delete;
    NeuralNetworkTeacher& operator=(NeuralNetworkTeacher&&) = delete;
    
    static void InitNeuralNetwork(NeuralNetwork& neuralNetwork, const float weightsFrom = -0.5,
                                    const float weightsTo = 0.5, const float biasFrom = -0.5,
                                    const float biasTo = 0.5);

    void Teach(NeuralNetwork& neuralNetwork, const LearningData& data);

    float GetMinRmsError() const;
    float GetLearningStep() const;
    void SetMinRmsError(const float minRmsError);
    void SetLearningStep(const float learningStep);
};
