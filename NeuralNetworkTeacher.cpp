#include "NeuralNetworkTeacher.h"
#include "NeuralNetwork.h"
#include <iostream>
#include <random>

float NeuralNetworkTeacher::GetMinRmsError() const
{
    return minRmsError_;
}

void NeuralNetworkTeacher::SetMinRmsError(const float minRmsError)
{
    minRmsError_ = minRmsError;
}

float NeuralNetworkTeacher::GetLearningStep() const
{
    return learningStep_;
}

void NeuralNetworkTeacher::SetLearningStep(const float learningStep)
{
    learningStep_ = learningStep;
}

void NeuralNetworkTeacher::InitNeuralNetwork(NeuralNetwork& neuralNetwork, const float weightsFrom,
                                             const float weightsTo, const float biasFrom, const float biasTo)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> randomWeight(weightsFrom, weightsTo);
    std::uniform_real_distribution<> randomBias(biasFrom, biasTo);

    for (auto& layer : neuralNetwork.neurons_)
    {
        for (auto& neuron : layer)
        {
            neuron.second.value = static_cast<float>(randomBias(gen));
        }
    }

    for (auto& neuronConnection : neuralNetwork.connections_)
    {
        neuronConnection.second = static_cast<float>(randomWeight(gen));
    }
}

void NeuralNetworkTeacher::Teach(NeuralNetwork& neuralNetwork, const LearningData& data, const unsigned int epochs)
{
    unsigned int learnDataIdx{0};
    unsigned int testDataIdx{0};
    unsigned int lastTestDataIdx{0};
    float standardError{0.0f};

    for (unsigned int epoch{0}; epoch < epochs; ++epoch)
    {
        for (unsigned int i{0}; i < 45; ++i)
        {
            // Assign incoming values to the first layer of neurons ----------------------------------------------------
            for (auto& neuron : neuralNetwork.neurons_[0])
            {
                neuron.second.value = data.GetTrainingLearnValAtAnyIdx(learnDataIdx++);
            }
            // ------------------------------------------------------------------------------------------------------------

            // Calculate the values of the output neurons --------------------------------------------------------------
            {
                auto lLayerIt = neuralNetwork.neurons_.begin();
                auto rLayerIt = ++neuralNetwork.neurons_.begin();

                while (rLayerIt != neuralNetwork.neurons_.end())
                {
                    auto rNeuronIt = rLayerIt->begin();
                    while (rNeuronIt != rLayerIt->end())
                    {
                        auto lNeuronIt = lLayerIt->begin();
                        rNeuronIt->second.value = 0.0f;
                        rNeuronIt->second.oldValue = 0.0f;
                        while (lNeuronIt != lLayerIt->end())
                        {
                            ConnectiontLoc keyVal{lNeuronIt->first, rNeuronIt->first};
                            auto connectionIt = neuralNetwork.connections_.find(keyVal);
                            if (connectionIt != neuralNetwork.connections_.end())
                            {
                                rNeuronIt->second.value += lNeuronIt->second.value * connectionIt->second;
                            }
                            ++lNeuronIt;
                        }
                        rNeuronIt->second.value -= rNeuronIt->second.bias;
                        ++rNeuronIt;
                    }
                    ++lLayerIt;
                    ++rLayerIt;
                }
            }
            // ------------------------------------------------------------------------------------------------------------

            // Calculate the standard error and calculate the bias of the output neurons -------------------------------

            auto rLayerIt = --neuralNetwork.neurons_.end();
            auto lLayerIt = neuralNetwork.neurons_.end() - 2;

            lastTestDataIdx = testDataIdx;
            standardError = 0.0f;
            for (auto& neuron : *rLayerIt)
            {
                standardError += (neuron.second.value - data.GetTrainingTestValAtAnyIdx(testDataIdx))
                    * (neuron.second.value - data.GetTrainingTestValAtAnyIdx(testDataIdx));

                neuron.second.bias -= learningStep_ * (neuron.second.value
                    - data.GetTrainingTestValAtAnyIdx(testDataIdx++));
            }
            standardError *= 0.5f;
            testDataIdx = lastTestDataIdx;

            // --------------------------------------------------------------------------------------------------------

            // Adjusting the values of the links of the last layer ----------------------------------------------------
            for (auto lNeuronIt = lLayerIt->begin(); lNeuronIt != lLayerIt->end(); ++lNeuronIt)
            {
                lastTestDataIdx = testDataIdx;
                for (auto rNeuronIt = rLayerIt->begin(); rNeuronIt != rLayerIt->end(); ++rNeuronIt)
                {
                    ConnectiontLoc keyVal{lNeuronIt->first, rNeuronIt->first};
                    auto connectionIt = neuralNetwork.connections_.find(keyVal);
                    if (connectionIt != neuralNetwork.connections_.end())
                    {
                        connectionIt->second -= learningStep_ * lNeuronIt->second.value
                            * (rNeuronIt->second.value - data.GetTrainingTestValAtAnyIdx(testDataIdx++));
                    }
                }
                testDataIdx = lastTestDataIdx;
            }
            // ------------------------------------------------------------------------------------------------------------

            // Adjust the values of the left layer of neurons and save their old values for use in the formula ------------
            for (auto lNeuronIt = lLayerIt->begin(); lNeuronIt != lLayerIt->end(); ++lNeuronIt)
            {
                lastTestDataIdx = testDataIdx;
                lNeuronIt->second.oldValue = lNeuronIt->second.value;
                lNeuronIt->second.value = 0.0f;
                for (auto rNeuronIt = rLayerIt->begin(); rNeuronIt != rLayerIt->end(); ++rNeuronIt)
                {
                    ConnectiontLoc keyVal{lNeuronIt->first, rNeuronIt->first};
                    auto connectionIt = neuralNetwork.connections_.find(keyVal);
                    if (connectionIt != neuralNetwork.connections_.end())
                    {
                        lNeuronIt->second.value += data.GetTrainingTestValAtAnyIdx(testDataIdx++) * connectionIt->second;
                    }
                }
                testDataIdx = lastTestDataIdx;
            }
            // ------------------------------------------------------------------------------------------------------------

            // Adjust all other values in the hidden layers ---------------------------------------------------------------
            if (lLayerIt != neuralNetwork.neurons_.begin())
            {
                --lLayerIt;
                --rLayerIt;

                while (true)
                {
                    // Adjusting the values of the links
                    for (auto lNeuronIt = lLayerIt->begin(); lNeuronIt != lLayerIt->end(); ++lNeuronIt)
                    {
                        for (auto rNeuronIt = rLayerIt->begin(); rNeuronIt != rLayerIt->end(); ++rNeuronIt)
                        {
                            ConnectiontLoc keyVal{lNeuronIt->first, rNeuronIt->first};
                            auto connectionIt = neuralNetwork.connections_.find(keyVal);
                            if (connectionIt != neuralNetwork.connections_.end())
                            {
                                connectionIt->second -= learningStep_ * lNeuronIt->second.value
                                    * (rNeuronIt->second.oldValue - rNeuronIt->second.value);
                            }
                        }
                    }

                    // We adjust the values of the left layer of neurons and keep the old one
                    for (auto lNeuronIt = lLayerIt->begin(); lNeuronIt != lLayerIt->end(); ++lNeuronIt)
                    {
                        lNeuronIt->second.oldValue = lNeuronIt->second.value;
                        lNeuronIt->second.value = 0.0f;
                        for (auto rNeuronIt = rLayerIt->begin(); rNeuronIt != rLayerIt->end(); ++rNeuronIt)
                        {
                            ConnectiontLoc keyVal{lNeuronIt->first, rNeuronIt->first};
                            auto connectionIt = neuralNetwork.connections_.find(keyVal);
                            if (connectionIt != neuralNetwork.connections_.end())
                            {
                                lNeuronIt->second.value += rNeuronIt->second.value * connectionIt->second;
                            }
                        }
                    }

                    if (lLayerIt == neuralNetwork.neurons_.begin()) break;

                    --lLayerIt;
                    --rLayerIt;
                }
            }
            // ------------------------------------------------------------------------------------------------------------

            // Shifting the index of the test sample to the appropriate value
            testDataIdx += (--neuralNetwork.neurons_.end())->size();
       } 
        // Output information about the training parameters
        std::cout << "Epoch: " << epoch + 1 << '\n';
        std::cout << "Standard error: " << standardError << '\n';

        for(auto &i : neuralNetwork.connections_)
        {
            std::cout << i.second << '\n';
        }
    }
}
