#include "NeuralNetworkTeacher.h"
#include "NeuralNetwork.h"
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

void NeuralNetworkTeacher::Teach(NeuralNetwork& neuralNetwork, const LearningData& data)
{
    int learnDataIdx{0};
    int testDataIdx{0};
    {
        // Feeding input data to a neural network ---------------------------------------------------------------------
        for (auto& neuron : neuralNetwork.neurons_[0])
        {
            neuron.second.value = data.GetTrainingLearnValAtAnyIdx(learnDataIdx++);
        }
        // ------------------------------------------------------------------------------------------------------------

        // The output elements of the neural network are being calculated ---------------------------------------------
        auto lLayerIt = neuralNetwork.neurons_.begin();
        auto rLayerIt = ++neuralNetwork.neurons_.begin();

        while (rLayerIt != neuralNetwork.neurons_.end())
        {
            auto rNeuronIt = rLayerIt->begin();
            while (rNeuronIt != rLayerIt->end())
            {
                auto lNeuronIt = lLayerIt->begin();
                while (lNeuronIt != lLayerIt->end())
                {
                    ConnectiontLoc keyVal{lNeuronIt->first, rNeuronIt->first};
                    rNeuronIt->second.value += lNeuronIt->second.value * neuralNetwork.connections_.at(keyVal);
                    ++lNeuronIt;
                }
                rNeuronIt->second.value -= rNeuronIt->second.bias;
                ++rNeuronIt;
            }
            ++lLayerIt;
            ++rLayerIt;
        }
        // ------------------------------------------------------------------------------------------------------------

        // Calculation of the root mean square error of a neural network ----------------------------------------------
        const unsigned int last_layer_idx{
            static_cast<unsigned int>(neuralNetwork.layerNeuronsCount_.size()) - 1
        };
        const unsigned int last_layer_neurons_count{neuralNetwork.layerNeuronsCount_.at(last_layer_idx)};

        std::vector<float> standard_errors;
        const auto& neurons{neuralNetwork.neurons_};

        for (unsigned int idx{0}; idx < last_layer_neurons_count; ++idx)
        {
            const float right_value{data.GetTrainingTestValAtAnyIdx(testDataIdx++)};
            const float predicted_value{neurons.at({last_layer_idx, idx}).value};
            const float buff_val{(predicted_value - right_value)};
            const float standard_error{0.5 * (buff_val * buff_val)};

            standard_errors.push_back(standard_error);
        }
        // ------------------------------------------------------------------------------------------------------------

        // The weights and threshold of the neural network are changed ------------------------------------------------
        for (unsigned int right_layer_idx = layers_count - 1; right_layer_idx > 0; --right_layer_idx)
        {
            const unsigned int right_layer_neuron_count{neuralNetwork.layerNeuronsCount_.at(right_layer_idx)};

            for (unsigned int right_neuron_idx{0}; right_neuron_idx < right_layer_neuron_count; ++right_neuron_idx)
            {
                const unsigned int left_layer_neuron_count{neuralNetwork.layerNeuronsCount_.at(right_layer_idx - 1)};
                const unsigned int left_layer_idx{right_layer_idx - 1};

                for (unsigned int left_neuron_idx{0}; left_neuron_idx < left_layer_neuron_count; ++left_neuron_idx)
                {
                    WeightLoc weight_location{
                        neuron_loc_t(left_layer_idx, left_neuron_idx),
                        neuron_loc_t(right_layer_idx, right_neuron_idx)
                    };

                    auto it = neuralNetwork.neuronConnections_.find(weight_location);

                    if (it != neuralNetwork.neuronConnections_.end())
                    {
                        const Neuron buff_left_neuron{neuralNetwork.GetNeuron(left_layer_idx, left_neuron_idx)};
                        Neuron buff_right_neuron{neuralNetwork.GetNeuron(right_layer_idx, right_neuron_idx)};
                        //w(t+1)=w(t) - α(y - e) * x	
                        it->second = it->second - learningStep_ * (buff_right_neuron.value)
                            * buff_left_neuron.value;

                        // T_1 (t+1) = T_1(t) - α * (y - e)
                        buff_left_neuron.bias = buff_left_neuron.bias - learningStep_ * (buff_right_neuron.value
                            - buff_left_neuron.value);
                        neuralNetwork.SetNeuron(right_layer_idx, right_neuron_idx, buff_right_neuron);
                    }
                }
            }

            for (auto& neuron_val : neuralNetwork.neurons_)
            {
                neuron_val.second.value -= neuron_val.second.bias;
            }
        }
        // ------------------------------------------------------------------------------------------------------------
    }
}
