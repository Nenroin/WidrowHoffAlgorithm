#pragma once
#include "NeuralNetworkStructs.h"
#include <vector>

// Stores training data for further transmission to neural_network_teacher
class LearningData
{
protected:
    LData trainingData_;
    LData testData_;

public:
    LearningData() = default;
    ~LearningData() = default;
    LearningData(const LearningData&) = delete;
    LearningData(LearningData&&) = delete;
    LearningData& operator=(const LearningData&) = delete;
    LearningData& operator=(LearningData&&) = delete;

    void AddTrainingVal(const std::vector<double> &learn, const std::vector<double> &test);
    void AddTestVal(const std::vector<double> &learn, const std::vector<double> &test);

    // When transmitting an index that goes beyond the array,
    // it starts transmitting data starting from the first element
    double GetTrainingLearnValAtAnyIdx(const unsigned int idx) const;
    double GetTrainingTestValAtAnyIdx(const unsigned int idx) const;
    double GetTestLearnValAtAnyIdx(const unsigned int idx) const;
    double GetTestTestValAtAnyIdx(const unsigned int idx) const;
};
