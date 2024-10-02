#include "LearningData.h"

void LearningData::AddTrainingVal(const std::vector<float> &learn, const std::vector<float> &test)
{
    trainingData_.learn = learn;
    trainingData_.test = test;
}

void LearningData::AddTestVal(const std::vector<float> &learn, const std::vector<float> &test)
{
    testData_.learn = learn;
    testData_.test = test;
}

float LearningData::GetTrainingLearnValAtAnyIdx(const unsigned int idx) const
{
    const unsigned int wrappedIdx{idx % static_cast<unsigned int>(trainingData_.learn.size())};
    return trainingData_.learn.at(wrappedIdx);
}

float LearningData::GetTrainingTestValAtAnyIdx(const unsigned int idx) const
{
    const unsigned int wrappedIdx{idx % static_cast<unsigned int>(trainingData_.test.size())};
    return trainingData_.test.at(wrappedIdx);
}

float LearningData::GetTestLearnValAtAnyIdx(const unsigned int idx) const
{
    const unsigned int wrappedIdx{idx % static_cast<unsigned int>(testData_.learn.size())};
    return testData_.learn.at(wrappedIdx);
}

float LearningData::GetTestTestValAtAnyIdx(const unsigned int idx) const
{
    const unsigned int wrappedIdx{idx % static_cast<unsigned int>(testData_.test.size())};
    return testData_.test.at(wrappedIdx);
}