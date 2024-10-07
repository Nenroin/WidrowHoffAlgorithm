#include "LearningData.h"

void LearningData::AddTrainingVal(const std::vector<double> &learn, const std::vector<double> &test)
{
    trainingData_.learn = learn;
    trainingData_.test = test;
}

void LearningData::AddTestVal(const std::vector<double> &learn, const std::vector<double> &test)
{
    testData_.learn = learn;
    testData_.test = test;
}

double LearningData::GetTrainingLearnValAtAnyIdx(const unsigned int idx) const
{
    const unsigned int wrappedIdx{idx % static_cast<unsigned int>(trainingData_.learn.size())};
    return trainingData_.learn.at(wrappedIdx);
}

double LearningData::GetTrainingTestValAtAnyIdx(const unsigned int idx) const
{
    const unsigned int wrappedIdx{idx % static_cast<unsigned int>(trainingData_.test.size())};
    return trainingData_.test.at(wrappedIdx);
}

double LearningData::GetTestLearnValAtAnyIdx(const unsigned int idx) const
{
    const unsigned int wrappedIdx{idx % static_cast<unsigned int>(testData_.learn.size())};
    return testData_.learn.at(wrappedIdx);
}

double LearningData::GetTestTestValAtAnyIdx(const unsigned int idx) const
{
    const unsigned int wrappedIdx{idx % static_cast<unsigned int>(testData_.test.size())};
    return testData_.test.at(wrappedIdx);
}

unsigned int LearningData::GetTrainingDataSize() const
{
    return static_cast<unsigned int>(trainingData_.learn.size()) + static_cast<unsigned int>(trainingData_.test.size());
}

unsigned int LearningData::GetTestDataSize() const
{
    return static_cast<unsigned int>(testData_.learn.size()) + static_cast<unsigned int>(testData_.test.size());
}