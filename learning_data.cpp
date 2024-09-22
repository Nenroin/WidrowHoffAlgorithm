#include "learning_data.h"

void learning_data::add_training_val(const double value)
{
    training_data_.emplace_back(value);
}

void learning_data::add_test_val(const double value)
{
    test_data_.emplace_back(value);
}

double learning_data::get_training_val_at(const unsigned long long int idx) const
{
    const unsigned long long int wrapped_idx{idx % training_data_.size()};
    return training_data_.at(wrapped_idx);
}

double learning_data::get_test_val_at(const unsigned long long int idx) const
{
    const unsigned long long int wrapped_idx{idx % test_data_.size()};
    return test_data_.at(wrapped_idx);
}

int learning_data::get_input_neurons_num() const
{
    return input_neurons_num_;
}

int learning_data::get_end_neurons_num() const
{
    return end_neurons_num_;
}
