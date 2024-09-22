#pragma once
#include <vector>

class learning_data
{
protected:
    int input_neurons_num_;
    int end_neurons_num_;

    std::vector<double> training_data_;
    std::vector<double> test_data_;

public:
    explicit learning_data(const int input_neurons_num = 0, const int end_neurons_num = 0)
        : input_neurons_num_{input_neurons_num}, end_neurons_num_{end_neurons_num}
    {
    }

    ~learning_data() = default;
    learning_data(const learning_data&) = default;
    learning_data(learning_data&&) = default;
    learning_data& operator=(const learning_data&) = default;
    learning_data& operator=(learning_data&&) = default;

    void add_training_val(const double value);
    void add_test_val(const double value);
    double get_training_val_at(const unsigned long long int idx) const;
    double get_test_val_at(const unsigned long long int idx) const;

    int get_input_neurons_num() const;
    int get_end_neurons_num() const;
};
