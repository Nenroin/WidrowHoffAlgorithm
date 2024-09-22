#include "neural_network_teacher.h"

double neural_network_teacher::get_min_rms_error() const
{
    return min_rms_error_;
}

void neural_network_teacher::set_min_rms_error(const double min_rms_error)
{
    min_rms_error_ = min_rms_error;
}

double neural_network_teacher::get_learning_step() const
{
    return learning_step_;
}

void neural_network_teacher::set_learning_step(const double learning_step)
{
    learning_step_ = learning_step;
}
