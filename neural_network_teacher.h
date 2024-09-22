#pragma once
#include "learning_data.h"

class linear_artif_neural_network;

// Implements the neural network learning functionality
class neural_network_teacher
{
protected:
    double learning_step_;
    double min_rms_error_;

public:
    explicit neural_network_teacher(const double learning_step = 0.0, const double min_rms_error = 0.0)
        : learning_step_{learning_step}, min_rms_error_{min_rms_error}
    {
    }

    ~neural_network_teacher() = default;
    neural_network_teacher(const neural_network_teacher&) = default;
    neural_network_teacher(neural_network_teacher&&) = default;
    neural_network_teacher& operator=(const neural_network_teacher&) = default;
    neural_network_teacher& operator=(neural_network_teacher&&) = default;

    static void init_neural_network(linear_artif_neural_network& neural_network, const double weights_from = -0.5,
                                    const double weights_to = 0.5, const double bias_from = -0.5,
                                    const double bias_to = 0.5);

    void teach(linear_artif_neural_network& neural_network, const learning_data& data)
    {
        
    }

    double get_min_rms_error() const;
    double get_learning_step() const;
    void set_min_rms_error(const double min_rms_error);
    void set_learning_step(const double learning_step);
};
