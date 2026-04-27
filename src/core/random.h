#pragma once

#include <random>

namespace renderer {

class Random {
public:
    explicit Random(unsigned int seed = 1) : engine_(seed) {}

    double next_double() {
        return distribution_(engine_);
    }

    double next_double(double min_value, double max_value) {
        return min_value + (max_value - min_value) * next_double();
    }

private:
    std::mt19937 engine_;
    std::uniform_real_distribution<double> distribution_{0.0, 1.0};
};

}  // namespace renderer
