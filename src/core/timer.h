#pragma once

#include <chrono>

namespace renderer {

class Timer {
public:
    Timer() : start_(Clock::now()) {}

    void reset() {
        start_ = Clock::now();
    }

    float elapsed_seconds() const {
        return std::chrono::duration<float>(Clock::now() - start_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

}  // namespace renderer
