#pragma once

#include <cstdint>
#include <random>

namespace renderer {

class PcgRandom {
public:
    explicit PcgRandom(std::uint64_t seed = 1) {
        // 固定种子的随机序列能让测试和图片对比可复现；同一个 seed 必须得到同一张噪声图。
        state_ = 0;
        increment_ = (seed << 1U) | 1U;
        next_u32();
        state_ += seed;
        next_u32();
    }

    std::uint32_t next_u32() {
        const std::uint64_t old_state = state_;
        state_ = old_state * 6364136223846793005ULL + increment_;
        const auto xorshifted = static_cast<std::uint32_t>(((old_state >> 18U) ^ old_state) >> 27U);
        const auto rot = static_cast<std::uint32_t>(old_state >> 59U);
        return (xorshifted >> rot) | (xorshifted << ((0U - rot) & 31U));
    }

    double next_double() {
        constexpr double scale = 1.0 / 4294967296.0;
        return static_cast<double>(next_u32()) * scale;
    }

private:
    std::uint64_t state_ = 0;
    std::uint64_t increment_ = 1;
};

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
