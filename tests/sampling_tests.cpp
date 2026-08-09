#include "test_framework.h"

#include "scene/environment.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    constexpr int width = 8;
    constexpr int height = 4;
    constexpr float pi = 3.14159265358979323846f;
    std::vector<renderer::Color> pixels;
    pixels.reserve(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float value = 0.1f + static_cast<float>(1 + x + 3 * y);
            pixels.emplace_back(value, value * 0.7f, value * 0.3f);
        }
    }
    const renderer::EnvironmentMap environment(
        width,
        height,
        std::move(pixels));

    float integrated_pdf = 0.0f;
    for (int y = 0; y < height; ++y) {
        const float theta0 = pi * static_cast<float>(y) / height;
        const float theta1 = pi * static_cast<float>(y + 1) / height;
        const float solid_angle =
            (2.0f * pi / width) *
            (std::cos(theta0) - std::cos(theta1));
        for (int x = 0; x < width; ++x) {
            const renderer::Vec2 uv(
                (static_cast<float>(x) + 0.5f) / width,
                (static_cast<float>(y) + 0.5f) / height);
            integrated_pdf += environment.direction_pdf(
                renderer::EnvironmentMap::uv_to_direction(uv)) * solid_angle;
        }
    }
    RENDER_CHECK(nearly_equal(integrated_pdf, 1.0f, 2.0e-5f));

    constexpr int sample_count = 32768;
    std::vector<int> counts(width * height, 0);
    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
        const renderer::EnvironmentMapSample sample = environment.sample(
            (static_cast<float>(sample_index) + 0.5f) / sample_count,
            0.375f,
            0.625f);
        const renderer::Vec2 uv =
            renderer::EnvironmentMap::direction_to_uv(sample.direction);
        const int x = std::clamp(static_cast<int>(uv.x() * width), 0, width - 1);
        const int y = std::clamp(static_cast<int>(uv.y() * height), 0, height - 1);
        ++counts[static_cast<std::size_t>(y * width + x)];
    }
    const auto& pmf = environment.importance_pmf();
    for (std::size_t index = 0; index < pmf.size(); ++index) {
        const float measured = static_cast<float>(counts[index]) / sample_count;
        RENDER_CHECK(std::abs(measured - pmf[index]) < 1.0e-4f);
    }
    std::cout << "sampling_tests: all tests passed\n";
}
