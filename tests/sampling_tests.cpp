#include "test_framework.h"

#include "scene/environment.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace {

renderer::Color integrate_environment(const renderer::EnvironmentMap& environment) {
    constexpr float pi = 3.14159265358979323846f;
    renderer::Color integral = renderer::Color::Zero();
    for (int y = 0; y < environment.height(); ++y) {
        const float theta0 = pi * static_cast<float>(y) / environment.height();
        const float theta1 =
            pi * static_cast<float>(y + 1) / environment.height();
        const float solid_angle =
            (2.0f * pi / environment.width()) *
            (std::cos(theta0) - std::cos(theta1));
        for (int x = 0; x < environment.width(); ++x) {
            integral += environment.pixels()[static_cast<std::size_t>(
                y * environment.width() + x)] * solid_angle;
        }
    }
    return integral;
}

RENDER_TEST(test_dominant_environment_light_extraction) {
    constexpr int width = 16;
    constexpr int height = 8;
    std::vector<renderer::Color> pixels(
        static_cast<std::size_t>(width * height),
        renderer::Color(0.05f, 0.05f, 0.05f));
    const renderer::Color dominant_color(40.0f, 20.0f, 5.0f);
    const auto set_pixel = [&](int x, int y, const renderer::Color& color) {
        pixels[static_cast<std::size_t>(y * width + x)] = color;
    };

    // The brightest connected component deliberately crosses the longitude seam.
    for (const int y : {3, 4}) {
        set_pixel(0, y, dominant_color);
        set_pixel(width - 1, y, dominant_color);
        set_pixel(width / 2, y, renderer::Color(12.0f, 12.0f, 12.0f));
    }

    const auto environment = std::make_shared<const renderer::EnvironmentMap>(
        width,
        height,
        std::move(pixels));
    const renderer::Color original_integral = integrate_environment(*environment);
    const renderer::DominantEnvironmentLight dominant =
        renderer::extract_dominant_environment_light(environment, 3.0f, 0.01f);

    RENDER_CHECK(dominant.valid);
    RENDER_CHECK(dominant.residual_map != nullptr);
    RENDER_CHECK(dominant.residual_map != environment);
    RENDER_CHECK(dominant.direction.x() < -0.98f);
    RENDER_CHECK(std::abs(dominant.direction.y()) < 0.05f);
    RENDER_CHECK(std::abs(dominant.direction.z()) < 0.05f);
    RENDER_CHECK(dominant.integrated_radiance.x() >
        dominant.integrated_radiance.y());
    RENDER_CHECK(dominant.integrated_radiance.y() >
        dominant.integrated_radiance.z());
    RENDER_CHECK(dominant.energy_fraction > 0.5f);
    RENDER_CHECK(dominant.energy_fraction < 1.0f);
    RENDER_CHECK(dominant.solid_angle > 0.0f);
    RENDER_CHECK(dominant.angular_radius_radians > 0.0f);
    for (const int y : {3, 4}) {
        RENDER_CHECK(dominant.residual_map->pixels()[static_cast<std::size_t>(
            y * width)].isZero());
        RENDER_CHECK(dominant.residual_map->pixels()[static_cast<std::size_t>(
            y * width + width - 1)].isZero());
        RENDER_CHECK(!dominant.residual_map->pixels()[static_cast<std::size_t>(
            y * width + width / 2)].isZero());
    }
    const renderer::Color reconstructed =
        integrate_environment(*dominant.residual_map) +
        dominant.integrated_radiance;
    RENDER_CHECK(reconstructed.isApprox(original_integral, 2.0e-5f));

    const renderer::DominantEnvironmentLight rejected_by_energy =
        renderer::extract_dominant_environment_light(environment, 3.0f, 1.0f);
    RENDER_CHECK(!rejected_by_energy.valid);
    RENDER_CHECK(rejected_by_energy.residual_map == environment);
    const renderer::DominantEnvironmentLight tight_threshold =
        renderer::extract_dominant_environment_light(environment, 0.0f, 0.01f);
    RENDER_CHECK(tight_threshold.valid);
    RENDER_CHECK(tight_threshold.direction.x() < -0.98f);
    const renderer::DominantEnvironmentLight broad_threshold =
        renderer::extract_dominant_environment_light(environment, 20.0f, 0.01f);
    RENDER_CHECK(!broad_threshold.valid);
    RENDER_CHECK(broad_threshold.residual_map == environment);

    const auto flat_environment =
        std::make_shared<const renderer::EnvironmentMap>(
            width,
            height,
            std::vector<renderer::Color>(
                static_cast<std::size_t>(width * height),
                renderer::Color::Ones()));
    const renderer::DominantEnvironmentLight flat =
        renderer::extract_dominant_environment_light(flat_environment);
    RENDER_CHECK(!flat.valid);
    RENDER_CHECK(flat.residual_map == flat_environment);
    RENDER_CHECK(!renderer::extract_dominant_environment_light(nullptr).valid);
    RENDER_CHECK(!renderer::extract_dominant_environment_light(
        environment,
        std::numeric_limits<float>::quiet_NaN(),
        0.01f).valid);
}

}  // namespace

RENDER_TEST(test_environment_importance_pdf_and_sampling) {
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
}
