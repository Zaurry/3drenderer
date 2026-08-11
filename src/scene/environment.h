#pragma once

#include "core/color.h"
#include "core/math/types.h"

#include <array>
#include <filesystem>
#include <memory>
#include <vector>

namespace renderer {

struct EnvironmentMapSample {
    Vec3 direction = Vec3(0.0f, 1.0f, 0.0f);
    Color radiance = Color::Zero();
    float pdf = 0.0f;
};

struct DominantEnvironmentLight {
    bool valid = false;
    // Direction from a shaded point toward the extracted environment region.
    Vec3 direction = Vec3(0.0f, 1.0f, 0.0f);
    // Solid-angle integral of the selected region's RGB radiance.
    Color integrated_radiance = Color::Zero();
    float solid_angle = 0.0f;
    float angular_radius_radians = 0.0f;
    float energy_fraction = 0.0f;
    std::shared_ptr<const class EnvironmentMap> residual_map;
};

class EnvironmentMap {
public:
    static std::shared_ptr<const EnvironmentMap> load(
        const std::filesystem::path& path);

    EnvironmentMap(
        int width,
        int height,
        std::vector<Color> pixels,
        std::filesystem::path source_path = {});

    int width() const;
    int height() const;
    const std::vector<Color>& pixels() const;
    const std::filesystem::path& source_path() const;
    const std::array<Color, 9>& radiance_sh() const;
    const std::vector<float>& importance_pmf() const;
    const std::vector<float>& importance_cdf() const;

    Color sample_direction(const Vec3& direction) const;
    Color diffuse_irradiance(const Vec3& normal) const;
    float direction_pdf(const Vec3& direction) const;
    EnvironmentMapSample sample(float select, float jitter_u, float jitter_v) const;

    static Vec2 direction_to_uv(const Vec3& direction);
    static Vec3 uv_to_direction(const Vec2& uv);

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<Color> pixels_;
    std::filesystem::path source_path_;
    std::vector<float> pmf_;
    std::vector<float> cdf_;
    std::array<Color, 9> radiance_sh_{};

    void build_sampling_data();
    const Color& pixel(int x, int y) const;
    float texel_solid_angle(int y) const;
};

Color environment_radiance(
    const Color& fallback,
    const std::shared_ptr<const EnvironmentMap>& map,
    float intensity,
    float rotation_degrees,
    const Vec3& world_direction);

float environment_pdf(
    const std::shared_ptr<const EnvironmentMap>& map,
    float rotation_degrees,
    const Vec3& world_direction);

EnvironmentMapSample sample_environment(
    const Color& fallback,
    const std::shared_ptr<const EnvironmentMap>& map,
    float intensity,
    float rotation_degrees,
    float select,
    float jitter_u,
    float jitter_v);

DominantEnvironmentLight extract_dominant_environment_light(
    const std::shared_ptr<const EnvironmentMap>& map,
    float peak_threshold_ev = 3.0f,
    float minimum_energy_fraction = 0.01f);

}  // namespace renderer
