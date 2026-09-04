#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace renderer {

// Kulla--Conty directional albedo for a unit-Fresnel GGX BRDF using the
// height-correlated Smith masking-shadowing function.  Values are stored at
// the endpoints of a regular (NdotV, perceptual roughness) grid.
inline constexpr int kGgxEnergyLutSize = 32;
inline constexpr std::array<float, kGgxEnergyLutSize * kGgxEnergyLutSize>
    kGgxDirectionalAlbedoLut{
#include "render/generated/ggx_directional_albedo_lut.inc"
    };
inline constexpr std::array<float, kGgxEnergyLutSize> kGgxAverageAlbedoLut{
#include "render/generated/ggx_average_albedo_lut.inc"
};

inline float ggx_directional_albedo(float n_dot_v, float roughness) {
    const float x = std::clamp(n_dot_v, 0.0f, 1.0f) *
        static_cast<float>(kGgxEnergyLutSize - 1);
    const float y = std::clamp(roughness, 0.0f, 1.0f) *
        static_cast<float>(kGgxEnergyLutSize - 1);
    const int x0 = std::min(static_cast<int>(x), kGgxEnergyLutSize - 2);
    const int y0 = std::min(static_cast<int>(y), kGgxEnergyLutSize - 2);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const auto value = [](int row, int column) {
        return kGgxDirectionalAlbedoLut[static_cast<std::size_t>(
            row * kGgxEnergyLutSize + column)];
    };
    const float lower = value(y0, x0) + (value(y0, x1) - value(y0, x0)) * tx;
    const float upper = value(y1, x0) + (value(y1, x1) - value(y1, x0)) * tx;
    return std::clamp(lower + (upper - lower) * ty, 0.0f, 1.0f);
}

inline float ggx_average_albedo(float roughness) {
    const float position = std::clamp(roughness, 0.0f, 1.0f) *
        static_cast<float>(kGgxEnergyLutSize - 1);
    const int lower = std::min(
        static_cast<int>(position),
        kGgxEnergyLutSize - 2);
    const float blend = position - static_cast<float>(lower);
    return std::clamp(
        kGgxAverageAlbedoLut[static_cast<std::size_t>(lower)] +
            (kGgxAverageAlbedoLut[static_cast<std::size_t>(lower + 1)] -
             kGgxAverageAlbedoLut[static_cast<std::size_t>(lower)]) *
                blend,
        0.0f,
        1.0f);
}

}  // namespace renderer
