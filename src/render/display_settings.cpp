#include "render/display_settings.h"

#include <algorithm>
#include <cmath>

namespace renderer {

namespace {

float reinhard(float value) {
    if (!std::isfinite(value)) {
        return value > 0.0f ? 1.0f : 0.0f;
    }
    return value / (1.0f + value);
}

float aces(float value) {
    if (!std::isfinite(value)) {
        return value > 0.0f ? 1.0f : 0.0f;
    }
    constexpr float a = 2.51f;
    constexpr float b = 0.03f;
    constexpr float c = 2.43f;
    constexpr float d = 0.59f;
    constexpr float e = 0.14f;
    return std::clamp(
        (value * (a * value + b)) / (value * (c * value + d) + e),
        0.0f,
        1.0f);
}

}  // namespace

Color apply_display_transform(const Color& linear_color, const DisplaySettings& settings) {
    const float exposure = std::isfinite(settings.exposure_ev)
        ? std::exp2(settings.exposure_ev)
        : 1.0f;
    Color transformed = (linear_color * exposure).cwiseMax(0.0f);
    if (settings.tone_mapper == ToneMapper::Reinhard) {
        transformed = transformed.unaryExpr([](float value) { return reinhard(value); });
    } else if (settings.tone_mapper == ToneMapper::Aces) {
        transformed = transformed.unaryExpr([](float value) { return aces(value); });
    }
    return transformed;
}

Rgb8 to_display_rgb8(const Color& linear_color, const DisplaySettings& settings) {
    return to_rgb8(apply_display_transform(linear_color, settings));
}

}  // namespace renderer
