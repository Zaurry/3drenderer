#pragma once

#include "core/math/vec3.h"

#include <algorithm>
#include <cmath>

namespace renderer {

using Color = Vec3;

struct Rgb8 {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

inline Color clamp_color(const Color& color) {
    return Color(
        std::clamp(color.x(), 0.0f, 1.0f),
        std::clamp(color.y(), 0.0f, 1.0f),
        std::clamp(color.z(), 0.0f, 1.0f));
}

inline unsigned char channel_to_rgb8(float linear_channel) {
    if (!std::isfinite(linear_channel)) {
        return linear_channel > 0.0f ? 255 : 0;
    }

    const float clamped = std::clamp(linear_channel, 0.0f, 1.0f);
    const float srgb = clamped <= 0.0031308f
        ? clamped * 12.92f
        : 1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f;
    return static_cast<unsigned char>(std::round(srgb * 255.0f));
}

inline Rgb8 to_rgb8(const Color& linear_color) {
    return Rgb8{
        channel_to_rgb8(linear_color.x()),
        channel_to_rgb8(linear_color.y()),
        channel_to_rgb8(linear_color.z())};
}

}  // namespace renderer
