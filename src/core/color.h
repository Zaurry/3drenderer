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

inline unsigned char channel_to_rgb8(double linear_channel) {
    if (!std::isfinite(linear_channel)) {
        return linear_channel > 0.0 ? 255 : 0;
    }

    const double clamped = std::clamp(linear_channel, 0.0, 1.0);
    const double srgb = clamped <= 0.0031308
        ? clamped * 12.92
        : 1.055 * std::pow(clamped, 1.0 / 2.4) - 0.055;
    return static_cast<unsigned char>(std::round(srgb * 255.0));
}

inline Rgb8 to_rgb8(const Color& linear_color) {
    return Rgb8{
        channel_to_rgb8(linear_color.x()),
        channel_to_rgb8(linear_color.y()),
        channel_to_rgb8(linear_color.z())};
}

}  // namespace renderer
