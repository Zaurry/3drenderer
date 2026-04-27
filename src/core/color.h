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
        std::clamp(color.x, 0.0, 1.0),
        std::clamp(color.y, 0.0, 1.0),
        std::clamp(color.z, 0.0, 1.0));
}

inline unsigned char channel_to_rgb8(double linear_channel) {
    const double clamped = std::clamp(linear_channel, 0.0, 1.0);
    // Gamma 校正：渲染计算在线性空间中进行，写入图片前转成接近屏幕显示的 sRGB 亮度。
    const double gamma_corrected = std::sqrt(clamped);
    return static_cast<unsigned char>(std::round(gamma_corrected * 255.0));
}

inline Rgb8 to_rgb8(const Color& linear_color) {
    return Rgb8{
        channel_to_rgb8(linear_color.x),
        channel_to_rgb8(linear_color.y),
        channel_to_rgb8(linear_color.z)};
}

}  // namespace renderer
