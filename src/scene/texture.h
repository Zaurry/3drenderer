#pragma once

#include "core/color.h"
#include "core/math/vec2.h"
#include "core/math/vec3.h"

#include <cmath>

namespace renderer {

struct ConstantTexture {
    Color color;

    Color sample(const Vec2& uv, const Vec3& p) const {
        (void)uv;
        (void)p;
        return color;
    }
};

struct CheckerTexture {
    Color even;
    Color odd;
    double scale = 8.0;

    Color sample(const Vec2& uv, const Vec3& p) const {
        (void)uv;
        const double checker = std::floor(p.x * scale) + std::floor(p.y * scale) + std::floor(p.z * scale);
        return static_cast<int>(checker) % 2 == 0 ? even : odd;
    }
};

}  // namespace renderer
