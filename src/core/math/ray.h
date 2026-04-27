#pragma once

#include "core/math/vec3.h"

namespace renderer {

struct Ray {
    Vec3 origin;
    Vec3 direction;

    constexpr Ray(const Vec3& ray_origin, const Vec3& ray_direction)
        : origin(ray_origin), direction(ray_direction) {}

    constexpr Vec3 at(double t) const {
        return origin + direction * t;
    }
};

}  // namespace renderer
