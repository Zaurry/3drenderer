#pragma once

#include "core/math/vec3.h"

namespace renderer {

struct Ray {
    Vec3 origin;
    Vec3 direction;

    Ray(const Vec3& ray_origin, const Vec3& ray_direction)
        : origin(ray_origin), direction(ray_direction) {}

    Vec3 at(float t) const {
        return origin + direction * t;
    }
};

}  // namespace renderer
