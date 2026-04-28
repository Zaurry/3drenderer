#pragma once

#include "core/color.h"
#include "core/math/vec3.h"

namespace renderer {

struct PointLight {
    Vec3 position;
    Color intensity;
};

struct DirectionalLight {
    Vec3 direction;
    Color radiance;
};

}  // namespace renderer
