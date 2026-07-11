#pragma once

#include "core/color.h"
#include "core/math/vec3.h"

namespace renderer {

struct PointLight {
    Vec3 position = Vec3::Zero();
    Color intensity = Color::Zero();
};

struct DirectionalLight {
    Vec3 direction = Vec3::Zero();
    Color radiance = Color::Zero();
};

}  // namespace renderer
