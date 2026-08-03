#pragma once

#include "core/color.h"
#include "core/math/types.h"

namespace renderer {

struct PointLight {
    Vec3 position = Vec3::Zero();
    Color intensity = Color::Zero();
    float range = 0.0f;
};

struct DirectionalLight {
    Vec3 direction = Vec3::Zero();
    Color radiance = Color::Zero();
};

struct SpotLight {
    Vec3 position = Vec3::Zero();
    Vec3 direction = Vec3(0.0f, 0.0f, -1.0f);
    Color intensity = Color::Zero();
    float range = 0.0f;
    float inner_cone_radians = 0.0f;
    float outer_cone_radians = 0.7853981634f;
};

}  // namespace renderer
