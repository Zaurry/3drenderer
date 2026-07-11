#pragma once

#include "core/math/vec2.h"
#include "core/math/vec3.h"

#include <array>
#include <vector>

namespace renderer {

struct RasterVertex {
    Vec3 view = Vec3::Zero();
    Vec3 world = Vec3::Zero();
    Vec2 uv = Vec2::Zero();
    Vec3 normal = Vec3::Zero();
};

std::vector<RasterVertex> clip_triangle_to_near_plane(
    const std::array<RasterVertex, 3>& triangle,
    double near_z);

Vec3 perspective_correct_weights(
    const Vec3& screen_weights,
    const Vec3& view_depths);

}  // namespace renderer
