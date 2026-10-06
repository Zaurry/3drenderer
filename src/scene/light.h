#pragma once

#include "core/color.h"
#include "core/math/types.h"

#include <cmath>
#include <cstdint>

namespace renderer {

struct PointLight {
    Vec3 position = Vec3::Zero();
    Color intensity = Color::Zero();
    float range = 0.0f;
    float source_radius = 0.05f;
    bool casts_shadows = true;
    int shadow_priority = 0;
    std::uint64_t stable_id = 0;
};

struct DirectionalLight {
    Vec3 direction = Vec3::Zero();
    Color radiance = Color::Zero();
    float angular_radius_radians = 0.00464257581f;
    bool casts_shadows = true;
    int shadow_priority = 0;
    std::uint64_t stable_id = 0;
};

struct SpotLight {
    Vec3 position = Vec3::Zero();
    Vec3 direction = Vec3(0.0f, 0.0f, -1.0f);
    Color intensity = Color::Zero();
    float range = 0.0f;
    float inner_cone_radians = 0.0f;
    float outer_cone_radians = 0.7853981634f;
    float source_radius = 0.05f;
    bool casts_shadows = true;
    int shadow_priority = 0;
    std::uint64_t stable_id = 0;
};

struct RectAreaLight {
    Vec3 position = Vec3::Zero();
    Vec3 axis_u = Vec3(0.5f, 0.0f, 0.0f);
    Vec3 axis_v = Vec3(0.0f, 0.5f, 0.0f);
    Color radiance = Color::Zero();
    bool two_sided = false;
    bool casts_shadows = true;
    int shadow_priority = 0;
    std::uint64_t stable_id = 0;
};

// The canonical visible quad is wound toward local -Z. Consequently the
// emitting side is axis_v x axis_u (not axis_u x axis_v). Keep this helper as
// the single CPU-side definition so scene import, CUDA geometry and editor
// diagnostics cannot silently choose opposite faces.
inline Vec3 rect_area_light_emission_direction(const RectAreaLight& light) {
    Vec3 direction = light.axis_v.cross(light.axis_u);
    const float length = direction.norm();
    if (!direction.allFinite() || !std::isfinite(length) || length <= 1.0e-10f) {
        return Vec3::Zero();
    }
    return direction / length;
}

inline bool rect_area_light_emits_toward(
    const RectAreaLight& light,
    const Vec3& receiver_position) {
    const Vec3 direction = rect_area_light_emission_direction(light);
    if (direction.squaredNorm() <= 0.0f) {
        return false;
    }
    if (light.two_sided) {
        return true;
    }
    const Vec3 to_receiver = receiver_position - light.position;
    return to_receiver.allFinite() &&
        direction.dot(to_receiver) > 0.0f;
}

}  // namespace renderer
