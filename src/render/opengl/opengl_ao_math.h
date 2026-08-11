#pragma once

#include "core/math/types.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace renderer {

inline Vec3 open_gl_ao_reconstruct_view_position(
    const Vec2& uv,
    float linear_depth,
    const Vec2& camera_viewport) {
    const Vec2 ndc = uv * 2.0f - Vec2::Ones();
    return Vec3(
        ndc.x() * 0.5f * camera_viewport.x() * linear_depth,
        ndc.y() * 0.5f * camera_viewport.y() * linear_depth,
        -linear_depth);
}

inline float open_gl_ao_projected_radius_pixels(
    float radius_world,
    float linear_depth,
    float camera_viewport_height,
    int render_height) {
    return std::max(radius_world, 0.0f) *
        static_cast<float>(std::max(render_height, 1)) /
        std::max(linear_depth * camera_viewport_height, 1.0e-8f);
}

inline float open_gl_ssao_range_weight(
    float radius_world,
    float depth_delta) {
    const float ratio = std::max(radius_world, 0.0f) /
        std::max(std::abs(depth_delta), 1.0e-8f);
    const float value = std::clamp(ratio, 0.0f, 1.0f);
    return value * value * (3.0f - 2.0f * value);
}

inline float open_gl_gtao_slice_visibility(
    float normal_angle,
    float horizon_negative,
    float horizon_positive,
    float projected_normal_length = 1.0f) {
    const float cosine_normal = std::cos(normal_angle);
    const auto side_visibility = [&](float horizon) {
        return projected_normal_length *
            (cosine_normal + 2.0f * horizon * std::sin(normal_angle) -
             std::cos(2.0f * horizon - normal_angle)) *
            0.25f;
    };
    return std::clamp(
        side_visibility(horizon_negative) +
            side_visibility(horizon_positive),
        0.0f,
        1.0f);
}

inline Vec3 open_gl_gtao_slice_bent_local(
    float normal_angle,
    float horizon_negative,
    float horizon_positive,
    const Vec2& slice_direction) {
    const float t0 = (
        6.0f * std::sin(horizon_negative - normal_angle) -
        std::sin(3.0f * horizon_negative - normal_angle) +
        6.0f * std::sin(horizon_positive - normal_angle) -
        std::sin(3.0f * horizon_positive - normal_angle) +
        16.0f * std::sin(normal_angle) -
        3.0f * (std::sin(horizon_negative + normal_angle) +
                std::sin(horizon_positive + normal_angle))) /
        12.0f;
    const float t1 = (
        -std::cos(3.0f * horizon_negative - normal_angle) -
        std::cos(3.0f * horizon_positive - normal_angle) +
        8.0f * std::cos(normal_angle) -
        3.0f * (std::cos(horizon_negative + normal_angle) +
                std::cos(horizon_positive + normal_angle))) /
        12.0f;
    return Vec3(
        slice_direction.x() * t0,
        slice_direction.y() * t0,
        t1);
}

inline Mat3 open_gl_ao_rotation_positive_z_to(const Vec3& target_value) {
    const Vec3 from = Vec3::UnitZ();
    const Vec3 target = target_value.normalized();
    const float cosine = std::clamp(from.dot(target), -1.0f, 1.0f);
    if (cosine < -0.9999f) {
        Mat3 rotation = Mat3::Identity();
        rotation(1, 1) = -1.0f;
        rotation(2, 2) = -1.0f;
        return rotation;
    }
    const Vec3 cross_value = from.cross(target);
    Mat3 skew;
    skew <<
        0.0f, -cross_value.z(), cross_value.y(),
        cross_value.z(), 0.0f, -cross_value.x(),
        -cross_value.y(), cross_value.x(), 0.0f;
    return Mat3::Identity() + skew +
        (skew * skew) / std::max(1.0f + cosine, 1.0e-5f);
}

inline float open_gl_gtso_visibility(
    float bent_reflection_alignment,
    float roughness,
    float ambient_visibility,
    float normal_view_alignment) {
    roughness = std::clamp(roughness, 0.0f, 1.0f);
    ambient_visibility = std::clamp(ambient_visibility, 0.0f, 1.0f);
    normal_view_alignment = std::clamp(normal_view_alignment, 0.0f, 1.0f);
    if (ambient_visibility >= 0.9999f) {
        return 1.0f;
    }
    const float scalar_visibility = std::clamp(
        std::pow(
            std::max(normal_view_alignment + ambient_visibility, 0.0f),
            std::exp2(-16.0f * roughness - 1.0f)) -
            1.0f + ambient_visibility,
        0.0f,
        1.0f);
    const float cone_cosine = std::sqrt(
        std::max(0.0f, 1.0f - ambient_visibility));
    const float lobe_width =
        0.04f + (1.0f - 0.04f) * roughness * roughness;
    const float edge0 = cone_cosine - lobe_width;
    const float edge1 = cone_cosine + lobe_width;
    const float normalized = std::clamp(
        (bent_reflection_alignment - edge0) /
            std::max(edge1 - edge0, 1.0e-8f),
        0.0f,
        1.0f);
    const float directional_visibility =
        normalized * normalized * (3.0f - 2.0f * normalized);
    return std::clamp(
        (directional_visibility * ambient_visibility) * (1.0f - roughness) +
            scalar_visibility * roughness,
        0.0f,
        1.0f);
}

}  // namespace renderer
