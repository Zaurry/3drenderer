#pragma once

#include "core/math/bounds.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace renderer {

struct OpenGlDirectionalShadowFit {
    Vec3 position = Vec3::Zero();
    Vec3 direction = Vec3(0.0f, -1.0f, 0.0f);
    Vec3 right = Vec3::UnitX();
    Vec3 up = Vec3::UnitZ();
    float half_width = 1.0f;
    float half_height = 1.0f;
    float near_plane = 0.01f;
    float far_plane = 1.0f;
};

inline std::array<Vec3, 8> open_gl_shadow_bounds_corners(
    const Bounds3& bounds) {
    std::array<Vec3, 8> corners{};
    for (int mask = 0; mask < 8; ++mask) {
        corners[static_cast<std::size_t>(mask)] = Vec3(
            (mask & 1) != 0 ? bounds.max.x() : bounds.min.x(),
            (mask & 2) != 0 ? bounds.max.y() : bounds.min.y(),
            (mask & 4) != 0 ? bounds.max.z() : bounds.min.z());
    }
    return corners;
}

inline OpenGlDirectionalShadowFit open_gl_fit_directional_shadow(
    const Bounds3& source_bounds,
    const Vec3& source_direction,
    float padding_fraction,
    int resolution) {
    const Bounds3 bounds =
        source_bounds.min.allFinite() && source_bounds.max.allFinite()
        ? source_bounds
        : Bounds3(-Vec3::Ones(), Vec3::Ones());
    OpenGlDirectionalShadowFit fit;
    fit.direction = source_direction.allFinite() &&
            source_direction.squaredNorm() > 1.0e-20f
        ? source_direction.normalized()
        : Vec3(0.0f, -1.0f, 0.0f);
    const Vec3 preferred_up = std::abs(fit.direction.y()) < 0.98f
        ? Vec3::UnitY()
        : Vec3::UnitX();
    fit.right = fit.direction.cross(preferred_up).normalized();
    fit.up = fit.right.cross(fit.direction).normalized();

    float minimum_x = std::numeric_limits<float>::infinity();
    float maximum_x = -minimum_x;
    float minimum_y = minimum_x;
    float maximum_y = -minimum_x;
    float minimum_depth = minimum_x;
    float maximum_depth = -minimum_x;
    for (const Vec3& corner : open_gl_shadow_bounds_corners(bounds)) {
        minimum_x = std::min(minimum_x, fit.right.dot(corner));
        maximum_x = std::max(maximum_x, fit.right.dot(corner));
        minimum_y = std::min(minimum_y, fit.up.dot(corner));
        maximum_y = std::max(maximum_y, fit.up.dot(corner));
        minimum_depth = std::min(minimum_depth, fit.direction.dot(corner));
        maximum_depth = std::max(maximum_depth, fit.direction.dot(corner));
    }
    const float diagonal = std::max(
        (bounds.max - bounds.min).norm(),
        1.0e-3f);
    const float padding = std::max(
        diagonal * std::max(0.0f, padding_fraction),
        diagonal * 1.0e-4f);
    fit.half_width = std::max(
        (maximum_x - minimum_x) * 0.5f + padding,
        1.0e-3f);
    fit.half_height = std::max(
        (maximum_y - minimum_y) * 0.5f + padding,
        1.0e-3f);
    float center_x = (minimum_x + maximum_x) * 0.5f;
    float center_y = (minimum_y + maximum_y) * 0.5f;
    const float safe_resolution = static_cast<float>(std::max(resolution, 1));
    // Reserve one pre-snap texel so rounding the projection center can never
    // move an extreme scene corner outside the orthographic footprint.
    fit.half_width += 2.0f * fit.half_width / safe_resolution;
    fit.half_height += 2.0f * fit.half_height / safe_resolution;
    const float texel_x = 2.0f * fit.half_width / safe_resolution;
    const float texel_y = 2.0f * fit.half_height / safe_resolution;
    center_x = std::round(center_x / texel_x) * texel_x;
    center_y = std::round(center_y / texel_y) * texel_y;
    fit.position = fit.right * center_x + fit.up * center_y +
        fit.direction * (minimum_depth - padding);
    fit.near_plane = padding;
    fit.far_plane = std::max(
        maximum_depth - minimum_depth + 2.0f * padding,
        fit.near_plane + 1.0e-3f);
    return fit;
}

inline float open_gl_normalized_linear_shadow_depth(
    float distance,
    float near_plane,
    float far_plane) {
    if (!std::isfinite(distance) || !std::isfinite(near_plane) ||
        !std::isfinite(far_plane) || far_plane <= near_plane) {
        return 0.0f;
    }
    return std::clamp(
        (distance - near_plane) / (far_plane - near_plane),
        0.0f,
        1.0f);
}

inline float open_gl_reconstruct_linear_shadow_distance(
    float normalized_depth,
    float near_plane,
    float far_plane) {
    if (!std::isfinite(normalized_depth) || !std::isfinite(near_plane) ||
        !std::isfinite(far_plane) || far_plane <= near_plane) {
        return near_plane;
    }
    return near_plane +
        std::clamp(normalized_depth, 0.0f, 1.0f) *
        (far_plane - near_plane);
}

inline float open_gl_pcss_penumbra_texels(
    float receiver_distance,
    float blocker_distance,
    float projected_source_radius_texels,
    float maximum_penumbra_texels) {
    if (!std::isfinite(receiver_distance) ||
        !std::isfinite(blocker_distance) ||
        !std::isfinite(projected_source_radius_texels) ||
        !std::isfinite(maximum_penumbra_texels) ||
        receiver_distance <= blocker_distance ||
        blocker_distance <= 0.0f ||
        projected_source_radius_texels <= 0.0f ||
        maximum_penumbra_texels <= 0.0f) {
        return 0.0f;
    }
    return std::clamp(
        (receiver_distance - blocker_distance) / blocker_distance *
            projected_source_radius_texels,
        0.0f,
        maximum_penumbra_texels);
}

inline bool open_gl_shadow_budget_precedes(
    int left_priority,
    float left_contribution,
    int right_priority,
    float right_contribution) {
    if (left_priority != right_priority) {
        return left_priority > right_priority;
    }
    return left_contribution > right_contribution;
}

}  // namespace renderer
