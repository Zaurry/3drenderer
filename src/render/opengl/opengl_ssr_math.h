#pragma once

#include "core/math/types.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>

namespace renderer {

struct OpenGlSsrDepthRange {
    float minimum = std::numeric_limits<float>::max();
    float maximum = 0.0f;

    bool valid() const {
        return std::isfinite(minimum) && std::isfinite(maximum) &&
            minimum > 0.0f && maximum >= minimum;
    }
};

struct OpenGlSsrReductionCoverage {
    int first = 0;
    int last = 0;
};

enum class OpenGlSsrDepthIntervalRelation {
    Empty,
    InFront,
    Overlap,
    Behind,
};

inline int open_gl_ssr_hiz_level_count(int width, int height) {
    int extent = std::max({width, height, 1});
    int levels = 1;
    while (extent > 1) {
        extent = std::max(extent / 2, 1);
        ++levels;
    }
    return levels;
}

inline OpenGlSsrReductionCoverage open_gl_ssr_reduction_coverage(
    int destination_texel,
    int source_extent,
    int destination_extent) {
    source_extent = std::max(source_extent, 1);
    destination_extent = std::max(destination_extent, 1);
    destination_texel = std::clamp(
        destination_texel, 0, destination_extent - 1);
    const int first = destination_texel * source_extent /
        destination_extent;
    const int exclusive_end =
        ((destination_texel + 1) * source_extent + destination_extent - 1) /
        destination_extent;
    return {
        std::clamp(first, 0, source_extent - 1),
        std::clamp(exclusive_end - 1, first, source_extent - 1)};
}

inline OpenGlSsrDepthRange open_gl_ssr_reduce_depth_ranges(
    std::span<const OpenGlSsrDepthRange> ranges) {
    OpenGlSsrDepthRange result;
    for (const OpenGlSsrDepthRange& range : ranges) {
        if (!range.valid()) {
            continue;
        }
        result.minimum = std::min(result.minimum, range.minimum);
        result.maximum = std::max(result.maximum, range.maximum);
    }
    return result;
}

inline OpenGlSsrDepthIntervalRelation
open_gl_ssr_depth_interval_relation(
    float ray_minimum_depth,
    float ray_maximum_depth,
    const OpenGlSsrDepthRange& cell,
    float thickness) {
    if (!cell.valid() || !std::isfinite(ray_minimum_depth) ||
        !std::isfinite(ray_maximum_depth) || !std::isfinite(thickness)) {
        return OpenGlSsrDepthIntervalRelation::Empty;
    }
    const float ray_min = std::min(ray_minimum_depth, ray_maximum_depth);
    const float ray_max = std::max(ray_minimum_depth, ray_maximum_depth);
    thickness = std::max(thickness, 0.0f);
    if (ray_max < cell.minimum) {
        return OpenGlSsrDepthIntervalRelation::InFront;
    }
    if (ray_min > cell.maximum + thickness) {
        return OpenGlSsrDepthIntervalRelation::Behind;
    }
    return OpenGlSsrDepthIntervalRelation::Overlap;
}

inline bool open_gl_ssr_hiz_intersects(
    float ray_minimum_depth,
    float ray_maximum_depth,
    const OpenGlSsrDepthRange& cell,
    float thickness) {
    return open_gl_ssr_depth_interval_relation(
        ray_minimum_depth,
        ray_maximum_depth,
        cell,
        thickness) == OpenGlSsrDepthIntervalRelation::Overlap;
}

inline Vec3 open_gl_ssr_cosine_hemisphere_sample(const Vec2& sample) {
    constexpr float two_pi = 6.28318530717958647692f;
    const float u = std::clamp(sample.x(), 0.0f, 1.0f);
    const float v = std::clamp(sample.y(), 0.0f, 1.0f);
    const float radius = std::sqrt(u);
    const float phi = two_pi * v;
    return Vec3(
        radius * std::cos(phi),
        radius * std::sin(phi),
        std::sqrt(std::max(1.0f - u, 0.0f)));
}

inline bool open_gl_ssr_history_valid(
    float predicted_depth,
    float history_depth,
    float thickness,
    float world_normal_alignment) {
    if (!std::isfinite(predicted_depth) || !std::isfinite(history_depth) ||
        !std::isfinite(thickness) ||
        !std::isfinite(world_normal_alignment) || predicted_depth <= 0.0f ||
        history_depth <= 0.0f) {
        return false;
    }
    const float tolerance = std::max(
        2.0f * std::max(thickness, 0.0f),
        0.01f * predicted_depth);
    return std::abs(predicted_depth - history_depth) <= tolerance &&
        world_normal_alignment >= 0.85f;
}

inline float open_gl_ssr_history_weight(
    float history_length,
    int maximum_history_frames) {
    const float length = std::max(history_length, 0.0f);
    const int maximum = std::max(maximum_history_frames, 1);
    const float running_average = length / (length + 1.0f);
    const float cap = 1.0f - 1.0f / static_cast<float>(maximum);
    return std::clamp(std::min(running_average, cap), 0.0f, 1.0f);
}

inline float open_gl_ssr_projected_boundary_distance(
    const Vec3& origin,
    const Vec3& direction,
    float boundary_uv,
    float camera_viewport_extent,
    bool x_axis) {
    if (!origin.allFinite() || !direction.allFinite() ||
        !std::isfinite(boundary_uv) ||
        !std::isfinite(camera_viewport_extent) ||
        camera_viewport_extent <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }
    const float coordinate = x_axis ? origin.x() : origin.y();
    const float slope = x_axis ? direction.x() : direction.y();
    const float projected = (boundary_uv * 2.0f - 1.0f) *
        0.5f * camera_viewport_extent;
    const float denominator = slope + projected * direction.z();
    if (std::abs(denominator) <= 1.0e-8f) {
        return std::numeric_limits<float>::infinity();
    }
    const float distance = (-projected * origin.z() - coordinate) /
        denominator;
    return distance > 0.0f
        ? distance
        : std::numeric_limits<float>::infinity();
}

inline float open_gl_ssr_edge_fade(
    float uv_x,
    float uv_y,
    float edge_fade) {
    const float distance_to_edge = 1.0f - std::max(
        std::abs(uv_x * 2.0f - 1.0f),
        std::abs(uv_y * 2.0f - 1.0f));
    const float fade_width = std::clamp(edge_fade, 0.0f, 0.5f);
    if (fade_width <= 0.0f) {
        return 1.0f;
    }
    const float value = std::clamp(
        distance_to_edge / fade_width,
        0.0f,
        1.0f);
    return value * value * (3.0f - 2.0f * value);
}

inline bool open_gl_ssr_depth_crossing(
    float previous_depth_delta,
    float depth_delta) {
    return std::isfinite(previous_depth_delta) && std::isfinite(depth_delta) &&
        previous_depth_delta < 0.0f && depth_delta >= 0.0f;
}

inline bool open_gl_ssr_depth_within_thickness(
    float ray_depth,
    float surface_depth,
    float thickness) {
    if (!std::isfinite(ray_depth) || !std::isfinite(surface_depth) ||
        !std::isfinite(thickness) || surface_depth <= 0.0f ||
        thickness < 0.0f) {
        return false;
    }
    const float depth_delta = ray_depth - surface_depth;
    return depth_delta >= 0.0f && depth_delta <= thickness;
}

}  // namespace renderer
