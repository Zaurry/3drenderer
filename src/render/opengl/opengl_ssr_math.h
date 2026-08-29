#pragma once

#include <algorithm>
#include <cmath>

namespace renderer {

inline constexpr float kOpenGlSsrMaxAnisotropy = 8.0f;
inline constexpr int kOpenGlSsrMaxFilterTaps = 9;

// C++ mirror of the GLSL screen_edge_fade() helper in shaders/opengl/ssr.frag
// so the screen-space reflection edge falloff can be verified without a live
// OpenGL context (see tests/opengl_contract_tests.cpp).
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

inline float open_gl_ssr_ggx_cone_tangent(float roughness) {
    if (!std::isfinite(roughness)) {
        return 0.0f;
    }
    roughness = std::clamp(roughness, 0.0f, 1.0f);
    const float alpha = roughness * roughness;
    const float half_maximum_slope = alpha * 0.6435942529f;
    return std::min(
        2.0f * half_maximum_slope /
            std::max(
                1.0f - half_maximum_slope * half_maximum_slope,
                1.0e-4f),
        8.0f);
}

struct OpenGlSsrLobeRadii {
    float major = 0.0f;
    float minor = 0.0f;
};

inline OpenGlSsrLobeRadii open_gl_ssr_lobe_radii(
    float roughness,
    float hit_distance,
    float n_dot_v) {
    if (!std::isfinite(roughness) || !std::isfinite(hit_distance) ||
        !std::isfinite(n_dot_v) || hit_distance <= 0.0f) {
        return {};
    }
    const float major = hit_distance *
        open_gl_ssr_ggx_cone_tangent(roughness);
    const float minor_scale = std::max(
        std::clamp(n_dot_v, 0.0f, 1.0f),
        1.0f / kOpenGlSsrMaxAnisotropy);
    return {major, major * minor_scale};
}

inline int open_gl_ssr_filter_tap_count(
    float major_radius_pixels,
    float minor_radius_pixels) {
    if (!std::isfinite(major_radius_pixels) ||
        !std::isfinite(minor_radius_pixels)) {
        return 1;
    }
    const float effective_minor = std::max(minor_radius_pixels, 1.0f);
    const float anisotropy = std::clamp(
        major_radius_pixels / effective_minor,
        1.0f,
        kOpenGlSsrMaxAnisotropy);
    const int half_taps = static_cast<int>(
        std::ceil((anisotropy - 1.0f) * 0.5f));
    return std::min(
        1 + 2 * half_taps,
        kOpenGlSsrMaxFilterTaps);
}

inline float open_gl_ssr_reflection_lod(
    float roughness,
    float hit_distance,
    float hit_view_depth,
    float camera_viewport_width,
    float camera_viewport_height,
    int texture_width,
    int texture_height,
    int mip_levels) {
    const float maximum_lod =
        static_cast<float>(std::max(mip_levels - 1, 0));
    if (!std::isfinite(roughness) || !std::isfinite(hit_distance) ||
        !std::isfinite(hit_view_depth) ||
        !std::isfinite(camera_viewport_width) ||
        !std::isfinite(camera_viewport_height) || hit_distance <= 0.0f ||
        hit_view_depth <= 0.0f || camera_viewport_width <= 0.0f ||
        camera_viewport_height <= 0.0f || texture_width <= 0 ||
        texture_height <= 0 || maximum_lod <= 0.0f) {
        return 0.0f;
    }
    const float radius_view = open_gl_ssr_lobe_radii(
        roughness,
        hit_distance,
        1.0f).major;
    const float radius_pixels_x = radius_view *
        static_cast<float>(texture_width) /
        (camera_viewport_width * hit_view_depth);
    const float radius_pixels_y = radius_view *
        static_cast<float>(texture_height) /
        (camera_viewport_height * hit_view_depth);
    const float footprint = std::max(
        std::max(radius_pixels_x, radius_pixels_y),
        1.0f);
    return std::clamp(std::log2(footprint), 0.0f, maximum_lod);
}

inline float open_gl_ssr_cone_edge_fade(
    float uv_x,
    float uv_y,
    float cone_radius_uv_x,
    float cone_radius_uv_y) {
    if (!std::isfinite(uv_x) || !std::isfinite(uv_y) ||
        !std::isfinite(cone_radius_uv_x) ||
        !std::isfinite(cone_radius_uv_y)) {
        return 0.0f;
    }
    cone_radius_uv_x = std::max(cone_radius_uv_x, 0.0f);
    cone_radius_uv_y = std::max(cone_radius_uv_y, 0.0f);
    if (std::max(cone_radius_uv_x, cone_radius_uv_y) <= 1.0e-8f) {
        return 1.0f;
    }
    const float edge_distance_x = std::max(
        std::min(uv_x, 1.0f - uv_x),
        0.0f);
    const float edge_distance_y = std::max(
        std::min(uv_y, 1.0f - uv_y),
        0.0f);
    const float support_x = cone_radius_uv_x > 1.0e-8f
        ? edge_distance_x / cone_radius_uv_x
        : 1.0f;
    const float support_y = cone_radius_uv_y > 1.0e-8f
        ? edge_distance_y / cone_radius_uv_y
        : 1.0f;
    const float value = std::clamp(
        std::min(support_x, support_y),
        0.0f,
        1.0f);
    return value * value * (3.0f - 2.0f * value);
}

}  // namespace renderer
