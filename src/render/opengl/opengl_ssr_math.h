#pragma once

#include <algorithm>
#include <cmath>

namespace renderer {

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

inline float open_gl_ssr_reflection_lod(float roughness, int mip_levels) {
    const float alpha = std::clamp(roughness, 0.0f, 1.0f);
    return alpha * alpha * static_cast<float>(std::max(mip_levels - 1, 0));
}

}  // namespace renderer
