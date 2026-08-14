#pragma once

#include <array>
#include <cmath>

// Canonical shading constants and small formulas shared across the host C++
// code. The CUDA and GLSL copies of these formulas are source-text pinned by
// tests/opengl_shader_lint_tests.cpp; change values here and update the
// shader sources together.

namespace renderer {

// Rec. 709 luminance weights.
inline constexpr std::array<float, 3> kLuminanceWeights{
    0.2126f, 0.7152f, 0.0722f};

// Physically-based punctual light falloff used by the CUDA path tracer and
// the OpenGL forward pass (raster.frag). Kept here as the reference copy.
inline float punctual_range_attenuation(float distance, float range) {
    if (!(range > 0.0f)) {
        return 1.0f;
    }
    const float ratio = distance / range;
    const float squared = ratio * ratio;
    const float cutoff = std::max(0.0f, 1.0f - squared * squared);
    return cutoff * cutoff;
}

}  // namespace renderer
