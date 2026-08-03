#pragma once

#include "core/color.h"
#include "core/math/types.h"

namespace renderer {

struct PbrSurface {
    Color diffuse_color = Color(0.8f, 0.8f, 0.8f);
    Color specular_f0 = Color::Constant(0.04f);
    Color specular_f90 = Color::Ones();
    Color diffuse_fresnel_f0 = Color::Constant(0.04f);
    Color diffuse_fresnel_f90 = Color::Ones();
    float roughness = 1.0f;
    bool diffuse_fresnel_uses_max = false;
};

struct PbrEvaluation {
    Color brdf = Color::Zero();
    float pdf = 0.0f;
};

struct PbrSample {
    Vec3 direction = Vec3::Zero();
    Color weight = Color::Zero();
    float pdf = 0.0f;
    bool valid = false;
};

PbrEvaluation evaluate_pbr(
    const PbrSurface& surface,
    const Vec3& normal,
    const Vec3& outgoing,
    const Vec3& incoming);

PbrSample sample_pbr(
    const PbrSurface& surface,
    const Vec3& normal,
    const Vec3& outgoing,
    float component_sample,
    float sample_x,
    float sample_y);

float power_heuristic(float pdf_a, float pdf_b);

}  // namespace renderer
