#pragma once

#include "core/color.h"
#include "scene/primitive.h"

namespace renderer {

struct Material;
struct Scene;

struct SurfaceMaterialSample {
    Color base_color = Color::Zero();
    Color diffuse_color = Color::Zero();
    Color specular_f0 = Color::Constant(0.04f);
    Color specular_f90 = Color::Ones();
    Color diffuse_fresnel_f0 = Color::Constant(0.04f);
    Color diffuse_fresnel_f90 = Color::Ones();
    bool diffuse_fresnel_uses_max = false;
    Color emission = Color::Zero();
    float opacity = 1.0f;
    float metallic = 0.0f;
    float roughness = 1.0f;
    float occlusion = 1.0f;
    Vec3 shading_normal = Vec3::Zero();
};

float sample_material_opacity(
    const Scene& scene,
    const Material& material,
    const HitRecord& hit);

SurfaceMaterialSample evaluate_surface_material(
    const Scene& scene,
    const Material& material,
    const HitRecord& hit);

}  // namespace renderer
