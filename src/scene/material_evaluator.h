#pragma once

#include "core/color.h"
#include "scene/primitive.h"

namespace renderer {

struct Material;
struct Scene;

struct SurfaceMaterialSample {
    Color base_color = Color::Zero();
    double opacity = 1.0;
    Vec3 shading_normal = Vec3::Zero();
};

double sample_material_opacity(
    const Scene& scene,
    const Material& material,
    const Vec2& uv);

SurfaceMaterialSample evaluate_surface_material(
    const Scene& scene,
    const Material& material,
    const HitRecord& hit);

}  // namespace renderer
