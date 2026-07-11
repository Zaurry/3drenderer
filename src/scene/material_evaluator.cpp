#include "scene/material_evaluator.h"

#include "scene/material.h"
#include "scene/scene.h"
#include "scene/texture.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace renderer {

namespace {

bool valid_texture_id(const Scene& scene, int texture_id) {
    return texture_id >= 0 && static_cast<std::size_t>(texture_id) < scene.textures.size();
}

Vec3 bumped_normal(
    const Scene& scene,
    const Material& material,
    const HitRecord& hit) {
    if (!valid_texture_id(scene, material.bump_texture_id) ||
        !hit.has_valid_uv_basis ||
        !usable_direction(hit.shading_normal) ||
        !usable_direction(hit.tangent) ||
        !usable_direction(hit.bitangent)) {
        return hit.shading_normal;
    }

    const ImageTexture& texture = scene.textures[static_cast<std::size_t>(material.bump_texture_id)];
    const Vec2 step = texture.texel_size();
    if (step.x() <= 0.0 || step.y() <= 0.0) {
        return hit.shading_normal;
    }

    const double left = texture.sample_scalar(Vec2(hit.uv.x() - step.x(), hit.uv.y()));
    const double right = texture.sample_scalar(Vec2(hit.uv.x() + step.x(), hit.uv.y()));
    const double down = texture.sample_scalar(Vec2(hit.uv.x(), hit.uv.y() - step.y()));
    const double up = texture.sample_scalar(Vec2(hit.uv.x(), hit.uv.y() + step.y()));
    const double dh_du = (right - left) * 0.5;
    const double dh_dv = (up - down) * 0.5;
    const Vec3 gradient = hit.tangent * dh_du + hit.bitangent * dh_dv;
    Vec3 perturbed = normalize(hit.shading_normal - gradient * material.bump_scale);
    if (!usable_direction(perturbed)) {
        return hit.shading_normal;
    }
    if (dot(perturbed, hit.geometric_normal) < 0.0) {
        perturbed = -perturbed;
    }
    return perturbed;
}

}  // namespace

double sample_material_opacity(
    const Scene& scene,
    const Material& material,
    const Vec2& uv) {
    double opacity = std::clamp(material.opacity, 0.0, 1.0);
    if (valid_texture_id(scene, material.opacity_texture_id)) {
        opacity *= scene.textures[static_cast<std::size_t>(material.opacity_texture_id)].sample_scalar(uv);
    }
    return std::clamp(opacity, 0.0, 1.0);
}

SurfaceMaterialSample evaluate_surface_material(
    const Scene& scene,
    const Material& material,
    const HitRecord& hit) {
    SurfaceMaterialSample sample;
    sample.base_color = sample_material_base_color(scene, material, hit.uv);
    sample.opacity = sample_material_opacity(scene, material, hit.uv);
    sample.shading_normal = bumped_normal(scene, material, hit);
    return sample;
}

}  // namespace renderer
