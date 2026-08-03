#include "render/scene_intersector.h"

#include "scene/material_evaluator.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace renderer {

namespace {

constexpr int max_transparent_layers = 64;
constexpr float kMinimumRayOriginScale = 1.0f;
// Triangle intersection and hit reconstruction accumulate several float
// roundoff steps. One representable step is not enough to keep secondary rays
// above the surface, so reserve a small ULP-scaled error budget.
constexpr float kRayOriginOffsetScale = 32.0f * std::numeric_limits<float>::epsilon();
constexpr float kOffsetNormalSquaredThreshold = 1e-24f;

bool material_exists(const Scene& scene, int material_id) {
    return material_id >= 0 && static_cast<std::size_t>(material_id) < scene.materials.size();
}

AlphaMode effective_alpha_mode(const Material& material) {
    if (material.type == MaterialType::Pbr) {
        return material.alpha_mode;
    }
    return material.opacity_texture_id >= 0 || material.opacity < 1.0f
        ? AlphaMode::Mask
        : AlphaMode::Opaque;
}

float stochastic_visibility_sample(const Ray& ray, const HitRecord& hit) {
    auto mix = [](std::uint32_t value) {
        value ^= value >> 16U;
        value *= 0x7feb352dU;
        value ^= value >> 15U;
        value *= 0x846ca68bU;
        return value ^ (value >> 16U);
    };
    std::uint32_t hash = 0x9e3779b9U;
    for (int axis = 0; axis < 3; ++axis) {
        hash ^= mix(std::bit_cast<std::uint32_t>(ray.direction[axis]) +
            static_cast<std::uint32_t>(axis) * 0x85ebca6bU);
        hash ^= mix(std::bit_cast<std::uint32_t>(hit.position[axis]) +
            static_cast<std::uint32_t>(axis) * 0xc2b2ae35U);
    }
    hash ^= mix(std::bit_cast<std::uint32_t>(hit.t));
    return static_cast<float>(hash >> 8U) * (1.0f / 16777216.0f);
}

}  // namespace

SceneIntersector::SceneIntersector(const Scene& scene)
    : scene_(scene) {
    bvh_.build(scene.triangles);
}

bool SceneIntersector::intersect(
    const Ray& ray,
    float t_min,
    float t_max,
    HitRecord& hit) const {
    float search_min = t_min;
    for (int layer = 0; layer < max_transparent_layers; ++layer) {
        HitRecord candidate;
        if (!intersect_nearest(ray, search_min, t_max, candidate)) {
            return false;
        }

        if (!material_exists(scene_, candidate.material_id)) {
            hit = candidate;
            return true;
        }

        const Material& material = scene_.materials[static_cast<std::size_t>(candidate.material_id)];
        const bool visible_side = material.two_sided || candidate.front_face;
        const float opacity = sample_material_opacity(scene_, material, candidate);
        const AlphaMode alpha_mode = effective_alpha_mode(material);
        const bool alpha_visible = alpha_mode == AlphaMode::Opaque ||
            alpha_mode == AlphaMode::Blend || opacity >= material.alpha_cutoff;
        if (visible_side && alpha_visible) {
            hit = candidate;
            return true;
        }

        const float advanced = std::nextafter(candidate.t, t_max);
        if (!(advanced > search_min)) {
            return false;
        }
        search_min = advanced;
    }
    return false;
}

bool SceneIntersector::occluded(const Ray& ray, float t_min, float t_max) const {
    float search_min = t_min;
    for (int layer = 0; layer < max_transparent_layers; ++layer) {
        HitRecord candidate;
        if (!intersect_nearest(ray, search_min, t_max, candidate)) {
            return false;
        }
        if (!material_exists(scene_, candidate.material_id)) {
            return true;
        }
        const Material& material =
            scene_.materials[static_cast<std::size_t>(candidate.material_id)];
        const bool visible_side = material.two_sided || candidate.front_face;
        const float opacity = sample_material_opacity(scene_, material, candidate);
        const AlphaMode alpha_mode = effective_alpha_mode(material);
        const bool alpha_visible = alpha_mode == AlphaMode::Opaque ||
            (alpha_mode == AlphaMode::Mask && opacity >= material.alpha_cutoff) ||
            (alpha_mode == AlphaMode::Blend &&
             stochastic_visibility_sample(ray, candidate) < opacity);
        if (visible_side && alpha_visible) {
            return true;
        }
        const float advanced = std::nextafter(candidate.t, t_max);
        if (!(advanced > search_min)) {
            return false;
        }
        search_min = advanced;
    }
    return false;
}

bool SceneIntersector::intersect_nearest(
    const Ray& ray,
    float t_min,
    float t_max,
    HitRecord& hit) const {
    bool hit_anything = false;
    float closest_t = t_max;

    for (const Sphere& sphere : scene_.spheres) {
        HitRecord candidate;
        if (sphere.intersect(ray, t_min, closest_t, candidate)) {
            hit_anything = true;
            closest_t = candidate.t;
            hit = candidate;
        }
    }

    HitRecord triangle_hit;
    if (bvh_.intersect(ray, t_min, closest_t, triangle_hit)) {
        hit_anything = true;
        hit = triangle_hit;
    }
    return hit_anything;
}

Vec3 offset_ray_origin(
    const Vec3& position,
    const Vec3& geometric_normal,
    const Vec3& direction) {
    if (!geometric_normal.allFinite() ||
        geometric_normal.squaredNorm() <= kOffsetNormalSquaredThreshold) {
        return position;
    }
    const float scale = std::max(kMinimumRayOriginScale, position.cwiseAbs().maxCoeff());
    const float sign = direction.dot(geometric_normal) >= 0.0f ? 1.0f : -1.0f;
    Vec3 offset_position =
        position + geometric_normal * (sign * scale * kRayOriginOffsetScale);
    const float infinity = std::numeric_limits<float>::infinity();
    for (int axis = 0; axis < 3; ++axis) {
        const float selected_normal_component = sign * geometric_normal[axis];
        if (selected_normal_component != 0.0f &&
            offset_position[axis] == position[axis]) {
            offset_position[axis] = std::nextafter(
                position[axis],
                selected_normal_component > 0.0f ? infinity : -infinity);
        }
    }
    return offset_position;
}

}  // namespace renderer
