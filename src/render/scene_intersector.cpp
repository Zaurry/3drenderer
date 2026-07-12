#include "render/scene_intersector.h"

#include "scene/material_evaluator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace renderer {

namespace {

constexpr int max_transparent_layers = 64;
constexpr float kMinimumRayOriginScale = 1.0f;
constexpr float kRayOriginOffsetScale = 1e-7f;
constexpr float kOffsetNormalSquaredThreshold = 1e-24f;

bool material_exists(const Scene& scene, int material_id) {
    return material_id >= 0 && static_cast<std::size_t>(material_id) < scene.materials.size();
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
        const float opacity = sample_material_opacity(scene_, material, candidate.uv);
        if (visible_side && opacity >= material.alpha_cutoff) {
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
    HitRecord hit;
    return intersect(ray, t_min, t_max, hit);
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
