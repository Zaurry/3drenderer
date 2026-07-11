#include "render/scene_intersector.h"

#include "scene/material_evaluator.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace renderer {

namespace {

constexpr int max_transparent_layers = 64;

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
    double t_min,
    double t_max,
    HitRecord& hit) const {
    double search_min = t_min;
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

        const float advanced = std::nextafter(candidate.t, static_cast<float>(t_max));
        if (!(advanced > search_min)) {
            return false;
        }
        search_min = advanced;
    }
    return false;
}

bool SceneIntersector::occluded(const Ray& ray, double t_min, double t_max) const {
    HitRecord hit;
    return intersect(ray, t_min, t_max, hit);
}

bool SceneIntersector::intersect_nearest(
    const Ray& ray,
    double t_min,
    double t_max,
    HitRecord& hit) const {
    bool hit_anything = false;
    double closest_t = t_max;

    for (const Sphere& sphere : scene_.spheres) {
        HitRecord candidate;
        if (sphere.intersect(
                ray,
                static_cast<float>(t_min),
                static_cast<float>(closest_t),
                candidate)) {
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
    if (!usable_direction(geometric_normal)) {
        return position;
    }
    const double scale = std::max({
        1.0,
        static_cast<double>(std::abs(position.x())),
        static_cast<double>(std::abs(position.y())),
        static_cast<double>(std::abs(position.z()))});
    const double sign = dot(direction, geometric_normal) >= 0.0 ? 1.0 : -1.0;
    return position + geometric_normal * static_cast<float>(sign * scale * 1e-7);
}

}  // namespace renderer
