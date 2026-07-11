#pragma once

#include "acceleration/bvh.h"
#include "scene/scene.h"

namespace renderer {

class SceneIntersector {
public:
    explicit SceneIntersector(const Scene& scene);

    bool intersect(
        const Ray& ray,
        float t_min,
        float t_max,
        HitRecord& hit) const;
    bool occluded(const Ray& ray, float t_min, float t_max) const;

private:
    bool intersect_nearest(
        const Ray& ray,
        float t_min,
        float t_max,
        HitRecord& hit) const;

    const Scene& scene_;
    Bvh bvh_;
};

Vec3 offset_ray_origin(
    const Vec3& position,
    const Vec3& geometric_normal,
    const Vec3& direction);

}  // namespace renderer
