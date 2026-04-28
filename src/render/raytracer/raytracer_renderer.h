#pragma once

#include "acceleration/bvh.h"
#include "render/renderer.h"

namespace renderer {

class RayTracerRenderer final : public IRenderer {
public:
    RenderResult render(const Scene& scene, const Camera& camera, const RenderSettings& settings) override;

private:
    Color trace_ray(const Ray& ray, const Scene& scene, const Bvh& bvh, int depth, const RenderSettings& settings) const;
    bool hit_scene(const Ray& ray, const Scene& scene, const Bvh& bvh, double t_min, double t_max, HitRecord& hit) const;
};

}  // namespace renderer
