#pragma once

#include "acceleration/bvh.h"
#include "core/random.h"
#include "render/renderer.h"

namespace renderer {

class PathTracerRenderer final : public IRenderer {
public:
    RenderResult render(const Scene& scene, const Camera& camera, const RenderSettings& settings) override;

private:
    Color trace_path(const Ray& ray, const Scene& scene, const Bvh& bvh, PcgRandom& rng, int depth) const;
    bool scatter(
        const Ray& ray,
        const HitRecord& hit,
        const Material& material,
        PcgRandom& rng,
        Color& attenuation,
        Ray& scattered) const;
    bool hit_scene(const Ray& ray, const Scene& scene, const Bvh& bvh, double t_min, double t_max, HitRecord& hit) const;
};

}  // namespace renderer
