#pragma once

#include "core/random.h"
#include "render/renderer.h"
#include "render/scene_intersector.h"
#include "scene/material_evaluator.h"

namespace renderer {

class PathTracerRenderer final : public IRenderer {
public:
    RenderResult render(const Scene& scene, const Camera& camera, const RenderSettings& settings) override;

private:
    Color trace_path(
        const Ray& ray,
        const Scene& scene,
        const SceneIntersector& intersector,
        PcgRandom& rng,
        int depth) const;
    bool scatter(
        const Ray& ray,
        const HitRecord& hit,
        const Material& material,
        const SurfaceMaterialSample& surface,
        PcgRandom& rng,
        Color& attenuation,
        Ray& scattered) const;
};

}  // namespace renderer
