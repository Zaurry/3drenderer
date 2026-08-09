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
    RenderResult render_cpu(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings);
    Color trace_path(
        const Ray& ray,
        const Scene& scene,
        const SceneIntersector& intersector,
        PcgRandom& rng,
        const PathRenderSettings& path_settings) const;
    bool scatter(
        const Ray& ray,
        const HitRecord& hit,
        const Material& material,
        const SurfaceMaterialSample& surface,
        PcgRandom& rng,
        Color& attenuation,
        Ray& scattered,
        float& pdf,
        bool& delta) const;
    Color estimate_direct_lighting(
        const Scene& scene,
        const SceneIntersector& intersector,
        const HitRecord& hit,
        const SurfaceMaterialSample& surface,
        const Vec3& outgoing) const;
    Color estimate_environment_lighting(
        const Scene& scene,
        const SceneIntersector& intersector,
        const HitRecord& hit,
        const SurfaceMaterialSample& surface,
        const Vec3& outgoing,
        PcgRandom& rng) const;
};

}  // namespace renderer
