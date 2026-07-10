#pragma once

#include "render/renderer.h"
#include "render/scene_intersector.h"

namespace renderer {

class RayTracerRenderer final : public IRenderer {
public:
    RenderResult render(const Scene& scene, const Camera& camera, const RenderSettings& settings) override;

private:
    Color trace_ray(
        const Ray& ray,
        const Scene& scene,
        const SceneIntersector& intersector,
        int depth,
        const RenderSettings& settings) const;
};

}  // namespace renderer
