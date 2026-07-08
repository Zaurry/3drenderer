#pragma once

#include "render/interactive/interactive_render_session.h"
#include "render/raytracer/raytracer_renderer.h"

namespace renderer {

class RayInteractiveSession final : public InteractiveRenderSession {
public:
    void reset(const Scene& scene, const RenderSettings& settings) override;
    void render_next_frame(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target) override;

private:
    RayTracerRenderer renderer_;
};

}  // namespace renderer
