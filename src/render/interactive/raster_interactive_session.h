#pragma once

#include "render/interactive/interactive_render_session.h"
#include "render/rasterizer/rasterizer_renderer.h"

namespace renderer {

class RasterInteractiveSession final : public InteractiveRenderSession {
public:
    void reset(const Scene& scene, const RenderSettings& settings) override;
    void render_next_frame(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target) override;

private:
    RasterizerRenderer renderer_;
};

}  // namespace renderer
