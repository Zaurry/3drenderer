#pragma once

#include "core/color.h"
#include "render/interactive/interactive_render_session.h"
#include "render/pathtracer/pathtracer_renderer.h"
#include "render/pathtracer/cuda_pathtracer.h"

#include <memory>
#include <vector>

namespace renderer {

class PathInteractiveSession final : public InteractiveRenderSession {
public:
    void reset(const Scene& scene, const RenderSettings& settings) override;
    void render_next_frame(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target) override;

    int accumulated_samples() const;
    ExecutionBackend active_backend() const;

private:
    PathTracerRenderer renderer_;
    std::vector<Color> accumulated_;
    int width_ = 0;
    int height_ = 0;
    int accumulated_samples_ = 0;
    PathBackend requested_backend_ = PathBackend::Auto;
    ExecutionBackend active_backend_ = ExecutionBackend::Cpu;
    std::unique_ptr<CudaPathInteractiveRenderer> cuda_renderer_;

    void reset_accumulation(int width, int height);
};

}  // namespace renderer
