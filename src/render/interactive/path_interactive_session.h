#pragma once

#include "render/interactive/interactive_render_session.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "scene/instanced_scene.h"

#include <memory>

namespace renderer {

class PathInteractiveSession final : public InteractiveRenderSession {
public:
    void reset(
        const RenderSceneSnapshot& snapshot,
        const RenderSettings& settings) override;
    void render_next_frame(
        const RenderSceneSnapshot& snapshot,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target) override;
    void render_next_frame_to_cuda_surface(
        const RenderSceneSnapshot& snapshot,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        CudaSurfaceHandle surface);
    void download_current_cuda_frame(Framebuffer& target);

    int accumulated_samples() const;
    CudaStreamHandle cuda_stream_handle() const;
    const CudaPathStatistics* cuda_statistics() const;
    void set_cuda_presentation_state(bool interop_active, bool fallback_active);

private:
    int width_ = 0;
    int height_ = 0;
    int accumulated_samples_ = 0;
    int cuda_device_ = -1;
    std::unique_ptr<CudaPathInteractiveRenderer> cuda_renderer_;

    void update_cuda_frame_state(const RenderSettings& settings);
};

}  // namespace renderer
