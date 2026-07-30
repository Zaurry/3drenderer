#pragma once

#include "core/color.h"
#include "render/interactive/interactive_render_session.h"
#include "render/pathtracer/pathtracer_renderer.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "scene/instanced_scene.h"

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
    void render_next_frame_to_cuda_surface(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        CudaSurfaceHandle surface,
        const InstancedSceneView* instanced_scene = nullptr);
    void reset_instanced(
        const Scene& scene,
        const RenderSettings& settings,
        const InstancedSceneView& instanced_scene);
    void render_next_frame_instanced(
        const Scene& scene,
        const InstancedSceneView& instanced_scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target);
    void download_current_cuda_frame(Framebuffer& target);

    int accumulated_samples() const;
    ExecutionBackend active_backend() const;
    CudaStreamHandle cuda_stream_handle() const;
    const CudaPathStatistics* cuda_statistics() const;
    void set_cuda_presentation_state(bool interop_active, bool fallback_active);

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
    void update_cuda_frame_state(const RenderSettings& settings);
};

}  // namespace renderer
