#pragma once

#include "render/interactive/interactive_render_session.h"
#include "render/renderer.h"

#include <memory>
#include <string>

namespace renderer {

bool cuda_path_backend_compiled();
bool cuda_path_backend_available(std::string* reason = nullptr);

RenderResult render_cuda_path(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings);

class CudaPathInteractiveRenderer {
public:
    CudaPathInteractiveRenderer();
    ~CudaPathInteractiveRenderer();
    CudaPathInteractiveRenderer(CudaPathInteractiveRenderer&&) noexcept;
    CudaPathInteractiveRenderer& operator=(CudaPathInteractiveRenderer&&) noexcept;

    CudaPathInteractiveRenderer(const CudaPathInteractiveRenderer&) = delete;
    CudaPathInteractiveRenderer& operator=(const CudaPathInteractiveRenderer&) = delete;

    void reset(const Scene& scene, const RenderSettings& settings);
    void render_next_frame(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target);
    int accumulated_samples() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace renderer
