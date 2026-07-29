#pragma once

#include "render/interactive/interactive_render_session.h"
#include "render/renderer.h"

#include <cstdint>
#include <memory>
#include <string>

namespace renderer {

using CudaSurfaceHandle = std::uint64_t;
using CudaStreamHandle = std::uint64_t;

struct CudaPathStatistics {
    float trace_milliseconds = 0.0f;
    float reset_milliseconds = 0.0f;
    float upload_milliseconds = 0.0f;
    std::uint64_t geometry_upload_bytes = 0;
    std::uint64_t material_binding_upload_bytes = 0;
    std::uint64_t material_upload_bytes = 0;
    std::uint64_t texture_upload_bytes = 0;
    std::uint64_t lighting_upload_bytes = 0;
    std::uint64_t bvh_upload_bytes = 0;
    std::uint64_t allocation_generation = 0;
    std::uint64_t framebuffer_downloads = 0;
    bool interop_active = false;
    bool fallback_active = false;
};

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
    void render_next_frame_to_surface(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        CudaSurfaceHandle surface);
    void download_current_frame(Framebuffer& target);
    int accumulated_samples() const;
    CudaStreamHandle stream_handle() const;
    const CudaPathStatistics& statistics() const;
    void set_presentation_state(bool interop_active, bool fallback_active);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace renderer
