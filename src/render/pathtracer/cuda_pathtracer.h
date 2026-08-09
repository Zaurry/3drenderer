#pragma once

#include "render/interactive/interactive_render_session.h"
#include "render/renderer.h"
#include "scene/instanced_scene.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace renderer {

using CudaSurfaceHandle = std::uint64_t;
using CudaStreamHandle = std::uint64_t;

enum class CudaPathWorkMode {
    FullFrame,
    InteractionPreview,
    NativeTile,
};

struct CudaPathStatistics {
    float trace_milliseconds = 0.0f;
    float reset_milliseconds = 0.0f;
    float upload_milliseconds = 0.0f;
    float instance_upload_milliseconds = 0.0f;
    float tlas_refit_milliseconds = 0.0f;
    float blas_build_milliseconds = 0.0f;
    std::uint64_t geometry_upload_bytes = 0;
    std::uint64_t material_binding_upload_bytes = 0;
    std::uint64_t material_upload_bytes = 0;
    std::uint64_t texture_upload_bytes = 0;
    std::uint64_t lighting_upload_bytes = 0;
    std::uint64_t bvh_upload_bytes = 0;
    std::uint64_t instance_upload_bytes = 0;
    std::uint64_t tlas_upload_bytes = 0;
    std::uint64_t blas_build_count = 0;
    std::uint64_t tlas_build_count = 0;
    std::uint64_t tlas_refit_count = 0;
    std::uint64_t allocation_generation = 0;
    std::uint64_t framebuffer_downloads = 0;
    CudaPathWorkMode work_mode = CudaPathWorkMode::FullFrame;
    int internal_width = 0;
    int internal_height = 0;
    int tile_y = 0;
    int tile_rows = 0;
    float sweep_progress = 0.0f;
    float complete_sweeps_per_second = 0.0f;
    float presentation_milliseconds = 0.0f;
    float traversal_milliseconds = 0.0f;
    float sort_milliseconds = 0.0f;
    bool presentation_updated = false;
    bool interop_active = false;
    bool fallback_active = false;
};

struct CudaPathDiagnosticProfile {
    static constexpr std::size_t kBounceBins = 65;

    std::uint64_t primary_rays = 0;
    std::uint64_t continuation_rays = 0;
    std::uint64_t directional_shadow_rays = 0;
    std::uint64_t point_shadow_rays = 0;
    std::uint64_t spot_shadow_rays = 0;
    std::uint64_t emissive_shadow_rays = 0;
    std::uint64_t environment_shadow_rays = 0;
    std::uint64_t primary_hits = 0;
    std::uint64_t primary_misses = 0;
    std::array<std::uint64_t, kBounceBins> rays_by_bounce{};
    std::array<std::uint64_t, kBounceBins> termination_by_bounce{};
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

    void reset(
        const Scene& scene,
        const RenderSettings& settings,
        const InstancedSceneView* instanced_scene = nullptr);
    void render_next_frame(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target,
        const InstancedSceneView* instanced_scene = nullptr);
    void render_next_frame_to_surface(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        CudaSurfaceHandle surface,
        const InstancedSceneView* instanced_scene = nullptr);
    void download_current_frame(Framebuffer& target);
    int accumulated_samples() const;
    CudaStreamHandle stream_handle() const;
    const CudaPathStatistics& statistics() const;
    CudaPathDiagnosticProfile download_diagnostic_profile();
    void set_presentation_state(bool interop_active, bool fallback_active);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace renderer
