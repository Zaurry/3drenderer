#pragma once

#include "render/interactive/render_frame_output.h"
#include "render/interactive/render_mode.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "render/pathtracer/path_backend.h"
#include "scene/instanced_scene.h"

#include <filesystem>
#include <memory>
#include <string>

namespace renderer {

struct ViewerRenderBackendStatistics {
    int accumulated_samples = 0;
    ExecutionBackend path_backend = ExecutionBackend::Cpu;
    bool shader_valid = false;
    bool shader_auto_reload = true;
    std::string shader_error;
    std::string shader_vertex_path;
    std::string shader_fragment_path;
    std::string interop_status = "unavailable";
    std::string interop_detail;
    CudaPathStatistics cuda;
};

class ViewerRenderBackend {
public:
    virtual ~ViewerRenderBackend() = default;

    virtual InteractiveRenderMode mode() const = 0;
    virtual RenderModeCapability capabilities() const = 0;
    virtual void reset(
        const Scene& scene,
        const RenderSettings& settings,
        const InstancedSceneView* instanced_scene = nullptr) = 0;
    virtual const RenderFrameOutput& render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        const InstancedSceneView* instanced_scene = nullptr) = 0;
    virtual const RenderFrameOutput& output() const = 0;
    virtual ViewerRenderBackendStatistics statistics() const = 0;

    virtual void set_shader_auto_reload(bool) {}
    virtual void request_shader_reload() {}
};

std::unique_ptr<ViewerRenderBackend> make_viewer_render_backend(
    InteractiveRenderMode mode,
    const std::filesystem::path& vertex_shader_path,
    const std::filesystem::path& fragment_shader_path);

}  // namespace renderer
