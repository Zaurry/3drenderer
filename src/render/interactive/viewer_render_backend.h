#pragma once

#include "render/interactive/render_frame_output.h"
#include "render/interactive/render_mode.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "render/opengl/opengl_raster_renderer.h"
#include "scene/instanced_scene.h"

#include <filesystem>
#include <memory>
#include <string>
#include <variant>

namespace renderer {

struct OpenGlViewerStatistics {
    bool shader_valid = false;
    bool shader_auto_reload = true;
    std::string shader_error;
    std::string shader_vertex_path;
    std::string shader_fragment_path;
    OpenGlTechniqueDiagnostics techniques;
};

struct CudaPathViewerStatistics {
    int accumulated_samples = 0;
    std::string interop_status = "unavailable";
    std::string interop_detail;
    CudaPathStatistics cuda;
};

using ViewerRenderBackendStatistics = std::variant<
    OpenGlViewerStatistics,
    CudaPathViewerStatistics>;

class OpenGlShaderControl {
public:
    virtual ~OpenGlShaderControl() = default;
    virtual void set_shader_auto_reload(bool enabled) = 0;
    virtual void request_shader_reload() = 0;
};

class ViewerRenderBackend {
public:
    virtual ~ViewerRenderBackend() = default;

    virtual InteractiveRenderMode mode() const = 0;
    virtual RenderModeCapability capabilities() const = 0;
    virtual void reset(
        const RenderSceneSnapshot& snapshot,
        const RenderSettings& settings) = 0;
    virtual const RenderFrameOutput& render(
        const RenderSceneSnapshot& snapshot,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state) = 0;
    // The last completed frame, valid until the next render() call. When
    // rendering is skipped (e.g. paused progressive accumulation), callers
    // present this same handle to show the frozen frame on purpose.
    virtual const RenderFrameOutput& output() const = 0;
    virtual ViewerRenderBackendStatistics statistics() const = 0;
};

OpenGlShaderControl* open_gl_shader_control(ViewerRenderBackend& backend);

std::unique_ptr<ViewerRenderBackend> make_viewer_render_backend(
    InteractiveRenderMode mode,
    const std::filesystem::path& vertex_shader_path,
    const std::filesystem::path& fragment_shader_path);

}  // namespace renderer
