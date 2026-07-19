#pragma once

#include "core/math/bounds.h"
#include "interactive/frame_rate_counter.h"
#include "interactive/free_camera_controller.h"
#include "interactive/orbit_camera_controller.h"
#include "render/display_settings.h"
#include "render/interactive/interactive_render_session.h"
#include "render/renderer.h"
#include "scene/scene.h"

#include <string>

namespace renderer {

enum class ViewerCameraMode {
    Orbit,
    Free,
};

struct ViewerUiState {
    InteractiveRenderMode mode = InteractiveRenderMode::Raster;
    ViewerCameraMode camera_mode = ViewerCameraMode::Orbit;
    DisplaySettings display;
    float render_scale = 1.0f;
    float ui_font_scale = 1.0f;
    bool path_accumulation_paused = false;
    bool panel_visible = true;
};

struct OpenGlShaderUiState {
    std::string vertex_path;
    std::string fragment_path;
    std::string error;
    bool auto_reload = true;
    bool valid = false;
};

struct CudaOpenGlInteropUiState {
    std::string status = "unavailable";
    std::string detail;
};

struct ViewerUiActions {
    bool mode_changed = false;
    bool path_backend_changed = false;
    bool camera_mode_changed = false;
    bool camera_parameters_changed = false;
    bool camera_reset_requested = false;
    bool lighting_changed = false;
    bool render_scale_changed = false;
    bool reset_requested = false;
    bool display_changed = false;
    bool ui_style_changed = false;
    bool shader_reload_requested = false;
    bool shader_auto_reload_changed = false;

    bool resets_path_accumulation() const {
        return mode_changed ||
            path_backend_changed ||
            camera_mode_changed ||
            camera_parameters_changed ||
            camera_reset_requested ||
            lighting_changed ||
            render_scale_changed ||
            reset_requested;
    }
};

class ViewerUi {
public:
    ViewerUiActions draw(
        ViewerUiState& state,
        RenderSettings& render_settings,
        Scene& scene,
        OrbitCameraController& orbit_camera,
        FreeCameraController& free_camera,
        const Bounds3& scene_bounds,
        const FrameRateSnapshot& performance,
        int accumulated_path_samples,
        ExecutionBackend active_path_backend,
        const CudaOpenGlInteropUiState& interop_state,
        OpenGlShaderUiState& shader_state);
};

}  // namespace renderer
