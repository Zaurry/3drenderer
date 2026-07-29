#pragma once

#include "core/math/bounds.h"
#include "interactive/frame_rate_counter.h"
#include "interactive/free_camera_controller.h"
#include "interactive/orbit_camera_controller.h"
#include "render/display_settings.h"
#include "render/interactive/interactive_render_session.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "render/renderer.h"
#include "scene/scene.h"
#include "scene/scene_document.h"

#include <cstddef>
#include <string>
#include <vector>

namespace renderer {

enum class ViewerCameraMode {
    Orbit,
    Free,
};

struct ViewerUiState {
    InteractiveRenderMode mode = InteractiveRenderMode::OpenGl;
    ViewerCameraMode camera_mode = ViewerCameraMode::Orbit;
    DisplaySettings display;
    float render_scale = 1.0f;
    float ui_font_scale = 1.0f;
    bool automatic_interaction_quality = true;
    bool path_accumulation_paused = false;
    bool show_point_light_markers = true;
    bool panel_visible = true;
    bool scene_panel_visible = true;
    bool inspector_panel_visible = true;
    bool rendering_panel_visible = true;
    bool camera_lighting_panel_visible = true;
    std::vector<ObjectId> selected_objects;
    ObjectId active_object = kInvalidObjectId;
    int gizmo_operation = 0;
    bool gizmo_local = false;
    bool gizmo_was_using = false;
    bool gizmo_hovered = false;
    ObjectId material_editor_object = kInvalidObjectId;
    std::size_t selected_material_slot = 0;
    std::string scene_status;
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
    SceneChangeSet scene_changes = SceneChange::None;
    bool render_scale_changed = false;
    bool automatic_interaction_quality_changed = false;
    bool reset_requested = false;
    bool display_changed = false;
    bool ui_style_changed = false;
    bool shader_reload_requested = false;
    bool shader_auto_reload_changed = false;
    bool import_files_requested = false;
    bool import_folder_requested = false;
    bool open_scene_requested = false;
    bool save_scene_requested = false;
    bool save_scene_as_requested = false;
    ObjectId focus_object = kInvalidObjectId;

    bool resets_path_accumulation() const {
        return mode_changed ||
            path_backend_changed ||
            camera_mode_changed ||
            camera_parameters_changed ||
            camera_reset_requested ||
            scene_changes != SceneChange::None ||
            render_scale_changed ||
            automatic_interaction_quality_changed ||
            reset_requested;
    }
};

class ViewerUi {
public:
    ViewerUiActions draw(
        ViewerUiState& state,
        RenderSettings& render_settings,
        SceneDocument& document,
        OrbitCameraController& orbit_camera,
        FreeCameraController& free_camera,
        const Bounds3& scene_bounds,
        const FrameRateSnapshot& performance,
        int accumulated_path_samples,
        ExecutionBackend active_path_backend,
        const CudaOpenGlInteropUiState& interop_state,
        const CudaPathStatistics& cuda_statistics,
        OpenGlShaderUiState& shader_state,
        bool scene_shortcuts_enabled);

    bool draw_scene_gizmo(
        ViewerUiState& state,
        SceneDocument& document,
        const Camera& camera,
        const Bounds3& scene_bounds);

    void draw_scene_selection(
        const ViewerUiState& state,
        const SceneDocument& document,
        const Camera& camera) const;

    void draw_point_light_markers(
        const ViewerUiState& state,
        const Scene& scene,
        const Camera& camera) const;
};

}  // namespace renderer
