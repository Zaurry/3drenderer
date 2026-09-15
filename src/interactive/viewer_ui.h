#pragma once

#include "core/math/bounds.h"
#include "interactive/frame_rate_counter.h"
#include "interactive/free_camera_controller.h"
#include "interactive/orbit_camera_controller.h"
#include "render/display_settings.h"
#include "render/interactive/interactive_render_session.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "render/opengl/opengl_raster_renderer.h"
#include "render/renderer.h"
#include "scene/scene.h"
#include "scene/scene_document.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace renderer {

enum class ViewerCameraMode {
    Orbit,
    Free,
};

struct ViewerUiState {
    InteractiveRenderMode mode = InteractiveRenderMode::Rtrt;
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
    bool techniques_panel_visible = true;
    std::vector<ObjectId> selected_objects;
    ObjectId active_object = kInvalidObjectId;
    int gizmo_operation = 0;
    bool gizmo_local = false;
    bool gizmo_was_using = false;
    bool gizmo_hovered = false;
    ObjectId material_editor_object = kInvalidObjectId;
    std::size_t selected_material_slot = 0;
    std::array<char, 160> scene_filter{};
    std::string scene_status;
};

inline void select_viewer_object(
    ViewerUiState& state,
    ObjectId id,
    bool additive) {
    if (!additive) {
        state.selected_objects.clear();
    }
    const auto found = std::find(
        state.selected_objects.begin(),
        state.selected_objects.end(),
        id);
    if (additive && found != state.selected_objects.end()) {
        state.selected_objects.erase(found);
        if (state.active_object == id) {
            state.active_object = state.selected_objects.empty()
                ? kInvalidObjectId
                : state.selected_objects.back();
        }
        return;
    }
    if (found == state.selected_objects.end()) {
        state.selected_objects.push_back(id);
    }
    state.active_object = id;
}

struct OpenGlShaderUiState {
    DdgiStatistics ddgi;
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

// Frame-local UI action flags. They are advisory hints for the main loop:
// the render backends treat scene revisions and the progressive render key
// as the authoritative change source. Flags must be consumed the same frame;
// they do not latch.
struct ViewerUiActions {
    bool mode_changed = false;
    bool camera_mode_changed = false;
    bool camera_parameters_changed = false;
    bool camera_reset_requested = false;
    SceneChangeSet scene_changes = SceneChange::None;
    bool render_scale_changed = false;
    bool automatic_interaction_quality_changed = false;
    bool path_depth_changed = false;
    bool path_roulette_changed = false;
    bool reset_requested = false;
    bool shader_reload_requested = false;
    bool shader_auto_reload_changed = false;
    bool import_files_requested = false;
    bool import_folder_requested = false;
    bool open_scene_requested = false;
    bool save_scene_requested = false;
    bool save_scene_as_requested = false;
    bool load_environment_requested = false;
    ObjectId focus_object = kInvalidObjectId;
    ObjectId look_through_camera = kInvalidObjectId;

    bool resets_path_accumulation() const {
        return mode_changed ||
            camera_mode_changed ||
            camera_parameters_changed ||
            camera_reset_requested ||
            scene_changes != SceneChange::None ||
            render_scale_changed ||
            automatic_interaction_quality_changed ||
            path_depth_changed ||
            path_roulette_changed ||
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
        const CudaOpenGlInteropUiState& interop_state,
        const CudaPathStatistics& cuda_statistics,
        const OpenGlTechniqueDiagnostics& technique_diagnostics,
        OpenGlShaderUiState& shader_state,
        bool scene_shortcuts_enabled);

    SceneChangeSet draw_scene_gizmo(
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
        const SceneDocument& document,
        const Camera& camera) const;

    void draw_directional_light_indicator(
        const ViewerUiState& state,
        const SceneDocument& document,
        const Camera& camera,
        const Bounds3& scene_bounds) const;
};

}  // namespace renderer
