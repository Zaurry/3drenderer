#pragma once

#include "interactive/viewer_ui.h"
#include "render/render_settings.h"
#include "scene/scene_document.h"

#include <filesystem>

namespace renderer {

struct ViewerCameraSessionState {
    Vec3 eye = Vec3(0.0f, 0.0f, 1.0f);
    Vec3 forward = Vec3(0.0f, 0.0f, -1.0f);
    Vec3 up = Vec3(0.0f, 1.0f, 0.0f);
    float vertical_fov_degrees = 45.0f;
    float orbit_distance = 1.0f;
    float free_movement_speed = 1.0f;
};

struct ViewerSessionState {
    SceneDocument document;
    std::filesystem::path document_path;
    bool document_dirty = false;
    int window_width = 960;
    int window_height = 540;
    ViewerUiState ui;
    RenderSettings render_settings;
    ViewerCameraSessionState camera;
};

class ViewerSessionStore {
public:
    static ViewerSessionState load(
        const std::filesystem::path& path);
    static void save(
        const std::filesystem::path& path,
        const SceneDocument& document,
        const ViewerSessionState& state);
};

}  // namespace renderer
