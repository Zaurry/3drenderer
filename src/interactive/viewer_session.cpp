#include "interactive/viewer_session.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace renderer {

namespace {

nlohmann::json vec3_json(const Vec3& value) {
    return nlohmann::json::array({value.x(), value.y(), value.z()});
}

Vec3 parse_vec3(const nlohmann::json& value, const char* field) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(std::string(field) + " must contain three numbers");
    }
    const Vec3 result(
        value.at(0).get<float>(),
        value.at(1).get<float>(),
        value.at(2).get<float>());
    if (!result.allFinite()) {
        throw std::runtime_error(std::string(field) + " must be finite");
    }
    return result;
}

const char* render_mode_name(InteractiveRenderMode mode) {
    return render_mode_descriptor(mode).cli_name;
}

InteractiveRenderMode parse_render_mode(const std::string& value) {
    try {
        return parse_interactive_render_mode(value);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("unknown viewer render mode: " + value);
    }
}

const char* camera_mode_name(ViewerCameraMode mode) {
    return mode == ViewerCameraMode::Free ? "free" : "orbit";
}

ViewerCameraMode parse_camera_mode(const std::string& value) {
    if (value == "orbit") {
        return ViewerCameraMode::Orbit;
    }
    if (value == "free") {
        return ViewerCameraMode::Free;
    }
    throw std::runtime_error("unknown viewer camera mode: " + value);
}

const char* path_backend_name(PathBackend backend) {
    switch (backend) {
        case PathBackend::Auto:
            return "auto";
        case PathBackend::Cpu:
            return "cpu";
        case PathBackend::Cuda:
            return "cuda";
    }
    return "auto";
}

PathBackend parse_stored_path_backend(const std::string& value) {
    if (value == "auto") {
        return PathBackend::Auto;
    }
    if (value == "cpu") {
        return PathBackend::Cpu;
    }
    if (value == "cuda") {
        return PathBackend::Cuda;
    }
    throw std::runtime_error("unknown stored path backend: " + value);
}

const char* tone_mapper_name(ToneMapper tone_mapper) {
    switch (tone_mapper) {
        case ToneMapper::None:
            return "none";
        case ToneMapper::Reinhard:
            return "reinhard";
        case ToneMapper::Aces:
            return "aces";
    }
    return "none";
}

ToneMapper parse_tone_mapper(const std::string& value) {
    if (value == "none") {
        return ToneMapper::None;
    }
    if (value == "reinhard") {
        return ToneMapper::Reinhard;
    }
    if (value == "aces") {
        return ToneMapper::Aces;
    }
    throw std::runtime_error("unknown tone mapper: " + value);
}

float finite_clamped(
    const nlohmann::json& object,
    const char* field,
    float minimum,
    float maximum) {
    const float value = object.at(field).get<float>();
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(field) + " must be finite");
    }
    return std::clamp(value, minimum, maximum);
}

void replace_file_atomically(
    const std::filesystem::path& path,
    const nlohmann::json& root) {
    if (path.empty()) {
        throw std::runtime_error("viewer session path is empty");
    }
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
        throw std::runtime_error(
            "failed to create viewer session directory: " + error.message());
    }
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "failed to open viewer session for writing: " + temporary.string());
    }
    output << root.dump(2) << '\n';
    output.close();
    if (!output) {
        throw std::runtime_error(
            "failed to write viewer session: " + temporary.string());
    }
#ifdef _WIN32
    if (!MoveFileExW(
            temporary.c_str(),
            path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const unsigned long windows_error = GetLastError();
        std::filesystem::remove(temporary, error);
        throw std::runtime_error(
            "failed to replace viewer session, Windows error " +
            std::to_string(windows_error));
    }
#else
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        throw std::runtime_error(
            "failed to replace viewer session: " + error.message());
    }
#endif
}

}  // namespace

ViewerSessionState ViewerSessionStore::load(
    const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "failed to open viewer session: " + path.string());
    }
    nlohmann::json root;
    input >> root;
    if (root.value("version", 0) != 1) {
        throw std::runtime_error("unsupported viewer session version");
    }

    ViewerSessionState state;
    const auto& window = root.at("window");
    state.window_width = std::clamp(window.at("width").get<int>(), 320, 8192);
    state.window_height = std::clamp(window.at("height").get<int>(), 240, 8192);

    const auto& document = root.at("document");
    state.document_path = document.value("file_path", std::string());
    state.document_dirty = document.value("dirty", false);
    state.document = SceneDocument::from_session_snapshot(
        document.at("snapshot"),
        state.window_width,
        state.window_height);
    state.document.restore_file_state(
        state.document_path,
        state.document_dirty);
    if (state.document.objects().empty()) {
        throw std::runtime_error("viewer session contains no restorable objects");
    }

    const auto& view = root.at("view");
    state.ui.mode = parse_render_mode(view.at("mode").get<std::string>());
    state.ui.camera_mode =
        parse_camera_mode(view.at("camera_mode").get<std::string>());
    state.ui.render_scale =
        finite_clamped(view, "render_scale", 0.25f, 1.0f);
    state.ui.ui_font_scale =
        finite_clamped(view, "ui_font_scale", 0.75f, 2.0f);
    state.ui.path_accumulation_paused =
        view.value("path_accumulation_paused", false);
    state.ui.show_point_light_markers =
        view.value("show_point_light_markers", true);
    state.ui.panel_visible = view.value("panel_visible", true);
    state.ui.scene_panel_visible = view.value("scene_panel_visible", true);
    state.ui.inspector_panel_visible = view.value("inspector_panel_visible", true);
    state.ui.rendering_panel_visible = view.value("rendering_panel_visible", true);
    state.ui.camera_lighting_panel_visible = view.value("camera_lighting_panel_visible", true);
    state.ui.active_object =
        view.value("active_object", kInvalidObjectId);
    state.ui.material_editor_object =
        view.value("material_editor_object", kInvalidObjectId);
    state.ui.selected_material_slot =
        view.value("selected_material_slot", std::size_t{0});
    state.ui.gizmo_operation =
        std::clamp(view.value("gizmo_operation", 0), 0, 2);
    state.ui.gizmo_local = view.value("gizmo_local", false);
    if (view.contains("selected_objects")) {
        for (const auto& id_json : view.at("selected_objects")) {
            const ObjectId id = id_json.get<ObjectId>();
            if (state.document.find(id)) {
                state.ui.selected_objects.push_back(id);
            }
        }
    }
    if (!state.document.find(state.ui.active_object)) {
        state.ui.active_object = kInvalidObjectId;
    }
    const SceneMeshAsset* material_asset =
        state.document.asset_for_object(state.ui.material_editor_object);
    if (!material_asset ||
        state.ui.selected_material_slot >= material_asset->local_scene.materials.size()) {
        state.ui.material_editor_object = kInvalidObjectId;
        state.ui.selected_material_slot = 0;
    }

    const auto& display = root.at("display");
    state.ui.display.exposure_ev =
        finite_clamped(display, "exposure_ev", -20.0f, 20.0f);
    state.ui.display.tone_mapper =
        parse_tone_mapper(display.at("tone_mapper").get<std::string>());

    const auto& render = root.at("render");
    state.render_settings.path.tile_size =
        std::clamp(render.at("tile_size").get<int>(), 4, 64);
    state.render_settings.path.thread_count =
        std::clamp(render.at("thread_count").get<int>(), 0, 4096);
    state.render_settings.path.backend =
        parse_stored_path_backend(render.at("path_backend").get<std::string>());
    state.render_settings.path.samples_per_pixel = 1;

    const auto& camera = root.at("camera");
    state.camera.eye = parse_vec3(camera.at("eye"), "camera.eye");
    state.camera.forward = parse_vec3(camera.at("forward"), "camera.forward");
    state.camera.up = parse_vec3(camera.at("up"), "camera.up");
    state.camera.vertical_fov_degrees =
        finite_clamped(camera, "vertical_fov_degrees", 5.0f, 175.0f);
    state.camera.orbit_distance =
        finite_clamped(camera, "orbit_distance", 0.05f, 1.0e9f);
    state.camera.free_movement_speed =
        finite_clamped(camera, "free_movement_speed", 1.0e-4f, 1.0e9f);
    if (state.camera.forward.squaredNorm() < 1.0e-12f ||
        state.camera.up.squaredNorm() < 1.0e-12f ||
        state.camera.forward.cross(state.camera.up).squaredNorm() < 1.0e-12f) {
        throw std::runtime_error("viewer session camera basis is invalid");
    }
    state.camera.forward.normalize();
    state.camera.up.normalize();
    return state;
}

void ViewerSessionStore::save(
    const std::filesystem::path& path,
    const SceneDocument& document,
    const ViewerSessionState& state) {
    nlohmann::json root;
    root["version"] = 1;
    root["document"] = {
        {"file_path", state.document_path.generic_string()},
        {"dirty", state.document_dirty},
        {"snapshot", document.session_snapshot()},
    };
    root["window"] = {
        {"width", std::clamp(state.window_width, 320, 8192)},
        {"height", std::clamp(state.window_height, 240, 8192)},
    };
    root["view"] = {
        {"mode", render_mode_name(state.ui.mode)},
        {"camera_mode", camera_mode_name(state.ui.camera_mode)},
        {"render_scale", state.ui.render_scale},
        {"ui_font_scale", state.ui.ui_font_scale},
        {"path_accumulation_paused", state.ui.path_accumulation_paused},
        {"show_point_light_markers", state.ui.show_point_light_markers},
        {"panel_visible", state.ui.panel_visible},
        {"scene_panel_visible", state.ui.scene_panel_visible},
        {"inspector_panel_visible", state.ui.inspector_panel_visible},
        {"rendering_panel_visible", state.ui.rendering_panel_visible},
        {
            "camera_lighting_panel_visible",
            state.ui.camera_lighting_panel_visible,
        },
        {"selected_objects", state.ui.selected_objects},
        {"active_object", state.ui.active_object},
        {"material_editor_object", state.ui.material_editor_object},
        {"selected_material_slot", state.ui.selected_material_slot},
        {"gizmo_operation", state.ui.gizmo_operation},
        {"gizmo_local", state.ui.gizmo_local},
    };
    root["display"] = {
        {"exposure_ev", state.ui.display.exposure_ev},
        {"tone_mapper", tone_mapper_name(state.ui.display.tone_mapper)},
    };
    root["render"] = {
        {"tile_size", state.render_settings.path.tile_size},
        {"thread_count", state.render_settings.path.thread_count},
        {"path_backend", path_backend_name(state.render_settings.path.backend)},
    };
    root["camera"] = {
        {"eye", vec3_json(state.camera.eye)},
        {"forward", vec3_json(state.camera.forward)},
        {"up", vec3_json(state.camera.up)},
        {"vertical_fov_degrees", state.camera.vertical_fov_degrees},
        {"orbit_distance", state.camera.orbit_distance},
        {"free_movement_speed", state.camera.free_movement_speed},
    };
    replace_file_atomically(path, root);
}

}  // namespace renderer
