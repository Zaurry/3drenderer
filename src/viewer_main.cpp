#include "interactive/frame_rate_counter.h"
#include "interactive/free_camera_controller.h"
#include "interactive/orbit_camera_controller.h"
#include "interactive/viewer_session.h"
#include "interactive/viewer_ui.h"
#include "platform/opengl/cuda_opengl_interop.h"
#include "platform/sdl/sdl_display_backend.h"
#include "render/interactive/render_mode.h"
#include "render/interactive/viewer_render_backend.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "scene/scene.h"
#include "scene/scene_asset_loader.h"
#include "scene/scene_document.h"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct ViewerOptions {
    std::string scene = "asset";
    std::vector<std::string> asset_paths;
    std::filesystem::path scene_file;
    renderer::InteractiveRenderMode mode = renderer::InteractiveRenderMode::OpenGl;
    int width = 960;
    int height = 540;
    int frame_limit = 0;
    int cuda_device = -1;
    std::filesystem::path gl_vertex_shader;
    std::filesystem::path gl_fragment_shader;
    std::filesystem::path environment_path;
    float environment_intensity = 1.0f;
    float environment_yaw_degrees = 0.0f;
    bool environment_background_visible = true;
    bool restore_last_session = true;
};

struct ViewerScene {
    renderer::SceneDocument document;
    renderer::Bounds3 bounds;
    renderer::Camera camera;
    bool camera_from_scene = false;
    float camera_vertical_fov_degrees = 45.0f;
};

struct SceneCameraView {
    renderer::Camera camera;
    float vertical_fov_degrees = 45.0f;
};

std::optional<SceneCameraView> make_scene_camera_view(
    const renderer::SceneDocument& document,
    renderer::ObjectId object_id,
    float viewport_aspect_ratio) {
    const renderer::SceneObject* object = document.find(object_id);
    if (!object || object->type != renderer::SceneObjectType::Camera) {
        return std::nullopt;
    }

    const renderer::Mat4 world = document.world_matrix(object_id);
    const renderer::Vec3 eye = world.block<3, 1>(0, 3);
    renderer::Vec3 forward =
        world.block<3, 3>(0, 0) * renderer::Vec3(0.0f, 0.0f, -1.0f);
    renderer::Vec3 up =
        world.block<3, 3>(0, 0) * renderer::Vec3(0.0f, 1.0f, 0.0f);
    if (!eye.allFinite() || !forward.allFinite() || !up.allFinite() ||
        forward.squaredNorm() <= 1.0e-12f || up.squaredNorm() <= 1.0e-12f) {
        return std::nullopt;
    }
    forward.normalize();
    up.normalize();
    if (std::abs(forward.dot(up)) > 0.999f) {
        up = renderer::Vec3(0.0f, 1.0f, 0.0f);
        if (std::abs(forward.dot(up)) > 0.999f) {
            up = renderer::Vec3(1.0f, 0.0f, 0.0f);
        }
    }

    const float fov = object->camera_projection ==
            renderer::SceneCameraProjection::Perspective
        ? std::clamp(object->camera_vertical_fov_degrees, 1.0f, 179.0f)
        : 45.0f;
    const float aspect = object->camera_aspect_ratio > 0.0f
        ? object->camera_aspect_ratio
        : viewport_aspect_ratio;
    try {
        return SceneCameraView{
            renderer::Camera(eye, eye + forward, up, fov, aspect),
            fov};
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

std::optional<SceneCameraView> first_scene_camera_view(
    const renderer::SceneDocument& document,
    float viewport_aspect_ratio) {
    for (const renderer::SceneObject& object : document.objects()) {
        if (object.type == renderer::SceneObjectType::Camera) {
            if (auto camera = make_scene_camera_view(
                    document, object.id, viewport_aspect_ratio)) {
                return camera;
            }
        }
    }
    return std::nullopt;
}

void print_help() {
    std::cout
        << "3D Renderer Viewer\n\n"
        << "Usage:\n"
        << "  viewer --scene builtin|asset [--asset path\\to\\model-or-directory ...] --mode opengl|path [options]\n\n"
        << "Options:\n"
        << "  --asset path      OBJ/glTF/GLB file or directory; may be repeated\n"
        << "  --scene-file path open a saved .rscene document\n"
        << "  --environment path  2:1 HDR/EXR/PNG/JPG environment map\n"
        << "  --environment-intensity value  environment multiplier, default 1\n"
        << "  --environment-yaw degrees  rotate the environment around Y\n"
        << "  --hide-environment-background  light the scene without drawing the map\n"
        << "  --width integer    window width, default 960\n"
        << "  --height integer   window height, default 540\n"
        << "  --frames integer   render N frames then exit, default unlimited\n"
        << "  --cuda-device N    CUDA device index for Path mode; default GL-compatible\n"
        << "  --gl-vertex-shader path    OpenGL vertex shader override\n"
        << "  --gl-fragment-shader path  OpenGL fragment shader override\n"
        << "  --no-restore-last  start the default scene without restoring the last session\n"
        << "  --help             show this help\n";
}

std::string require_value(int argc, char** argv, int& i, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(flag + " requires a value");
    }
    ++i;
    return argv[i];
}

int parse_nonnegative_int(const std::string& value, const std::string& name) {
    std::size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(name + " must be an integer");
    }
    if (consumed != value.size() || parsed < 0) {
        throw std::invalid_argument(name + " must be a non-negative integer");
    }
    return parsed;
}

int parse_positive_int(const std::string& value, const std::string& name) {
    const int parsed = parse_nonnegative_int(value, name);
    if (parsed <= 0) {
        throw std::invalid_argument(name + " must be positive");
    }
    return parsed;
}

float parse_finite_float(const std::string& value, const std::string& name) {
    std::size_t consumed = 0;
    float parsed = 0.0f;
    try {
        parsed = std::stof(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(name + " must be a number");
    }
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::invalid_argument(name + " must be a finite number");
    }
    return parsed;
}

renderer::InteractiveRenderMode parse_mode(const std::string& value) {
    return renderer::parse_interactive_render_mode(value);
}

ViewerOptions parse_args(int argc, char** argv) {
    ViewerOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            print_help();
            std::exit(0);
        } else if (arg == "--scene") {
            options.scene = require_value(argc, argv, i, arg);
        } else if (arg == "--asset") {
            options.asset_paths.push_back(require_value(argc, argv, i, arg));
        } else if (arg == "--scene-file") {
            options.scene_file = require_value(argc, argv, i, arg);
        } else if (arg == "--mode") {
            options.mode = parse_mode(require_value(argc, argv, i, arg));
        } else if (arg == "--width") {
            options.width = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--height") {
            options.height = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--frames") {
            options.frame_limit = parse_nonnegative_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--cuda-device") {
            options.cuda_device = parse_nonnegative_int(
                require_value(argc, argv, i, arg),
                arg);
        } else if (arg == "--gl-vertex-shader") {
            options.gl_vertex_shader = require_value(argc, argv, i, arg);
        } else if (arg == "--gl-fragment-shader") {
            options.gl_fragment_shader = require_value(argc, argv, i, arg);
        } else if (arg == "--environment") {
            options.environment_path = require_value(argc, argv, i, arg);
        } else if (arg == "--environment-intensity") {
            options.environment_intensity =
                parse_finite_float(require_value(argc, argv, i, arg), arg);
            if (options.environment_intensity < 0.0f) {
                throw std::invalid_argument(arg + " must be non-negative");
            }
        } else if (arg == "--environment-yaw") {
            options.environment_yaw_degrees =
                parse_finite_float(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--hide-environment-background") {
            options.environment_background_visible = false;
        } else if (arg == "--no-restore-last") {
            options.restore_last_session = false;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.scene != "builtin" && options.scene != "asset") {
        throw std::invalid_argument("--scene must be builtin or asset");
    }
    if (!options.scene_file.empty() && options.scene == "builtin") {
        throw std::invalid_argument("--scene-file cannot be combined with --scene builtin");
    }
    return options;
}

std::filesystem::path resolve_shader_path(
    const std::filesystem::path& explicit_path,
    const std::filesystem::path& executable_path,
    const std::filesystem::path& relative_path) {
    if (!explicit_path.empty()) {
        return std::filesystem::absolute(explicit_path).lexically_normal();
    }
    const std::filesystem::path working_copy =
        std::filesystem::absolute(relative_path).lexically_normal();
    if (std::filesystem::exists(working_copy)) {
        return working_copy;
    }
    const std::filesystem::path executable_copy =
        std::filesystem::absolute(executable_path).parent_path() / relative_path;
    return executable_copy.lexically_normal();
}

renderer::Bounds3 scene_bounds(const renderer::Scene& scene) {
    renderer::Bounds3 bounds;
    for (const renderer::Sphere& sphere : scene.spheres) {
        bounds.expand(sphere.bounds());
    }
    for (const renderer::Triangle& triangle : scene.triangles) {
        bounds.expand(triangle.bounds());
    }
    return bounds;
}

renderer::Camera make_camera_from_bounds(const renderer::Bounds3& bounds, int width, int height) {
    const renderer::Vec3 center = (bounds.min + bounds.max) * 0.5f;
    const float radius = std::max(0.5f, (bounds.max - bounds.min).norm() * 0.5f);
    return renderer::Camera(
        center + renderer::Vec3(0.0f, radius * 0.15f, radius * 2.4f),
        center,
        renderer::Vec3(0.0f, 1.0f, 0.0f),
        45.0f,
        static_cast<float>(width) / static_cast<float>(height));
}

ViewerScene load_viewer_scene(
    const ViewerOptions& options,
    const std::filesystem::path& executable_path) {
    std::filesystem::path default_asset =
        std::filesystem::path("Computer Graphics Archive") /
        "CornellBox" /
        "CornellBox-Original.obj";
    if (!std::filesystem::exists(default_asset)) {
        const std::filesystem::path repository_candidate =
            executable_path.parent_path().parent_path().parent_path().parent_path() /
            default_asset;
        if (std::filesystem::exists(repository_candidate)) {
            default_asset = repository_candidate;
        }
    }

    if (options.scene == "asset") {
        renderer::SceneDocument document = options.scene_file.empty()
            ? renderer::SceneDocument()
            : renderer::SceneDocument::load(options.scene_file, options.width, options.height);
        std::vector<std::string> paths = options.asset_paths;
        if (paths.empty() && options.scene_file.empty()) {
            paths.push_back(default_asset.string());
        }
        for (const std::string& path : paths) {
            document.import_path(path, options.width, options.height);
        }
        if (!options.environment_path.empty()) {
            document.set_environment_map(options.environment_path);
        }
        document.set_environment_intensity(options.environment_intensity);
        document.set_environment_rotation_degrees(options.environment_yaw_degrees);
        document.set_environment_background_visible(
            options.environment_background_visible);
        for (const std::string& warning : document.warnings()) {
            std::cerr << "warning: " << warning << '\n';
        }
        const renderer::Bounds3 bounds = document.scene_bounds();
        renderer::Camera camera =
            make_camera_from_bounds(bounds, options.width, options.height);
        bool camera_from_scene = false;
        float camera_vertical_fov_degrees = 45.0f;
        if (auto scene_camera = first_scene_camera_view(
                document,
                static_cast<float>(options.width) /
                    static_cast<float>(options.height))) {
            camera = scene_camera->camera;
            camera_from_scene = true;
            camera_vertical_fov_degrees =
                scene_camera->vertical_fov_degrees;
        }
        return ViewerScene{
            std::move(document),
            bounds,
            std::move(camera),
            camera_from_scene,
            camera_vertical_fov_degrees};
    }

    renderer::Scene scene = renderer::make_cornell_box_scene();
    renderer::Bounds3 bounds = scene_bounds(scene);
    renderer::SceneDocument document =
        renderer::SceneDocument::from_scene(
            scene,
            "Builtin Cornell Box",
            "cornell_box");
    if (!options.environment_path.empty()) {
        document.set_environment_map(options.environment_path);
    }
    document.set_environment_intensity(options.environment_intensity);
    document.set_environment_rotation_degrees(options.environment_yaw_degrees);
    document.set_environment_background_visible(
        options.environment_background_visible);
    return ViewerScene{
        std::move(document),
        bounds,
        make_camera_from_bounds(bounds, options.width, options.height)};
}

const char* mode_name(renderer::InteractiveRenderMode mode) {
    return renderer::render_mode_descriptor(mode).cli_name;
}

const char* camera_mode_name(renderer::ViewerCameraMode mode) {
    return mode == renderer::ViewerCameraMode::Orbit ? "orbit" : "free";
}

bool has_extension(const std::filesystem::path& path, std::string extension) {
    std::string actual = path.extension().string();
    std::transform(actual.begin(), actual.end(), actual.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return actual == extension;
}

int scaled_dimension(int window_dimension, float render_scale) {
    return std::max(1, static_cast<int>(std::lround(
        static_cast<float>(window_dimension) * std::clamp(render_scale, 0.25f, 1.0f))));
}

void transition_camera_mode(
    renderer::ViewerCameraMode previous,
    renderer::ViewerCameraMode next,
    renderer::OrbitCameraController& orbit_camera,
    renderer::FreeCameraController& free_camera) {
    if (previous == next) {
        return;
    }
    if (next == renderer::ViewerCameraMode::Free) {
        free_camera.set_camera(orbit_camera.camera());
    } else {
        orbit_camera.set_camera(free_camera.camera());
    }
}

struct ViewerSessionSignature {
    renderer::InteractiveRenderMode mode =
        renderer::InteractiveRenderMode::OpenGl;
    renderer::ViewerCameraMode camera_mode =
        renderer::ViewerCameraMode::Orbit;
    renderer::ToneMapper tone_mapper = renderer::ToneMapper::None;
    int cuda_device = 0;
    int max_bounces = 64;
    int russian_roulette_start_bounce = 3;
    float russian_roulette_min_probability = 0.05f;
    float russian_roulette_max_probability = 0.95f;
    renderer::OpenGlRenderSettings opengl;
    int logical_width = 0;
    int logical_height = 0;
    float render_scale = 1.0f;
    float ui_font_scale = 1.0f;
    float exposure_ev = 0.0f;
    bool automatic_interaction_quality = true;
    bool path_accumulation_paused = false;
    bool show_point_light_markers = true;
    bool panel_visible = true;
    bool scene_panel_visible = true;
    bool inspector_panel_visible = true;
    bool rendering_panel_visible = true;
    bool camera_lighting_panel_visible = true;
    bool techniques_panel_visible = true;
    std::vector<renderer::ObjectId> selected_objects;
    renderer::ObjectId active_object = renderer::kInvalidObjectId;
    renderer::ObjectId material_editor_object = renderer::kInvalidObjectId;
    std::size_t selected_material_slot = 0;
    int gizmo_operation = 0;
    bool gizmo_local = false;
    std::filesystem::path document_path;
    bool document_dirty = false;

    bool operator==(const ViewerSessionSignature&) const = default;
};

ViewerSessionSignature make_session_signature(
    const renderer::ViewerUiState& ui,
    const renderer::RenderSettings& settings,
    const renderer::SceneDocument& document,
    std::pair<int, int> logical_size) {
    ViewerSessionSignature signature;
    signature.mode = ui.mode;
    signature.camera_mode = ui.camera_mode;
    signature.tone_mapper = ui.display.tone_mapper;
    signature.cuda_device = settings.path.cuda_device;
    signature.max_bounces = settings.path.max_bounces;
    signature.russian_roulette_start_bounce =
        settings.path.russian_roulette_start_bounce;
    signature.russian_roulette_min_probability =
        settings.path.russian_roulette_min_probability;
    signature.russian_roulette_max_probability =
        settings.path.russian_roulette_max_probability;
    signature.opengl = settings.opengl;
    signature.logical_width = logical_size.first;
    signature.logical_height = logical_size.second;
    signature.render_scale = ui.render_scale;
    signature.ui_font_scale = ui.ui_font_scale;
    signature.exposure_ev = ui.display.exposure_ev;
    signature.automatic_interaction_quality =
        ui.automatic_interaction_quality;
    signature.path_accumulation_paused = ui.path_accumulation_paused;
    signature.show_point_light_markers = ui.show_point_light_markers;
    signature.panel_visible = ui.panel_visible;
    signature.scene_panel_visible = ui.scene_panel_visible;
    signature.inspector_panel_visible = ui.inspector_panel_visible;
    signature.rendering_panel_visible = ui.rendering_panel_visible;
    signature.camera_lighting_panel_visible = ui.camera_lighting_panel_visible;
    signature.techniques_panel_visible = ui.techniques_panel_visible;
    signature.selected_objects = ui.selected_objects;
    signature.active_object = ui.active_object;
    signature.material_editor_object = ui.material_editor_object;
    signature.selected_material_slot = ui.selected_material_slot;
    signature.gizmo_operation = ui.gizmo_operation;
    signature.gizmo_local = ui.gizmo_local;
    signature.document_path = document.file_path();
    signature.document_dirty = document.dirty();
    return signature;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        ViewerOptions options = parse_args(argc, argv);
        bool session_enabled = argc == 1 && options.restore_last_session;
        std::filesystem::path session_path;
        std::optional<renderer::ViewerSessionState> restored_session;
        std::string startup_status;
        if (session_enabled) {
            try {
                session_path =
                    renderer::SdlDisplayBackend::preferred_session_path();
                if (std::filesystem::exists(session_path)) {
                    restored_session =
                        renderer::ViewerSessionStore::load(session_path);
                    options.width = restored_session->window_width;
                    options.height = restored_session->window_height;
                    options.mode = restored_session->ui.mode;
                    startup_status = "Restored last viewer session";
                    if (!restored_session->migration_warning.empty()) {
                        startup_status = restored_session->migration_warning;
                    }
                }
            } catch (const std::exception& error) {
                startup_status =
                    "Last session restore failed; loaded the default scene";
                std::cerr << "warning: " << startup_status << ": "
                          << error.what() << '\n';
                restored_session.reset();
            }
        }
        if (session_path.empty()) {
            session_enabled = false;
        }
        const std::filesystem::path executable_path = std::filesystem::absolute(argv[0]);
        options.gl_vertex_shader = resolve_shader_path(
            options.gl_vertex_shader,
            executable_path,
            std::filesystem::path("shaders") / "opengl" / "raster.vert");
        options.gl_fragment_shader = resolve_shader_path(
            options.gl_fragment_shader,
            executable_path,
            std::filesystem::path("shaders") / "opengl" / "raster.frag");
        ViewerScene viewer_scene = [&]() {
            if (!restored_session) {
                return load_viewer_scene(options, executable_path);
            }
            renderer::SceneDocument document =
                std::move(restored_session->document);
            const renderer::Bounds3 bounds = document.scene_bounds();
            const renderer::ViewerCameraSessionState& saved_camera =
                restored_session->camera;
            renderer::Camera camera(
                saved_camera.eye,
                saved_camera.eye + saved_camera.forward,
                saved_camera.up,
                saved_camera.vertical_fov_degrees,
                static_cast<float>(options.width) /
                    static_cast<float>(options.height));
            return ViewerScene{
                std::move(document),
                bounds,
                std::move(camera)};
        }();
        if (restored_session) {
            std::cout << "session=restored path="
                      << session_path.string() << '\n';
            for (const std::string& warning :
                 viewer_scene.document.warnings()) {
                std::cerr << "warning: " << warning << '\n';
            }
        }

        renderer::SdlDisplayBackend display;
        if (!display.initialize(options.width, options.height, "3D Renderer Viewer")) {
            throw std::runtime_error(display.last_error());
        }
        if (restored_session && !display.constrain_window_to_display()) {
            std::cerr << "warning: failed to constrain restored window size: "
                      << display.last_error() << '\n';
        }
        renderer::ViewerUiState ui_state = restored_session
            ? restored_session->ui
            : renderer::ViewerUiState{};
        if (!restored_session) {
            ui_state.mode = options.mode;
            if (viewer_scene.camera_from_scene) {
                ui_state.camera_mode = renderer::ViewerCameraMode::Free;
                ui_state.scene_status =
                    "Using the first imported glTF scene camera";
            }
        }
        if (!startup_status.empty()) {
            ui_state.scene_status = startup_status;
        }
        renderer::ViewerUi viewer_ui;

        const auto initial_drawable_size = display.drawable_size();
        int window_width = std::max(1, initial_drawable_size.first);
        int window_height = std::max(1, initial_drawable_size.second);
        renderer::RenderSettings settings = restored_session
            ? restored_session->render_settings
            : renderer::RenderSettings{};
        settings.width = scaled_dimension(window_width, ui_state.render_scale);
        settings.height = scaled_dimension(window_height, ui_state.render_scale);
        settings.path.samples_per_pixel = 1;
        if (!restored_session) {
            settings.path.cuda_device = options.cuda_device;
        }
        if (ui_state.mode == renderer::InteractiveRenderMode::Path) {
            std::string reason;
            auto device_context =
                renderer::select_cuda_device_for_current_opengl_context(
                    settings.path.cuda_device,
                    &reason);
            if (!device_context && restored_session) {
                device_context =
                    renderer::select_cuda_device_for_current_opengl_context(
                        -1,
                        &reason);
                if (device_context) {
                    ui_state.scene_status =
                        "Saved CUDA device is not GL-compatible; selected device " +
                        std::to_string(device_context->device_id());
                }
            }
            if (device_context) {
                settings.path.cuda_device = device_context->device_id();
            } else {
                settings.path.cuda_device = 0;
                ui_state.mode = renderer::InteractiveRenderMode::OpenGl;
                ui_state.scene_status =
                    "CUDA Path unavailable; switched to OpenGL: " + reason;
                std::cerr << "warning: " << ui_state.scene_status << '\n';
                // Stable machine-readable marker; CI smoke tests match this
                // line instead of the localized status text.
                std::cout << "path-mode-unavailable: switched to OpenGL ("
                          << reason << ")\n";
            }
        }

        const float camera_aspect_ratio =
            static_cast<float>(window_width) / static_cast<float>(window_height);
        const float scene_radius = std::max(
            0.5f,
            (viewer_scene.bounds.max - viewer_scene.bounds.min).norm() * 0.5f);
        renderer::OrbitCameraController orbit_camera(viewer_scene.bounds, camera_aspect_ratio);
        renderer::FreeCameraController free_camera(
            viewer_scene.camera,
            camera_aspect_ratio,
            scene_radius);
        if (!restored_session && viewer_scene.camera_from_scene) {
            orbit_camera.set_camera(viewer_scene.camera);
            orbit_camera.set_vertical_fov_degrees(
                viewer_scene.camera_vertical_fov_degrees);
            free_camera.set_vertical_fov_degrees(
                viewer_scene.camera_vertical_fov_degrees);
        }
        if (restored_session) {
            const renderer::ViewerCameraSessionState& saved_camera =
                restored_session->camera;
            const renderer::Camera restored_camera(
                saved_camera.eye,
                saved_camera.eye + saved_camera.forward,
                saved_camera.up,
                saved_camera.vertical_fov_degrees,
                camera_aspect_ratio);
            orbit_camera.set_distance(saved_camera.orbit_distance);
            orbit_camera.set_camera(restored_camera);
            orbit_camera.set_vertical_fov_degrees(
                saved_camera.vertical_fov_degrees);
            free_camera.set_camera(restored_camera);
            free_camera.set_vertical_fov_degrees(
                saved_camera.vertical_fov_degrees);
            free_camera.set_movement_speed(
                saved_camera.free_movement_speed);
        }
        renderer::OrbitCameraController initial_orbit_camera = orbit_camera;
        renderer::FreeCameraController initial_free_camera = free_camera;
        const auto reset_cameras_for_scene = [&]() {
            const float aspect_ratio =
                static_cast<float>(settings.width) /
                static_cast<float>(settings.height);
            const float radius = std::max(
                0.5f,
                (viewer_scene.bounds.max - viewer_scene.bounds.min).norm() * 0.5f);
            orbit_camera = renderer::OrbitCameraController(
                viewer_scene.bounds,
                aspect_ratio);
            free_camera = renderer::FreeCameraController(
                orbit_camera.camera(),
                aspect_ratio,
                radius);
            initial_orbit_camera = orbit_camera;
            initial_free_camera = free_camera;
            ui_state.camera_mode = renderer::ViewerCameraMode::Orbit;
        };

        std::unique_ptr<renderer::ViewerRenderBackend> render_backend =
            renderer::make_viewer_render_backend(
                ui_state.mode,
                options.gl_vertex_shader,
                options.gl_fragment_shader);
        const auto current_render_scene_snapshot =
            [&]() -> const renderer::RenderSceneSnapshot& {
                return viewer_scene.document.render_scene_snapshot();
            };
        render_backend->reset(
            current_render_scene_snapshot(),
            settings);
        renderer::OpenGlShaderUiState shader_ui_state;
        shader_ui_state.vertex_path = options.gl_vertex_shader.string();
        shader_ui_state.fragment_path = options.gl_fragment_shader.string();
        renderer::CudaOpenGlInteropUiState interop_ui_state;
        renderer::FrameRateCounter frame_rate_counter;
        const auto set_viewer_title = [&](int path_samples) {
            std::string title = renderer::format_viewer_title(
                ui_state.mode,
                frame_rate_counter.snapshot(),
                path_samples);
            if (ui_state.mode == renderer::InteractiveRenderMode::Path) {
                title += " - cuda";
            }
            title += std::string(" - camera=") + camera_mode_name(ui_state.camera_mode);
            display.set_title(title);
        };
        set_viewer_title(0);

        const auto save_session_now = [&]() {
            if (!session_enabled) {
                return true;
            }
            try {
                renderer::ViewerSessionState state;
                state.document_path = viewer_scene.document.file_path();
                state.document_dirty = viewer_scene.document.dirty();
                const auto logical_size = display.logical_window_size();
                state.window_width = logical_size.first > 0
                    ? logical_size.first
                    : options.width;
                state.window_height = logical_size.second > 0
                    ? logical_size.second
                    : options.height;
                state.ui = ui_state;
                state.render_settings = settings;
                const renderer::Camera camera =
                    ui_state.camera_mode == renderer::ViewerCameraMode::Orbit
                    ? orbit_camera.camera()
                    : free_camera.camera();
                state.camera.eye = camera.eye();
                state.camera.forward = camera.forward();
                state.camera.up = camera.up();
                state.camera.vertical_fov_degrees =
                    ui_state.camera_mode == renderer::ViewerCameraMode::Orbit
                    ? orbit_camera.vertical_fov_degrees()
                    : free_camera.vertical_fov_degrees();
                state.camera.orbit_distance = orbit_camera.distance();
                state.camera.free_movement_speed =
                    free_camera.movement_speed();
                renderer::ViewerSessionStore::save(
                    session_path,
                    viewer_scene.document,
                    state);
                return true;
            } catch (const std::exception& error) {
                ui_state.scene_status =
                    "Session auto-save failed: " + std::string(error.what());
                std::cerr << "warning: " << ui_state.scene_status << '\n';
                return false;
            }
        };
        ViewerSessionSignature last_session_signature =
            make_session_signature(
                ui_state,
                settings,
                viewer_scene.document,
                display.logical_window_size());
        bool session_save_pending = session_enabled;
        auto session_changed_at = std::chrono::steady_clock::now();

        int rendered_frames = 0;
        std::string reported_interop_reason;
        bool running = true;
        auto previous_time = std::chrono::steady_clock::now();
        while (running) {
            const auto frame_begin = std::chrono::steady_clock::now();
            // Clamp the frame delta: dialog pauses, shader compiles, or a
            // dragged window would otherwise produce spikes that teleport the
            // free camera.
            const float delta_seconds = std::min(
                std::chrono::duration<float>(frame_begin - previous_time).count(),
                0.1f);
            previous_time = frame_begin;

            const renderer::InputState input = display.poll_input();
            running = !input.quit_requested;
            if (input.toggle_ui) {
                ui_state.panel_visible = !ui_state.panel_visible;
            }
            renderer::SceneChangeSet external_scene_changes =
                renderer::SceneChange::None;
            const auto import_asset_path = [&](const std::filesystem::path& path) {
                try {
                    viewer_scene.document.import_path(
                        path,
                        settings.width,
                        settings.height);
                    viewer_scene.bounds = viewer_scene.document.scene_bounds();
                    ui_state.scene_status = "Imported " + path.filename().string();
                    external_scene_changes = renderer::SceneChange::All;
                } catch (const std::exception& error) {
                    ui_state.scene_status =
                        "Import failed: " + std::string(error.what());
                }
            };
            const auto open_scene_path = [&](const std::filesystem::path& path) {
                try {
                    viewer_scene.document = renderer::SceneDocument::load(
                        path,
                        settings.width,
                        settings.height);
                    viewer_scene.bounds = viewer_scene.document.scene_bounds();
                    ui_state.selected_objects.clear();
                    ui_state.active_object = renderer::kInvalidObjectId;
                    reset_cameras_for_scene();
                    ui_state.scene_status = "Opened " + path.filename().string();
                    external_scene_changes = renderer::SceneChange::All;
                } catch (const std::exception& error) {
                    ui_state.scene_status =
                        "Open failed: " + std::string(error.what());
                }
            };
            const auto save_scene_path = [&](std::filesystem::path path) {
                try {
                    if (!has_extension(path, ".rscene")) {
                        path += ".rscene";
                    }
                    viewer_scene.document.save(path);
                    ui_state.scene_status = "Saved " + path.filename().string();
                } catch (const std::exception& error) {
                    ui_state.scene_status =
                        "Save failed: " + std::string(error.what());
                }
            };
            const auto load_environment_path = [&](const std::filesystem::path& path) {
                try {
                    viewer_scene.document.checkpoint();
                    viewer_scene.document.set_environment_map(path);
                    ui_state.scene_status =
                        "Loaded environment " + path.filename().string();
                    external_scene_changes |= renderer::SceneChange::Environment;
                } catch (const std::exception& error) {
                    ui_state.scene_status =
                        "Environment load failed: " + std::string(error.what());
                }
            };
            for (const std::string& dropped : input.dropped_paths) {
                const std::filesystem::path path(dropped);
                if (has_extension(path, ".rscene")) {
                    open_scene_path(path);
                } else if (has_extension(path, ".hdr") ||
                           has_extension(path, ".exr") ||
                           has_extension(path, ".png") ||
                           has_extension(path, ".jpg") ||
                           has_extension(path, ".jpeg")) {
                    load_environment_path(path);
                } else {
                    import_asset_path(path);
                }
            }
            for (const renderer::FileDialogResult& result : input.dialog_results) {
                if (!result.error.empty()) {
                    ui_state.scene_status = "Dialog failed: " + result.error;
                    continue;
                }
                for (const std::string& selected_path : result.paths) {
                    if (result.kind == renderer::FileDialogKind::OpenScene) {
                        open_scene_path(selected_path);
                    } else if (result.kind == renderer::FileDialogKind::SaveScene) {
                        save_scene_path(selected_path);
                    } else if (result.kind == renderer::FileDialogKind::OpenEnvironment) {
                        load_environment_path(selected_path);
                    } else {
                        import_asset_path(selected_path);
                    }
                }
            }
            display.begin_ui_frame();

            const renderer::InteractiveRenderMode previous_mode = ui_state.mode;
            const renderer::ViewerCameraMode previous_camera_mode = ui_state.camera_mode;
            const renderer::ViewerRenderBackendStatistics backend_statistics =
                render_backend->statistics();
            int accumulated_samples = 0;
            renderer::CudaPathStatistics cuda_statistics;
            renderer::OpenGlTechniqueDiagnostics technique_diagnostics;
            if (const auto* gl = std::get_if<renderer::OpenGlViewerStatistics>(
                    &backend_statistics)) {
                shader_ui_state.auto_reload = gl->shader_auto_reload;
                shader_ui_state.valid = gl->shader_valid;
                shader_ui_state.error = gl->shader_error;
                technique_diagnostics = gl->techniques;
                if (!gl->shader_vertex_path.empty()) {
                    shader_ui_state.vertex_path = gl->shader_vertex_path;
                }
                if (!gl->shader_fragment_path.empty()) {
                    shader_ui_state.fragment_path = gl->shader_fragment_path;
                }
            } else if (const auto* path =
                           std::get_if<renderer::CudaPathViewerStatistics>(
                               &backend_statistics)) {
                accumulated_samples = path->accumulated_samples;
                interop_ui_state.status = path->interop_status;
                interop_ui_state.detail = path->interop_detail;
                cuda_statistics = path->cuda;
            }
            renderer::ViewerUiActions ui_actions = viewer_ui.draw(
                ui_state,
                settings,
                viewer_scene.document,
                orbit_camera,
                free_camera,
                viewer_scene.bounds,
                frame_rate_counter.snapshot(),
                accumulated_samples,
                interop_ui_state,
                cuda_statistics,
                technique_diagnostics,
                shader_ui_state,
                display.main_window_has_keyboard_focus());

            // Escape quits only when neither ImGui nor the viewer wants the
            // keyboard (dismissing a popup/text edit must not close the app).
            if (input.escape_pressed &&
                !ImGui::GetIO().WantCaptureKeyboard &&
                !display.wants_keyboard_capture()) {
                running = false;
            }

            if (ui_actions.import_files_requested &&
                !display.show_import_files_dialog()) {
                ui_state.scene_status = display.last_error();
            }
            if (ui_actions.import_folder_requested &&
                !display.show_import_folder_dialog()) {
                ui_state.scene_status = display.last_error();
            }
            if (ui_actions.open_scene_requested &&
                !display.show_open_scene_dialog()) {
                ui_state.scene_status = display.last_error();
            }
            if (ui_actions.load_environment_requested &&
                !display.show_environment_dialog(
                    viewer_scene.document.environment_path().string())) {
                ui_state.scene_status = display.last_error();
            }
            if (ui_actions.save_scene_as_requested ||
                (ui_actions.save_scene_requested &&
                 viewer_scene.document.file_path().empty())) {
                const std::string location = viewer_scene.document.file_path().empty()
                    ? (std::filesystem::current_path() / "scene.rscene").string()
                    : viewer_scene.document.file_path().string();
                if (!display.show_save_scene_dialog(location)) {
                    ui_state.scene_status = display.last_error();
                }
            } else if (ui_actions.save_scene_requested) {
                save_scene_path(viewer_scene.document.file_path());
            }

            renderer::SceneChangeSet scene_changes =
                external_scene_changes | ui_actions.scene_changes;
            bool document_scene_changed =
                scene_changes != renderer::SceneChange::None;
            if (document_scene_changed) {
                viewer_scene.bounds = viewer_scene.document.scene_bounds();
            }

            const bool keyboard_available = !display.wants_keyboard_capture();
            const bool mouse_available = !display.wants_mouse_capture();
            if (keyboard_available) {
                renderer::InteractiveRenderMode hotkey_mode = ui_state.mode;
                if (input.render_mode_hotkey != 0) {
                    hotkey_mode = renderer::interactive_render_mode_from_hotkey(
                        input.render_mode_hotkey);
                }
                if (hotkey_mode != ui_state.mode) {
                    if (hotkey_mode == renderer::InteractiveRenderMode::Path) {
                        std::string reason;
                        if (!renderer::cuda_path_backend_available(&reason)) {
                            ui_state.scene_status =
                                "CUDA Path unavailable: " + reason;
                        } else {
                            ui_state.mode = hotkey_mode;
                            ui_actions.mode_changed = true;
                        }
                    } else {
                        ui_state.mode = hotkey_mode;
                        ui_actions.mode_changed = true;
                    }
                }
                if (input.toggle_camera_mode) {
                    ui_state.camera_mode = ui_state.camera_mode == renderer::ViewerCameraMode::Orbit
                        ? renderer::ViewerCameraMode::Free
                        : renderer::ViewerCameraMode::Orbit;
                    ui_actions.camera_mode_changed = true;
                }
                ui_actions.reset_requested = ui_actions.reset_requested || input.reset_render;
                ui_actions.shader_reload_requested =
                    ui_actions.shader_reload_requested || input.reload_shaders;
            }

            if (input.window_resized) {
                window_width = input.window_width;
                window_height = input.window_height;
            }

            const bool mode_changed = ui_actions.mode_changed || previous_mode != ui_state.mode;
            if (mode_changed) {
                render_backend = renderer::make_viewer_render_backend(
                    ui_state.mode,
                    options.gl_vertex_shader,
                    options.gl_fragment_shader);
                render_backend->reset(
                    current_render_scene_snapshot(),
                    settings);
                frame_rate_counter.reset();
                std::cout << "mode=" << mode_name(ui_state.mode) << '\n';
            }
            if (ui_actions.shader_auto_reload_changed) {
                if (renderer::OpenGlShaderControl* shader_control =
                        renderer::open_gl_shader_control(*render_backend)) {
                    shader_control->set_shader_auto_reload(
                        shader_ui_state.auto_reload);
                }
            }
            if (ui_actions.shader_reload_requested) {
                if (renderer::OpenGlShaderControl* shader_control =
                        renderer::open_gl_shader_control(*render_backend)) {
                    shader_control->request_shader_reload();
                }
            }

            bool camera_changed = ui_actions.camera_parameters_changed;
            if (ui_actions.focus_object != renderer::kInvalidObjectId) {
                const renderer::Bounds3 focus_bounds =
                    viewer_scene.document.world_bounds(ui_actions.focus_object);
                if (focus_bounds.min.allFinite() && focus_bounds.max.allFinite()) {
                    const float aspect_ratio =
                        static_cast<float>(settings.width) /
                        static_cast<float>(settings.height);
                    orbit_camera = renderer::OrbitCameraController(
                        focus_bounds,
                        aspect_ratio);
                    free_camera.set_camera(orbit_camera.camera());
                    ui_state.camera_mode = renderer::ViewerCameraMode::Orbit;
                    camera_changed = true;
                }
            }
            if (ui_actions.camera_reset_requested) {
                orbit_camera = initial_orbit_camera;
                free_camera = initial_free_camera;
                ui_state.camera_mode = renderer::ViewerCameraMode::Orbit;
                camera_changed = true;
            } else if (ui_actions.camera_mode_changed || previous_camera_mode != ui_state.camera_mode) {
                transition_camera_mode(
                    previous_camera_mode,
                    ui_state.camera_mode,
                    orbit_camera,
                    free_camera);
                camera_changed = true;
                std::cout << "camera=" << camera_mode_name(ui_state.camera_mode) << '\n';
            }
            if (ui_actions.look_through_camera != renderer::kInvalidObjectId) {
                const float aspect_ratio =
                    static_cast<float>(settings.width) /
                    static_cast<float>(settings.height);
                if (auto scene_camera = make_scene_camera_view(
                        viewer_scene.document,
                        ui_actions.look_through_camera,
                        aspect_ratio)) {
                    free_camera.set_camera(scene_camera->camera);
                    free_camera.set_vertical_fov_degrees(
                        scene_camera->vertical_fov_degrees);
                    ui_state.camera_mode = renderer::ViewerCameraMode::Free;
                    ui_actions.camera_mode_changed = true;
                    camera_changed = true;
                    ui_state.scene_status = "Looking through imported scene camera";
                } else {
                    ui_state.scene_status = "Selected scene camera is invalid";
                }
            }

            renderer::InteractiveFrameState frame_state;
            frame_state.delta_seconds = delta_seconds;
            frame_state.scene_changes = scene_changes;
            frame_state.reset_requested = ui_actions.reset_requested;
            frame_state.reset_requested =
                frame_state.reset_requested ||
                ui_actions.path_depth_changed ||
                ui_actions.path_roulette_changed;
            frame_state.automatic_interaction_quality =
                ui_state.automatic_interaction_quality;

            if (input.window_resized || ui_actions.render_scale_changed) {
                settings.width = scaled_dimension(window_width, ui_state.render_scale);
                settings.height = scaled_dimension(window_height, ui_state.render_scale);
                const float aspect_ratio =
                    static_cast<float>(settings.width) / static_cast<float>(settings.height);
                orbit_camera.set_aspect_ratio(aspect_ratio);
                free_camera.set_aspect_ratio(aspect_ratio);
                frame_state.framebuffer_resized = true;
            }

            const bool desired_relative_mouse =
                ui_state.camera_mode == renderer::ViewerCameraMode::Free &&
                input.right_mouse_down &&
                mouse_available;
            if (!display.set_relative_mouse_mode(desired_relative_mouse)) {
                // Recoverable input-mode issue (e.g. focus lost mid-drag);
                // degrade to absolute input instead of terminating.
                ui_state.scene_status = display.last_error();
            }

            const bool mouse_moved =
                input.mouse_delta_x != 0.0f || input.mouse_delta_y != 0.0f;
            if (!ui_actions.camera_reset_requested && mouse_available &&
                ui_state.camera_mode == renderer::ViewerCameraMode::Orbit) {
                if (input.left_mouse_down && mouse_moved &&
                    !ui_state.gizmo_hovered && !ui_state.gizmo_was_using) {
                    orbit_camera.orbit(input.mouse_delta_x, input.mouse_delta_y);
                    camera_changed = true;
                }
                if (input.right_mouse_down && mouse_moved &&
                    !ui_state.gizmo_hovered && !ui_state.gizmo_was_using) {
                    orbit_camera.pan(
                        input.mouse_delta_x,
                        input.mouse_delta_y,
                        static_cast<float>(std::max(1, window_height)));
                    camera_changed = true;
                }
                if (input.wheel_delta != 0.0f) {
                    orbit_camera.zoom(input.wheel_delta);
                    camera_changed = true;
                }
            } else if (!ui_actions.camera_reset_requested &&
                       ui_state.camera_mode == renderer::ViewerCameraMode::Free) {
                if (mouse_available && input.right_mouse_down && mouse_moved) {
                    free_camera.look(input.mouse_delta_x, input.mouse_delta_y);
                    camera_changed = true;
                }
                if (keyboard_available) {
                    const float forward_axis =
                        static_cast<float>(input.move_forward) -
                        static_cast<float>(input.move_backward);
                    const float right_axis =
                        static_cast<float>(input.move_right) -
                        static_cast<float>(input.move_left);
                    const float up_axis =
                        static_cast<float>(input.move_up) -
                        static_cast<float>(input.move_down);
                    camera_changed = free_camera.move(
                        forward_axis,
                        right_axis,
                        up_axis,
                        delta_seconds) || camera_changed;
                }
            }
            frame_state.camera_changed = camera_changed;

            const renderer::Camera camera = ui_state.camera_mode == renderer::ViewerCameraMode::Orbit
                ? orbit_camera.camera()
                : free_camera.camera();
            if (input.left_mouse_clicked && mouse_available &&
                !ui_state.gizmo_hovered && !ui_state.gizmo_was_using) {
                const float u = std::clamp(
                    input.mouse_x / static_cast<float>(std::max(1, window_width)),
                    0.0f,
                    1.0f);
                const float v = std::clamp(
                    1.0f - input.mouse_y / static_cast<float>(std::max(1, window_height)),
                    0.0f,
                    1.0f);
                const auto picked = viewer_scene.document.pick(camera.generate_ray(u, v));
                const bool additive = ImGui::GetIO().KeyCtrl;
                if (picked) {
                    renderer::select_viewer_object(
                        ui_state,
                        picked->object_id,
                        additive);
                } else if (!additive) {
                    ui_state.selected_objects.clear();
                    ui_state.active_object = renderer::kInvalidObjectId;
                }
            }
            const renderer::SceneChangeSet gizmo_changes =
                viewer_ui.draw_scene_gizmo(
                    ui_state,
                    viewer_scene.document,
                    camera,
                    viewer_scene.bounds);
            if (gizmo_changes != renderer::SceneChange::None) {
                viewer_scene.bounds = viewer_scene.document.scene_bounds();
                document_scene_changed = true;
                scene_changes |= gizmo_changes;
            }
            frame_state.scene_changes = scene_changes;
            viewer_ui.draw_scene_selection(
                ui_state,
                viewer_scene.document,
                camera);
            viewer_ui.draw_directional_light_indicator(
                ui_state,
                viewer_scene.document,
                camera,
                viewer_scene.bounds);
            viewer_ui.draw_point_light_markers(
                ui_state,
                viewer_scene.document,
                camera);
            const bool path_needs_preview =
                rendered_frames == 0 ||
                mode_changed ||
                ui_actions.automatic_interaction_quality_changed ||
                ui_actions.path_depth_changed ||
                ui_actions.path_roulette_changed ||
                frame_state.camera_changed ||
                frame_state.scene_changes != renderer::SceneChange::None ||
                frame_state.framebuffer_resized ||
                frame_state.reset_requested;
            const bool should_render =
                ui_state.mode != renderer::InteractiveRenderMode::Path ||
                !ui_state.path_accumulation_paused ||
                path_needs_preview;
            if (should_render) {
                render_backend->render(
                    current_render_scene_snapshot(),
                    camera,
                    settings,
                    frame_state);
            }
            const renderer::ViewerRenderBackendStatistics current_statistics =
                render_backend->statistics();
            const auto* current_path =
                std::get_if<renderer::CudaPathViewerStatistics>(
                    &current_statistics);
            if (current_path &&
                current_path->interop_status == "fallback" &&
                !current_path->interop_detail.empty() &&
                current_path->interop_detail != reported_interop_reason) {
                reported_interop_reason = current_path->interop_detail;
                std::cerr << "warning: CUDA/OpenGL interop unavailable, using host staging: "
                          << reported_interop_reason << '\n';
            }
            display.present(render_backend->output(), ui_state.display);

            if (session_enabled) {
                const ViewerSessionSignature signature =
                    make_session_signature(
                        ui_state,
                        settings,
                        viewer_scene.document,
                        display.logical_window_size());
                if (!(signature == last_session_signature) ||
                    document_scene_changed ||
                    camera_changed) {
                    last_session_signature = signature;
                    session_save_pending = true;
                    session_changed_at = std::chrono::steady_clock::now();
                }
                if (session_save_pending &&
                    std::chrono::steady_clock::now() - session_changed_at >=
                        std::chrono::seconds(1)) {
                    save_session_now();
                    session_save_pending = false;
                }
            }

            ++rendered_frames;
            const int path_samples = current_path
                ? current_path->accumulated_samples
                : 0;
            const auto frame_end = std::chrono::steady_clock::now();
            const float frame_seconds =
                std::chrono::duration<float>(frame_end - frame_begin).count();
            if (frame_rate_counter.tick(frame_seconds) ||
                mode_changed ||
                ui_actions.camera_mode_changed ||
                ui_actions.camera_reset_requested ||
                rendered_frames == 1) {
                set_viewer_title(path_samples);
            }
            if (options.frame_limit > 0 && rendered_frames >= options.frame_limit) {
                running = false;
            }
        }

        if (session_enabled) {
            save_session_now();
        }
        std::cout << "viewer mode=" << mode_name(ui_state.mode)
                  << " frames=" << rendered_frames
                  << " size=" << settings.width << "x" << settings.height;
        const renderer::ViewerRenderBackendStatistics final_statistics =
            render_backend->statistics();
        if (ui_state.mode == renderer::InteractiveRenderMode::OpenGl) {
            const auto& gl =
                std::get<renderer::OpenGlViewerStatistics>(final_statistics);
            std::cout << " shader="
                      << (gl.shader_valid ? "active" : "invalid");
            if (!gl.shader_error.empty()) {
                std::cerr << "\nshader error: " << gl.shader_error;
            }
        } else if (ui_state.mode == renderer::InteractiveRenderMode::Path) {
            const auto& path =
                std::get<renderer::CudaPathViewerStatistics>(final_statistics);
            std::cout << " interop="
                      << path.interop_status;
            std::cout << " trace_ms=" << path.cuda.trace_milliseconds
                      << " present_ms="
                      << path.cuda.presentation_milliseconds
                      << " published="
                      << (path.cuda.presentation_updated ? 1 : 0)
                      << " upload_ms=" << path.cuda.upload_milliseconds
                      << " instance_ms="
                      << path.cuda.instance_upload_milliseconds
                      << " tlas_refit_ms="
                      << path.cuda.tlas_refit_milliseconds
                      << " blas_builds="
                      << path.cuda.blas_build_count
                      << " tlas_refits="
                      << path.cuda.tlas_refit_count
                      << " allocations="
                      << path.cuda.allocation_generation
                      << " downloads="
                      << path.cuda.framebuffer_downloads;
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n\n";
        print_help();
        return 1;
    }
}
