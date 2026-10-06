#include "render/optix/optix_realtime_renderer.h"
#include "interactive/frame_rate_counter.h"
#include "interactive/free_camera_controller.h"
#include "interactive/orbit_camera_controller.h"
#include "interactive/viewer_session.h"
#include "interactive/viewer_ui.h"
#include "platform/opengl/cuda_opengl_interop.h"
#include "render/ddgi/ddgi_settings_json.h"
#include "platform/sdl/sdl_display_backend.h"
#include "render/interactive/render_mode.h"
#include "render/interactive/viewer_render_backend.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "render/realtime/realtime_settings_json.h"
#include "render/dxr/dxr_settings_json.h"
#include <fstream>
#include "scene/scene.h"
#include "scene/scene_asset_loader.h"
#include "scene/scene_document.h"

#include <imgui.h>
#include <glad/gl.h>

#include <algorithm>
#include <array>
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
    std::optional<bool> ddgi;
    std::optional<renderer::OpenGlRenderStyle> style;
    std::filesystem::path capture_path;
    std::string scene = "asset";
    std::vector<std::string> asset_paths;
    std::filesystem::path scene_file;
    renderer::InteractiveRenderMode mode = renderer::InteractiveRenderMode::Rtrt;
    std::filesystem::path realtime_config;
    std::filesystem::path dxr_config;
    bool strict_dxr = false;
    std::filesystem::path frame_report, camera_preset;
    int warmup_frames = 60;
    bool dxr_settle_before_warmup = false;
    bool camera_motion = false;
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
    bool environment_intensity_set=false,environment_yaw_set=false,environment_background_set=false;
    bool restore_last_session = true;
    bool disable_cuda_interop = false;
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
        << "  viewer --scene builtin|asset [--asset path\\to\\model-or-directory ...] --mode opengl|rtrt|dxr [options]\n\n"
        << "Options:\n"
        << "  --asset path      OBJ/glTF/GLB file or directory; may be repeated\n"
        << "  --scene many-lights  deterministic Cornell scene with 256 point lights\n"
        << "  --scene-file path open a saved .rscene document\n"
        << "  --environment path  2:1 HDR/EXR/PNG/JPG environment map\n"
        << "  --environment-intensity value  environment multiplier, default 1\n"
        << "  --environment-yaw degrees  rotate the environment around Y\n"
        << "  --hide-environment-background  light the scene without drawing the map\n"
        << "  --width integer    window width, default 960\n"
        << "  --height integer   window height, default 540\n"
        << "  --frames integer   render N frames then exit, default unlimited\n"
        << "  --ddgi on|off      enable or disable OpenGL dynamic irradiance probes\n"
        << "  --frame-report path.json  measure full frames through GPU presentation completion\n"
        << "  --warmup-frames integer  exclude initial frames from report, default 60\n"
        << "  --dxr-settle-before-warmup  finish OMM/SER initialization before the scheduled frames\n"
        << "  --camera-preset path.json  read the camera object from a benchmark case\n"
        << "  --camera-motion   repeat 600 frames of lateral motion and 300 settled frames\n"
        << "  --style realistic|toon|sketch  OpenGL rendering style\n"
        << "  --capture path.png  save the final render without UI (use with --frames)\n"
        << "  --no-cuda-interop  use host presentation (diagnose interop compatibility)\n"
        << "  --rtrt-config path.json  RTRT settings object (same fields as session render.realtime)\n"
        << "  --dxr-config path.json  independent DXR settings object\n"
        << "  --strict-dxr       fail if DXR cannot run; never fall back\n"
        << "  --cuda-device N    CUDA device index for RTRT/DDGI; default GL-compatible\n"
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
        } else if (arg == "--no-cuda-interop") {
            options.disable_cuda_interop = true;
        } else if (arg == "--rtrt-config") {
            options.realtime_config = require_value(argc, argv, i, arg);
        } else if (arg == "--dxr-config") {
            options.dxr_config = require_value(argc, argv, i, arg);
        } else if (arg == "--strict-dxr") {
            options.strict_dxr = true;
        } else if (arg == "--style") {
            const std::string style = require_value(argc, argv, i, arg);
            if (style == "realistic") options.style = renderer::OpenGlRenderStyle::Realistic;
            else if (style == "toon") options.style = renderer::OpenGlRenderStyle::Toon;
            else if (style == "sketch") options.style = renderer::OpenGlRenderStyle::Sketch;
            else throw std::invalid_argument("--style must be realistic, toon, or sketch");
        } else if (arg == "--ddgi") {
            const auto value = require_value(argc, argv, i, arg);
            if (value != "on" && value != "off") throw std::invalid_argument("--ddgi expects on or off");
            options.ddgi = value == "on";
        } else if (arg == "--capture") {
            options.capture_path = require_value(argc, argv, i, arg);
        } else if (arg == "--width") {
            options.width = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--height") {
            options.height = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--frames") {
            options.frame_limit = parse_nonnegative_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--frame-report") {
            options.frame_report = require_value(argc, argv, i, arg);
        } else if (arg == "--warmup-frames") {
            options.warmup_frames = parse_nonnegative_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--dxr-settle-before-warmup") {
            options.dxr_settle_before_warmup = true;
        } else if (arg == "--camera-preset") {
            options.camera_preset = require_value(argc, argv, i, arg);
        } else if (arg == "--camera-motion") {
            options.camera_motion = true;
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
            options.environment_intensity_set=true;
            options.environment_intensity =
                parse_finite_float(require_value(argc, argv, i, arg), arg);
            if (options.environment_intensity < 0.0f) {
                throw std::invalid_argument(arg + " must be non-negative");
            }
        } else if (arg == "--environment-yaw") {
            options.environment_yaw_set=true;
            options.environment_yaw_degrees =
                parse_finite_float(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--hide-environment-background") {
            options.environment_background_set=true;
            options.environment_background_visible = false;
        } else if (arg == "--no-restore-last") {
            options.restore_last_session = false;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.scene != "builtin" && options.scene != "asset" && options.scene != "many-lights") {
        throw std::invalid_argument("--scene must be builtin, asset, or many-lights");
    }
    if (!options.scene_file.empty() && options.scene != "asset") {
        throw std::invalid_argument("--scene-file requires --scene asset");
    }
    if (options.dxr_settle_before_warmup && (options.mode != renderer::InteractiveRenderMode::Dxr ||
        options.frame_report.empty() || options.frame_limit <= options.warmup_frames)) {
        throw std::invalid_argument("--dxr-settle-before-warmup requires --mode dxr, --frame-report, and --frames greater than --warmup-frames");
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
        if(options.scene_file.empty() || options.environment_intensity_set)document.set_environment_intensity(options.environment_intensity);
        if(options.scene_file.empty() || options.environment_yaw_set)document.set_environment_rotation_degrees(options.environment_yaw_degrees);
        if(options.scene_file.empty() || options.environment_background_set)document.set_environment_background_visible(options.environment_background_visible);
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
    if(options.scene=="many-lights") {
        for(int layer=0;layer<2;++layer)for(int y=0;y<8;++y)for(int x=0;x<16;++x) {
            renderer::PointLight light;light.position=renderer::Vec3(-.9f+1.8f*x/15.f,-.8f+1.6f*y/7.f,-.3f-1.4f*layer);
            light.intensity=.025f*renderer::Color(.3f+.7f*x/15.f,.3f+.7f*y/7.f,.4f+.6f*layer);scene.point_lights.push_back(light);
        }
    }
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
    renderer::RealtimeRenderSettings realtime;
    renderer::DxrRenderSettings dxr;
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
    signature.realtime = settings.realtime;
    signature.dxr = settings.dxr;
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
        nlohmann::json camera_preset;
        if(!options.camera_preset.empty()) {
            std::ifstream input(options.camera_preset);
            if(!input) throw std::runtime_error("Cannot open camera preset: "+options.camera_preset.string());
            nlohmann::json document;input>>document;camera_preset=document.at("camera");
        }
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
        const bool startup_dxr=(restored_session?restored_session->ui.mode:options.mode)==renderer::InteractiveRenderMode::Dxr;
        bool dxr_startup_failed=false;
        if (!display.initialize(options.width, options.height, "3D Renderer Viewer",startup_dxr,options.frame_limit==0)) {
            if(!startup_dxr || options.strict_dxr)throw std::runtime_error(display.last_error());
            startup_status="DXR unavailable; switched to OpenGL: "+display.last_error();dxr_startup_failed=true;display.switch_presentation(false);
            std::cout<<"dxr-mode-unavailable: "<<startup_status<<'\n';
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
        if(dxr_startup_failed)ui_state.mode=renderer::InteractiveRenderMode::OpenGl;
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
        if (options.ddgi) settings.opengl.ddgi.enabled = *options.ddgi;
        if (!options.realtime_config.empty()) {
            std::ifstream config_file(options.realtime_config);
            if (!config_file) throw std::runtime_error("cannot open RTRT config");
            nlohmann::json config; config_file >> config;
            settings.realtime = renderer::parse_realtime_settings(config);
        }
        if(!options.dxr_config.empty()) {
            std::ifstream input(options.dxr_config);if(!input)throw std::runtime_error("cannot open DXR config");
            nlohmann::json config;input>>config;settings.dxr=renderer::parse_dxr_settings(config);
        }
        if (options.style) {
            settings.opengl.npr.style = *options.style;
            settings.opengl.shadow_map.debug_view = renderer::OpenGlShadowDebugView::Final;
            settings.opengl.ambient_occlusion.debug_view = renderer::OpenGlAmbientOcclusionDebugView::Final;
            settings.opengl.ssr.debug_view = renderer::OpenGlSsrDebugView::Final;
        }
        if (!restored_session) {
            settings.path.cuda_device = options.cuda_device;
        }
        if (ui_state.mode == renderer::InteractiveRenderMode::Rtrt) {
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
            if (device_context && renderer::optix_realtime_available(device_context->device_id(),&reason)) {
                settings.path.cuda_device = device_context->device_id();
            } else {
                settings.path.cuda_device = 0;
                ui_state.mode = renderer::InteractiveRenderMode::OpenGl;
                ui_state.scene_status =
                    "OptiX RTRT unavailable; switched to OpenGL: " + reason;
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
        const auto reset_cameras_for_scene = [&]() {
            viewer_scene.bounds=viewer_scene.document.scene_bounds();
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
            ui_state.camera_mode = renderer::ViewerCameraMode::Orbit;
        };

        std::unique_ptr<renderer::ViewerRenderBackend> render_backend;
        const auto current_render_scene_snapshot =
            [&]() -> const renderer::RenderSceneSnapshot& {
                return viewer_scene.document.render_scene_snapshot();
            };
        const auto reset_render_backend = [&]() {
            const auto create = [&]() {
                render_backend.reset();
                display.switch_presentation(ui_state.mode==renderer::InteractiveRenderMode::Dxr);
                auto next = renderer::make_viewer_render_backend(
                    ui_state.mode, options.gl_vertex_shader,
                    options.gl_fragment_shader, options.disable_cuda_interop,display.dxr_context());
                next->reset(current_render_scene_snapshot(), settings);
                render_backend = std::move(next);
            };
            try { create(); }
            catch (const std::exception& error) {
                const bool dxr=ui_state.mode==renderer::InteractiveRenderMode::Dxr;
                if (ui_state.mode==renderer::InteractiveRenderMode::OpenGl || (dxr && options.strict_dxr)) throw;
                ui_state.mode = renderer::InteractiveRenderMode::OpenGl;
                ui_state.scene_status = std::string(dxr?"DXR":"OptiX RTRT")+" unavailable; switched to OpenGL: " + error.what();
                std::cerr << "warning: " << ui_state.scene_status << '\n';
                std::cout << "path-mode-unavailable: switched to OpenGL (" << error.what() << ")\n";
                create();
            }
        };
        reset_render_backend();
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
            if (ui_state.mode == renderer::InteractiveRenderMode::Rtrt) {
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
        int startup_settling_frames = 0;
        bool startup_settling = options.dxr_settle_before_warmup;
        double startup_settling_ms = 0;
        const auto startup_begin = std::chrono::steady_clock::now();
        nlohmann::json measured_frames=nlohmann::json::array();
        std::string reported_interop_reason;
        bool running = true;
        auto previous_time = std::chrono::steady_clock::now();
        while (running) {
            if(render_backend->mode()!=ui_state.mode){reset_render_backend();frame_rate_counter.reset();}
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
            renderer::DxrStatistics dxr_statistics;
            if(const auto* dxr=std::get_if<renderer::DxrStatistics>(&backend_statistics))dxr_statistics=*dxr;
            renderer::OpenGlTechniqueDiagnostics technique_diagnostics;
            if (const auto* gl = std::get_if<renderer::OpenGlViewerStatistics>(
                    &backend_statistics)) {
                shader_ui_state.auto_reload = gl->shader_auto_reload;
                shader_ui_state.valid = gl->shader_valid;
                shader_ui_state.error = gl->shader_error;
                technique_diagnostics = gl->techniques;
                shader_ui_state.ddgi = gl->ddgi;
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
                display.main_window_has_keyboard_focus(),dxr_statistics);

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
                    if (hotkey_mode == renderer::InteractiveRenderMode::Rtrt) {
                        std::string reason;
                        if (!renderer::optix_realtime_available(settings.path.cuda_device,&reason)) {
                            ui_state.scene_status =
                                "OptiX RTRT unavailable: " + reason;
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
                // The next iteration rebuilds presentation before ImGui starts a frame.
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
                reset_cameras_for_scene();
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
            frame_state.camera_cut = ui_actions.camera_reset_requested ||
                ui_actions.look_through_camera != renderer::kInvalidObjectId;
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

            renderer::Camera camera = ui_state.camera_mode == renderer::ViewerCameraMode::Orbit
                ? orbit_camera.camera()
                : free_camera.camera();
            if(!camera_preset.is_null()) {
                auto vector=[&](const char* key) {
                    const auto values=camera_preset.at(key).get<std::array<float,3>>();
                    return renderer::Vec3(values[0],values[1],values[2]);
                };
                camera=renderer::Camera(vector("eye"),vector("eye")+vector("forward"),vector("up"),
                    camera_preset.at("vertical_fov_degrees").get<float>(),float(settings.width)/settings.height);
            }
            const int scheduled_frame=rendered_frames-startup_settling_frames;
            const bool scripted_motion=options.camera_motion && !startup_settling && scheduled_frame%900<600;
            if(options.camera_motion && !startup_settling) {
                const float shift=scripted_motion?.5f*std::sin(float(scheduled_frame%900)*2*3.14159265358979323846f/600):0;
                const auto eye=camera.eye()+camera.right()*shift;
                const float fov=2*std::atan(camera.viewport_height()*.5f)*180/3.14159265358979323846f;
                camera=renderer::Camera(eye,eye+camera.forward(),camera.up(),fov,float(settings.width)/settings.height);
                frame_state.camera_changed=true;
            }
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
            const auto snapshot_begin=std::chrono::steady_clock::now();
            const auto& render_scene=current_render_scene_snapshot();
            const auto snapshot_end=std::chrono::steady_clock::now();
            render_backend->render(render_scene, camera, settings, frame_state);
            const auto render_end=std::chrono::steady_clock::now();
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
            const auto present_end=std::chrono::steady_clock::now();

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
            // Profiling only: close the GL/CUDA queue so the measured frame
            // includes completed rendering, UI composition and swap submission.
            // Normal interactive rendering remains asynchronous.
            const auto completion_wait_begin=std::chrono::steady_clock::now();
            if(!options.frame_report.empty()) display.wait_for_frame();
            const auto frame_end = std::chrono::steady_clock::now();
            const float frame_seconds =
                std::chrono::duration<float>(frame_end - frame_begin).count();
            if(startup_settling) {
                ++startup_settling_frames;
                startup_settling_ms=std::chrono::duration<double,std::milli>(frame_end-startup_begin).count();
                const auto* dxr=std::get_if<renderer::DxrStatistics>(&current_statistics);
                if(!dxr || !dxr->active) throw std::runtime_error("DXR startup settling requires an active DXR backend");
                const bool ser_ready=!settings.dxr.shader_execution_reordering || !dxr->device.ser_supported ||
                    !dxr->device.ser_reorders || dxr->ser_probe_complete;
                if(dxr->omm_pending==0 && ser_ready) {
                    startup_settling=false;
                    std::cout<<"dxr_startup_settling frames="<<startup_settling_frames<<" milliseconds="<<startup_settling_ms<<'\n';
                } else if(startup_settling_ms>120000) {
                    throw std::runtime_error("DXR startup did not settle within 120 seconds (OMM pending="+
                        std::to_string(dxr->omm_pending)+", SER probe complete="+std::to_string(dxr->ser_probe_complete)+")");
                }
            }
            const int completed_scheduled_frames=rendered_frames-startup_settling_frames;
            if(!options.frame_report.empty() && completed_scheduled_frames>options.warmup_frames) {
                nlohmann::json sample={{"frame",completed_scheduled_frames},{"rendered_frame",rendered_frames},
                    {"moving",scripted_motion},{"frame_ms",1000*frame_seconds}};
                const auto ms=[](auto begin,auto end){return std::chrono::duration<double,std::milli>(end-begin).count();};
                sample.update({{"cpu_ui_input_ms",ms(frame_begin,snapshot_begin)},
                    {"cpu_scene_snapshot_ms",ms(snapshot_begin,snapshot_end)},
                    {"cpu_render_record_ms",ms(snapshot_end,render_end)},
                    {"cpu_present_ms",ms(render_end,present_end)},
                    {"cpu_session_ms",ms(present_end,completion_wait_begin)},
                    {"cpu_completion_wait_ms",ms(completion_wait_begin,frame_end)}});
                const auto statistics=render_backend->statistics();
                if(const auto* path=std::get_if<renderer::CudaPathViewerStatistics>(&statistics)) {
                    const auto& rt=path->cuda.realtime;
                    sample.update({{"cuda_ms",rt.total_ms},{"primary_ms",rt.gbuffer_ms},{"lighting_ms",rt.lighting_ms},
                        {"hardware_rt",rt.hardware_ray_tracing_active},{"hardware_detail",rt.hardware_ray_tracing_detail},
                        {"rt_core_version",rt.rt_core_version},{"ser_supported",rt.ser_supported},{"ser_active",rt.ser_active},
                        {"gas_builds",rt.gas_builds},{"ias_builds",rt.ias_builds},{"ias_updates",rt.ias_updates},{"acceleration_ms",rt.acceleration_ms},
                        {"denoiser",!settings.realtime.denoise?"off":(rt.optix_denoiser_active?"optix":"svgf")},
                        {"optix_denoiser_temporal",rt.optix_denoiser_temporal},{"optix_denoiser_bytes",rt.optix_denoiser_bytes},
                        {"optix_denoiser_detail",rt.optix_denoiser_detail},
                        {"temporal_ms",rt.temporal_ms},{"filter_ms",rt.filter_ms},{"reconstruction_ms",rt.reconstruction_ms},
                        {"internal_width",path->cuda.internal_width},{"internal_height",path->cuda.internal_height},
                        {"framebuffer_bytes",rt.framebuffer_bytes},{"hardware_bytes",rt.hardware_ray_tracing_bytes},
                        {"allocations",path->cuda.allocation_generation},
                        {"downloads",path->cuda.framebuffer_downloads},{"history_resets",rt.history_resets}});
                }
                if(const auto* gl=std::get_if<renderer::OpenGlViewerStatistics>(&statistics)) {
                    const auto& d=gl->ddgi;
                    sample.update({{"ddgi",d.status},{"probe_trace_ms",d.trace_ms},{"probe_blend_ms",d.blend_ms},
                        {"probe_gather_ms",d.gather_ms},{"probe_export_ms",d.export_ms},{"probe_count",d.probe_count},{"active_probes",d.active_probes},
                        {"updated_probes",d.updated_probes},{"maximum_probe_age",d.maximum_age},
                        {"probe_memory_bytes",d.memory_bytes},{"atlas_downloads",d.atlas_downloads},{"probe_resets",d.reset_count}});
                }
                if(const auto* d=std::get_if<renderer::DxrStatistics>(&statistics)) {
                    sample.update({{"adapter",d->device.adapter},{"dxr_tier",d->device.raytracing_tier},{"shader_model",d->device.shader_model},
                        {"ser_supported",d->device.ser_supported},{"ser_active",d->ser_active},{"omm_supported",d->device.omm_supported},{"omm_active",d->omm_active},
                        {"omm_builds",d->omm_builds},{"omm_pending",d->omm_pending},{"omm_states",d->omm_states},
                        {"enhanced_barriers_supported",d->device.enhanced_barriers},{"enhanced_barriers_active",d->enhanced_barriers_active},
                        {"restir_di",d->restir_di_active},{"restir_pt",d->restir_pt_active},{"reconstruction",d->reconstruction},{"detail",d->detail},
                        {"acceleration_ms",d->acceleration_ms},{"gbuffer_ms",d->gbuffer_ms},{"direct_ms",d->direct_ms},{"indirect_ms",d->indirect_ms},{"reconstruction_ms",d->reconstruction_ms},{"gpu_ms",d->total_ms},
                        {"allocated_bytes",d->allocated_bytes},{"internal_width",d->internal_width},{"internal_height",d->internal_height},{"readbacks",d->readbacks},{"history_resets",d->history_resets}});
                    sample.update({{"pooled_bytes",d->pooled_bytes},{"resource_creations",d->resource_creations},{"resource_reuses",d->resource_reuses},{"pipeline_cache_hits",d->pipeline_cache_hits}});
                    sample.update({{"video_memory_available",d->video_memory_available},{"video_memory_usage",d->video_memory_usage},{"video_memory_budget",d->video_memory_budget}});
                    sample["presentation_ms"]=d->presentation_ms;
                    sample.update({{"blas_builds",d->blas_builds},{"tlas_builds",d->tlas_builds},{"tlas_updates",d->tlas_updates}});
                    sample.update({{"pt_initial_ms",d->pt_initial_ms},{"pt_temporal_ms",d->pt_temporal_ms},{"pt_spatial_ms",d->pt_spatial_ms},
                        {"debug_layer_active",d->device.debug_layer_active},{"gpu_validation_active",d->device.gpu_validation_active}});
                    sample.update({{"ser_actually_reorders",d->device.ser_reorders},{"ser_probe_complete",d->ser_probe_complete},
                        {"ser_measured_speedup",d->ser_measured_speedup},{"ser_probe_ms",d->ser_probe_ms},{"trace_probe_ms",d->trace_probe_ms}});
                }
                measured_frames.push_back(std::move(sample));
            }
            if (frame_rate_counter.tick(frame_seconds) ||
                mode_changed ||
                ui_actions.camera_mode_changed ||
                ui_actions.camera_reset_requested ||
                rendered_frames == 1) {
                set_viewer_title(path_samples);
            }
            if (options.frame_limit > 0 && completed_scheduled_frames >= options.frame_limit) {
                running = false;
            }
        }

        if (session_enabled) {
            save_session_now();
        }
        if(!options.frame_report.empty()) {
            if(startup_settling) throw std::runtime_error("DXR startup settling was interrupted before measurement");
            if(measured_frames.empty()) throw std::runtime_error("Frame report has no samples after warmup");
            std::vector<double> times;for(const auto& frame:measured_frames)times.push_back(frame.at("frame_ms").get<double>());
            std::sort(times.begin(),times.end());
            const auto quantile=[&](double p){return times[std::min(times.size()-1,std::size_t(std::ceil(p*times.size()))-1)];};
            const auto& scene=current_render_scene_snapshot();std::size_t triangles=0;
            for(const auto& asset:scene.assets)if(asset.local_scene)triangles+=asset.local_scene->triangles.size();
            nlohmann::json report={{"width",settings.width},{"height",settings.height},{"scene_file",options.scene_file.generic_string()},
                {"camera_preset",options.camera_preset.generic_string()},{"camera_motion",options.camera_motion},{"triangles",triangles},
                {"warmup_frames",options.warmup_frames},{"settings",renderer::realtime_settings_json(settings.realtime)},
                {"measurement","CPU frame start through GL finish after presentation; swap interval 0"},
                {"count",times.size()},{"p50_ms",quantile(.5)},{"p95_ms",quantile(.95)},{"p99_ms",quantile(.99)},
                {"over_16_667_ms",std::count_if(times.begin(),times.end(),[](double ms){return ms>1000.0/60;})},
                {"samples",std::move(measured_frames)}};
            report["mode"] = mode_name(ui_state.mode);
            report["startup_settling_enabled"]=options.dxr_settle_before_warmup;
            report["startup_settling_frames"]=startup_settling_frames;
            report["startup_settling_ms"]=startup_settling_ms;
            report["rendered_frames"]=rendered_frames;
            if(ui_state.mode==renderer::InteractiveRenderMode::Dxr){
                report["settings"]=renderer::dxr_settings_json(settings.dxr);report["measurement"]="CPU frame start through D3D12 fence after DXGI presentation submission; vsync off";
                const auto stats=std::get<renderer::DxrStatistics>(render_backend->statistics());
                report["driver_version"]=stats.device.driver_version;report["dlss_runtime_pinned"]=stats.dlss_runtime_pinned;
                report["runtime_modules"]=nlohmann::json::array();
                for(const auto& module:stats.runtime_modules)report["runtime_modules"].push_back({{"name",module.name},{"path",module.path},{"version",module.version},{"sha256",module.sha256},{"expected_sha256",module.expected_sha256},{"pinned",module.pinned}});
                report["sdk_versions"]={{"agility","1.619.6"},{"dxc","1.9.2609"},{"rtxdi","3.1.0"},{"rtxdi_library","f12037fa8e97ebc08e9e3edfd2de528ed1772a4b"},{"nrd","4.17.3"},{"streamline","2.14.1"}};
#ifdef _MSC_FULL_VER
                report["compiler"]="MSVC "+std::to_string(_MSC_FULL_VER);
#else
                report["compiler"]=__VERSION__;
#endif
            }
            report["ddgi_settings"] = renderer::ddgi_settings_json(settings.opengl.ddgi);
            if(!options.frame_report.parent_path().empty())std::filesystem::create_directories(options.frame_report.parent_path());
            std::ofstream output(options.frame_report);output<<report.dump(2);
            if(!output)throw std::runtime_error("Failed to write frame report");
            std::cout<<"frame_report count="<<times.size()<<" p50_ms="<<quantile(.5)<<" p95_ms="<<quantile(.95)<<" p99_ms="<<quantile(.99)<<'\n';
        }
        if (!options.capture_path.empty()) {
            renderer::Framebuffer frame(1,1);render_backend->readback(frame);
            const int capture_width=frame.width(),capture_height=frame.height();
            renderer::Image capture(capture_width,capture_height);
            for (int y=0;y<capture_height;++y) {
                for (int x=0;x<capture_width;++x) {
                    const renderer::Color color=frame.pixel(x,y);
                    if (!color.allFinite()) throw std::runtime_error("Capture contains non-finite pixels");
                    capture.set_pixel(x,y,renderer::apply_display_transform(color,ui_state.display));
                }
            }
            if (!capture.write_png(options.capture_path.string()))
                throw std::runtime_error("Failed to write capture: " + options.capture_path.string());
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
            std::cout << " ddgi=" << gl.ddgi.status << " probe_frames=" << gl.ddgi.frame_index
                      << " probes=" << gl.ddgi.active_probes << '/' << gl.ddgi.probe_count
                      << " updated=" << gl.ddgi.updated_probes << " probe_resets=" << gl.ddgi.reset_count
                      << " probe_trace_ms=" << gl.ddgi.trace_ms << " probe_blend_ms=" << gl.ddgi.blend_ms
                      << " probe_export_ms=" << gl.ddgi.export_ms << " probe_gather_ms=" << gl.ddgi.gather_ms
                      << " atlas_downloads=" << gl.ddgi.atlas_downloads;
            if (!gl.ddgi.detail.empty()) std::cout << " ddgi_detail=" << gl.ddgi.detail;
            if (!gl.shader_error.empty()) {
                std::cerr << "\nshader error: " << gl.shader_error;
            }
        } else if (ui_state.mode == renderer::InteractiveRenderMode::Dxr) {
            const auto& d=std::get<renderer::DxrStatistics>(final_statistics);
            std::cout<<" adapter="<<d.device.adapter<<" dxr="<<d.device.raytracing_tier<<" ser_active="<<d.ser_active<<" omm_active="<<d.omm_active
                <<" reconstruction="<<d.reconstruction<<" internal="<<d.internal_width<<'x'<<d.internal_height<<" gpu_ms="<<d.total_ms<<" readbacks="<<d.readbacks;
        } else if (ui_state.mode == renderer::InteractiveRenderMode::Rtrt) {
            const auto& path =
                std::get<renderer::CudaPathViewerStatistics>(final_statistics);
            std::cout << " interop="
                      << path.interop_status;
            const auto& rt = path.cuda.realtime;
            std::cout << " rt_frames=" << rt.frames << " history_resets=" << rt.history_resets
                      << " hardware_rt=" << rt.hardware_ray_tracing_active << " hardware_detail=" << rt.hardware_ray_tracing_detail
                      << " denoiser=" << (!settings.realtime.denoise?"off":(rt.optix_denoiser_active?"optix":"svgf"))
                      << " denoiser_detail=" << rt.optix_denoiser_detail
                      << " primary=optix" << " rt_core_version=" << rt.rt_core_version
                      << " ser_supported=" << rt.ser_supported << " ser_active=" << rt.ser_active
                      << " gas_builds=" << rt.gas_builds << " ias_builds=" << rt.ias_builds << " ias_updates=" << rt.ias_updates
                      << " acceleration_ms=" << rt.acceleration_ms << " acceleration_bytes=" << rt.hardware_ray_tracing_bytes
                      << " internal=" << path.cuda.internal_width << 'x' << path.cuda.internal_height
                      << " gbuffer_ms=" << rt.gbuffer_ms << " lighting_ms=" << rt.lighting_ms
                      << " temporal_ms=" << rt.temporal_ms << " filter_ms=" << rt.filter_ms
                      << " reconstruction_ms=" << rt.reconstruction_ms << " total_ms=" << rt.total_ms
                      << " framebuffer_mib=" << double(rt.framebuffer_bytes)/1048576;
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
