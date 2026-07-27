#include "interactive/frame_rate_counter.h"
#include "interactive/free_camera_controller.h"
#include "interactive/orbit_camera_controller.h"
#include "interactive/viewer_session.h"
#include "interactive/viewer_ui.h"
#include "platform/opengl/cuda_opengl_interop.h"
#include "platform/sdl/sdl_display_backend.h"
#include "render/framebuffer.h"
#include "render/interactive/path_interactive_session.h"
#include "render/interactive/raster_interactive_session.h"
#include "render/interactive/ray_interactive_session.h"
#include "render/opengl/opengl_raster_renderer.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "render/pathtracer/path_backend.h"
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
    renderer::InteractiveRenderMode mode = renderer::InteractiveRenderMode::Raster;
    int width = 960;
    int height = 540;
    int frame_limit = 0;
    renderer::PathBackend path_backend = renderer::PathBackend::Auto;
    std::filesystem::path gl_vertex_shader;
    std::filesystem::path gl_fragment_shader;
    bool restore_last_session = true;
};

struct ViewerScene {
    renderer::SceneDocument document;
    renderer::Scene scene;
    renderer::Bounds3 bounds;
    renderer::Camera camera;
};

void print_help() {
    std::cout
        << "CPU 3D Renderer viewer\n\n"
        << "Usage:\n"
        << "  viewer --scene builtin|asset [--asset path\\to\\model-or-directory ...] --mode raster|ray|path|opengl [options]\n\n"
        << "Options:\n"
        << "  --asset path      OBJ file or directory; may be repeated\n"
        << "  --scene-file path open a saved .rscene document\n"
        << "  --width integer    window width, default 960\n"
        << "  --height integer   window height, default 540\n"
        << "  --frames integer   render N frames then exit, default unlimited\n"
        << "  --path-backend auto|cpu|cuda  path execution backend, default auto\n"
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

renderer::InteractiveRenderMode parse_mode(const std::string& value) {
    if (value == "raster") {
        return renderer::InteractiveRenderMode::Raster;
    }
    if (value == "ray") {
        return renderer::InteractiveRenderMode::Ray;
    }
    if (value == "path") {
        return renderer::InteractiveRenderMode::Path;
    }
    if (value == "opengl" || value == "gl") {
        return renderer::InteractiveRenderMode::OpenGl;
    }
    throw std::invalid_argument("unknown mode: " + value);
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
        } else if (arg == "--path-backend") {
            options.path_backend = renderer::parse_path_backend(require_value(argc, argv, i, arg));
        } else if (arg == "--gl-vertex-shader") {
            options.gl_vertex_shader = require_value(argc, argv, i, arg);
        } else if (arg == "--gl-fragment-shader") {
            options.gl_fragment_shader = require_value(argc, argv, i, arg);
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
        for (const std::string& warning : document.warnings()) {
            std::cerr << "warning: " << warning << '\n';
        }
        const renderer::Bounds3 bounds = document.scene_bounds();
        renderer::Scene scene = document.render_scene();
        return ViewerScene{
            std::move(document),
            std::move(scene),
            bounds,
            make_camera_from_bounds(bounds, options.width, options.height)};
    }

    renderer::Scene scene = renderer::make_cornell_box_scene();
    renderer::Bounds3 bounds = scene_bounds(scene);
    renderer::SceneDocument document =
        renderer::SceneDocument::from_scene(
            scene,
            "Builtin Cornell Box",
            "cornell_box");
    return ViewerScene{
        std::move(document),
        std::move(scene),
        bounds,
        make_camera_from_bounds(bounds, options.width, options.height)};
}

std::unique_ptr<renderer::InteractiveRenderSession> make_session(renderer::InteractiveRenderMode mode) {
    if (mode == renderer::InteractiveRenderMode::Raster) {
        return std::make_unique<renderer::RasterInteractiveSession>();
    }
    if (mode == renderer::InteractiveRenderMode::Ray) {
        return std::make_unique<renderer::RayInteractiveSession>();
    }
    if (mode == renderer::InteractiveRenderMode::Path) {
        return std::make_unique<renderer::PathInteractiveSession>();
    }
    return nullptr;
}

const char* mode_name(renderer::InteractiveRenderMode mode) {
    if (mode == renderer::InteractiveRenderMode::Raster) {
        return "raster";
    }
    if (mode == renderer::InteractiveRenderMode::Ray) {
        return "ray";
    }
    if (mode == renderer::InteractiveRenderMode::Path) {
        return "path";
    }
    return "opengl";
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

const char* interop_state_name(renderer::CudaOpenGlInteropState state) {
    if (state == renderer::CudaOpenGlInteropState::Ready) {
        return "ready";
    }
    if (state == renderer::CudaOpenGlInteropState::Active) {
        return "active";
    }
    if (state == renderer::CudaOpenGlInteropState::Fallback) {
        return "fallback";
    }
    return "unavailable";
}

int scaled_dimension(int window_dimension, float render_scale) {
    return std::max(1, static_cast<int>(std::lround(
        static_cast<float>(window_dimension) * std::clamp(render_scale, 0.25f, 1.0f))));
}

int accumulated_samples(const renderer::InteractiveRenderSession* session) {
    const auto* path_session = dynamic_cast<const renderer::PathInteractiveSession*>(session);
    return path_session ? path_session->accumulated_samples() : 0;
}

renderer::ExecutionBackend active_path_backend(const renderer::InteractiveRenderSession* session) {
    const auto* path_session = dynamic_cast<const renderer::PathInteractiveSession*>(session);
    return path_session ? path_session->active_backend() : renderer::ExecutionBackend::Cpu;
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
        renderer::InteractiveRenderMode::Raster;
    renderer::ViewerCameraMode camera_mode =
        renderer::ViewerCameraMode::Orbit;
    renderer::PathBackend path_backend = renderer::PathBackend::Auto;
    renderer::ToneMapper tone_mapper = renderer::ToneMapper::None;
    int max_depth = 0;
    int tile_size = 0;
    int thread_count = 0;
    int logical_width = 0;
    int logical_height = 0;
    float render_scale = 1.0f;
    float ui_font_scale = 1.0f;
    float exposure_ev = 0.0f;
    bool path_accumulation_paused = false;
    bool show_point_light_markers = true;
    bool panel_visible = true;
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
    signature.path_backend = settings.path_backend;
    signature.tone_mapper = ui.display.tone_mapper;
    signature.max_depth = settings.max_depth;
    signature.tile_size = settings.tile_size;
    signature.thread_count = settings.thread_count;
    signature.logical_width = logical_size.first;
    signature.logical_height = logical_size.second;
    signature.render_scale = ui.render_scale;
    signature.ui_font_scale = ui.ui_font_scale;
    signature.exposure_ev = ui.display.exposure_ev;
    signature.path_accumulation_paused = ui.path_accumulation_paused;
    signature.show_point_light_markers = ui.show_point_light_markers;
    signature.panel_visible = ui.panel_visible;
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
                    options.path_backend =
                        restored_session->render_settings.path_backend;
                    startup_status = "Restored last viewer session";
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
            renderer::Scene scene = document.render_scene();
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
                std::move(scene),
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
        renderer::CudaOpenGlInteropTexture cuda_gl_interop;

        renderer::ViewerUiState ui_state = restored_session
            ? restored_session->ui
            : renderer::ViewerUiState{};
        if (!restored_session) {
            ui_state.mode = options.mode;
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
        settings.samples_per_pixel = 1;
        if (!restored_session) {
            settings.max_depth = 4;
            settings.thread_count = 1;
            settings.path_backend = options.path_backend;
        } else if (settings.path_backend == renderer::PathBackend::Cuda) {
            std::string reason;
            if (!renderer::cuda_path_backend_available(&reason)) {
                settings.path_backend = renderer::PathBackend::Auto;
                ui_state.scene_status =
                    "Saved CUDA backend is unavailable; using automatic fallback";
                std::cerr << "warning: " << ui_state.scene_status
                          << ": " << reason << '\n';
            }
        }
        bool cuda_gl_interop_initialized = false;
        const auto initialize_cuda_gl_interop = [&]() {
            if (!cuda_gl_interop_initialized &&
                settings.path_backend != renderer::PathBackend::Cpu) {
                cuda_gl_interop.initialize();
                cuda_gl_interop_initialized = true;
            }
        };
        if (ui_state.mode == renderer::InteractiveRenderMode::Path &&
            settings.path_backend == renderer::PathBackend::Auto) {
            initialize_cuda_gl_interop();
            std::string reason;
            if (!renderer::cuda_path_backend_available(&reason)) {
                std::cerr << "warning: CUDA path backend unavailable, using CPU: "
                          << reason << '\n';
            }
        }

        renderer::Framebuffer framebuffer(settings.width, settings.height);
        const float camera_aspect_ratio =
            static_cast<float>(window_width) / static_cast<float>(window_height);
        const float scene_radius = std::max(
            0.5f,
            (viewer_scene.bounds.max - viewer_scene.bounds.min).norm() * 0.5f);
        renderer::OrbitCameraController orbit_camera(viewer_scene.bounds, camera_aspect_ratio);
        renderer::FreeCameraController free_camera(
            orbit_camera.camera(),
            camera_aspect_ratio,
            scene_radius);
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

        std::unique_ptr<renderer::InteractiveRenderSession> session;
        std::unique_ptr<renderer::OpenGlRasterRenderer> opengl_renderer;
        const auto ensure_opengl_renderer = [&]() -> renderer::OpenGlRasterRenderer& {
            if (!opengl_renderer) {
                opengl_renderer = std::make_unique<renderer::OpenGlRasterRenderer>(
                    options.gl_vertex_shader,
                    options.gl_fragment_shader);
                opengl_renderer->reset(viewer_scene.scene);
            }
            return *opengl_renderer;
        };
        if (ui_state.mode == renderer::InteractiveRenderMode::OpenGl) {
            ensure_opengl_renderer();
        } else {
            if (ui_state.mode == renderer::InteractiveRenderMode::Path) {
                initialize_cuda_gl_interop();
            }
            session = make_session(ui_state.mode);
            session->reset(viewer_scene.scene, settings);
        }
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
                title += std::string(" - ") +
                    renderer::execution_backend_name(active_path_backend(session.get()));
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
        bool path_presented_from_interop = false;
        std::string reported_interop_reason;
        bool running = true;
        auto previous_time = std::chrono::steady_clock::now();
        while (running) {
            const auto frame_begin = std::chrono::steady_clock::now();
            const float delta_seconds =
                std::chrono::duration<float>(frame_begin - previous_time).count();
            previous_time = frame_begin;

            const renderer::InputState input = display.poll_input();
            running = !input.quit_requested;
            if (input.toggle_ui) {
                ui_state.panel_visible = !ui_state.panel_visible;
            }
            bool external_scene_changed = false;
            const auto import_asset_path = [&](const std::filesystem::path& path) {
                try {
                    viewer_scene.document.import_path(
                        path,
                        settings.width,
                        settings.height);
                    viewer_scene.scene = viewer_scene.document.render_scene();
                    viewer_scene.bounds = viewer_scene.document.scene_bounds();
                    ui_state.scene_status = "Imported " + path.filename().string();
                    external_scene_changed = true;
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
                    viewer_scene.scene = viewer_scene.document.render_scene();
                    viewer_scene.bounds = viewer_scene.document.scene_bounds();
                    ui_state.selected_objects.clear();
                    ui_state.active_object = renderer::kInvalidObjectId;
                    reset_cameras_for_scene();
                    ui_state.scene_status = "Opened " + path.filename().string();
                    external_scene_changed = true;
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
            for (const std::string& dropped : input.dropped_paths) {
                const std::filesystem::path path(dropped);
                if (has_extension(path, ".rscene")) {
                    open_scene_path(path);
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
                    } else {
                        import_asset_path(selected_path);
                    }
                }
            }
            display.begin_ui_frame();

            const renderer::InteractiveRenderMode previous_mode = ui_state.mode;
            const renderer::ViewerCameraMode previous_camera_mode = ui_state.camera_mode;
            if (opengl_renderer) {
                shader_ui_state.auto_reload = opengl_renderer->auto_reload();
                shader_ui_state.valid = opengl_renderer->has_valid_shader();
                shader_ui_state.error = opengl_renderer->shader_error();
            }
            interop_ui_state.status = interop_state_name(cuda_gl_interop.state());
            interop_ui_state.detail = cuda_gl_interop.reason();
            renderer::ViewerUiActions ui_actions = viewer_ui.draw(
                ui_state,
                settings,
                viewer_scene.document,
                orbit_camera,
                free_camera,
                viewer_scene.bounds,
                frame_rate_counter.snapshot(),
                accumulated_samples(session.get()),
                active_path_backend(session.get()),
                interop_ui_state,
                shader_ui_state);

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

            bool document_scene_changed =
                external_scene_changed || ui_actions.scene_changed;
            if (document_scene_changed) {
                viewer_scene.document.rebuild_render_scene();
                viewer_scene.scene = viewer_scene.document.render_scene();
                viewer_scene.bounds = viewer_scene.document.scene_bounds();
            }

            const bool keyboard_available = !display.wants_keyboard_capture();
            const bool mouse_available = !display.wants_mouse_capture();
            if (keyboard_available) {
                renderer::InteractiveRenderMode hotkey_mode = ui_state.mode;
                if (input.select_raster) {
                    hotkey_mode = renderer::InteractiveRenderMode::Raster;
                } else if (input.select_ray) {
                    hotkey_mode = renderer::InteractiveRenderMode::Ray;
                } else if (input.select_path) {
                    hotkey_mode = renderer::InteractiveRenderMode::Path;
                } else if (input.select_opengl) {
                    hotkey_mode = renderer::InteractiveRenderMode::OpenGl;
                }
                if (hotkey_mode != ui_state.mode) {
                    ui_state.mode = hotkey_mode;
                    ui_actions.mode_changed = true;
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
                if (ui_state.mode == renderer::InteractiveRenderMode::OpenGl) {
                    session.reset();
                    ensure_opengl_renderer();
                } else {
                    if (ui_state.mode == renderer::InteractiveRenderMode::Path) {
                        initialize_cuda_gl_interop();
                    }
                    session = make_session(ui_state.mode);
                    session->reset(viewer_scene.scene, settings);
                }
                if (ui_state.mode != renderer::InteractiveRenderMode::Path) {
                    path_presented_from_interop = false;
                    cuda_gl_interop.release_texture();
                }
                frame_rate_counter.reset();
                std::cout << "mode=" << mode_name(ui_state.mode) << '\n';
            } else if (ui_actions.path_backend_changed &&
                       ui_state.mode == renderer::InteractiveRenderMode::Path) {
                initialize_cuda_gl_interop();
                session->reset(viewer_scene.scene, settings);
                path_presented_from_interop = false;
                if (active_path_backend(session.get()) != renderer::ExecutionBackend::Cuda) {
                    cuda_gl_interop.release_texture();
                }
            }
            if (ui_state.mode == renderer::InteractiveRenderMode::OpenGl) {
                renderer::OpenGlRasterRenderer& active_opengl = ensure_opengl_renderer();
                if (ui_actions.shader_auto_reload_changed) {
                    active_opengl.set_auto_reload(shader_ui_state.auto_reload);
                }
                if (ui_actions.shader_reload_requested) {
                    active_opengl.request_shader_reload();
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

            renderer::InteractiveFrameState frame_state;
            frame_state.delta_seconds = delta_seconds;
            frame_state.lighting_changed = ui_actions.lighting_changed;
            frame_state.reset_requested = ui_actions.reset_requested;

            if (input.window_resized || ui_actions.render_scale_changed) {
                settings.width = scaled_dimension(window_width, ui_state.render_scale);
                settings.height = scaled_dimension(window_height, ui_state.render_scale);
                framebuffer.resize(settings.width, settings.height);
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
                throw std::runtime_error(display.last_error());
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
                if (!additive) {
                    ui_state.selected_objects.clear();
                }
                if (picked) {
                    const auto found = std::find(
                        ui_state.selected_objects.begin(),
                        ui_state.selected_objects.end(),
                        picked->object_id);
                    if (additive && found != ui_state.selected_objects.end()) {
                        ui_state.selected_objects.erase(found);
                    } else if (found == ui_state.selected_objects.end()) {
                        ui_state.selected_objects.push_back(picked->object_id);
                    }
                    ui_state.active_object = picked->object_id;
                } else if (!additive) {
                    ui_state.active_object = renderer::kInvalidObjectId;
                }
            }
            if (viewer_ui.draw_scene_gizmo(
                    ui_state,
                    viewer_scene.document,
                    camera,
                    viewer_scene.bounds)) {
                viewer_scene.scene = viewer_scene.document.render_scene();
                viewer_scene.bounds = viewer_scene.document.scene_bounds();
                document_scene_changed = true;
            }
            frame_state.scene_changed = document_scene_changed;
            if (document_scene_changed && opengl_renderer) {
                opengl_renderer->reset(viewer_scene.scene);
            }
            viewer_ui.draw_scene_selection(
                ui_state,
                viewer_scene.document,
                camera);
            viewer_ui.draw_point_light_markers(
                ui_state,
                viewer_scene.scene,
                camera);
            const bool path_needs_preview =
                rendered_frames == 0 ||
                mode_changed ||
                ui_actions.path_backend_changed ||
                frame_state.camera_changed ||
                frame_state.lighting_changed ||
                frame_state.scene_changed ||
                frame_state.framebuffer_resized ||
                frame_state.reset_requested;
            const bool should_render =
                ui_state.mode != renderer::InteractiveRenderMode::Path ||
                !ui_state.path_accumulation_paused ||
                path_needs_preview;
            if (should_render) {
                if (ui_state.mode == renderer::InteractiveRenderMode::OpenGl) {
                    ensure_opengl_renderer().render(
                        viewer_scene.scene,
                        camera,
                        settings,
                        frame_state);
                } else if (
                    ui_state.mode == renderer::InteractiveRenderMode::Path &&
                    active_path_backend(session.get()) == renderer::ExecutionBackend::Cuda &&
                    cuda_gl_interop.state() != renderer::CudaOpenGlInteropState::Fallback &&
                    cuda_gl_interop.state() != renderer::CudaOpenGlInteropState::Unavailable) {
                    auto* path_session = dynamic_cast<renderer::PathInteractiveSession*>(session.get());
                    if (!path_session) {
                        throw std::logic_error("CUDA/OpenGL interop requires a path session");
                    }
                    renderer::CudaSurfaceHandle surface = 0;
                    if (cuda_gl_interop.begin_frame(settings.width, settings.height, surface)) {
                        try {
                            path_session->render_next_frame_to_cuda_surface(
                                viewer_scene.scene,
                                camera,
                                settings,
                                frame_state,
                                surface);
                        } catch (...) {
                            cuda_gl_interop.cancel_frame();
                            throw;
                        }
                        if (cuda_gl_interop.end_frame()) {
                            path_presented_from_interop = true;
                        } else {
                            path_session->download_current_cuda_frame(framebuffer);
                            path_presented_from_interop = false;
                        }
                    } else {
                        session->render_next_frame(
                            viewer_scene.scene,
                            camera,
                            settings,
                            frame_state,
                            framebuffer);
                        path_presented_from_interop = false;
                    }
                } else {
                    session->render_next_frame(
                        viewer_scene.scene,
                        camera,
                        settings,
                        frame_state,
                        framebuffer);
                    if (ui_state.mode == renderer::InteractiveRenderMode::Path) {
                        path_presented_from_interop = false;
                    }
                }
            }
            if (ui_state.mode == renderer::InteractiveRenderMode::Path &&
                active_path_backend(session.get()) == renderer::ExecutionBackend::Cuda &&
                cuda_gl_interop.state() == renderer::CudaOpenGlInteropState::Fallback &&
                !cuda_gl_interop.reason().empty() &&
                cuda_gl_interop.reason() != reported_interop_reason) {
                reported_interop_reason = cuda_gl_interop.reason();
                std::cerr << "warning: CUDA/OpenGL interop unavailable, using CPU staging: "
                          << reported_interop_reason << '\n';
            }
            if (ui_state.mode == renderer::InteractiveRenderMode::OpenGl) {
                const renderer::OpenGlRasterRenderer& active_opengl = ensure_opengl_renderer();
                display.present_texture(
                    active_opengl.output_texture(),
                    active_opengl.output_width(),
                    active_opengl.output_height(),
                    false,
                    ui_state.display);
            } else if (
                ui_state.mode == renderer::InteractiveRenderMode::Path &&
                path_presented_from_interop &&
                cuda_gl_interop.texture() != 0) {
                display.present_texture(
                    cuda_gl_interop.texture(),
                    cuda_gl_interop.width(),
                    cuda_gl_interop.height(),
                    true,
                    ui_state.display);
            } else {
                display.present(framebuffer, ui_state.display);
            }

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
            const int path_samples = accumulated_samples(session.get());
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
        if (ui_state.mode == renderer::InteractiveRenderMode::OpenGl && opengl_renderer) {
            std::cout << " shader="
                      << (opengl_renderer->has_valid_shader() ? "active" : "invalid");
            if (!opengl_renderer->shader_error().empty()) {
                std::cerr << "\nshader error: " << opengl_renderer->shader_error();
            }
        } else if (
            ui_state.mode == renderer::InteractiveRenderMode::Path &&
            active_path_backend(session.get()) == renderer::ExecutionBackend::Cuda) {
            std::cout << " interop="
                      << (path_presented_from_interop ? "active" : "fallback");
        }
        std::cout << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n\n";
        print_help();
        return 1;
    }
}
