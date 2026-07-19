#include "interactive/frame_rate_counter.h"
#include "interactive/free_camera_controller.h"
#include "interactive/orbit_camera_controller.h"
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct ViewerOptions {
    std::string scene = "asset";
    std::string asset_path;
    renderer::InteractiveRenderMode mode = renderer::InteractiveRenderMode::Raster;
    int width = 960;
    int height = 540;
    int frame_limit = 0;
    renderer::PathBackend path_backend = renderer::PathBackend::Auto;
    std::filesystem::path gl_vertex_shader;
    std::filesystem::path gl_fragment_shader;
};

struct ViewerScene {
    renderer::Scene scene;
    renderer::Bounds3 bounds;
    renderer::Camera camera;
};

void print_help() {
    std::cout
        << "CPU 3D Renderer viewer\n\n"
        << "Usage:\n"
        << "  viewer --scene builtin|asset --asset path\\to\\scene.obj --mode raster|ray|path|opengl [options]\n\n"
        << "Options:\n"
        << "  --width integer    window width, default 960\n"
        << "  --height integer   window height, default 540\n"
        << "  --frames integer   render N frames then exit, default unlimited\n"
        << "  --path-backend auto|cpu|cuda  path execution backend, default auto\n"
        << "  --gl-vertex-shader path    OpenGL vertex shader override\n"
        << "  --gl-fragment-shader path  OpenGL fragment shader override\n"
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
            options.asset_path = require_value(argc, argv, i, arg);
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
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.scene != "builtin" && options.scene != "asset") {
        throw std::invalid_argument("--scene must be builtin or asset");
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

ViewerScene load_viewer_scene(const ViewerOptions& options) {
    const std::filesystem::path default_asset =
        std::filesystem::path("Computer Graphics Archive") /
        "CornellBox" /
        "CornellBox-Original.obj";

    if (options.scene == "asset") {
        const std::string asset_path = options.asset_path.empty()
            ? default_asset.string()
            : options.asset_path;
        if (std::filesystem::exists(asset_path)) {
            renderer::LoadedScene loaded = renderer::load_scene_asset(asset_path, options.width, options.height);
            for (const std::string& warning : loaded.warnings) {
                std::cerr << "warning: " << warning << '\n';
            }
            return ViewerScene{loaded.scene, loaded.bounds, loaded.camera};
        }
        if (!options.asset_path.empty()) {
            throw std::runtime_error("asset path does not exist: " + asset_path);
        }
    }

    renderer::Scene scene = renderer::make_cornell_box_scene();
    renderer::Bounds3 bounds = scene_bounds(scene);
    return ViewerScene{scene, bounds, make_camera_from_bounds(bounds, options.width, options.height)};
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

}  // namespace

int main(int argc, char** argv) {
    try {
        ViewerOptions options = parse_args(argc, argv);
        const std::filesystem::path executable_path = std::filesystem::absolute(argv[0]);
        options.gl_vertex_shader = resolve_shader_path(
            options.gl_vertex_shader,
            executable_path,
            std::filesystem::path("shaders") / "opengl" / "raster.vert");
        options.gl_fragment_shader = resolve_shader_path(
            options.gl_fragment_shader,
            executable_path,
            std::filesystem::path("shaders") / "opengl" / "raster.frag");
        ViewerScene viewer_scene = load_viewer_scene(options);

        renderer::SdlDisplayBackend display;
        if (!display.initialize(options.width, options.height, "3D Renderer Viewer")) {
            throw std::runtime_error(display.last_error());
        }
        renderer::CudaOpenGlInteropTexture cuda_gl_interop;

        renderer::ViewerUiState ui_state;
        ui_state.mode = options.mode;
        renderer::ViewerUi viewer_ui;

        int window_width = options.width;
        int window_height = options.height;
        renderer::RenderSettings settings;
        settings.width = options.width;
        settings.height = options.height;
        settings.samples_per_pixel = 1;
        settings.max_depth = 4;
        settings.thread_count = 1;
        settings.path_backend = options.path_backend;
        bool cuda_gl_interop_initialized = false;
        const auto initialize_cuda_gl_interop = [&]() {
            if (!cuda_gl_interop_initialized &&
                settings.path_backend != renderer::PathBackend::Cpu) {
                cuda_gl_interop.initialize();
                cuda_gl_interop_initialized = true;
            }
        };
        if (ui_state.mode == renderer::InteractiveRenderMode::Path &&
            options.path_backend == renderer::PathBackend::Auto) {
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
        const renderer::OrbitCameraController initial_orbit_camera = orbit_camera;
        const renderer::FreeCameraController initial_free_camera = free_camera;

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
                viewer_scene.scene,
                orbit_camera,
                free_camera,
                viewer_scene.bounds,
                frame_rate_counter.snapshot(),
                accumulated_samples(session.get()),
                active_path_backend(session.get()),
                interop_ui_state,
                shader_ui_state);

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
                if (input.left_mouse_down && mouse_moved) {
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
            const bool path_needs_preview =
                mode_changed ||
                ui_actions.path_backend_changed ||
                frame_state.camera_changed ||
                frame_state.lighting_changed ||
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
