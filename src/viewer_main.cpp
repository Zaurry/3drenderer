#include "interactive/frame_rate_counter.h"
#include "interactive/free_camera_controller.h"
#include "interactive/orbit_camera_controller.h"
#include "interactive/viewer_ui.h"
#include "platform/sdl/sdl_display_backend.h"
#include "render/framebuffer.h"
#include "render/interactive/path_interactive_session.h"
#include "render/interactive/raster_interactive_session.h"
#include "render/interactive/ray_interactive_session.h"
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
        << "  viewer --scene builtin|asset --asset path\\to\\scene.obj --mode raster|ray|path [options]\n\n"
        << "Options:\n"
        << "  --width integer    window width, default 960\n"
        << "  --height integer   window height, default 540\n"
        << "  --frames integer   render N frames then exit, default unlimited\n"
        << "  --path-backend auto|cpu|cuda  path execution backend, default auto\n"
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
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (options.scene != "builtin" && options.scene != "asset") {
        throw std::invalid_argument("--scene must be builtin or asset");
    }
    return options;
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
    return std::make_unique<renderer::PathInteractiveSession>();
}

const char* mode_name(renderer::InteractiveRenderMode mode) {
    if (mode == renderer::InteractiveRenderMode::Raster) {
        return "raster";
    }
    if (mode == renderer::InteractiveRenderMode::Ray) {
        return "ray";
    }
    return "path";
}

const char* camera_mode_name(renderer::ViewerCameraMode mode) {
    return mode == renderer::ViewerCameraMode::Orbit ? "orbit" : "free";
}

int scaled_dimension(int window_dimension, float render_scale) {
    return std::max(1, static_cast<int>(std::lround(
        static_cast<float>(window_dimension) * std::clamp(render_scale, 0.25f, 1.0f))));
}

int accumulated_samples(const renderer::InteractiveRenderSession& session) {
    const auto* path_session = dynamic_cast<const renderer::PathInteractiveSession*>(&session);
    return path_session ? path_session->accumulated_samples() : 0;
}

renderer::ExecutionBackend active_path_backend(const renderer::InteractiveRenderSession& session) {
    const auto* path_session = dynamic_cast<const renderer::PathInteractiveSession*>(&session);
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
        const ViewerOptions options = parse_args(argc, argv);
        ViewerScene viewer_scene = load_viewer_scene(options);

        renderer::SdlDisplayBackend display;
        if (!display.initialize(options.width, options.height, "CPU 3D Renderer Viewer")) {
            throw std::runtime_error(display.last_error());
        }

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
        if (ui_state.mode == renderer::InteractiveRenderMode::Path &&
            options.path_backend == renderer::PathBackend::Auto) {
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

        std::unique_ptr<renderer::InteractiveRenderSession> session = make_session(ui_state.mode);
        session->reset(viewer_scene.scene, settings);
        renderer::FrameRateCounter frame_rate_counter;
        const auto set_viewer_title = [&](int path_samples) {
            std::string title = renderer::format_viewer_title(
                ui_state.mode,
                frame_rate_counter.snapshot(),
                path_samples);
            if (ui_state.mode == renderer::InteractiveRenderMode::Path) {
                title += std::string(" - ") +
                    renderer::execution_backend_name(active_path_backend(*session));
            }
            title += std::string(" - camera=") + camera_mode_name(ui_state.camera_mode);
            display.set_title(title);
        };
        set_viewer_title(0);

        int rendered_frames = 0;
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
            renderer::ViewerUiActions ui_actions = viewer_ui.draw(
                ui_state,
                settings,
                viewer_scene.scene,
                orbit_camera,
                free_camera,
                viewer_scene.bounds,
                frame_rate_counter.snapshot(),
                accumulated_samples(*session),
                active_path_backend(*session));

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
            }

            if (input.window_resized) {
                window_width = input.window_width;
                window_height = input.window_height;
            }

            const bool mode_changed = ui_actions.mode_changed || previous_mode != ui_state.mode;
            if (mode_changed) {
                session = make_session(ui_state.mode);
                session->reset(viewer_scene.scene, settings);
                frame_rate_counter.reset();
                std::cout << "mode=" << mode_name(ui_state.mode) << '\n';
            } else if (ui_actions.path_backend_changed &&
                       ui_state.mode == renderer::InteractiveRenderMode::Path) {
                session->reset(viewer_scene.scene, settings);
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
                session->render_next_frame(
                    viewer_scene.scene,
                    camera,
                    settings,
                    frame_state,
                    framebuffer);
            }
            display.present(framebuffer, ui_state.display);

            ++rendered_frames;
            const int path_samples = accumulated_samples(*session);
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
                  << " size=" << settings.width << "x" << settings.height << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n\n";
        print_help();
        return 1;
    }
}
