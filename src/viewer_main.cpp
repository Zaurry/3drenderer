#include "interactive/orbit_camera_controller.h"
#include "interactive/frame_rate_counter.h"
#include "platform/sdl/sdl_display_backend.h"
#include "render/framebuffer.h"
#include "render/interactive/path_interactive_session.h"
#include "render/interactive/raster_interactive_session.h"
#include "render/interactive/ray_interactive_session.h"
#include "scene/scene_asset_loader.h"
#include "scene/scene.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct ViewerOptions {
    std::string scene = "asset";
    std::string asset_path;
    renderer::InteractiveRenderMode mode = renderer::InteractiveRenderMode::Raster;
    int width = 960;
    int height = 540;
    int frame_limit = 0;
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
    const double radius = std::max(
        0.5, static_cast<double>(renderer::length(bounds.max - bounds.min)) * 0.5);
    return renderer::Camera(
        center + renderer::Vec3(
            0.0f,
            static_cast<float>(radius * 0.15),
            static_cast<float>(radius * 2.4)),
        center,
        renderer::Vec3(0.0f, 1.0f, 0.0f),
        45.0,
        static_cast<double>(width) / static_cast<double>(height));
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

}  // namespace

int main(int argc, char** argv) {
    try {
        ViewerOptions options = parse_args(argc, argv);
        ViewerScene viewer_scene = load_viewer_scene(options);

        renderer::SdlDisplayBackend display;
        if (!display.initialize(options.width, options.height, "CPU 3D Renderer Viewer")) {
            throw std::runtime_error(display.last_error());
        }

        renderer::Framebuffer framebuffer(options.width, options.height);
        renderer::RenderSettings settings;
        settings.width = options.width;
        settings.height = options.height;
        settings.samples_per_pixel = 1;
        settings.max_depth = 4;
        settings.thread_count = 1;

        renderer::OrbitCameraController camera_controller(viewer_scene.bounds, static_cast<double>(options.width) / options.height);
        std::unique_ptr<renderer::InteractiveRenderSession> session = make_session(options.mode);
        session->reset(viewer_scene.scene, settings);
        renderer::FrameRateCounter frame_rate_counter;
        display.set_title(renderer::format_viewer_title(options.mode, frame_rate_counter.snapshot(), 0));

        int rendered_frames = 0;
        bool running = true;
        auto previous_time = std::chrono::steady_clock::now();
        while (running) {
            auto now = std::chrono::steady_clock::now();
            const double delta_seconds = std::chrono::duration<double>(now - previous_time).count();
            previous_time = now;

            const renderer::InputState input = display.poll_input();
            running = !input.quit_requested;

            bool mode_changed = false;
            if (input.select_raster) {
                options.mode = renderer::InteractiveRenderMode::Raster;
                mode_changed = true;
            } else if (input.select_ray) {
                options.mode = renderer::InteractiveRenderMode::Ray;
                mode_changed = true;
            } else if (input.select_path) {
                options.mode = renderer::InteractiveRenderMode::Path;
                mode_changed = true;
            }
            if (mode_changed) {
                session = make_session(options.mode);
                session->reset(viewer_scene.scene, settings);
                frame_rate_counter.reset();
                std::cout << "mode=" << mode_name(options.mode) << "\n";
            }

            renderer::InteractiveFrameState frame_state;
            frame_state.delta_seconds = delta_seconds;
            frame_state.scene_changed = mode_changed;
            frame_state.reset_requested = input.reset_render;

            if (input.window_resized) {
                settings.width = input.window_width;
                settings.height = input.window_height;
                framebuffer.resize(settings.width, settings.height);
                camera_controller.set_aspect_ratio(static_cast<double>(settings.width) / settings.height);
                frame_state.framebuffer_resized = true;
            }
            if (input.left_mouse_down && (input.mouse_delta_x != 0.0 || input.mouse_delta_y != 0.0)) {
                camera_controller.orbit(input.mouse_delta_x, input.mouse_delta_y);
                frame_state.camera_changed = true;
            }
            if (input.wheel_delta != 0.0) {
                camera_controller.zoom(input.wheel_delta);
                frame_state.camera_changed = true;
            }

            const renderer::Camera camera = camera_controller.camera();
            session->render_next_frame(viewer_scene.scene, camera, settings, frame_state, framebuffer);
            display.present(framebuffer);

            ++rendered_frames;
            int accumulated_path_samples = 0;
            if (const auto* path_session = dynamic_cast<const renderer::PathInteractiveSession*>(session.get())) {
                accumulated_path_samples = path_session->accumulated_samples();
            }
            const auto frame_end = std::chrono::steady_clock::now();
            const double frame_seconds = std::chrono::duration<double>(frame_end - now).count();
            if (frame_rate_counter.tick(frame_seconds) || mode_changed || rendered_frames == 1) {
                display.set_title(renderer::format_viewer_title(
                    options.mode,
                    frame_rate_counter.snapshot(),
                    accumulated_path_samples));
            }
            if (options.frame_limit > 0 && rendered_frames >= options.frame_limit) {
                running = false;
            }
        }

        std::cout << "viewer mode=" << mode_name(options.mode)
                  << " frames=" << rendered_frames
                  << " size=" << settings.width << "x" << settings.height << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n\n";
        print_help();
        return 1;
    }
}
