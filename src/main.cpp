#include "render/pathtracer/pathtracer_renderer.h"
#include "render/pathtracer/path_backend.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "scene/scene_asset_loader.h"
#include "scene/scene.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct SceneBundle {
    renderer::Scene scene;
    renderer::Camera camera;
};

struct CliOptions {
    std::string mode = "path";
    std::string scene = "gradient_sphere";
    std::string asset_path;
    std::filesystem::path environment_path;
    float environment_intensity = 1.0f;
    float environment_yaw_degrees = 0.0f;
    bool environment_background_visible = true;
    std::string output_path;
    int width = 512;
    int height = 512;
    int samples_per_pixel = 1;
    int thread_count = 0;
    renderer::PathBackend path_backend = renderer::PathBackend::Auto;
    bool help = false;
};

void print_help() {
    std::cout
        << "3D Path Renderer v0.1\n"
        << "\n"
        << "Usage:\n"
        << "  renderer --mode path --scene gradient_sphere|triangle|mirror_spheres|cornell_box|asset_viewer --output file.png [options]\n"
        << "\n"
        << "Options:\n"
        << "  --asset path        OBJ/glTF/GLB file for --scene asset_viewer\n"
        << "  --obj path          backward-compatible alias for --asset\n"
        << "  --environment path  2:1 HDR/EXR/PNG/JPG environment map\n"
        << "  --environment-intensity value  environment multiplier, default 1\n"
        << "  --environment-yaw degrees  rotate the environment around Y\n"
        << "  --hide-environment-background  light without drawing the environment\n"
        << "  --width integer     image width, default 512\n"
        << "  --height integer    image height, default 512\n"
        << "  --spp integer       samples per pixel, default 1\n"
        << "  --threads integer   path tracer worker threads, default hardware threads\n"
        << "  --path-backend auto|cpu|cuda  path execution backend, default auto\n"
        << "  --help              show this help\n";
}

int parse_positive_int(const std::string& value, const std::string& name) {
    std::size_t consumed = 0;
    int parsed = 0;
    try {
        parsed = std::stoi(value, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(name + " must be an integer");
    }
    if (consumed != value.size() || parsed <= 0) {
        throw std::invalid_argument(name + " must be a positive integer");
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

std::string require_value(int argc, char** argv, int& i, const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(flag + " requires a value");
    }
    ++i;
    return argv[i];
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            options.help = true;
        } else if (arg == "--mode") {
            options.mode = require_value(argc, argv, i, arg);
        } else if (arg == "--scene") {
            options.scene = require_value(argc, argv, i, arg);
        } else if (arg == "--obj" || arg == "--asset") {
            options.asset_path = require_value(argc, argv, i, arg);
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
        } else if (arg == "--width") {
            options.width = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--height") {
            options.height = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--spp") {
            options.samples_per_pixel = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--threads") {
            options.thread_count = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--path-backend") {
            options.path_backend = renderer::parse_path_backend(require_value(argc, argv, i, arg));
        } else if (arg == "--output") {
            options.output_path = require_value(argc, argv, i, arg);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (!options.help && options.output_path.empty()) {
        throw std::invalid_argument("--output is required");
    }
    if (!options.help && options.mode != "path") {
        throw std::invalid_argument(
            "unknown mode: " + options.mode + " (only path is supported)");
    }
    return options;
}

renderer::Camera make_camera(
    const renderer::Vec3& eye,
    const renderer::Vec3& target,
    float vertical_fov_degrees,
    int width,
    int height) {
    return renderer::Camera(
        eye,
        target,
        renderer::Vec3(0.0f, 1.0f, 0.0f),
        vertical_fov_degrees,
        static_cast<float>(width) / static_cast<float>(height));
}

SceneBundle make_asset_scene(const CliOptions& options) {
    if (options.asset_path.empty()) {
        throw std::invalid_argument("--scene asset_viewer requires --asset");
    }

    const renderer::LoadedScene loaded = renderer::load_scene_asset(
        options.asset_path,
        options.width,
        options.height);
    for (const std::string& warning : loaded.warnings) {
        std::cerr << "warning: " << warning << '\n';
    }
    return SceneBundle{loaded.scene, loaded.camera};
}

SceneBundle make_scene_bundle(const CliOptions& options) {
    if (options.scene == "gradient_sphere") {
        return SceneBundle{
            renderer::make_gradient_sphere_scene(),
            make_camera(
                renderer::Vec3(0.0f, 0.0f, 2.0f),
                renderer::Vec3(0.0f, 0.0f, -1.0f),
                45.0f,
                options.width,
                options.height)};
    }
    if (options.scene == "triangle") {
        return SceneBundle{
            renderer::make_triangle_scene(),
            make_camera(
                renderer::Vec3(0.0f, 0.0f, 2.0f),
                renderer::Vec3(0.0f, 0.0f, 0.0f),
                45.0f,
                options.width,
                options.height)};
    }
    if (options.scene == "mirror_spheres") {
        return SceneBundle{
            renderer::make_mirror_spheres_scene(),
            make_camera(
                renderer::Vec3(0.0f, 0.65f, 2.4f),
                renderer::Vec3(0.0f, -0.05f, -1.0f),
                42.0f,
                options.width,
                options.height)};
    }
    if (options.scene == "cornell_box") {
        return SceneBundle{
            renderer::make_cornell_box_scene(),
            make_camera(
                renderer::Vec3(0.0f, 0.15f, 1.5f),
                renderer::Vec3(0.0f, 0.15f, -2.0f),
                45.0f,
                options.width,
                options.height)};
    }
    if (options.scene == "obj_viewer" || options.scene == "asset_viewer") {
        return make_asset_scene(options);
    }

    throw std::invalid_argument("unknown scene: " + options.scene);
}

renderer::RenderSettings make_settings(const CliOptions& options) {
    renderer::RenderSettings settings;
    settings.width = options.width;
    settings.height = options.height;
    settings.path.samples_per_pixel = options.samples_per_pixel;
    settings.path.thread_count = options.thread_count;
    settings.path.backend = options.path_backend;
    return settings;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_args(argc, argv);
        if (options.help) {
            print_help();
            return 0;
        }

        SceneBundle bundle = make_scene_bundle(options);
        if (!options.environment_path.empty()) {
            bundle.scene.environment_map =
                renderer::EnvironmentMap::load(options.environment_path);
            bundle.scene.environment = renderer::Color::Ones();
        }
        bundle.scene.environment_intensity = options.environment_intensity;
        bundle.scene.environment_rotation_degrees = options.environment_yaw_degrees;
        bundle.scene.environment_background_visible =
            options.environment_background_visible;
        if (options.path_backend == renderer::PathBackend::Auto) {
            std::string reason;
            if (!renderer::cuda_path_backend_available(&reason)) {
                std::cerr << "warning: CUDA path backend unavailable, using CPU: "
                          << reason << '\n';
            }
        }
        renderer::PathTracerRenderer renderer_instance;
        const renderer::RenderSettings settings = make_settings(options);
        const renderer::RenderResult result =
            renderer_instance.render(bundle.scene, bundle.camera, settings);

        if (!result.image.write_png(options.output_path)) {
            throw std::runtime_error("failed to write PNG: " + options.output_path);
        }

        std::cout << "mode=" << options.mode
                  << " scene=" << options.scene
                  << " size=" << settings.width << "x" << settings.height
                  << " spp=" << settings.path.samples_per_pixel
                  << " backend=" << renderer::execution_backend_name(result.backend);
        std::cout << " seconds=" << result.seconds
                  << " output=" << options.output_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n\n";
        print_help();
        return 1;
    }
}
