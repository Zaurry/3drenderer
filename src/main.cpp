#include "render/pathtracer/pathtracer_renderer.h"
#include "render/rasterizer/rasterizer_renderer.h"
#include "render/raytracer/raytracer_renderer.h"
#include "scene/obj_loader.h"
#include "scene/scene.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct SceneBundle {
    renderer::Scene scene;
    renderer::Camera camera;
};

struct CliOptions {
    std::string mode = "ray";
    std::string scene = "gradient_sphere";
    std::string obj_path;
    std::string output_path;
    int width = 512;
    int height = 512;
    int samples_per_pixel = 1;
    int max_depth = 5;
    int thread_count = 0;
    bool help = false;
};

void print_help() {
    std::cout
        << "CPU 3D Renderer v0.1\n"
        << "\n"
        << "Usage:\n"
        << "  renderer --mode raster|ray|path --scene gradient_sphere|raster_triangle|mirror_spheres|cornell_box|obj_viewer --output file.png [options]\n"
        << "\n"
        << "Options:\n"
        << "  --obj path          OBJ file for --scene obj_viewer\n"
        << "  --width integer     image width, default 512\n"
        << "  --height integer    image height, default 512\n"
        << "  --spp integer       samples per pixel, default 1\n"
        << "  --max-depth integer bounce depth, default 5\n"
        << "  --threads integer   path tracer worker threads, default hardware threads\n"
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
        } else if (arg == "--obj") {
            options.obj_path = require_value(argc, argv, i, arg);
        } else if (arg == "--width") {
            options.width = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--height") {
            options.height = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--spp") {
            options.samples_per_pixel = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--max-depth") {
            options.max_depth = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--threads") {
            options.thread_count = parse_positive_int(require_value(argc, argv, i, arg), arg);
        } else if (arg == "--output") {
            options.output_path = require_value(argc, argv, i, arg);
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }

    if (!options.help && options.output_path.empty()) {
        throw std::invalid_argument("--output is required");
    }
    return options;
}

renderer::Camera make_camera(
    const renderer::Vec3& eye,
    const renderer::Vec3& target,
    double vertical_fov_degrees,
    int width,
    int height) {
    return renderer::Camera(
        eye,
        target,
        renderer::Vec3(0.0, 1.0, 0.0),
        vertical_fov_degrees,
        static_cast<double>(width) / static_cast<double>(height));
}

renderer::Bounds3 mesh_bounds(const renderer::Mesh& mesh) {
    renderer::Bounds3 bounds;
    for (const renderer::Triangle& triangle : mesh.triangles) {
        bounds.expand(triangle.bounds());
    }
    return bounds;
}

SceneBundle make_obj_scene(const CliOptions& options) {
    if (options.obj_path.empty()) {
        throw std::invalid_argument("--scene obj_viewer requires --obj");
    }

    renderer::Mesh mesh = renderer::load_obj_mesh(options.obj_path, 0);
    if (mesh.triangles.empty()) {
        throw std::runtime_error("OBJ contains no triangles");
    }

    renderer::Scene scene;
    scene.environment = renderer::Color(0.04, 0.05, 0.07);
    renderer::Material material;
    material.type = renderer::MaterialType::Diffuse;
    material.base_color = renderer::Color(0.72, 0.74, 0.78);
    scene.materials.push_back(material);
    scene.triangles = mesh.triangles;
    scene.directional_lights.push_back(
        renderer::DirectionalLight{renderer::normalize(renderer::Vec3(-1.0, -1.0, -1.0)), renderer::Color(1.5, 1.5, 1.4)});

    const renderer::Bounds3 bounds = mesh_bounds(mesh);
    const renderer::Vec3 center = (bounds.min + bounds.max) * 0.5;
    const double radius = std::max(0.5, renderer::length(bounds.max - bounds.min) * 0.5);
    return SceneBundle{
        scene,
        make_camera(center + renderer::Vec3(0.0, 0.0, radius * 3.0), center, 45.0, options.width, options.height)};
}

SceneBundle make_scene_bundle(const CliOptions& options) {
    if (options.scene == "gradient_sphere") {
        return SceneBundle{
            renderer::make_gradient_sphere_scene(),
            make_camera(
                renderer::Vec3(0.0, 0.0, 2.0),
                renderer::Vec3(0.0, 0.0, -1.0),
                45.0,
                options.width,
                options.height)};
    }
    if (options.scene == "raster_triangle") {
        return SceneBundle{
            renderer::make_raster_triangle_scene(),
            make_camera(
                renderer::Vec3(0.0, 0.0, 2.0),
                renderer::Vec3(0.0, 0.0, 0.0),
                45.0,
                options.width,
                options.height)};
    }
    if (options.scene == "mirror_spheres") {
        return SceneBundle{
            renderer::make_mirror_spheres_scene(),
            make_camera(
                renderer::Vec3(0.0, 0.65, 2.4),
                renderer::Vec3(0.0, -0.05, -1.0),
                42.0,
                options.width,
                options.height)};
    }
    if (options.scene == "cornell_box") {
        return SceneBundle{
            renderer::make_cornell_box_scene(),
            make_camera(
                renderer::Vec3(0.0, 0.15, 1.5),
                renderer::Vec3(0.0, 0.15, -2.0),
                45.0,
                options.width,
                options.height)};
    }
    if (options.scene == "obj_viewer") {
        return make_obj_scene(options);
    }

    throw std::invalid_argument("unknown scene: " + options.scene);
}

std::unique_ptr<renderer::IRenderer> make_renderer(const std::string& mode) {
    if (mode == "raster") {
        return std::make_unique<renderer::RasterizerRenderer>();
    }
    if (mode == "ray") {
        return std::make_unique<renderer::RayTracerRenderer>();
    }
    if (mode == "path") {
        return std::make_unique<renderer::PathTracerRenderer>();
    }
    throw std::invalid_argument("unknown mode: " + mode);
}

renderer::RenderSettings make_settings(const CliOptions& options) {
    renderer::RenderSettings settings;
    settings.width = options.width;
    settings.height = options.height;
    settings.samples_per_pixel = options.samples_per_pixel;
    settings.max_depth = options.max_depth;
    settings.thread_count = options.thread_count;
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
        std::unique_ptr<renderer::IRenderer> renderer_instance = make_renderer(options.mode);
        const renderer::RenderSettings settings = make_settings(options);
        const renderer::RenderResult result =
            renderer_instance->render(bundle.scene, bundle.camera, settings);

        if (!result.image.write_png(options.output_path)) {
            throw std::runtime_error("failed to write PNG: " + options.output_path);
        }

        std::cout << "mode=" << options.mode
                  << " scene=" << options.scene
                  << " size=" << settings.width << "x" << settings.height
                  << " spp=" << settings.samples_per_pixel
                  << " max_depth=" << settings.max_depth
                  << " seconds=" << result.seconds
                  << " output=" << options.output_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n\n";
        print_help();
        return 1;
    }
}
