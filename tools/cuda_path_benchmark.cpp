#include "render/framebuffer.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "render/render_settings.h"
#include "scene/camera.h"
#include "scene/scene_document.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct BenchmarkCase {
    const char* name;
    const char* asset;
    renderer::Vec3 eye;
    renderer::Vec3 forward;
    renderer::Vec3 up;
};

constexpr int kWidth = 2418;
constexpr int kHeight = 1343;

BenchmarkCase benchmark_case(const std::string& name) {
    if (name == "sponza") {
        return BenchmarkCase{
            "sponza",
            "Computer Graphics Archive/sponza/sponza.obj",
            renderer::Vec3(12.0652f, 1.5227f, 0.6981f),
            renderer::Vec3(-0.9934f, 0.0699f, -0.0904f),
            renderer::Vec3(0.069655f, 0.997551f, 0.006342f)};
    }
    if (name == "san-miguel") {
        return BenchmarkCase{
            "san-miguel",
            "Computer Graphics Archive/San_Miguel/san-miguel-low-poly.obj",
            renderer::Vec3(25.2773f, 1.3083f, 2.3746f),
            renderer::Vec3(-0.9717f, 0.2085f, -0.1112f),
            renderer::Vec3(0.207110f, 0.978031f, 0.023692f)};
    }
    throw std::invalid_argument(
        "benchmark case must be 'sponza' or 'san-miguel'");
}

float median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string case_name = argc > 1 ? argv[1] : "sponza";
        const int iterations = argc > 2 ? std::max(1, std::stoi(argv[2])) : 5;
        const std::string work_mode = argc > 3 ? argv[3] : "full";
        const bool automatic = work_mode != "full";
        const BenchmarkCase test = benchmark_case(case_name);
        const std::filesystem::path source_root(RENDERER_SOURCE_DIR);

        std::string unavailable_reason;
        if (!renderer::cuda_path_backend_available(&unavailable_reason)) {
            throw std::runtime_error(
                "CUDA path backend unavailable: " + unavailable_reason);
        }

        renderer::SceneDocument document;
        const std::filesystem::path floor =
            source_root / "Computer Graphics Archive/hw1/floor/floor.obj";
        if (std::filesystem::exists(floor)) {
            document.import_path(floor, kWidth, kHeight);
        }
        const std::vector<renderer::ObjectId> imported =
            document.import_path(
                source_root / test.asset,
                kWidth,
                kHeight);
        if (imported.empty()) {
            throw std::runtime_error("benchmark asset did not create an instance");
        }
        const renderer::ObjectId moved_object = imported.front();
        const renderer::Mat4 initial_world =
            document.world_matrix(moved_object);
        renderer::Scene scene_placeholder;
        std::size_t triangle_count = 0;
        for (const renderer::InstancedSceneAssetView& asset :
             document.instanced_render_scene().assets) {
            triangle_count += asset.local_scene
                ? asset.local_scene->triangles.size()
                : 0;
        }

        renderer::RenderSettings settings;
        settings.width = kWidth;
        settings.height = kHeight;
        settings.path.backend = renderer::PathBackend::Cuda;
        settings.path.samples_per_pixel = 1;
        const renderer::Camera camera(
            test.eye,
            test.eye + test.forward,
            test.up,
            45.0f,
            static_cast<float>(kWidth) / static_cast<float>(kHeight));

        renderer::CudaPathInteractiveRenderer renderer;
        renderer.reset(
            scene_placeholder,
            settings,
            &document.instanced_render_scene());
        renderer::Framebuffer framebuffer(kWidth, kHeight);
        renderer::InteractiveFrameState frame_state;
        frame_state.automatic_interaction_quality = automatic;
        std::vector<float> trace_milliseconds;
        trace_milliseconds.reserve(static_cast<std::size_t>(iterations));

        for (int iteration = 0; iteration < iterations; ++iteration) {
            frame_state.scene_changes =
                renderer::SceneChange::None;
            frame_state.camera_changed =
                work_mode == "interaction" || iteration == 0;
            if (work_mode == "drag") {
                renderer::Mat4 moved = initial_world;
                moved(0, 3) +=
                    0.01f * static_cast<float>(iteration + 1);
                if (!document.set_world_matrix(
                        moved_object,
                        moved)) {
                    throw std::runtime_error(
                        "failed to move benchmark instance");
                }
                frame_state.scene_changes =
                    renderer::SceneChange::InstanceTransforms;
                frame_state.camera_changed = false;
            }
            const auto wall_start = std::chrono::steady_clock::now();
            renderer.render_next_frame(
                scene_placeholder,
                camera,
                settings,
                frame_state,
                framebuffer,
                &document.instanced_render_scene());
            const auto wall_end = std::chrono::steady_clock::now();
            const renderer::CudaPathStatistics& statistics =
                renderer.statistics();
            trace_milliseconds.push_back(statistics.trace_milliseconds);
            const float wall_milliseconds =
                std::chrono::duration<float, std::milli>(
                    wall_end - wall_start).count();
            std::cout
                << "iteration=" << iteration
                << " trace_ms=" << statistics.trace_milliseconds
                << " wall_ms=" << wall_milliseconds
                << " spp=" << renderer.accumulated_samples()
                << " internal=" << statistics.internal_width
                << 'x' << statistics.internal_height
                << " quantum_rows=" << statistics.tile_rows
                << " sweep=" << statistics.sweep_progress
                << " complete_spp_s="
                << statistics.complete_sweeps_per_second
                << " published="
                << (statistics.presentation_updated ? 1 : 0)
                << " present_ms="
                << statistics.presentation_milliseconds
                << " instance_ms="
                << statistics.instance_upload_milliseconds
                << " tlas_refit_ms="
                << statistics.tlas_refit_milliseconds
                << " blas_builds="
                << statistics.blas_build_count
                << " tlas_refits="
                << statistics.tlas_refit_count
                << '\n';
        }

        std::cout
            << std::fixed << std::setprecision(3)
            << "case=" << test.name
            << " size=" << kWidth << 'x' << kHeight
            << " triangles=" << triangle_count
            << " median_trace_ms=" << median(trace_milliseconds)
            << " upload_ms=" << renderer.statistics().upload_milliseconds
            << " downloads=" << renderer.statistics().framebuffer_downloads
            << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "cuda_path_benchmark: " << error.what() << '\n';
        return 1;
    }
}
