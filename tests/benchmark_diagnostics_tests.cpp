#include "test_framework.h"

#include "render/framebuffer.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "render/render_settings.h"
#include "scene/camera.h"
#include "scene/instanced_scene.h"
#include "scene/scene.h"

#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>

RENDER_TEST(test_cuda_diagnostic_profile_contracts) {
    std::string reason;
    if (!renderer::cuda_path_backend_available(&reason)) {
        RENDER_SKIP("CUDA path backend unavailable: " + reason);
    }
    constexpr int width = 32;
    constexpr int height = 24;
    renderer::Scene scene = renderer::make_gradient_sphere_scene();
    renderer::RenderSettings settings;
    settings.width = width;
    settings.height = height;
    settings.path.samples_per_pixel = 1;
    renderer::Camera camera(
        renderer::Vec3(0.0f, 0.0f, 2.5f),
        renderer::Vec3::Zero(),
        renderer::Vec3(0.0f, 1.0f, 0.0f),
        45.0f,
        static_cast<float>(width) / static_cast<float>(height));
    renderer::CudaPathInteractiveRenderer renderer_instance;
    renderer::RenderSceneSnapshot snapshot =
        renderer::make_render_scene_snapshot(std::move(scene));
    renderer_instance.reset(snapshot, settings);
    renderer::Framebuffer framebuffer(width, height);
    renderer::InteractiveFrameState frame;
    frame.camera_changed = true;
    frame.automatic_interaction_quality = false;
    renderer_instance.render_next_frame(
        snapshot,
        camera,
        settings,
        frame,
        framebuffer);
    const renderer::CudaPathDiagnosticProfile profile =
        renderer_instance.download_diagnostic_profile();
    const std::uint64_t expected_primary =
        static_cast<std::uint64_t>(width) * height;
    RENDER_CHECK(profile.primary_rays == expected_primary);
    RENDER_CHECK(
        profile.primary_hits + profile.primary_misses ==
        profile.primary_rays);
    RENDER_CHECK(profile.rays_by_bounce[0] == profile.primary_rays);
    const std::uint64_t ray_histogram_sum = std::accumulate(
        profile.rays_by_bounce.begin(),
        profile.rays_by_bounce.end(),
        std::uint64_t{0});
    RENDER_CHECK(
        ray_histogram_sum ==
        profile.primary_rays + profile.continuation_rays);
    const std::uint64_t termination_sum = std::accumulate(
        profile.termination_by_bounce.begin(),
        profile.termination_by_bounce.end(),
        std::uint64_t{0});
    RENDER_CHECK(termination_sum == profile.primary_rays);
}
