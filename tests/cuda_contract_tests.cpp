#include "test_framework.h"

#include "render/pathtracer/cuda_device_context.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "scene/camera.h"
#include "scene/instanced_scene.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

RENDER_TEST(test_cuda_compile_flag_matches_availability) {
    RENDER_CHECK(
        renderer::cuda_path_backend_compiled() ==
        (RENDERER_HAS_CUDA == 1));
}

RENDER_TEST(test_cuda_device_selection_contracts) {
    const std::vector<int> compatible{2, 0};
    RENDER_CHECK(renderer::select_cuda_device_id(
        -1, 3, compatible, true).value() == 2);
    RENDER_CHECK(renderer::select_cuda_device_id(
        0, 3, compatible, true).value() == 0);
    RENDER_CHECK(!renderer::select_cuda_device_id(
        1, 3, compatible, true).has_value());
    RENDER_CHECK(renderer::select_cuda_device_id(
        -1, 3, {}, false).value() == 0);
    RENDER_CHECK(!renderer::select_cuda_device_id(
        3, 3, {}, false).has_value());
}

RENDER_TEST(test_cuda_availability_contracts) {
    std::string reason;
    RENDER_CHECK(!renderer::cuda_path_backend_available(-1, &reason));
    RENDER_CHECK(!reason.empty());
    reason.clear();
    RENDER_CHECK(!renderer::cuda_path_backend_available(999999, &reason));
    RENDER_CHECK(!reason.empty());
}

RENDER_TEST(test_cuda_stub_throws_uniformly_when_uncompiled) {
    if (renderer::cuda_path_backend_compiled()) {
        RENDER_SKIP("CUDA backend compiled; stub contract not exercised");
    }
    renderer::CudaPathInteractiveRenderer renderer_instance;
    bool threw = false;
    try {
        (void)renderer_instance.statistics();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    RENDER_CHECK(threw);
    threw = false;
    try {
        (void)renderer_instance.stream_handle();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    RENDER_CHECK(threw);
    threw = false;
    try {
        renderer_instance.set_presentation_state(false, false);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    RENDER_CHECK(threw);
}

RENDER_TEST(test_cuda_interactive_upload_contracts) {
    std::string reason;
    if (!renderer::cuda_path_backend_available(0, &reason)) {
        RENDER_CHECK(!reason.empty());
        RENDER_SKIP("CUDA path backend unavailable");
    }
    const renderer::CudaDeviceContext context =
        renderer::CudaDeviceContext::create(0);
    RENDER_CHECK(context.valid());
    RENDER_CHECK(context.device_id() == 0);
    RENDER_CHECK(context.is_current());
    renderer::CudaPathInteractiveRenderer interactive(context);
    RENDER_CHECK(interactive.device_id() == 0);

    auto first_asset = std::make_shared<renderer::Scene>();
    first_asset->materials.push_back(renderer::Material{});
    first_asset->triangles.emplace_back(
        renderer::Vec3(-0.4f, -0.4f, 0.0f),
        renderer::Vec3(0.4f, -0.4f, 0.0f),
        renderer::Vec3(0.0f, 0.4f, 0.0f),
        0);
    auto second_asset =
        std::make_shared<renderer::Scene>(*first_asset);
    renderer::RenderSceneSnapshot snapshot;
    snapshot.source_id = renderer::allocate_render_scene_source_id();
    snapshot.revisions = renderer::SceneRevisions{1, 1, 1, 1, 1, 1, 1, 1};
    renderer::RenderSceneAssetSnapshot first_view;
    first_view.asset_id = 101;
    first_view.geometry_revision = 1;
    first_view.local_scene = first_asset;
    first_view.local_bounds = first_asset->triangles[0].bounds();
    first_view.triangle_material_slots.push_back(
        renderer::MaterialSlot::bound(0));
    renderer::RenderSceneAssetSnapshot second_view = first_view;
    second_view.asset_id = 102;
    second_view.local_scene = second_asset;
    snapshot.assets.push_back(first_view);
    snapshot.assets.push_back(second_view);

    const auto make_instance = [](
        int asset_index,
        std::uint64_t object_id,
        float x,
        const renderer::Bounds3& local_bounds) {
        renderer::RenderSceneInstanceSnapshot instance;
        instance.object_id = object_id;
        instance.asset_index = asset_index;
        instance.object_to_world = renderer::Mat4::Identity();
        instance.object_to_world(0, 3) = x;
        instance.object_to_world(2, 3) = -2.0f;
        instance.world_to_object = instance.object_to_world.inverse();
        instance.normal_to_world = renderer::Mat3::Identity();
        instance.world_bounds = renderer::Bounds3(
            local_bounds.min + renderer::Vec3(x, 0.0f, -2.0f),
            local_bounds.max + renderer::Vec3(x, 0.0f, -2.0f));
        instance.materials.push_back(renderer::Material{});
        return instance;
    };
    snapshot.instances.push_back(make_instance(
        0, 201, -0.5f, first_view.local_bounds));
    snapshot.instances.push_back(make_instance(
        1, 202, 0.5f, second_view.local_bounds));

    renderer::RenderSettings settings;
    settings.width = 8;
    settings.height = 8;
    settings.path.cuda_device = 0;
    const renderer::Camera camera(
        renderer::Vec3::Zero(),
        -renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitY(),
        45.0f,
        1.0f);
    renderer::Framebuffer frame(settings.width, settings.height);
    renderer::InteractiveFrameState frame_state;
    interactive.reset(snapshot, settings);
    const renderer::CudaPathStatistics initial_statistics =
        interactive.statistics();
    RENDER_CHECK(initial_statistics.blas_build_count == 2);
    RENDER_CHECK(initial_statistics.tlas_build_count == 1);

    first_asset->triangles[0] = renderer::Triangle(
        renderer::Vec3(-0.45f, -0.4f, 0.0f),
        renderer::Vec3(0.45f, -0.4f, 0.0f),
        renderer::Vec3(0.0f, 0.45f, 0.0f),
        0);
    snapshot.assets[0].local_bounds =
        first_asset->triangles[0].bounds();
    snapshot.assets[0].geometry_revision++;
    snapshot.instances[0] = make_instance(
        0, 201, -0.5f, snapshot.assets[0].local_bounds);
    snapshot.revisions.geometry++;
    snapshot.revisions.material_bindings++;
    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        frame);
    const renderer::CudaPathStatistics after_geometry =
        interactive.statistics();
    RENDER_CHECK(
        after_geometry.blas_build_count ==
        initial_statistics.blas_build_count + 1);

    const std::uint64_t geometry_upload =
        after_geometry.geometry_upload_bytes;
    const std::uint64_t binding_upload =
        after_geometry.material_binding_upload_bytes;
    const std::uint64_t material_upload =
        after_geometry.material_upload_bytes;
    const std::uint64_t texture_upload =
        after_geometry.texture_upload_bytes;
    snapshot.instances[0] = make_instance(
        0, 201, -0.35f, snapshot.assets[0].local_bounds);
    snapshot.revisions.transforms++;
    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        frame);
    const renderer::CudaPathStatistics after_transform =
        interactive.statistics();
    RENDER_CHECK(after_transform.blas_build_count ==
        after_geometry.blas_build_count);
    RENDER_CHECK(after_transform.tlas_refit_count ==
        after_geometry.tlas_refit_count + 1);
    RENDER_CHECK(after_transform.geometry_upload_bytes == geometry_upload);
    RENDER_CHECK(after_transform.material_binding_upload_bytes == binding_upload);
    RENDER_CHECK(after_transform.material_upload_bytes == material_upload);
    RENDER_CHECK(after_transform.texture_upload_bytes == texture_upload);

    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        frame);
    RENDER_CHECK(interactive.accumulated_samples() == 2);
    settings.path.max_bounces++;
    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        frame);
    RENDER_CHECK(interactive.accumulated_samples() == 1);
    settings.path.russian_roulette_start_bounce++;
    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        frame);
    RENDER_CHECK(interactive.accumulated_samples() == 1);
    settings.path.russian_roulette_min_probability += 0.01f;
    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        frame);
    RENDER_CHECK(interactive.accumulated_samples() == 1);
    settings.path.russian_roulette_max_probability -= 0.01f;
    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        frame);
    RENDER_CHECK(interactive.accumulated_samples() == 1);
    renderer::InteractiveFrameState non_radiative_state;
    non_radiative_state.delta_seconds = 0.25f;
    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        non_radiative_state,
        frame);
    RENDER_CHECK(interactive.accumulated_samples() == 2);
    settings.path.sample_seed_offset++;
    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        frame);
    RENDER_CHECK(interactive.accumulated_samples() == 1);
    snapshot.environment = renderer::Color(0.2f, 0.1f, 0.05f);
    snapshot.revisions.environment++;
    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        frame);
    RENDER_CHECK(interactive.accumulated_samples() == 1);
    snapshot.source_id = renderer::allocate_render_scene_source_id();
    interactive.render_next_frame(
        snapshot,
        camera,
        settings,
        frame_state,
        frame);
    RENDER_CHECK(interactive.accumulated_samples() == 1);
}
