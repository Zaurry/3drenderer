#include "test_framework.h"

#include "acceleration/bvh.h"
#include "core/color.h"
#include "core/image.h"
#include "core/io/atomic_file.h"
#include "core/math/bounds.h"
#include "core/math/ray.h"
#include "core/math/types.h"
#include "core/math/transforms.h"
#include "core/random.h"
#include "core/timer.h"
#include "interactive/frame_rate_counter.h"
#include "interactive/free_camera_controller.h"
#include "interactive/orbit_camera_controller.h"
#include "interactive/viewer_session.h"
#include "interactive/viewer_ui.h"
#include "platform/sdl/sdl_display_backend.h"
#include "render/display_settings.h"
#include "render/renderer.h"
#include "render/render_settings.h"
#include "render/framebuffer.h"
#include "render/ggx_energy_compensation.h"
#include "render/interactive/interactive_render_session.h"
#include "render/interactive/path_interactive_session.h"
#include "render/pathtracer/cuda_pathtracer.h"
#include "render/pbr.h"
#include "render/scene_intersector.h"
#include "sampling/sampler.h"
#include "scene/camera.h"
#include "scene/environment.h"
#include "scene/gltf_loader.h"
#include "scene/instanced_scene.h"
#include "scene/material.h"
#include "scene/material_evaluator.h"
#include "scene/obj_loader.h"
#include "scene/primitive.h"
#include "scene/scene_asset_loader.h"
#include "scene/scene_document.h"
#include "scene/scene.h"
#include "scene/texture.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(std::is_same_v<renderer::Scalar, float>);
static_assert(std::is_same_v<renderer::Vec2, Eigen::Vector2f>);
static_assert(std::is_same_v<renderer::Vec3, Eigen::Vector3f>);
static_assert(std::is_same_v<renderer::Vec4, Eigen::Vector4f>);
static_assert(std::is_same_v<renderer::Mat3, Eigen::Matrix3f>);
static_assert(std::is_same_v<renderer::Mat4, Eigen::Matrix4f>);
static_assert(std::is_same_v<renderer::Color, Eigen::Vector3f>);
static_assert(std::is_same_v<decltype(renderer::PcgRandom().next_float()), float>);
static_assert(std::is_same_v<decltype(renderer::Timer().elapsed_seconds()), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::RenderResult>().seconds), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::FrameRateSnapshot>().frames_per_second), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::Camera>().viewport_width()), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::InteractiveFrameState>().delta_seconds), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::InputState>().mouse_delta_x), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::InputState>().mouse_delta_y), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::InputState>().wheel_delta), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::HitRecord>().t), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::Material>().roughness), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::Material>().opacity), float>);
static_assert(std::is_same_v<decltype(std::declval<renderer::SurfaceMaterialSample>().opacity), float>);
static_assert(std::is_same_v<
    decltype(&renderer::Bvh::intersect),
    bool (renderer::Bvh::*)(
        const renderer::Ray&,
        float,
        float,
        renderer::HitRecord&) const>);
static_assert(std::is_same_v<
    decltype(&renderer::SceneIntersector::intersect),
    bool (renderer::SceneIntersector::*)(
        const renderer::Ray&,
        float,
        float,
        renderer::HitRecord&) const>);
static_assert(std::is_same_v<
    decltype(&renderer::SceneIntersector::occluded),
    bool (renderer::SceneIntersector::*)(const renderer::Ray&, float, float) const>);
static_assert(std::is_same_v<
    decltype(&renderer::refract),
    bool (*)(const renderer::Vec3&, const renderer::Vec3&, float, renderer::Vec3&)>);
static_assert(std::is_same_v<
    decltype(renderer::offset_ray_origin(
        renderer::Vec3::Zero(), renderer::Vec3::UnitX(), renderer::Vec3::UnitX())),
    renderer::Vec3>);

renderer::RenderResult render_cuda_scene(
    const renderer::Scene& scene,
    const renderer::Camera& camera,
    const renderer::RenderSettings& settings) {
    return renderer::render_cuda_path(
        renderer::make_render_scene_snapshot(scene),
        camera,
        settings);
}

RENDER_TEST(test_vec3_arithmetic) {
    renderer::Vec3 a(1.0f, 2.0f, 3.0f);
    renderer::Vec3 b(4.0f, -2.0f, 0.5f);

    renderer::Vec3 sum = a + b;
    RENDER_CHECK(nearly_equal(sum.x(), 5.0f));
    RENDER_CHECK(nearly_equal(sum.y(), 0.0f));
    RENDER_CHECK(nearly_equal(sum.z(), 3.5f));

    renderer::Vec3 difference = a - b;
    RENDER_CHECK(nearly_equal(difference.x(), -3.0f));
    RENDER_CHECK(nearly_equal(difference.y(), 4.0f));
    RENDER_CHECK(nearly_equal(difference.z(), 2.5f));

    renderer::Vec3 scaled_right = a * 2.0f;
    RENDER_CHECK(nearly_equal(scaled_right.x(), 2.0f));
    RENDER_CHECK(nearly_equal(scaled_right.y(), 4.0f));
    RENDER_CHECK(nearly_equal(scaled_right.z(), 6.0f));

    renderer::Vec3 scaled_left = 0.5f * b;
    RENDER_CHECK(nearly_equal(scaled_left.x(), 2.0f));
    RENDER_CHECK(nearly_equal(scaled_left.y(), -1.0f));
    RENDER_CHECK(nearly_equal(scaled_left.z(), 0.25f));

    renderer::Vec3 divided = a / 2.0f;
    RENDER_CHECK(nearly_equal(divided.x(), 0.5f));
    RENDER_CHECK(nearly_equal(divided.y(), 1.0f));
    RENDER_CHECK(nearly_equal(divided.z(), 1.5f));

    renderer::Vec3 negated = -a;
    RENDER_CHECK(nearly_equal(negated.x(), -1.0f));
    RENDER_CHECK(nearly_equal(negated.y(), -2.0f));
    RENDER_CHECK(nearly_equal(negated.z(), -3.0f));

    renderer::Vec3 min_v = a.cwiseMin(b);
    RENDER_CHECK(nearly_equal(min_v.x(), 1.0f));
    RENDER_CHECK(nearly_equal(min_v.y(), -2.0f));
    RENDER_CHECK(nearly_equal(min_v.z(), 0.5f));

    renderer::Vec3 max_v = a.cwiseMax(b);
    RENDER_CHECK(nearly_equal(max_v.x(), 4.0f));
    RENDER_CHECK(nearly_equal(max_v.y(), 2.0f));
    RENDER_CHECK(nearly_equal(max_v.z(), 3.0f));

    RENDER_CHECK(nearly_equal(a.dot(b), 1.0f * 4.0f + 2.0f * -2.0f + 3.0f * 0.5f));

    renderer::Vec3 c = renderer::Vec3(1, 0, 0).cross(renderer::Vec3(0, 1, 0));
    RENDER_CHECK(nearly_equal(c.x(), 0.0f));
    RENDER_CHECK(nearly_equal(c.y(), 0.0f));
    RENDER_CHECK(nearly_equal(c.z(), 1.0f));

    renderer::Vec3 n = renderer::Vec3(0, 3, 4).normalized();
    RENDER_CHECK(nearly_equal(n.norm(), 1.0f));
    RENDER_CHECK(nearly_equal(n.y(), 0.6f, 1e-6f));
    RENDER_CHECK(nearly_equal(n.z(), 0.8f, 1e-6f));
}

RENDER_TEST(test_mat4_composition_order) {
    const renderer::Mat4 transform =
        (Eigen::Translation3f(renderer::Vec3(1.0f, 2.0f, 3.0f)) *
         Eigen::Scaling(2.0f, 3.0f, 4.0f))
            .matrix();
    const renderer::Vec4 p = transform * renderer::Vec4(1.0f, 1.0f, 1.0f, 1.0f);
    RENDER_CHECK(nearly_equal(p.x(), 3.0f));
    RENDER_CHECK(nearly_equal(p.y(), 5.0f));
    RENDER_CHECK(nearly_equal(p.z(), 7.0f));
    RENDER_CHECK(nearly_equal(p.w(), 1.0f));
}

RENDER_TEST(test_mat4_perspective_uses_degrees_and_ndc_depth) {
    const renderer::Mat4 p = renderer::make_perspective_matrix(90.0f, 1.0f, 1.0f, 10.0f);
    RENDER_CHECK(nearly_equal(p(0, 0), 1.0f));
    RENDER_CHECK(nearly_equal(p(1, 1), 1.0f));
    RENDER_CHECK(nearly_equal(p(2, 2), -11.0f / 9.0f));
    RENDER_CHECK(nearly_equal(p(2, 3), -20.0f / 9.0f));
    RENDER_CHECK(nearly_equal(p(3, 2), -1.0f));

    const renderer::Vec4 near_clip = p * renderer::Vec4(0.0f, 0.0f, -1.0f, 1.0f);
    RENDER_CHECK(nearly_equal(near_clip.z() / near_clip.w(), -1.0f, 1e-6f));

    const renderer::Vec4 far_clip = p * renderer::Vec4(0.0f, 0.0f, -10.0f, 1.0f);
    RENDER_CHECK(nearly_equal(far_clip.z() / far_clip.w(), 1.0f, 1e-6f));
}

void check_perspective_invalid_input_throws(
    float vertical_fov_degrees,
    float aspect,
    float near_z,
    float far_z) {
    bool threw = false;
    try {
        renderer::make_perspective_matrix(vertical_fov_degrees, aspect, near_z, far_z);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    RENDER_CHECK(threw);
}

RENDER_TEST(test_mat4_perspective_invalid_inputs_throw) {
    check_perspective_invalid_input_throws(0.0f, 1.0f, 1.0f, 10.0f);
    check_perspective_invalid_input_throws(-1.0f, 1.0f, 1.0f, 10.0f);
    check_perspective_invalid_input_throws(180.0f, 1.0f, 1.0f, 10.0f);
    check_perspective_invalid_input_throws(181.0f, 1.0f, 1.0f, 10.0f);
    check_perspective_invalid_input_throws(90.0f, 0.0f, 1.0f, 10.0f);
    check_perspective_invalid_input_throws(90.0f, -1.0f, 1.0f, 10.0f);
    check_perspective_invalid_input_throws(90.0f, 1.0f, 0.0f, 10.0f);
    check_perspective_invalid_input_throws(90.0f, 1.0f, -1.0f, 10.0f);
    check_perspective_invalid_input_throws(90.0f, 1.0f, 1.0f, 1.0f);
    check_perspective_invalid_input_throws(90.0f, 1.0f, 10.0f, 1.0f);
    check_perspective_invalid_input_throws(
        std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f, 10.0f);
    check_perspective_invalid_input_throws(
        90.0f, std::numeric_limits<float>::infinity(), 1.0f, 10.0f);
    check_perspective_invalid_input_throws(
        90.0f, 1.0f, std::numeric_limits<float>::quiet_NaN(), 10.0f);
    check_perspective_invalid_input_throws(
        90.0f, 1.0f, 1.0f, -std::numeric_limits<float>::infinity());
}

RENDER_TEST(test_mat4_look_at) {
    const renderer::Mat4 view = renderer::make_look_at_matrix(
        renderer::Vec3(0.0f, 0.0f, 0.0f),
        renderer::Vec3(0.0f, 0.0f, -1.0f),
        renderer::Vec3(0.0f, 1.0f, 0.0f));
    RENDER_CHECK(nearly_equal(view(0, 0), 1.0f));
    RENDER_CHECK(nearly_equal(view(1, 1), 1.0f));
    RENDER_CHECK(nearly_equal(view(2, 2), 1.0f));
    RENDER_CHECK(nearly_equal(view(3, 3), 1.0f));
    const renderer::Vec4 p = view * renderer::Vec4(0.0f, 0.0f, -1.0f, 1.0f);
    RENDER_CHECK(nearly_equal(p.z(), -1.0f));
}

RENDER_TEST(test_mat4_look_at_invalid_inputs_throw) {
    bool threw_eye_equals_target = false;
    try {
        renderer::make_look_at_matrix(
            renderer::Vec3(0.0f, 0.0f, 0.0f),
            renderer::Vec3(0.0f, 0.0f, 0.0f),
            renderer::Vec3(0.0f, 1.0f, 0.0f));
    } catch (const std::invalid_argument&) {
        threw_eye_equals_target = true;
    }
    RENDER_CHECK(threw_eye_equals_target);

    bool threw_zero_up = false;
    try {
        renderer::make_look_at_matrix(
            renderer::Vec3(0.0f, 0.0f, 0.0f),
            renderer::Vec3(0.0f, 0.0f, -1.0f),
            renderer::Vec3(0.0f, 0.0f, 0.0f));
    } catch (const std::invalid_argument&) {
        threw_zero_up = true;
    }
    RENDER_CHECK(threw_zero_up);

    bool threw_parallel_up = false;
    try {
        renderer::make_look_at_matrix(
            renderer::Vec3(0.0f, 0.0f, 0.0f),
            renderer::Vec3(0.0f, 0.0f, -1.0f),
            renderer::Vec3(0.0f, 0.0f, 1.0f));
    } catch (const std::invalid_argument&) {
        threw_parallel_up = true;
    }
    RENDER_CHECK(threw_parallel_up);

    bool threw_non_finite_eye = false;
    try {
        renderer::make_look_at_matrix(
            renderer::Vec3(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f),
            renderer::Vec3(0.0f, 0.0f, -1.0f),
            renderer::Vec3(0.0f, 1.0f, 0.0f));
    } catch (const std::invalid_argument&) {
        threw_non_finite_eye = true;
    }
    RENDER_CHECK(threw_non_finite_eye);

    bool threw_non_finite_target = false;
    try {
        renderer::make_look_at_matrix(
            renderer::Vec3(0.0f, 0.0f, 0.0f),
            renderer::Vec3(0.0f, std::numeric_limits<float>::infinity(), -1.0f),
            renderer::Vec3(0.0f, 1.0f, 0.0f));
    } catch (const std::invalid_argument&) {
        threw_non_finite_target = true;
    }
    RENDER_CHECK(threw_non_finite_target);

    bool threw_non_finite_up = false;
    try {
        renderer::make_look_at_matrix(
            renderer::Vec3(0.0f, 0.0f, 0.0f),
            renderer::Vec3(0.0f, 0.0f, -1.0f),
            renderer::Vec3(0.0f, 1.0f, -std::numeric_limits<float>::infinity()));
    } catch (const std::invalid_argument&) {
        threw_non_finite_up = true;
    }
    RENDER_CHECK(threw_non_finite_up);
}

RENDER_TEST(test_camera_center_ray_points_forward) {
    renderer::Camera camera(
        renderer::Vec3(0, 0, 0),
        renderer::Vec3(0, 0, -1),
        renderer::Vec3(0, 1, 0),
        60.0f,
        1.0f);

    renderer::Ray ray = camera.generate_ray(0.5f, 0.5f);
    RENDER_CHECK(nearly_equal(ray.direction.x(), 0.0f, 1e-6f));
    RENDER_CHECK(nearly_equal(ray.direction.y(), 0.0f, 1e-6f));
    RENDER_CHECK(ray.direction.z() < -0.999f);
}

RENDER_TEST(test_camera_rejects_non_finite_screen_coordinates) {
    renderer::Camera camera(
        renderer::Vec3(0, 0, 0),
        renderer::Vec3(0, 0, -1),
        renderer::Vec3(0, 1, 0),
        60.0f,
        1.0f);

    bool threw_nan_u = false;
    try {
        camera.generate_ray(std::numeric_limits<float>::quiet_NaN(), 0.5f);
    } catch (const std::invalid_argument&) {
        threw_nan_u = true;
    }
    RENDER_CHECK(threw_nan_u);

    bool threw_infinite_v = false;
    try {
        camera.generate_ray(0.5f, std::numeric_limits<float>::infinity());
    } catch (const std::invalid_argument&) {
        threw_infinite_v = true;
    }
    RENDER_CHECK(threw_infinite_v);
}

RENDER_TEST(test_ray_and_bounds_intersection) {
    renderer::Ray ray(renderer::Vec3(0, 0, -5), renderer::Vec3(0, 0, 1));
    renderer::Bounds3 box(renderer::Vec3(-1, -1, -1), renderer::Vec3(1, 1, 1));
    RENDER_CHECK(box.intersect(ray, 0.001f, 1000.0f));

    renderer::Ray miss(renderer::Vec3(5, 5, -5), renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!box.intersect(miss, 0.001f, 1000.0f));
}

RENDER_TEST(test_bounds_intersection_counts_corner_touch_as_hit) {
    renderer::Bounds3 box(renderer::Vec3(-1, -1, -1), renderer::Vec3(1, 1, 1));
    renderer::Ray corner_touch(renderer::Vec3(-2, -2, 1), renderer::Vec3(1, 1, 0));
    RENDER_CHECK(box.intersect(corner_touch, 0.001f, 1000.0f));
}

RENDER_TEST(test_image_invalid_dimensions_throw_invalid_argument) {
    bool threw_zero_width = false;
    try {
        renderer::Image image(0, 1);
    } catch (const std::invalid_argument&) {
        threw_zero_width = true;
    }
    RENDER_CHECK(threw_zero_width);

    bool threw_zero_height = false;
    try {
        renderer::Image image(1, 0);
    } catch (const std::invalid_argument&) {
        threw_zero_height = true;
    }
    RENDER_CHECK(threw_zero_height);

    bool threw_negative_width = false;
    try {
        renderer::Image image(-1, 1);
    } catch (const std::invalid_argument&) {
        threw_negative_width = true;
    }
    RENDER_CHECK(threw_negative_width);
}

RENDER_TEST(test_image_stores_gamma_corrected_pixels) {
    renderer::Image image(2, 1);
    image.set_pixel(0, 0, renderer::Color(1.0f, 0.25f, 0.0f));
    renderer::Rgb8 pixel = image.pixel_rgb8(0, 0);
    RENDER_CHECK(pixel.r == 255);
    RENDER_CHECK(pixel.g >= 135 && pixel.g <= 137);
    RENDER_CHECK(pixel.b == 0);
}

RENDER_TEST(test_image_and_framebuffer_bulk_pixel_assignment_validates_size) {
    const std::vector<renderer::Color> pixels{
        renderer::Color(1.0f, 0.0f, 0.0f),
        renderer::Color(0.0f, 1.0f, 0.0f)};

    renderer::Image image(2, 1);
    image.set_pixels(pixels);
    RENDER_CHECK(nearly_equal(image.pixel(0, 0).x(), 1.0f));
    RENDER_CHECK(nearly_equal(image.pixel(1, 0).y(), 1.0f));

    renderer::Framebuffer framebuffer(2, 1);
    framebuffer.set_pixels(pixels);
    RENDER_CHECK(nearly_equal(framebuffer.pixel(0, 0).x(), 1.0f));
    RENDER_CHECK(nearly_equal(framebuffer.pixel(1, 0).y(), 1.0f));

    bool image_threw = false;
    try {
        image.set_pixels(std::vector<renderer::Color>(1));
    } catch (const std::invalid_argument&) {
        image_threw = true;
    }
    RENDER_CHECK(image_threw);

    bool framebuffer_threw = false;
    try {
        framebuffer.set_pixels(std::vector<renderer::Color>(3));
    } catch (const std::invalid_argument&) {
        framebuffer_threw = true;
    }
    RENDER_CHECK(framebuffer_threw);
}

RENDER_TEST(test_to_rgb8_uses_standard_srgb_transfer_curve) {
    RENDER_CHECK(renderer::channel_to_rgb8(0.0031308f) == 10);
    RENDER_CHECK(renderer::channel_to_rgb8(0.5f) == 188);
}

RENDER_TEST(test_to_rgb8_sanitizes_non_finite_channels) {
    const renderer::Rgb8 nan_pixel = renderer::to_rgb8(
        renderer::Color(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f));
    RENDER_CHECK(nan_pixel.r == 0);

    const renderer::Rgb8 infinity_pixel = renderer::to_rgb8(
        renderer::Color(std::numeric_limits<float>::infinity(), 0.0f, 0.0f));
    RENDER_CHECK(infinity_pixel.r == 255);
}

RENDER_TEST(test_display_settings_apply_exposure_and_tone_mapping) {
    const renderer::Color source(0.25f, 0.5f, 2.0f);
    const renderer::DisplaySettings defaults;
    const renderer::Rgb8 default_pixel = renderer::to_display_rgb8(source, defaults);
    const renderer::Rgb8 legacy_pixel = renderer::to_rgb8(source);
    RENDER_CHECK(default_pixel.r == legacy_pixel.r);
    RENDER_CHECK(default_pixel.g == legacy_pixel.g);
    RENDER_CHECK(default_pixel.b == legacy_pixel.b);

    renderer::DisplaySettings exposed;
    exposed.exposure_ev = 1.0f;
    const renderer::Color doubled = renderer::apply_display_transform(source, exposed);
    RENDER_CHECK((doubled - source * 2.0f).cwiseAbs().maxCoeff() < 1e-6f);

    renderer::DisplaySettings reinhard;
    reinhard.tone_mapper = renderer::ToneMapper::Reinhard;
    const renderer::Color reinhard_value = renderer::apply_display_transform(
        renderer::Color::Constant(3.0f),
        reinhard);
    RENDER_CHECK((reinhard_value - renderer::Color::Constant(0.75f)).cwiseAbs().maxCoeff() < 1e-6f);

    renderer::DisplaySettings aces;
    aces.exposure_ev = 3.0f;
    aces.tone_mapper = renderer::ToneMapper::Aces;
    const renderer::Color aces_value = renderer::apply_display_transform(
        renderer::Color(0.0f, 1.0f, 10000.0f),
        aces);
    RENDER_CHECK(aces_value.allFinite());
    RENDER_CHECK(aces_value.minCoeff() >= 0.0f);
    RENDER_CHECK(aces_value.maxCoeff() <= 1.0f);

    renderer::Framebuffer framebuffer(1, 1);
    framebuffer.set_pixel(0, 0, renderer::Color::Constant(0.25f));
    const std::vector<std::uint8_t> original = framebuffer.to_rgba8();
    const std::vector<std::uint8_t> brighter = framebuffer.to_rgba8(exposed);
    RENDER_CHECK(original.size() == 4);
    RENDER_CHECK(brighter.size() == 4);
    RENDER_CHECK(brighter[0] > original[0]);
    RENDER_CHECK(brighter[3] == 255);
}

RENDER_TEST(test_viewer_ui_actions_classify_path_resets) {
    const renderer::CudaOpenGlInteropUiState interop_state;
    RENDER_CHECK(interop_state.status == "unavailable");
    RENDER_CHECK(interop_state.detail.empty());

    renderer::ViewerUiActions actions;
    actions.shader_reload_requested = true;
    RENDER_CHECK(!actions.resets_path_accumulation());
    actions = renderer::ViewerUiActions{};
    actions.shader_auto_reload_changed = true;
    RENDER_CHECK(!actions.resets_path_accumulation());

    actions.scene_changes = renderer::SceneChange::Lighting;
    RENDER_CHECK(actions.resets_path_accumulation());
    actions = renderer::ViewerUiActions{};
    actions.camera_parameters_changed = true;
    RENDER_CHECK(actions.resets_path_accumulation());
    actions = renderer::ViewerUiActions{};
    actions.render_scale_changed = true;
    RENDER_CHECK(actions.resets_path_accumulation());
    actions = renderer::ViewerUiActions{};
    actions.path_depth_changed = true;
    RENDER_CHECK(actions.resets_path_accumulation());
    actions = renderer::ViewerUiActions{};
    actions.path_roulette_changed = true;
    RENDER_CHECK(actions.resets_path_accumulation());
}

RENDER_TEST(test_orbit_camera_controller_zoom_and_orbit_change_camera) {
    renderer::Bounds3 bounds(renderer::Vec3(-1, 0, -1), renderer::Vec3(1, 2, 1));
    renderer::OrbitCameraController controller(bounds, 1.0f);
    renderer::Camera before = controller.camera();

    controller.orbit(0.5f, 0.25f);
    controller.zoom(-1.0f);
    renderer::Camera after = controller.camera();

    RENDER_CHECK((after.eye() - before.eye()).norm() > 0.001f);
    RENDER_CHECK(after.viewport_width() > 0.0f);
    controller.set_vertical_fov_degrees(60.0f);
    controller.set_distance(3.0f);
    RENDER_CHECK(nearly_equal(controller.vertical_fov_degrees(), 60.0f));
    RENDER_CHECK(nearly_equal(controller.distance(), 3.0f));
}

RENDER_TEST(test_orbit_camera_controller_horizontal_drag_tracks_scene_direction) {
    renderer::Bounds3 bounds(renderer::Vec3(-1, 0, -1), renderer::Vec3(1, 2, 1));
    renderer::OrbitCameraController controller(bounds, 1.0f);
    const float before_x = controller.camera().eye().x();

    controller.orbit(25.0f, 0.0f);
    const float after_x = controller.camera().eye().x();

    RENDER_CHECK(after_x < before_x);
}

RENDER_TEST(test_orbit_camera_controller_pan_moves_in_camera_plane) {
    renderer::Bounds3 bounds(renderer::Vec3(-1, 0, -1), renderer::Vec3(1, 2, 1));
    renderer::OrbitCameraController controller(bounds, 1.5f);
    controller.set_vertical_fov_degrees(60.0f);
    controller.set_distance(4.0f);
    const renderer::Camera before = controller.camera();
    constexpr float delta_x = 80.0f;
    constexpr float delta_y = 40.0f;
    constexpr float viewport_height = 800.0f;
    const float world_units_per_pixel =
        controller.distance() * before.viewport_height() / viewport_height;
    const renderer::Vec3 expected_translation =
        (-delta_x * before.right() + delta_y * before.up()) * world_units_per_pixel;

    controller.pan(delta_x, delta_y, viewport_height);
    const renderer::Camera after = controller.camera();

    RENDER_CHECK((after.eye() - before.eye() - expected_translation).norm() < 1e-5f);
    RENDER_CHECK((after.forward() - before.forward()).norm() < 1e-5f);
    RENDER_CHECK(nearly_equal(controller.distance(), 4.0f));
}

RENDER_TEST(test_free_camera_controller_looks_and_clamps_pitch) {
    const renderer::Camera initial(
        renderer::Vec3::Zero(),
        -renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitY(),
        45.0f,
        1.0f);
    renderer::FreeCameraController controller(initial, 1.0f, 2.0f);
    controller.set_vertical_fov_degrees(65.0f);
    controller.set_movement_speed(3.5f);
    RENDER_CHECK(nearly_equal(controller.vertical_fov_degrees(), 65.0f));
    RENDER_CHECK(nearly_equal(controller.movement_speed(), 3.5f));

    controller.look(10.0f, 0.0f);
    RENDER_CHECK(controller.camera().forward().x() > 0.09f);

    controller.look(0.0f, -10000.0f);
    const renderer::Camera pitched = controller.camera();
    RENDER_CHECK(pitched.forward().y() > 0.99f);
    RENDER_CHECK(pitched.forward().allFinite());
    RENDER_CHECK(pitched.right().allFinite());

    controller.set_aspect_ratio(2.0f);
    const renderer::Camera resized = controller.camera();
    RENDER_CHECK(nearly_equal(
        resized.viewport_width() / resized.viewport_height(),
        2.0f,
        1e-5f));
}

RENDER_TEST(test_free_camera_controller_moves_in_camera_and_world_directions) {
    const renderer::Camera initial(
        renderer::Vec3::Zero(),
        -renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitY(),
        45.0f,
        1.0f);
    renderer::FreeCameraController controller(initial, 1.0f, 2.0f);

    RENDER_CHECK(controller.move(1.0f, 0.0f, 0.0f, 0.5f));
    RENDER_CHECK((controller.camera().eye() - renderer::Vec3(0.0f, 0.0f, -1.0f)).norm() < 1e-5f);
    RENDER_CHECK(controller.move(0.0f, 1.0f, 0.0f, 0.5f));
    RENDER_CHECK((controller.camera().eye() - renderer::Vec3(1.0f, 0.0f, -1.0f)).norm() < 1e-5f);
    RENDER_CHECK(controller.move(0.0f, 0.0f, 1.0f, 0.5f));
    RENDER_CHECK((controller.camera().eye() - renderer::Vec3(1.0f, 1.0f, -1.0f)).norm() < 1e-5f);
    RENDER_CHECK(controller.move(0.0f, 0.0f, -1.0f, 0.5f));
    RENDER_CHECK((controller.camera().eye() - renderer::Vec3(1.0f, 0.0f, -1.0f)).norm() < 1e-5f);
    RENDER_CHECK(!controller.move(0.0f, 0.0f, 0.0f, 1.0f));

    renderer::FreeCameraController diagonal(initial, 1.0f, 2.0f);
    RENDER_CHECK(diagonal.move(1.0f, 1.0f, 0.0f, 0.5f));
    RENDER_CHECK(nearly_equal(diagonal.camera().eye().norm(), 1.0f, 1e-5f));
}

RENDER_TEST(test_camera_mode_switch_preserves_pose) {
    const renderer::Bounds3 bounds(
        renderer::Vec3(-1.0f, 0.0f, -1.0f),
        renderer::Vec3(1.0f, 2.0f, 1.0f));
    renderer::OrbitCameraController orbit(bounds, 1.0f);
    orbit.orbit(25.0f, -10.0f);
    orbit.zoom(1.0f);
    const renderer::Camera orbit_pose = orbit.camera();

    renderer::FreeCameraController free_camera(orbit_pose, 1.0f, 1.0f);
    const renderer::Camera free_pose = free_camera.camera();
    RENDER_CHECK((free_pose.eye() - orbit_pose.eye()).norm() < 1e-5f);
    RENDER_CHECK((free_pose.forward() - orbit_pose.forward()).norm() < 1e-5f);

    free_camera.look(12.0f, -7.0f);
    RENDER_CHECK(free_camera.move(1.0f, -1.0f, 1.0f, 0.25f));
    const renderer::Camera moved_free_pose = free_camera.camera();
    orbit.set_camera(moved_free_pose);
    const renderer::Camera restored_orbit_pose = orbit.camera();
    RENDER_CHECK((restored_orbit_pose.eye() - moved_free_pose.eye()).norm() < 1e-5f);
    RENDER_CHECK((restored_orbit_pose.forward() - moved_free_pose.forward()).norm() < 1e-5f);
}

RENDER_TEST(test_frame_rate_counter_reports_window_average) {
    renderer::FrameRateCounter counter(0.25f);
    RENDER_CHECK(!counter.snapshot().valid);
    RENDER_CHECK(!counter.tick(0.10f));

    RENDER_CHECK(counter.tick(0.15f));
    const renderer::FrameRateSnapshot snapshot = counter.snapshot();
    RENDER_CHECK(snapshot.valid);
    RENDER_CHECK(snapshot.frames == 2);
    RENDER_CHECK(nearly_equal(snapshot.frames_per_second, 8.0f));
    RENDER_CHECK(nearly_equal(snapshot.milliseconds_per_frame, 125.0f));

    counter.reset();
    RENDER_CHECK(!counter.snapshot().valid);
}

RENDER_TEST(test_viewer_title_format_includes_fps_without_progressive_samples) {
    renderer::FrameRateSnapshot warming_up;
    const std::string opengl_warming_title = renderer::format_viewer_title(
        renderer::InteractiveRenderMode::OpenGl,
        warming_up,
        0);
    RENDER_CHECK(opengl_warming_title.find("opengl") != std::string::npos);
    RENDER_CHECK(opengl_warming_title.find("FPS --") != std::string::npos);
    RENDER_CHECK(opengl_warming_title.find("spp") == std::string::npos);

    renderer::FrameRateSnapshot snapshot;
    snapshot.valid = true;
    snapshot.frames = 3;
    snapshot.frames_per_second = 60.0f;
    snapshot.milliseconds_per_frame = 16.666f;

    const std::string path_title = renderer::format_viewer_title(
        renderer::InteractiveRenderMode::Path,
        snapshot,
        12);
    RENDER_CHECK(path_title.find("rtrt") != std::string::npos);
    RENDER_CHECK(path_title.find("60.0 FPS") != std::string::npos);
    RENDER_CHECK(path_title.find("16.7 ms") != std::string::npos);
    RENDER_CHECK(path_title.find("spp") == std::string::npos);

    const std::string opengl_title = renderer::format_viewer_title(
        renderer::InteractiveRenderMode::OpenGl,
        snapshot,
        99);
    RENDER_CHECK(opengl_title.find("opengl") != std::string::npos);
    RENDER_CHECK(opengl_title.find("spp") == std::string::npos);
}

RENDER_TEST(test_interactive_mode_catalog_contains_opengl_and_rtrt) {
    const auto& modes = renderer::interactive_render_modes();
    RENDER_CHECK(modes.size() == 2);
    RENDER_CHECK(modes[0].mode == renderer::InteractiveRenderMode::OpenGl);
    RENDER_CHECK(modes[0].hotkey == 1);
    RENDER_CHECK(modes[1].mode == renderer::InteractiveRenderMode::Path);
    RENDER_CHECK(modes[1].hotkey == 2);
    RENDER_CHECK(
        renderer::interactive_render_mode_from_hotkey(1) ==
        renderer::InteractiveRenderMode::OpenGl);
    RENDER_CHECK(
        renderer::interactive_render_mode_from_hotkey(2) ==
        renderer::InteractiveRenderMode::Path);
    RENDER_CHECK(
        renderer::ViewerUiState{}.mode ==
        renderer::InteractiveRenderMode::Rtrt);
    for (const char* removed_mode : {"raster", "ray"}) {
        bool rejected = false;
        try {
            (void)renderer::parse_interactive_render_mode(removed_mode);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        RENDER_CHECK(rejected);
    }
}

RENDER_TEST(test_viewer_selection_keeps_active_object_selected) {
    renderer::ViewerUiState state;
    renderer::select_viewer_object(state, 10, false);
    RENDER_CHECK(state.selected_objects == std::vector<renderer::ObjectId>{10});
    RENDER_CHECK(state.active_object == 10);

    renderer::select_viewer_object(state, 20, true);
    RENDER_CHECK(
        state.selected_objects ==
        std::vector<renderer::ObjectId>({10, 20}));
    RENDER_CHECK(state.active_object == 20);

    // Ctrl-clicking the active object deselects it. The old viewport path left
    // active_object pointing at 20, so the gizmo was drawn for an unselected
    // light and its move/rotate/scale delta was applied elsewhere.
    renderer::select_viewer_object(state, 20, true);
    RENDER_CHECK(state.selected_objects == std::vector<renderer::ObjectId>{10});
    RENDER_CHECK(state.active_object == 10);

    renderer::select_viewer_object(state, 10, true);
    RENDER_CHECK(state.selected_objects.empty());
    RENDER_CHECK(state.active_object == renderer::kInvalidObjectId);
}

RENDER_TEST(test_framebuffer_clear_set_and_rgba8_conversion) {
    renderer::Framebuffer framebuffer(2, 1);
    framebuffer.clear(renderer::Color(0.25f, 0.0f, 1.0f));
    framebuffer.set_pixel(1, 0, renderer::Color(1.0f, 0.25f, 0.0f));

    RENDER_CHECK(framebuffer.width() == 2);
    RENDER_CHECK(framebuffer.height() == 1);
    RENDER_CHECK(nearly_equal(framebuffer.pixel(0, 0).z(), 1.0f));

    const std::vector<std::uint8_t> rgba = framebuffer.to_rgba8();
    RENDER_CHECK(rgba.size() == 8);
    RENDER_CHECK(rgba[3] == 255);
    RENDER_CHECK(rgba[4] == 255);
    RENDER_CHECK(rgba[7] == 255);

    const std::vector<float> rgba32f = framebuffer.to_rgba32f();
    RENDER_CHECK(rgba32f.size() == 8);
    RENDER_CHECK(nearly_equal(rgba32f[0], 0.25f));
    RENDER_CHECK(nearly_equal(rgba32f[2], 1.0f));
    RENDER_CHECK(nearly_equal(rgba32f[3], 1.0f));
    RENDER_CHECK(nearly_equal(rgba32f[4], 1.0f));
    RENDER_CHECK(nearly_equal(rgba32f[5], 0.25f));
    RENDER_CHECK(nearly_equal(rgba32f[7], 1.0f));
}

RENDER_TEST(test_sphere_intersection) {
    renderer::Material material;
    material.base_color = renderer::Color(1, 0, 0);
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0f, 0);

    renderer::Ray ray(renderer::Vec3(0, 0, -5), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(sphere.intersect(ray, 0.001f, 1000.0f, hit));
    RENDER_CHECK(nearly_equal(hit.t, 4.0f));
    RENDER_CHECK(hit.position.allFinite());
    RENDER_CHECK(hit.shading_normal.allFinite());
    RENDER_CHECK(hit.tangent.allFinite());
    RENDER_CHECK(hit.bitangent.allFinite());
    RENDER_CHECK(nearly_equal(hit.position.z(), -1.0f));
    RENDER_CHECK(nearly_equal(hit.shading_normal.norm(), 1.0f));
    RENDER_CHECK(hit.material_id == 0);
}

RENDER_TEST(test_sphere_rejects_zero_direction_ray) {
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0f, 0);
    renderer::Ray ray(renderer::Vec3(0, 0, -5), renderer::Vec3(0, 0, 0));
    renderer::HitRecord hit;
    RENDER_CHECK(!sphere.intersect(ray, 0.001f, 1000.0f, hit));
}

RENDER_TEST(test_sphere_invalid_radius_throws) {
    bool threw_zero_radius = false;
    try {
        renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 0.0f, 0);
    } catch (const std::invalid_argument&) {
        threw_zero_radius = true;
    }
    RENDER_CHECK(threw_zero_radius);

    bool threw_negative_radius = false;
    try {
        renderer::Sphere sphere(renderer::Vec3(0, 0, 0), -1.0f, 0);
    } catch (const std::invalid_argument&) {
        threw_negative_radius = true;
    }
    RENDER_CHECK(threw_negative_radius);

    bool threw_nan_radius = false;
    try {
        renderer::Sphere sphere(renderer::Vec3(0, 0, 0), std::numeric_limits<float>::quiet_NaN(), 0);
    } catch (const std::invalid_argument&) {
        threw_nan_radius = true;
    }
    RENDER_CHECK(threw_nan_radius);

    bool threw_infinite_radius = false;
    try {
        renderer::Sphere sphere(renderer::Vec3(0, 0, 0), std::numeric_limits<float>::infinity(), 0);
    } catch (const std::invalid_argument&) {
        threw_infinite_radius = true;
    }
    RENDER_CHECK(threw_infinite_radius);
}

RENDER_TEST(test_sphere_invalid_center_throws) {
    bool threw_nan_center = false;
    try {
        renderer::Sphere sphere(
            renderer::Vec3(std::numeric_limits<float>::quiet_NaN(), 0, 0),
            1.0f,
            0);
    } catch (const std::invalid_argument&) {
        threw_nan_center = true;
    }
    RENDER_CHECK(threw_nan_center);

    bool threw_infinite_center = false;
    try {
        renderer::Sphere sphere(
            renderer::Vec3(std::numeric_limits<float>::infinity(), 0, 0),
            1.0f,
            0);
    } catch (const std::invalid_argument&) {
        threw_infinite_center = true;
    }
    RENDER_CHECK(threw_infinite_center);
}

RENDER_TEST(test_sphere_rejects_non_finite_direction_rays) {
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0f, 0);

    renderer::HitRecord nan_hit;
    renderer::Ray nan_ray(
        renderer::Vec3(0, 0, -5),
        renderer::Vec3(0, 0, std::numeric_limits<float>::quiet_NaN()));
    RENDER_CHECK(!sphere.intersect(nan_ray, 0.001f, 1000.0f, nan_hit));

    renderer::HitRecord infinite_hit;
    renderer::Ray infinite_ray(
        renderer::Vec3(0, 0, -5),
        renderer::Vec3(0, 0, std::numeric_limits<float>::infinity()));
    RENDER_CHECK(!sphere.intersect(infinite_ray, 0.001f, 1000.0f, infinite_hit));
}

RENDER_TEST(test_sphere_rejects_non_finite_origin_rays) {
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0f, 0);

    renderer::HitRecord nan_hit;
    renderer::Ray nan_ray(
        renderer::Vec3(std::numeric_limits<float>::quiet_NaN(), 0, -5),
        renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!sphere.intersect(nan_ray, 0.001f, 1000.0f, nan_hit));

    renderer::HitRecord infinite_hit;
    renderer::Ray infinite_ray(
        renderer::Vec3(std::numeric_limits<float>::infinity(), 0, -5),
        renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!sphere.intersect(infinite_ray, 0.001f, 1000.0f, infinite_hit));
}

RENDER_TEST(test_sphere_inside_ray_reports_back_face) {
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0f, 1);
    renderer::Ray ray(renderer::Vec3(0, 0, 0), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(sphere.intersect(ray, 0.001f, 1000.0f, hit));
    RENDER_CHECK(!hit.front_face);
    RENDER_CHECK(nearly_equal(hit.t, 1.0f));
    RENDER_CHECK(nearly_equal(hit.shading_normal.z(), -1.0f));
}

RENDER_TEST(test_sphere_bounds_include_center_and_radius) {
    renderer::Sphere sphere(renderer::Vec3(1, 2, 3), 2.0f, 0);
    const renderer::Bounds3 bounds = sphere.bounds();
    RENDER_CHECK(nearly_equal(bounds.min.x(), -1.0f));
    RENDER_CHECK(nearly_equal(bounds.min.y(), 0.0f));
    RENDER_CHECK(nearly_equal(bounds.min.z(), 1.0f));
    RENDER_CHECK(nearly_equal(bounds.max.x(), 3.0f));
    RENDER_CHECK(nearly_equal(bounds.max.y(), 4.0f));
    RENDER_CHECK(nearly_equal(bounds.max.z(), 5.0f));
}

RENDER_TEST(test_triangle_intersection) {
    renderer::Triangle tri(
        renderer::Vec3(-1, 0, 0),
        renderer::Vec3(1, 0, 0),
        renderer::Vec3(0, 1, 0),
        2);

    renderer::Ray ray(renderer::Vec3(0, 0.25f, -2), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(tri.intersect(ray, 0.001f, 1000.0f, hit));
    RENDER_CHECK(nearly_equal(hit.position.x(), 0.0f));
    RENDER_CHECK(nearly_equal(hit.position.y(), 0.25f));
    RENDER_CHECK(hit.material_id == 2);
}

RENDER_TEST(test_triangle_interpolates_shading_normal_separately_from_geometry) {
    const renderer::Triangle triangle(
        renderer::TriangleVertex{
            renderer::Vec3(-1.0f, -1.0f, -1.0f),
            renderer::Vec2(0.0f, 0.0f),
            renderer::Vec3(0.0f, 1.0f, 1.0f).normalized(),
            true},
        renderer::TriangleVertex{
            renderer::Vec3(1.0f, -1.0f, -1.0f),
            renderer::Vec2(1.0f, 0.0f),
            renderer::Vec3(1.0f, 0.0f, 1.0f).normalized(),
            true},
        renderer::TriangleVertex{
            renderer::Vec3(0.0f, 1.0f, -1.0f),
            renderer::Vec2(0.5f, 1.0f),
            renderer::Vec3(0.0f, 0.0f, 1.0f),
            true},
        0);

    renderer::HitRecord hit;
    const renderer::Ray ray(renderer::Vec3(0.0f, 0.0f, 0.0f), renderer::Vec3(0.0f, 0.0f, -1.0f));
    RENDER_CHECK(triangle.intersect(ray, 1e-6f, 10.0f, hit));
    RENDER_CHECK(hit.geometric_normal.allFinite());
    RENDER_CHECK(hit.shading_normal.allFinite());
    RENDER_CHECK(hit.geometric_normal.dot(renderer::Vec3(0.0f, 0.0f, 1.0f)) > 0.999f);
    RENDER_CHECK(hit.shading_normal.dot(hit.geometric_normal) > 0.0f);
    RENDER_CHECK((hit.shading_normal - hit.geometric_normal).norm() > 0.01f);
}

RENDER_TEST(test_triangle_invalid_vertices_throw) {
    bool threw_nan_vertex = false;
    try {
        renderer::Triangle tri(
            renderer::Vec3(std::numeric_limits<float>::quiet_NaN(), 0, 0),
            renderer::Vec3(1, 0, 0),
            renderer::Vec3(0, 1, 0),
            0);
    } catch (const std::invalid_argument&) {
        threw_nan_vertex = true;
    }
    RENDER_CHECK(threw_nan_vertex);

    bool threw_infinite_vertex = false;
    try {
        renderer::Triangle tri(
            renderer::Vec3(0, 0, 0),
            renderer::Vec3(std::numeric_limits<float>::infinity(), 0, 0),
            renderer::Vec3(0, 1, 0),
            0);
    } catch (const std::invalid_argument&) {
        threw_infinite_vertex = true;
    }
    RENDER_CHECK(threw_infinite_vertex);
}

RENDER_TEST(test_triangle_rejects_non_finite_rays) {
    renderer::Triangle tri(
        renderer::Vec3(-1, 0, 0),
        renderer::Vec3(1, 0, 0),
        renderer::Vec3(0, 1, 0),
        2);

    renderer::HitRecord nan_origin_hit;
    renderer::Ray nan_origin_ray(
        renderer::Vec3(std::numeric_limits<float>::quiet_NaN(), 0.25f, -2.0f),
        renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!tri.intersect(nan_origin_ray, 0.001f, 1000.0f, nan_origin_hit));

    renderer::HitRecord infinite_direction_hit;
    renderer::Ray infinite_direction_ray(
        renderer::Vec3(0, 0.25f, -2),
        renderer::Vec3(0, 0, std::numeric_limits<float>::infinity()));
    RENDER_CHECK(!tri.intersect(infinite_direction_ray, 0.001f, 1000.0f, infinite_direction_hit));
}

RENDER_TEST(test_triangle_back_side_hit_reports_back_face) {
    renderer::Triangle tri(
        renderer::Vec3(-1, 0, 0),
        renderer::Vec3(1, 0, 0),
        renderer::Vec3(0, 1, 0),
        2);

    renderer::Ray ray(renderer::Vec3(0, 0.25f, -2), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(tri.intersect(ray, 0.001f, 1000.0f, hit));
    RENDER_CHECK(!hit.front_face);
    RENDER_CHECK(hit.shading_normal.dot(ray.direction) < 0.0f);
}

RENDER_TEST(test_triangle_boundary_hits_succeed) {
    renderer::Triangle tri(
        renderer::Vec3(-1, 0, 0),
        renderer::Vec3(1, 0, 0),
        renderer::Vec3(0, 1, 0),
        2);

    renderer::HitRecord vertex_hit;
    renderer::Ray vertex_ray(renderer::Vec3(-1, 0, -2), renderer::Vec3(0, 0, 1));
    RENDER_CHECK(tri.intersect(vertex_ray, 0.001f, 1000.0f, vertex_hit));
    RENDER_CHECK(nearly_equal(vertex_hit.position.x(), -1.0f));
    RENDER_CHECK(nearly_equal(vertex_hit.position.y(), 0.0f));

    renderer::HitRecord edge_hit;
    renderer::Ray edge_ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 0, 1));
    RENDER_CHECK(tri.intersect(edge_ray, 0.001f, 1000.0f, edge_hit));
    RENDER_CHECK(nearly_equal(edge_hit.position.x(), 0.0f));
    RENDER_CHECK(nearly_equal(edge_hit.position.y(), 0.0f));
}

RENDER_TEST(test_degenerate_triangle_misses) {
    renderer::Triangle tri(
        renderer::Vec3(0, 0, 0),
        renderer::Vec3(1, 1, 1),
        renderer::Vec3(2, 2, 2),
        2);

    renderer::Ray ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(!tri.intersect(ray, 0.001f, 1000.0f, hit));
}

RENDER_TEST(test_triangle_degenerate_normals_and_uvs_use_finite_fallbacks) {
    const renderer::Triangle triangle(
        renderer::TriangleVertex{
            renderer::Vec3(-1.0f, -1.0f, -1.0f),
            renderer::Vec2::Zero(),
            renderer::Vec3::Zero(),
            true},
        renderer::TriangleVertex{
            renderer::Vec3(1.0f, -1.0f, -1.0f),
            renderer::Vec2::Zero(),
            renderer::Vec3::Zero(),
            true},
        renderer::TriangleVertex{
            renderer::Vec3(0.0f, 1.0f, -1.0f),
            renderer::Vec2::Zero(),
            renderer::Vec3::Zero(),
            true},
        0);

    renderer::HitRecord hit;
    const renderer::Ray ray(
        renderer::Vec3::Zero(),
        renderer::Vec3(0.0f, 0.0f, -1.0f));
    RENDER_CHECK(triangle.intersect(ray, 1e-6f, 10.0f, hit));
    RENDER_CHECK(!hit.has_valid_uv_basis);
    RENDER_CHECK(hit.position.allFinite());
    RENDER_CHECK(hit.geometric_normal.allFinite());
    RENDER_CHECK(hit.shading_normal.allFinite());
    RENDER_CHECK(hit.tangent.allFinite());
    RENDER_CHECK(hit.bitangent.allFinite());
    RENDER_CHECK(hit.uv.allFinite());
}

RENDER_TEST(test_small_triangle_intersection_remains_valid) {
    const renderer::Triangle triangle(
        renderer::Vec3(-5e-6f, -5e-6f, -1.0f),
        renderer::Vec3(5e-6f, -5e-6f, -1.0f),
        renderer::Vec3(-5e-6f, 5e-6f, -1.0f),
        0);
    renderer::HitRecord hit;
    const renderer::Ray ray(renderer::Vec3::Zero(), renderer::Vec3(0.0f, 0.0f, -1.0f));

    RENDER_CHECK(triangle.intersect(ray, 1e-6f, 10.0f, hit));
    RENDER_CHECK(hit.position.allFinite());
    RENDER_CHECK(hit.shading_normal.allFinite());
}

RENDER_TEST(test_small_nonzero_uv_basis_remains_valid) {
    const renderer::Triangle triangle(
        renderer::Vec3(-1.0f, -1.0f, -1.0f),
        renderer::Vec3(1.0f, -1.0f, -1.0f),
        renderer::Vec3(-1.0f, 1.0f, -1.0f),
        0,
        renderer::Vec2::Zero(),
        renderer::Vec2(1e-5f, 0.0f),
        renderer::Vec2(0.0f, 1e-5f));
    renderer::HitRecord hit;
    const renderer::Ray ray(renderer::Vec3::Zero(), renderer::Vec3(0.0f, 0.0f, -1.0f));

    RENDER_CHECK(triangle.intersect(ray, 1e-6f, 10.0f, hit));
    RENDER_CHECK(hit.has_valid_uv_basis);
    RENDER_CHECK(hit.tangent.allFinite());
    RENDER_CHECK(hit.bitangent.allFinite());
}

bool brute_force_triangle_intersect(
    const std::vector<renderer::Triangle>& tris,
    const renderer::Ray& ray,
    float t_min,
    float t_max,
    renderer::HitRecord& closest_hit) {
    bool hit_anything = false;
    float closest_t = t_max;
    for (const renderer::Triangle& tri : tris) {
        renderer::HitRecord hit;
        if (tri.intersect(ray, t_min, closest_t, hit)) {
            hit_anything = true;
            closest_t = hit.t;
            closest_hit = hit;
        }
    }
    return hit_anything;
}

void check_bvh_matches_bruteforce(
    const renderer::Bvh& bvh,
    const std::vector<renderer::Triangle>& tris,
    const renderer::Ray& ray) {
    renderer::HitRecord brute_force_hit;
    renderer::HitRecord bvh_hit;
    const bool brute_force_found =
        brute_force_triangle_intersect(tris, ray, 0.001f, 1000.0f, brute_force_hit);
    const bool bvh_found = bvh.intersect(ray, 0.001f, 1000.0f, bvh_hit);
    RENDER_CHECK(bvh_found == brute_force_found);
    if (!brute_force_found) {
        return;
    }

    RENDER_CHECK(nearly_equal(static_cast<float>(bvh_hit.t), static_cast<float>(brute_force_hit.t)));
    RENDER_CHECK(bvh_hit.material_id == brute_force_hit.material_id);
    RENDER_CHECK(nearly_equal(bvh_hit.position.x(), brute_force_hit.position.x()));
    RENDER_CHECK(nearly_equal(bvh_hit.position.y(), brute_force_hit.position.y()));
    RENDER_CHECK(nearly_equal(bvh_hit.position.z(), brute_force_hit.position.z()));
    RENDER_CHECK(bvh_hit.shading_normal.dot(brute_force_hit.shading_normal) > 0.999f);
}

RENDER_TEST(test_empty_bvh_has_no_nodes_or_hits) {
    std::vector<renderer::Triangle> tris;

    renderer::Bvh bvh;
    bvh.build(tris);

    RENDER_CHECK(bvh.nodes().empty());
    RENDER_CHECK(bvh.primitive_indices().empty());

    renderer::HitRecord hit;
    RENDER_CHECK(!bvh.intersect(
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 0, 1)),
        0.001f,
        1000.0f,
        hit));
}

RENDER_TEST(test_bvh_matches_bruteforce_triangle_hit) {
    std::vector<renderer::Triangle> tris;
    tris.emplace_back(renderer::Vec3(-1, 0, 0), renderer::Vec3(1, 0, 0), renderer::Vec3(0, 1, 0), 0);
    tris.emplace_back(renderer::Vec3(-1, 0, 5), renderer::Vec3(1, 0, 5), renderer::Vec3(0, 1, 5), 0);

    renderer::Bvh bvh;
    bvh.build(tris);

    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(0, 0.25f, -2), renderer::Vec3(0, 0, 1)));
    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(3, 3, -2), renderer::Vec3(0, 0, 1)));
}

RENDER_TEST(test_bvh_splits_and_traverses_interior_nodes) {
    std::vector<renderer::Triangle> tris;
    for (int i = 0; i < 6; ++i) {
        const float x = static_cast<float>(i) * 3.0f;
        tris.emplace_back(
            renderer::Vec3(x - 1.0f, 0, 0),
            renderer::Vec3(x + 1.0f, 0, 0),
            renderer::Vec3(x, 1, 0),
            i);
    }

    renderer::Bvh bvh;
    bvh.build(tris);

    const std::vector<renderer::BvhNode>& nodes = bvh.nodes();
    RENDER_CHECK(nodes.size() > 1);
    RENDER_CHECK(!nodes[0].is_leaf());
    RENDER_CHECK(nodes[0].left >= 0);
    RENDER_CHECK(nodes[0].right >= 0);
    RENDER_CHECK(static_cast<std::size_t>(nodes[0].left) < nodes.size());
    RENDER_CHECK(static_cast<std::size_t>(nodes[0].right) < nodes.size());

    std::vector<int> sorted_indices = bvh.primitive_indices();
    std::sort(sorted_indices.begin(), sorted_indices.end());
    RENDER_CHECK(sorted_indices.size() == tris.size());
    for (std::size_t i = 0; i < sorted_indices.size(); ++i) {
        RENDER_CHECK(sorted_indices[i] == static_cast<int>(i));
    }

    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(0, 0.25f, -2), renderer::Vec3(0, 0, 1)));
    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(6, 0.25f, -2), renderer::Vec3(0, 0, 1)));
    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(15, 0.25f, -2), renderer::Vec3(0, 0, 1)));
    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(1.5f, 0.25f, -2), renderer::Vec3(0, 0, 1)));
}

RENDER_TEST(test_float_bvh_matches_bruteforce_at_large_coordinates) {
    constexpr float center = 100000.0f;
    std::vector<renderer::Triangle> tris;
    for (int i = 0; i < 6; ++i) {
        const float x = center + static_cast<float>(i) * 32.0f;
        tris.emplace_back(
            renderer::Vec3(x - 8.0f, center - 8.0f, center),
            renderer::Vec3(x + 8.0f, center - 8.0f, center),
            renderer::Vec3(x, center + 8.0f, center),
            20 + i);
    }

    renderer::Bvh bvh;
    bvh.build(tris);

    const auto compare = [&bvh, &tris](const renderer::Ray& ray) {
        renderer::HitRecord brute_force_hit;
        renderer::HitRecord bvh_hit;
        const bool brute_force_found =
            brute_force_triangle_intersect(tris, ray, 0.0f, 1000.0f, brute_force_hit);
        const bool bvh_found = bvh.intersect(ray, 0.0f, 1000.0f, bvh_hit);
        RENDER_CHECK(bvh_found == brute_force_found);
        if (bvh_found) {
            RENDER_CHECK(bvh_hit.material_id == brute_force_hit.material_id);
            RENDER_CHECK(std::abs(bvh_hit.t - brute_force_hit.t) <= 1e-2f);
            RENDER_CHECK(bvh_hit.shading_normal.dot(brute_force_hit.shading_normal) > 0.999f);
        }
    };

    compare(renderer::Ray(
        renderer::Vec3(center, center, center - 100.0f),
        renderer::Vec3::UnitZ()));
    compare(renderer::Ray(
        renderer::Vec3(center + 16.0f, center + 24.0f, center - 100.0f),
        renderer::Vec3::UnitZ()));
}

RENDER_TEST(test_checker_texture_is_deterministic_for_positive_and_negative_coordinates) {
    renderer::CheckerTexture texture;
    texture.even = renderer::Color(1, 0, 0);
    texture.odd = renderer::Color(0, 1, 0);
    texture.scale = 1.0f;

    const renderer::Color positive =
        texture.sample(renderer::Vec2::Zero(), renderer::Vec3(0.25f, 0.25f, 0.25f));
    RENDER_CHECK(nearly_equal(positive.x(), 1.0f));
    RENDER_CHECK(nearly_equal(positive.y(), 0.0f));
    RENDER_CHECK(nearly_equal(positive.z(), 0.0f));

    const renderer::Color negative =
        texture.sample(renderer::Vec2::Zero(), renderer::Vec3(-0.25f, 0.25f, 0.25f));
    RENDER_CHECK(nearly_equal(negative.x(), 0.0f));
    RENDER_CHECK(nearly_equal(negative.y(), 1.0f));
    RENDER_CHECK(nearly_equal(negative.z(), 0.0f));
}

RENDER_TEST(test_builtin_scene_contains_renderable_geometry) {
    renderer::Scene scene = renderer::make_gradient_sphere_scene();
    RENDER_CHECK(!scene.materials.empty());
    RENDER_CHECK(!scene.spheres.empty());
}

bool has_material_type(const renderer::Scene& scene, renderer::MaterialType type) {
    return std::any_of(
        scene.materials.begin(),
        scene.materials.end(),
        [type](const renderer::Material& material) {
            return material.type == type;
        });
}

bool has_red_like_material(const renderer::Scene& scene) {
    return std::any_of(
        scene.materials.begin(),
        scene.materials.end(),
        [](const renderer::Material& material) {
            return material.base_color.x() > material.base_color.y() &&
                   material.base_color.x() > material.base_color.z();
        });
}

bool has_green_like_material(const renderer::Scene& scene) {
    return std::any_of(
        scene.materials.begin(),
        scene.materials.end(),
        [](const renderer::Material& material) {
            return material.base_color.y() > material.base_color.x() &&
                   material.base_color.y() > material.base_color.z();
        });
}

bool has_white_diffuse_material(const renderer::Scene& scene) {
    return std::any_of(
        scene.materials.begin(),
        scene.materials.end(),
        [](const renderer::Material& material) {
            return material.type == renderer::MaterialType::Diffuse &&
                   material.base_color.x() > 0.5f &&
                   material.base_color.y() > 0.5f &&
                   material.base_color.z() > 0.5f;
        });
}

bool has_nonzero_emissive_material(const renderer::Scene& scene) {
    return std::any_of(
        scene.materials.begin(),
        scene.materials.end(),
        [](const renderer::Material& material) {
            return material.type == renderer::MaterialType::Emissive &&
                   material.emission.squaredNorm() > 0.0f;
        });
}

bool material_id_in_range(const renderer::Scene& scene, int material_id) {
    return material_id >= 0 &&
           static_cast<std::size_t>(material_id) < scene.materials.size();
}

bool intersect_closest_sphere(const renderer::Scene& scene, const renderer::Ray& ray, renderer::HitRecord& closest_hit) {
    bool hit_anything = false;
    float closest_t = 1000.0f;
    for (const renderer::Sphere& sphere : scene.spheres) {
        renderer::HitRecord hit;
        if (sphere.intersect(ray, 0.001f, closest_t, hit)) {
            hit_anything = true;
            closest_t = hit.t;
            closest_hit = hit;
        }
    }
    return hit_anything;
}

bool intersect_closest_triangle(const renderer::Scene& scene, const renderer::Ray& ray, renderer::HitRecord& closest_hit) {
    bool hit_anything = false;
    float closest_t = 1000.0f;
    for (const renderer::Triangle& triangle : scene.triangles) {
        renderer::HitRecord hit;
        if (triangle.intersect(ray, 0.001f, closest_t, hit)) {
            hit_anything = true;
            closest_t = hit.t;
            closest_hit = hit;
        }
    }
    return hit_anything;
}

void check_sphere_hit_material_in_range(const renderer::Scene& scene, const renderer::Ray& ray) {
    renderer::HitRecord hit;
    RENDER_CHECK(intersect_closest_sphere(scene, ray, hit));
    RENDER_CHECK(material_id_in_range(scene, hit.material_id));
}

void check_triangle_hit_material_in_range(const renderer::Scene& scene, const renderer::Ray& ray) {
    renderer::HitRecord hit;
    RENDER_CHECK(intersect_closest_triangle(scene, ray, hit));
    RENDER_CHECK(material_id_in_range(scene, hit.material_id));
}

RENDER_TEST(test_triangle_scene_contains_triangle_and_light) {
    renderer::Scene scene = renderer::make_triangle_scene();
    RENDER_CHECK(!scene.materials.empty());
    RENDER_CHECK(!scene.triangles.empty());
    RENDER_CHECK(!scene.point_lights.empty() || !scene.directional_lights.empty());
}

RENDER_TEST(test_mirror_spheres_scene_contains_metal_sphere_and_point_light) {
    renderer::Scene scene = renderer::make_mirror_spheres_scene();
    RENDER_CHECK(scene.spheres.size() >= 2);
    RENDER_CHECK(has_material_type(scene, renderer::MaterialType::Metal));
    RENDER_CHECK(!scene.point_lights.empty());
}

RENDER_TEST(test_cornell_box_scene_contains_walls_and_expected_materials) {
    renderer::Scene scene = renderer::make_cornell_box_scene();
    RENDER_CHECK(scene.triangles.size() >= 12);
    RENDER_CHECK(has_red_like_material(scene));
    RENDER_CHECK(has_green_like_material(scene));
    RENDER_CHECK(has_white_diffuse_material(scene));
    RENDER_CHECK(has_nonzero_emissive_material(scene));
}

RENDER_TEST(test_builtin_scene_probe_material_ids_are_in_range) {
    renderer::Scene gradient_scene = renderer::make_gradient_sphere_scene();
    check_sphere_hit_material_in_range(
        gradient_scene,
        renderer::Ray(renderer::Vec3(0, 0, 0), renderer::Vec3(0, 0, -1)));

    renderer::Scene triangle_scene = renderer::make_triangle_scene();
    check_triangle_hit_material_in_range(
        triangle_scene,
        renderer::Ray(renderer::Vec3(0, 0, 0), renderer::Vec3(0, 0, -1)));

    renderer::Scene mirror_scene = renderer::make_mirror_spheres_scene();
    check_sphere_hit_material_in_range(
        mirror_scene,
        renderer::Ray(renderer::Vec3(0, 0, 0), renderer::Vec3(0, 0, -1)));
    check_sphere_hit_material_in_range(
        mirror_scene,
        renderer::Ray(renderer::Vec3(2, 0, -1), renderer::Vec3(0, -1, 0)));

    renderer::Scene cornell_scene = renderer::make_cornell_box_scene();
    check_triangle_hit_material_in_range(
        cornell_scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, -1, 0)));
    check_triangle_hit_material_in_range(
        cornell_scene,
        renderer::Ray(renderer::Vec3(0.8f, 0, -2), renderer::Vec3(0, 1, 0)));
    check_triangle_hit_material_in_range(
        cornell_scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(-1, 0, 0)));
    check_triangle_hit_material_in_range(
        cornell_scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(1, 0, 0)));
    check_triangle_hit_material_in_range(
        cornell_scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 0, -1)));
    check_triangle_hit_material_in_range(
        cornell_scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 1, 0)));
}

void check_cornell_wall_hit(
    const renderer::Scene& scene,
    const renderer::Ray& ray,
    bool (*material_predicate)(const renderer::Material&)) {
    renderer::HitRecord hit;
    RENDER_CHECK(intersect_closest_triangle(scene, ray, hit));
    RENDER_CHECK(hit.front_face);
    RENDER_CHECK(material_id_in_range(scene, hit.material_id));
    RENDER_CHECK(material_predicate(scene.materials[hit.material_id]));
}

bool is_red_like_material(const renderer::Material& material) {
    return material.base_color.x() > material.base_color.y() &&
           material.base_color.x() > material.base_color.z();
}

bool is_green_like_material(const renderer::Material& material) {
    return material.base_color.y() > material.base_color.x() &&
           material.base_color.y() > material.base_color.z();
}

bool is_white_diffuse_material(const renderer::Material& material) {
    return material.type == renderer::MaterialType::Diffuse &&
           material.base_color.x() > 0.5f &&
           material.base_color.y() > 0.5f &&
           material.base_color.z() > 0.5f;
}

RENDER_TEST(test_cornell_box_wall_normals_face_inward) {
    renderer::Scene scene = renderer::make_cornell_box_scene();
    check_cornell_wall_hit(
        scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, -1, 0)),
        is_white_diffuse_material);
    check_cornell_wall_hit(
        scene,
        renderer::Ray(renderer::Vec3(0.8f, 0, -2), renderer::Vec3(0, 1, 0)),
        is_white_diffuse_material);
    check_cornell_wall_hit(
        scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(-1, 0, 0)),
        is_red_like_material);
    check_cornell_wall_hit(
        scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(1, 0, 0)),
        is_green_like_material);
    check_cornell_wall_hit(
        scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 0, -1)),
        is_white_diffuse_material);
}

RENDER_TEST(test_cornell_box_light_uses_emissive_material_and_faces_downward) {
    renderer::Scene scene = renderer::make_cornell_box_scene();
    renderer::HitRecord hit;
    RENDER_CHECK(intersect_closest_triangle(
        scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 1, 0)),
        hit));
    RENDER_CHECK(material_id_in_range(scene, hit.material_id));
    RENDER_CHECK(scene.materials[hit.material_id].type == renderer::MaterialType::Emissive);
    RENDER_CHECK(scene.materials[hit.material_id].emission.squaredNorm() > 0.0f);
    RENDER_CHECK(hit.front_face);
    RENDER_CHECK(hit.shading_normal.y() < -0.999f);
}

RENDER_TEST(test_cosine_sample_is_in_upper_hemisphere) {
    renderer::PcgRandom rng(42);
    for (int i = 0; i < 100; ++i) {
        renderer::Vec3 d = renderer::cosine_weighted_hemisphere(rng);
        RENDER_CHECK(d.z() >= -1e-9f);
        RENDER_CHECK(nearly_equal(d.norm(), 1.0f, 1e-6f));
    }
}

RENDER_TEST(test_reflect_preserves_unit_length) {
    const renderer::Vec3 incident = renderer::Vec3(1.0f, -1.0f, 0.0f).normalized();
    const renderer::Vec3 reflected = renderer::reflect(incident, renderer::Vec3::UnitY());

    RENDER_CHECK(nearly_equal(reflected.norm(), 1.0f, 1e-6f));
    RENDER_CHECK(reflected.y() > 0.0f);
}

RENDER_TEST(test_refract_returns_unit_direction) {
    const renderer::Vec3 incident(0.6f, -0.8f, 0.0f);
    renderer::Vec3 refracted = renderer::Vec3::Zero();

    RENDER_CHECK(renderer::refract(
        incident,
        renderer::Vec3::UnitY(),
        1.0f / 1.5f,
        refracted));
    RENDER_CHECK(nearly_equal(refracted.norm(), 1.0f, 1e-6f));
    RENDER_CHECK(refracted.y() < 0.0f);
}

RENDER_TEST(test_refract_rejects_total_internal_reflection) {
    const renderer::Vec3 incident(0.8f, 0.6f, 0.0f);
    renderer::Vec3 refracted = renderer::Vec3::Zero();

    RENDER_CHECK(!renderer::refract(
        incident,
        -renderer::Vec3::UnitY(),
        1.5f,
        refracted));
}

RENDER_TEST(test_render_settings_defaults_are_useful) {
    renderer::RenderSettings settings;
    RENDER_CHECK(settings.width == 512);
    RENDER_CHECK(settings.height == 512);
    RENDER_CHECK(settings.path.samples_per_pixel == 1);
    RENDER_CHECK(settings.path.max_bounces == 64);
    RENDER_CHECK(settings.path.russian_roulette_start_bounce == 3);
    RENDER_CHECK(nearly_equal(
        settings.path.russian_roulette_min_probability,
        0.05f));
    RENDER_CHECK(nearly_equal(
        settings.path.russian_roulette_max_probability,
        0.95f));
    RENDER_CHECK(settings.path.sample_seed_offset == 0);
    RENDER_CHECK(settings.path.cuda_device == 0);
    RENDER_CHECK(
        settings.opengl.ambient_occlusion.mode ==
        renderer::OpenGlAmbientOcclusionMode::Gtao);
    RENDER_CHECK(settings.opengl.ambient_occlusion.gtao.slice_count == 3);
    RENDER_CHECK(
        settings.opengl.ambient_occlusion.gtao.samples_per_side == 3);
    RENDER_CHECK(
        settings.opengl.ambient_occlusion.gtao.bent_normals_enabled);
    RENDER_CHECK(settings.opengl.ambient_occlusion.denoise.enabled);
    RENDER_CHECK(
        settings.opengl.ambient_occlusion.denoise.kernel_radius == 2);
    RENDER_CHECK(settings.opengl.ssr.enabled);
    RENDER_CHECK(settings.opengl.ssr.rays_per_pixel == 2);
    RENDER_CHECK(settings.opengl.ssr.max_steps == 64);
    RENDER_CHECK(settings.opengl.ssr.max_history_frames == 32);
    RENDER_CHECK(settings.opengl.ssr.denoise_passes == 3);
    RENDER_CHECK(
        settings.opengl.ssr.debug_view ==
        renderer::OpenGlSsrDebugView::Final);
}

bool image_colors_are_finite(const renderer::Image& image) {
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (!image.pixel(x, y).allFinite()) {
                return false;
            }
        }
    }
    return true;
}

bool framebuffer_colors_are_finite(const renderer::Framebuffer& framebuffer) {
    for (int y = 0; y < framebuffer.height(); ++y) {
        for (int x = 0; x < framebuffer.width(); ++x) {
            if (!framebuffer.pixel(x, y).allFinite()) {
                return false;
            }
        }
    }
    return true;
}

renderer::Triangle make_test_triangle_at_z(float depth, int material_id) {
    return renderer::Triangle(
        renderer::Vec3(-1.0f, -1.0f, depth),
        renderer::Vec3(1.0f, -1.0f, depth),
        renderer::Vec3(0.0f, 1.0f, depth),
        material_id);
}

RENDER_TEST(test_pathtracer_renders_emissive_scene) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }
    renderer::Scene scene = renderer::make_cornell_box_scene();
    renderer::Camera camera(
        renderer::Vec3(0, 1, 4),
        renderer::Vec3(0, 1, 0),
        renderer::Vec3(0, 1, 0),
        40.0f,
        1.0f);

    renderer::RenderSettings settings;
    settings.width = 16;
    settings.height = 16;
    settings.path.samples_per_pixel = 2;
    renderer::RenderResult result =
        render_cuda_scene(scene, camera, settings);
    RENDER_CHECK(image_colors_are_finite(result.image));

    float luminance_sum = 0.0f;
    for (int y = 0; y < result.image.height(); ++y) {
        for (int x = 0; x < result.image.width(); ++x) {
            renderer::Color c = result.image.pixel(x, y);
            luminance_sum += c.x() + c.y() + c.z();
        }
    }
    RENDER_CHECK(luminance_sum > 0.1f);
}

renderer::Scene make_path_direct_light_scene() {
    renderer::Scene scene;
    scene.environment = renderer::Color::Zero();
    renderer::Material material;
    material.type = renderer::MaterialType::Diffuse;
    material.base_color = renderer::Color(1.0f, 1.0f, 1.0f);
    scene.materials.push_back(material);
    scene.triangles.emplace_back(
        renderer::Vec3(-10.0f, -10.0f, -1.0f),
        renderer::Vec3(10.0f, -10.0f, -1.0f),
        renderer::Vec3(0.0f, 10.0f, -1.0f),
        0);
    return scene;
}

renderer::Color render_one_path_pixel(
    const renderer::Scene& scene) {
    const renderer::Camera camera(
        renderer::Vec3(0.0f, 0.0f, 0.0f),
        renderer::Vec3(0.0f, 0.0f, -1.0f),
        renderer::Vec3(0.0f, 1.0f, 0.0f),
        20.0f,
        1.0f);
    renderer::RenderSettings settings;
    settings.width = 1;
    settings.height = 1;
    settings.path.samples_per_pixel = 1;
    return render_cuda_scene(scene, camera, settings).image.pixel(0, 0);
}

RENDER_TEST(test_atomic_write_failure_preserves_original_bytes) {
    const std::filesystem::path directory = "test_atomic_write";
    const std::filesystem::path path = directory / "scene.rscene";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const std::string original = "original scene bytes\n";
    renderer::write_file_atomically(path, original);

    renderer::set_atomic_write_failure_for_testing(
        renderer::AtomicWriteFailurePoint::BeforeReplace);
    bool failed = false;
    try {
        renderer::write_file_atomically(path, "replacement bytes\n");
    } catch (const std::runtime_error&) {
        failed = true;
    }
    RENDER_CHECK(failed);
    std::ifstream input(path, std::ios::binary);
    const std::string restored{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};
    RENDER_CHECK(restored == original);
    input.close();
    std::size_t entry_count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        RENDER_CHECK(entry.path() == path);
        ++entry_count;
    }
    RENDER_CHECK(entry_count == 1);
    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_pathtracer_receives_directional_light) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }
    renderer::Scene dark = make_path_direct_light_scene();
    const renderer::Color unlit = render_one_path_pixel(dark);

    renderer::Scene lit = dark;
    lit.directional_lights.push_back(renderer::DirectionalLight{
        renderer::Vec3(0.0f, 0.0f, -1.0f),
        renderer::Color(2.0f, 2.0f, 2.0f)});
    const renderer::Color illuminated = render_one_path_pixel(lit);

    RENDER_CHECK(illuminated.x() > unlit.x() + 0.5f);
}

RENDER_TEST(test_pathtracer_point_light_uses_inverse_square_falloff) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }
    renderer::Scene near_scene = make_path_direct_light_scene();
    near_scene.point_lights.push_back(renderer::PointLight{
        renderer::Vec3(0.0f, 0.0f, 1.0f),
        renderer::Color(8.0f, 8.0f, 8.0f)});
    renderer::Scene far_scene = make_path_direct_light_scene();
    far_scene.point_lights.push_back(renderer::PointLight{
        renderer::Vec3(0.0f, 0.0f, 3.0f),
        renderer::Color(8.0f, 8.0f, 8.0f)});

    const renderer::Color near_value = render_one_path_pixel(near_scene);
    const renderer::Color far_value = render_one_path_pixel(far_scene);
    RENDER_CHECK(near_value.x() > far_value.x() * 3.5f);
}

RENDER_TEST(test_pathtracer_direct_light_respects_shadow_blockers) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }
    renderer::Scene visible = make_path_direct_light_scene();
    visible.point_lights.push_back(renderer::PointLight{
        renderer::Vec3(2.0f, 0.0f, 0.0f),
        renderer::Color(20.0f, 20.0f, 20.0f)});
    renderer::Scene blocked = visible;
    blocked.triangles.emplace_back(
        renderer::Vec3(1.0f, -10.0f, -2.0f),
        renderer::Vec3(1.0f, 10.0f, -2.0f),
        renderer::Vec3(1.0f, 0.0f, 1.0f),
        0);

    const renderer::Color visible_value = render_one_path_pixel(visible);
    const renderer::Color blocked_value = render_one_path_pixel(blocked);
    RENDER_CHECK(visible_value.x() > blocked_value.x() + 0.05f);
}

renderer::Scene make_path_roulette_layer_scene() {
    renderer::Scene scene;
    scene.environment = renderer::Color::Ones();

    renderer::Material dielectric;
    dielectric.type = renderer::MaterialType::Dielectric;
    dielectric.ior = 1.0f;
    scene.materials.push_back(dielectric);

    for (int layer = 1; layer <= 4; ++layer) {
        const float z = -static_cast<float>(layer);
        scene.triangles.emplace_back(
            renderer::Vec3(-10.0f, -10.0f, z),
            renderer::Vec3(10.0f, -10.0f, z),
            renderer::Vec3(0.0f, 10.0f, z),
            0);
    }
    return scene;
}

renderer::Color render_roulette_layer_sample(
    const renderer::Scene& scene,
    std::uint64_t seed_offset,
    int roulette_start_bounce = 3,
    float roulette_min_probability = 0.05f,
    float roulette_max_probability = 0.95f,
    int max_bounces = 64) {
    const renderer::Camera camera(
        renderer::Vec3::Zero(),
        -renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitY(),
        10.0f,
        1.0f);
    renderer::RenderSettings settings;
    settings.width = 1;
    settings.height = 1;
    settings.path.samples_per_pixel = 1;
    settings.path.sample_seed_offset = seed_offset;
    settings.path.max_bounces = max_bounces;
    settings.path.russian_roulette_start_bounce = roulette_start_bounce;
    settings.path.russian_roulette_min_probability =
        roulette_min_probability;
    settings.path.russian_roulette_max_probability =
        roulette_max_probability;
    return render_cuda_scene(scene, camera, settings).image.pixel(0, 0);
}

RENDER_TEST(test_pathtracer_russian_roulette_terminates_and_preserves_energy) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }
    const renderer::Scene scene = make_path_roulette_layer_scene();
    constexpr int sample_count = 1024;
    renderer::Color accumulated = renderer::Color::Zero();
    bool saw_terminated_path = false;
    bool saw_surviving_path = false;

    for (int sample = 0; sample < sample_count; ++sample) {
        const renderer::Color color = render_roulette_layer_sample(
            scene,
            static_cast<std::uint64_t>(sample + 1));
        RENDER_CHECK(color.allFinite());
        accumulated += color;

        if (color.maxCoeff() < 1e-6f) {
            saw_terminated_path = true;
        } else {
            saw_surviving_path = true;
            RENDER_CHECK(color.x() > 1.0f);
            RENDER_CHECK(nearly_equal(color.x(), color.y(), 1e-6f));
            RENDER_CHECK(nearly_equal(color.x(), color.z(), 1e-6f));
        }
    }

    const renderer::Color mean = accumulated / static_cast<float>(sample_count);
    RENDER_CHECK(saw_terminated_path);
    RENDER_CHECK(saw_surviving_path);
    RENDER_CHECK(mean.x() > 0.9f);
    RENDER_CHECK(mean.x() < 1.1f);

    for (int sample = 0; sample < 16; ++sample) {
        const renderer::Color delayed = render_roulette_layer_sample(
            scene,
            static_cast<std::uint64_t>(sample + 1),
            5);
        RENDER_CHECK(delayed.isApprox(renderer::Color::Ones(), 1e-6f));
    }

    const renderer::Color truncated = render_roulette_layer_sample(
        scene,
        1,
        5,
        0.05f,
        0.95f,
        4);
    RENDER_CHECK(truncated.isApprox(renderer::Color::Zero(), 1e-6f));
    const renderer::Color complete = render_roulette_layer_sample(
        scene,
        1,
        5,
        0.05f,
        0.95f,
        5);
    RENDER_CHECK(complete.isApprox(renderer::Color::Ones(), 1e-6f));
}

int count_lit_pixels(const renderer::Framebuffer& framebuffer) {
    int lit_pixels = 0;
    for (int y = 0; y < framebuffer.height(); ++y) {
        for (int x = 0; x < framebuffer.width(); ++x) {
            const renderer::Color c = framebuffer.pixel(x, y);
            if (c.x() + c.y() + c.z() > 0.05f) {
                ++lit_pixels;
            }
        }
    }
    return lit_pixels;
}

bool colors_nearly_equal(
    const renderer::Color& a,
    const renderer::Color& b,
    float tolerance = 1e-6f) {
    return (a - b).cwiseAbs().maxCoeff() <= tolerance;
}

bool framebuffers_differ(
    const renderer::Framebuffer& a,
    const renderer::Framebuffer& b,
    float tolerance) {
    if (a.width() != b.width() || a.height() != b.height()) {
        return true;
    }

    for (int y = 0; y < a.height(); ++y) {
        for (int x = 0; x < a.width(); ++x) {
            if (!colors_nearly_equal(a.pixel(x, y), b.pixel(x, y), tolerance)) {
                return true;
            }
        }
    }
    return false;
}

bool framebuffer_matches_image(
    const renderer::Framebuffer& framebuffer,
    const renderer::Image& image,
    float tolerance = 1e-6f) {
    if (framebuffer.width() != image.width() || framebuffer.height() != image.height()) {
        return false;
    }
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            if (!colors_nearly_equal(framebuffer.pixel(x, y), image.pixel(x, y), tolerance)) {
                return false;
            }
        }
    }
    return true;
}

bool framebuffer_matches_average(
    const renderer::Framebuffer& framebuffer,
    const renderer::Image& first,
    const renderer::Image& second,
    float tolerance = 1e-6f) {
    if (framebuffer.width() != first.width() || framebuffer.height() != first.height() ||
        first.width() != second.width() || first.height() != second.height()) {
        return false;
    }
    for (int y = 0; y < first.height(); ++y) {
        for (int x = 0; x < first.width(); ++x) {
            const renderer::Color expected = (first.pixel(x, y) + second.pixel(x, y)) * 0.5f;
            if (!colors_nearly_equal(framebuffer.pixel(x, y), expected, tolerance)) {
                return false;
            }
        }
    }
    return true;
}

RENDER_TEST(test_scene_intersector_skips_alpha_cutout_hits) {
    renderer::Scene scene;
    renderer::Material transparent;
    transparent.opacity = 0.0f;
    transparent.alpha_cutoff = 0.5f;
    scene.materials.push_back(transparent);
    scene.materials.push_back(renderer::Material());
    scene.triangles.push_back(make_test_triangle_at_z(-1.0f, 0));
    scene.triangles.push_back(make_test_triangle_at_z(-2.0f, 1));

    const renderer::SceneIntersector intersector(scene);
    renderer::HitRecord hit;
    const renderer::Ray ray(renderer::Vec3(0.0f, 0.0f, 0.0f), renderer::Vec3(0.0f, 0.0f, -1.0f));
    RENDER_CHECK(intersector.intersect(ray, 0.0f, 100.0f, hit));
    RENDER_CHECK(hit.material_id == 1);
    RENDER_CHECK(hit.position.z() < -1.5f);
}

RENDER_TEST(test_scene_intersector_continues_through_thin_alpha_layer) {
    renderer::Scene scene;
    renderer::Material transparent;
    transparent.opacity = 0.0f;
    transparent.alpha_cutoff = 0.5f;
    scene.materials.push_back(transparent);
    scene.materials.push_back(renderer::Material());
    scene.triangles.push_back(make_test_triangle_at_z(-1.0f, 0));
    scene.triangles.push_back(make_test_triangle_at_z(-1.00001f, 1));

    const renderer::SceneIntersector intersector(scene);
    renderer::HitRecord hit;
    const renderer::Ray ray(renderer::Vec3::Zero(), -renderer::Vec3::UnitZ());
    RENDER_CHECK(intersector.intersect(ray, 0.0f, 10.0f, hit));
    RENDER_CHECK(hit.material_id == 1);
    RENDER_CHECK(hit.t > 1.0f);
    RENDER_CHECK(hit.t < 1.001f);
}

RENDER_TEST(test_scene_intersector_respects_single_and_two_sided_materials) {
    renderer::Scene scene;
    renderer::Material material;
    material.two_sided = false;
    scene.materials.push_back(material);
    scene.triangles.push_back(make_test_triangle_at_z(-1.0f, 0));
    const renderer::Ray back_ray(
        renderer::Vec3(0.0f, 0.0f, -2.0f),
        renderer::Vec3(0.0f, 0.0f, 1.0f));

    renderer::HitRecord hit;
    RENDER_CHECK(!renderer::SceneIntersector(scene).intersect(back_ray, 0.0f, 100.0f, hit));

    scene.materials[0].two_sided = true;
    RENDER_CHECK(renderer::SceneIntersector(scene).intersect(back_ray, 0.0f, 100.0f, hit));
    RENDER_CHECK(hit.geometric_normal.dot(back_ray.direction) < 0.0f);
    RENDER_CHECK(hit.shading_normal.dot(hit.geometric_normal) > 0.0f);
}

RENDER_TEST(test_offset_ray_origin_is_finite_and_monotonic_across_scales) {
    for (const float coordinate : {1.0f, 100000.0f}) {
        const renderer::Vec3 position(coordinate, -coordinate, 0.5f * coordinate);
        const std::array<renderer::Vec3, 2> normals{
            renderer::Vec3::UnitX(),
            -renderer::Vec3::UnitX()};
        for (const renderer::Vec3& normal : normals) {
            const renderer::Vec3 outward =
                renderer::offset_ray_origin(position, normal, normal);
            const renderer::Vec3 inward =
                renderer::offset_ray_origin(position, normal, -normal);
            RENDER_CHECK(outward.allFinite());
            RENDER_CHECK(inward.allFinite());
            RENDER_CHECK((outward - position).dot(normal) > 0.0f);
            RENDER_CHECK((inward - position).dot(normal) < 0.0f);
            RENDER_CHECK(outward.x() != position.x());
            RENDER_CHECK(inward.x() != position.x());
        }
    }
}

RENDER_TEST(test_offset_ray_origin_uses_float_roundoff_budget_across_scales) {
    constexpr float offset_scale = 32.0f * std::numeric_limits<float>::epsilon();
    const float infinity = std::numeric_limits<float>::infinity();
    const renderer::Vec3 normal = renderer::Vec3::Ones().normalized();

    for (const float coordinate : {1.0f, 100000.0f}) {
        const renderer::Vec3 position = renderer::Vec3::Constant(coordinate);
        const float scale = std::max(1.0f, coordinate);
        for (const float selected_side : {1.0f, -1.0f}) {
            const renderer::Vec3 direction = selected_side * normal;
            const renderer::Vec3 nominal =
                position + normal * (selected_side * scale * offset_scale);
            const renderer::Vec3 offset =
                renderer::offset_ray_origin(position, normal, direction);

            RENDER_CHECK(offset.allFinite());
            RENDER_CHECK((offset.array() != position.array()).any());
            RENDER_CHECK((offset - position).dot(direction) > 0.0f);
            for (int axis = 0; axis < 3; ++axis) {
                const float next = std::nextafter(
                    position[axis],
                    selected_side > 0.0f ? infinity : -infinity);
                const float expected =
                    nominal[axis] == position[axis] ? next : nominal[axis];
                RENDER_CHECK(offset[axis] != position[axis]);
                RENDER_CHECK(offset[axis] == expected);
            }
        }
    }
}

void write_test_ppm_texture(const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n2 2\n255\n";
    const std::array<unsigned char, 12> pixels{
        255, 0, 0,
        0, 255, 0,
        0, 0, 255,
        255, 255, 255};
    out.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
}

void write_single_pixel_ppm(const std::string& path, unsigned char value) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n1 1\n255\n";
    const std::array<unsigned char, 3> pixel{value, value, value};
    out.write(reinterpret_cast<const char*>(pixel.data()), static_cast<std::streamsize>(pixel.size()));
}

renderer::Scene make_emissive_silhouette_scene() {
    renderer::Scene scene;
    scene.environment = renderer::Color(0.0f, 0.0f, 0.0f);

    renderer::Material light;
    light.type = renderer::MaterialType::Emissive;
    light.base_color = renderer::Color(1.0f, 1.0f, 1.0f);
    light.emission = renderer::Color(6.0f, 6.0f, 6.0f);
    scene.materials.push_back(light);
    scene.spheres.emplace_back(renderer::Vec3(0.0f, 0.0f, -1.0f), 0.55f, 0);

    return scene;
}

renderer::Image render_direct_path_sample(
    const renderer::Scene& scene,
    const renderer::Camera& camera,
    const renderer::RenderSettings& settings,
    std::uint64_t sample_seed_offset) {
    renderer::RenderSettings direct_settings = settings;
    direct_settings.path.samples_per_pixel = 1;
    direct_settings.path.sample_seed_offset = sample_seed_offset;
    return render_cuda_scene(scene, camera, direct_settings).image;
}

RENDER_TEST(test_path_interactive_session_matches_direct_samples_and_resets) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }
    const renderer::Scene scene = make_emissive_silhouette_scene();
    renderer::RenderSceneSnapshot snapshot =
        renderer::make_render_scene_snapshot(scene);
    const renderer::Camera camera(
        renderer::Vec3(0.0f, 0.0f, 2.0f),
        renderer::Vec3(0.0f, 0.0f, -1.0f),
        renderer::Vec3::UnitY(),
        45.0f,
        1.0f);
    renderer::RenderSettings settings;
    settings.width = 32;
    settings.height = 32;
    settings.path.samples_per_pixel = 1;
    settings.path.sample_seed_offset = 70;
    renderer::Framebuffer framebuffer(settings.width, settings.height);
    renderer::InteractiveFrameState frame_state;
    constexpr float reset_difference_tolerance = 1e-3f;

    renderer::PathInteractiveSession path;
    path.reset(snapshot, settings);
    path.render_next_frame(snapshot, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(framebuffer_colors_are_finite(framebuffer));
    RENDER_CHECK(count_lit_pixels(framebuffer) > 0);
    RENDER_CHECK(path.accumulated_samples() == 1);

    path.render_next_frame(snapshot, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(framebuffer_colors_are_finite(framebuffer));
    RENDER_CHECK(path.accumulated_samples() == 2);

    const renderer::Camera changed_camera(
        renderer::Vec3(0.0f, 0.0f, 2.0f),
        renderer::Vec3(-0.35f, 0.0f, -1.0f),
        renderer::Vec3::UnitY(),
        45.0f,
        1.0f);
    const renderer::Framebuffer accumulated_before_reset = framebuffer;
    frame_state.camera_changed = true;
    path.render_next_frame(snapshot, changed_camera, settings, frame_state, framebuffer);
    RENDER_CHECK(framebuffer_colors_are_finite(framebuffer));
    RENDER_CHECK(framebuffers_differ(
        accumulated_before_reset,
        framebuffer,
        reset_difference_tolerance));
    RENDER_CHECK(path.accumulated_samples() == 1);

    frame_state = renderer::InteractiveFrameState{};
    frame_state.scene_changes = renderer::SceneChange::Lighting;
    path.render_next_frame(snapshot, changed_camera, settings, frame_state, framebuffer);
    RENDER_CHECK(path.accumulated_samples() == 2);

    renderer::RenderSceneSnapshot relit_snapshot = snapshot;
    relit_snapshot.revisions.lighting++;
    path.render_next_frame(
        relit_snapshot,
        changed_camera,
        settings,
        renderer::InteractiveFrameState{},
        framebuffer);
    RENDER_CHECK(path.accumulated_samples() == 1);
}

RENDER_TEST(test_obj_loader_reads_single_triangle) {
    const std::string path = "test_single_triangle.obj";
    {
        std::ofstream out(path);
        out << "v 0 0 0\n";
        out << "v 1 0 0\n";
        out << "v 0 1 0\n";
        out << "f 1 2 3\n";
    }

    renderer::Mesh mesh = renderer::load_obj_mesh(path, 0);
    RENDER_CHECK(mesh.triangles.size() == 1);
    std::remove(path.c_str());
}

RENDER_TEST(test_scene_asset_loader_preserves_obj_vertex_normals) {
    const std::string obj_path = "test_smooth_normals.obj";
    {
        std::ofstream obj(obj_path);
        obj << "v -1 -1 -1\n";
        obj << "v 1 -1 -1\n";
        obj << "v 0 1 -1\n";
        obj << "vn 0 1 1\n";
        obj << "vn 1 0 1\n";
        obj << "vn 0 0 1\n";
        obj << "f 1//1 2//2 3//3\n";
    }

    const renderer::LoadedScene loaded = renderer::load_scene_asset(obj_path, 64, 64);
    renderer::HitRecord hit;
    const renderer::Ray ray(renderer::Vec3(0.0f, 0.0f, 0.0f), renderer::Vec3(0.0f, 0.0f, -1.0f));
    RENDER_CHECK(loaded.scene.triangles[0].intersect(ray, 1e-6f, 10.0f, hit));
    RENDER_CHECK(hit.position.allFinite());
    RENDER_CHECK(hit.shading_normal.allFinite());
    RENDER_CHECK((hit.shading_normal - hit.geometric_normal).norm() > 0.01f);
    RENDER_CHECK(hit.shading_normal.dot(hit.geometric_normal) > 0.0f);

    std::remove(obj_path.c_str());
}

RENDER_TEST(test_image_texture_samples_obj_uv_space) {
    const std::string texture_path = "test_map_kd.ppm";
    write_test_ppm_texture(texture_path);

    const renderer::ImageTexture texture = renderer::ImageTexture::load(texture_path);

    const renderer::Color top_left = texture.sample(renderer::Vec2(0.25f, 0.75f));
    const renderer::Color bottom_left = texture.sample(renderer::Vec2(0.25f, 0.25f));
    const renderer::Color wrapped = texture.sample(renderer::Vec2(-0.75f, 1.75f));
    RENDER_CHECK(top_left.allFinite());
    RENDER_CHECK(bottom_left.allFinite());
    RENDER_CHECK(wrapped.allFinite());
    RENDER_CHECK(top_left.x() > 0.9f);
    RENDER_CHECK(top_left.y() < 0.1f);
    RENDER_CHECK(bottom_left.z() > 0.9f);

    std::remove(texture_path.c_str());
}

RENDER_TEST(test_texture_encoding_distinguishes_srgb_from_linear) {
    const std::string texture_path = "test_texture_encoding.ppm";
    write_single_pixel_ppm(texture_path, 128);

    const renderer::ImageTexture srgb = renderer::ImageTexture::load(
        texture_path,
        renderer::TextureEncoding::Srgb);
    const renderer::ImageTexture linear = renderer::ImageTexture::load(
        texture_path,
        renderer::TextureEncoding::Linear);

    RENDER_CHECK(srgb.sample(renderer::Vec2::Zero()).x() < 0.25f);
    RENDER_CHECK(linear.sample(renderer::Vec2::Zero()).x() > 0.49f);
    RENDER_CHECK(linear.sample(renderer::Vec2::Zero()).x() < 0.51f);
    RENDER_CHECK(srgb.sample(renderer::Vec2::Zero()).allFinite());
    RENDER_CHECK(linear.sample(renderer::Vec2::Zero()).allFinite());

    std::remove(texture_path.c_str());
}

RENDER_TEST(test_material_evaluator_combines_opacity_and_perturbs_bump_normal) {
    renderer::Scene scene;
    scene.textures.emplace_back(
        4,
        2,
        std::vector<renderer::Color>{
            renderer::Color(0.0f, 0.0f, 0.0f),
            renderer::Color(0.25f, 0.25f, 0.25f),
            renderer::Color(0.5f, 0.5f, 0.5f),
            renderer::Color(1.0f, 1.0f, 1.0f),
            renderer::Color(0.0f, 0.0f, 0.0f),
            renderer::Color(0.25f, 0.25f, 0.25f),
            renderer::Color(0.5f, 0.5f, 0.5f),
            renderer::Color(1.0f, 1.0f, 1.0f)});

    renderer::Material material;
    material.opacity = 0.8f;
    material.opacity_texture_id = 0;
    material.bump_texture_id = 0;
    material.bump_scale = 1.0f;

    renderer::HitRecord hit;
    hit.uv = renderer::Vec2(0.375f, 0.25f);
    hit.geometric_normal = renderer::Vec3(0.0f, 0.0f, 1.0f);
    hit.shading_normal = renderer::Vec3(0.0f, 0.0f, 1.0f);
    hit.tangent = renderer::Vec3(1.0f, 0.0f, 0.0f);
    hit.bitangent = renderer::Vec3(0.0f, 1.0f, 0.0f);
    hit.has_valid_uv_basis = true;

    const renderer::SurfaceMaterialSample sample =
        renderer::evaluate_surface_material(scene, material, hit);
    RENDER_CHECK(sample.opacity < material.opacity);
    RENDER_CHECK((sample.shading_normal - hit.shading_normal).norm() > 0.01f);
    RENDER_CHECK(sample.shading_normal.dot(hit.geometric_normal) > 0.0f);
    RENDER_CHECK(sample.shading_normal.dot(hit.shading_normal) > 0.9f);
    RENDER_CHECK(sample.base_color.allFinite());
    RENDER_CHECK(std::isfinite(sample.opacity));
    RENDER_CHECK(sample.shading_normal.allFinite());

    hit.has_valid_uv_basis = false;
    const renderer::SurfaceMaterialSample fallback =
        renderer::evaluate_surface_material(scene, material, hit);
    RENDER_CHECK((fallback.shading_normal - hit.shading_normal).norm() < 1e-12f);
}

RENDER_TEST(test_scene_asset_loader_loads_map_kd_and_triangle_uvs) {
    const std::string obj_path = "test_textured_asset.obj";
    const std::string mtl_path = "test_textured_asset.mtl";
    const std::string texture_path = "test_textured_asset.ppm";
    write_test_ppm_texture(texture_path);
    {
        std::ofstream mtl(mtl_path);
        mtl << "newmtl textured\nKd 1 1 1\nmap_Kd " << texture_path << "\n";
    }
    {
        std::ofstream obj(obj_path);
        obj << "mtllib " << mtl_path << "\n";
        obj << "v -1 -1 -1\n";
        obj << "v 1 -1 -1\n";
        obj << "v -1 1 -1\n";
        obj << "vt 0 0\n";
        obj << "vt 1 0\n";
        obj << "vt 0 1\n";
        obj << "usemtl textured\n";
        obj << "f 1/1 2/2 3/3\n";
    }

    renderer::LoadedScene loaded = renderer::load_scene_asset(obj_path, 64, 64);
    RENDER_CHECK(loaded.scene.textures.size() == 1);
    RENDER_CHECK(!loaded.scene.materials.empty());
    RENDER_CHECK(loaded.scene.materials[0].diffuse_texture_id == 0);

    renderer::HitRecord hit;
    const bool did_hit = loaded.scene.triangles[0].intersect(
        renderer::Ray(renderer::Vec3(-0.5f, 0.5f, 0.0f), renderer::Vec3(0.0f, 0.0f, -1.0f)),
        0.001f,
        10.0f,
        hit);
    RENDER_CHECK(did_hit);
    RENDER_CHECK(nearly_equal(hit.uv.x(), 0.25f));
    RENDER_CHECK(nearly_equal(hit.uv.y(), 0.75f));

    const renderer::Color textured_color = renderer::sample_material_base_color(
        loaded.scene,
        loaded.scene.materials[0],
        hit.uv);
    RENDER_CHECK(textured_color.x() > 0.9f);
    RENDER_CHECK(textured_color.y() < 0.1f);
    RENDER_CHECK(textured_color.z() < 0.1f);

    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
    std::remove(texture_path.c_str());
}

RENDER_TEST(test_scene_asset_loader_imports_alpha_and_bump_maps) {
    const std::string obj_path = "test_surface_maps.obj";
    const std::string mtl_path = "test_surface_maps.mtl";
    const std::string texture_path = "test_surface_maps.ppm";
    write_single_pixel_ppm(texture_path, 128);
    {
        std::ofstream mtl(mtl_path);
        mtl << "newmtl surface\n";
        mtl << "Kd 1 1 1\n";
        mtl << "d 0.8\n";
        mtl << "map_Kd " << texture_path << "\n";
        mtl << "map_d " << texture_path << "\n";
        mtl << "bump -bm 0.25 " << texture_path << "\n";
    }
    {
        std::ofstream obj(obj_path);
        obj << "mtllib " << mtl_path << "\n";
        obj << "v -1 -1 -1\n";
        obj << "v 1 -1 -1\n";
        obj << "v 0 1 -1\n";
        obj << "vt 0 0\nvt 1 0\nvt 0.5 1\n";
        obj << "vn 0 0 1\n";
        obj << "usemtl surface\n";
        obj << "f 1/1/1 2/2/1 3/3/1\n";
    }

    const renderer::LoadedScene loaded = renderer::load_scene_asset(obj_path, 64, 64);
    const renderer::Material& material = loaded.scene.materials[0];
    RENDER_CHECK(nearly_equal(static_cast<float>(material.opacity), 0.8f, 1e-6f));
    RENDER_CHECK(material.diffuse_texture_id >= 0);
    RENDER_CHECK(material.opacity_texture_id >= 0);
    RENDER_CHECK(material.bump_texture_id >= 0);
    RENDER_CHECK(material.diffuse_texture_id != material.opacity_texture_id);
    RENDER_CHECK(material.opacity_texture_id == material.bump_texture_id);
    RENDER_CHECK(loaded.scene.textures.size() == 2);
    RENDER_CHECK(nearly_equal(static_cast<float>(material.bump_scale), 0.25f, 1e-6f));
    RENDER_CHECK(
        loaded.scene.textures[static_cast<std::size_t>(material.opacity_texture_id)]
            .sample_scalar(renderer::Vec2::Zero()) > 0.49f);

    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
    std::remove(texture_path.c_str());
}

RENDER_TEST(test_scene_asset_loader_warns_for_missing_optional_maps) {
    const std::string obj_path = "test_missing_maps.obj";
    const std::string mtl_path = "test_missing_maps.mtl";
    {
        std::ofstream mtl(mtl_path);
        mtl << "newmtl missing\nKd 0.8 0.8 0.8\n";
        mtl << "map_d missing-opacity.png\n";
        mtl << "bump missing-height.png\n";
    }
    {
        std::ofstream obj(obj_path);
        obj << "mtllib " << mtl_path << "\n";
        obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
        obj << "usemtl missing\nf 1 2 3\n";
    }

    const renderer::LoadedScene loaded = renderer::load_scene_asset(obj_path, 64, 64);
    RENDER_CHECK(loaded.scene.triangles.size() == 1);
    RENDER_CHECK(loaded.scene.materials[0].opacity_texture_id == -1);
    RENDER_CHECK(loaded.scene.materials[0].bump_texture_id == -1);
    bool saw_opacity_warning = false;
    bool saw_bump_warning = false;
    for (const std::string& warning : loaded.warnings) {
        saw_opacity_warning = saw_opacity_warning || warning.find("opacity") != std::string::npos;
        saw_bump_warning = saw_bump_warning || warning.find("bump") != std::string::npos;
    }
    RENDER_CHECK(saw_opacity_warning);
    RENDER_CHECK(saw_bump_warning);

    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
}

RENDER_TEST(test_scene_asset_loader_preserves_obj_mtl_materials) {
    const std::string obj_path = "test_asset_loader.obj";
    const std::string mtl_path = "test_asset_loader.mtl";
    {
        std::ofstream mtl(mtl_path);
        mtl << "newmtl red\nKd 0.8 0.1 0.1\nKs 0 0 0\nillum 2\n";
        mtl << "newmtl light\nKd 1 1 1\nKe 4 3 2\nillum 2\n";
        mtl << "newmtl mirror\nKd 0.02 0.02 0.02\nKs 0.95 0.95 0.95\nNs 1000\nillum 5\n";
        mtl << "newmtl glass\nKd 0.01 0.01 0.01\nKs 0.3 0.3 0.3\nNi 1.33\nillum 7\n";
    }
    {
        std::ofstream obj(obj_path);
        obj << "mtllib " << mtl_path << "\n";
        obj << "v 0 0 0\n";
        obj << "v 1 0 0\n";
        obj << "v 0 1 0\n";
        obj << "v 0 0 1\n";
        obj << "v 1 0 1\n";
        obj << "v 0 1 1\n";
        obj << "usemtl red\nf 1 2 3\n";
        obj << "usemtl light\nf 4 5 6\n";
        obj << "usemtl mirror\nf 1 4 2\n";
        obj << "usemtl glass\nf 2 5 3\n";
    }

    renderer::LoadedScene loaded = renderer::load_scene_asset(obj_path, 64, 64);
    RENDER_CHECK(loaded.scene.triangles.size() == 4);
    RENDER_CHECK(loaded.material_names.size() == loaded.scene.materials.size());
    RENDER_CHECK(loaded.material_names[0] == "red");
    RENDER_CHECK(loaded.material_names[1] == "light");
    RENDER_CHECK(loaded.material_names[2] == "mirror");
    RENDER_CHECK(loaded.material_names[3] == "glass");
    RENDER_CHECK(loaded.material_names[4] == "<default>");
    RENDER_CHECK(has_red_like_material(loaded.scene));
    RENDER_CHECK(has_nonzero_emissive_material(loaded.scene));
    RENDER_CHECK(has_material_type(loaded.scene, renderer::MaterialType::Metal));
    RENDER_CHECK(has_material_type(loaded.scene, renderer::MaterialType::Dielectric));
    RENDER_CHECK(loaded.camera.viewport_width() > 0.0f);

    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
}

RENDER_TEST(test_scene_asset_loader_promotes_emissive_quad_to_rect_area_light) {
    const std::string obj_path = "test_emissive_quad.obj";
    const std::string mtl_path = "test_emissive_quad.mtl";
    {
        std::ofstream mtl(mtl_path);
        mtl << "newmtl light\nKd 0.8 0.8 0.8\nKe 17 12 4\nillum 2\n";
    }
    {
        std::ofstream obj(obj_path);
        obj << "mtllib " << mtl_path << "\n";
        obj << "v -0.24 1.98 0.16\n";
        obj << "v -0.24 1.98 -0.22\n";
        obj << "v 0.23 1.98 -0.22\n";
        obj << "v 0.23 1.98 0.16\n";
        obj << "g light\nusemtl light\nf 1 2 3 4\n";
    }

    const renderer::LoadedScene loaded =
        renderer::load_scene_asset(obj_path, 64, 64);
    RENDER_CHECK(loaded.scene.rect_area_lights.size() == 1);
    RENDER_CHECK(loaded.scene.triangles.empty());
    RENDER_CHECK(loaded.scene.directional_lights.empty());
    const renderer::RectAreaLight& light = loaded.scene.rect_area_lights[0];
    RENDER_CHECK(light.position.isApprox(renderer::Vec3(-0.005f, 1.98f, -0.03f)));
    RENDER_CHECK(light.radiance.isApprox(renderer::Color(17.0f, 12.0f, 4.0f)));
    RENDER_CHECK(!light.two_sided);
    RENDER_CHECK(
        renderer::rect_area_light_emission_direction(light).y() < -0.999f);

    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
}

RENDER_TEST(test_scene_asset_loader_recovers_light_from_same_stem_mtl) {
    const std::string obj_path = "test_supplemental_light.obj";
    const std::string declared_mtl_path = "test_incomplete_materials.mtl";
    const std::string supplemental_mtl_path = "test_supplemental_light.mtl";
    {
        std::ofstream mtl(declared_mtl_path);
        mtl << "newmtl surface\nKd 0.8 0.8 0.8\nillum 2\n";
    }
    {
        std::ofstream mtl(supplemental_mtl_path);
        mtl << "newmtl light\nKd 1 1 1\nKe 2 3 4\nillum 2\n";
    }
    {
        std::ofstream obj(obj_path);
        obj << "mtllib " << declared_mtl_path << "\n";
        obj << "v -1 1 -1\nv -1 1 1\nv 1 1 1\nv 1 1 -1\n";
        obj << "g light\nusemtl light\nf 1 2 3 4\n";
    }

    const renderer::LoadedScene loaded =
        renderer::load_scene_asset(obj_path, 64, 64);
    RENDER_CHECK(loaded.scene.rect_area_lights.size() == 1);
    RENDER_CHECK(loaded.scene.rect_area_lights[0].radiance.isApprox(
        renderer::Color(2.0f, 3.0f, 4.0f)));
    RENDER_CHECK(std::any_of(
        loaded.warnings.begin(),
        loaded.warnings.end(),
        [](const std::string& warning) {
            return warning.find("same-stem library") != std::string::npos;
        }));

    std::remove(obj_path.c_str());
    std::remove(declared_mtl_path.c_str());
    std::remove(supplemental_mtl_path.c_str());
}

RENDER_TEST(test_scene_asset_loader_recovers_named_light_without_mtl_definition) {
    const std::string obj_path = "test_named_light.obj";
    const std::string mtl_path = "test_named_light_materials.mtl";
    {
        std::ofstream mtl(mtl_path);
        mtl << "newmtl surface\nKd 0.8 0.8 0.8\nillum 2\n";
    }
    {
        std::ofstream obj(obj_path);
        obj << "mtllib " << mtl_path << "\n";
        obj << "v -1 1 -1\nv -1 1 1\nv 1 1 1\nv 1 1 -1\n";
        obj << "g light\nusemtl light\nf 1 2 3 4\n";
    }

    const renderer::LoadedScene loaded =
        renderer::load_scene_asset(obj_path, 64, 64);
    RENDER_CHECK(loaded.scene.rect_area_lights.size() == 1);
    RENDER_CHECK(loaded.scene.rect_area_lights[0].radiance.isApprox(
        renderer::Color(10.0f, 10.0f, 10.0f)));
    RENDER_CHECK(std::any_of(
        loaded.warnings.begin(),
        loaded.warnings.end(),
        [](const std::string& warning) {
            return warning.find("white emissive light") != std::string::npos;
        }));

    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
}

RENDER_TEST(test_scene_asset_loader_does_not_treat_default_tf_as_transmission) {
    const std::string obj_path = "test_tf_materials.obj";
    const std::string mtl_path = "test_tf_materials.mtl";
    {
        std::ofstream mtl(mtl_path);
        mtl << "newmtl opaque\nKd 0.8 0.7 0.6\nKs 0 0 0\nd 1\nTf 1 1 1\nillum 2\n";
        for (const int illum : {4, 6, 7, 9}) {
            mtl << "newmtl glass" << illum << "\n";
            mtl << "Kd 0.1 0.1 0.1\nKs 0 0 0\nd 1\nTf 0.7 0.8 0.9\n";
            mtl << "illum " << illum << "\n";
        }
    }
    {
        std::ofstream obj(obj_path);
        obj << "mtllib " << mtl_path << "\n";
        obj << "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
        obj << "usemtl opaque\nf 1 2 3\n";
        for (const int illum : {4, 6, 7, 9}) {
            obj << "usemtl glass" << illum << "\nf 1 2 3\n";
        }
    }

    const renderer::LoadedScene loaded = renderer::load_scene_asset(obj_path, 64, 64);
    RENDER_CHECK(loaded.scene.triangles.size() == 5);
    const int opaque_id = loaded.scene.triangles[0].material_id();
    RENDER_CHECK(opaque_id >= 0);
    RENDER_CHECK(loaded.scene.materials[static_cast<std::size_t>(opaque_id)].type ==
        renderer::MaterialType::Diffuse);
    RENDER_CHECK(nearly_equal(
        loaded.scene.materials[static_cast<std::size_t>(opaque_id)].opacity,
        1.0f));
    for (std::size_t triangle_index = 1; triangle_index < loaded.scene.triangles.size(); ++triangle_index) {
        const int material_id = loaded.scene.triangles[triangle_index].material_id();
        RENDER_CHECK(material_id >= 0);
        RENDER_CHECK(loaded.scene.materials[static_cast<std::size_t>(material_id)].type ==
            renderer::MaterialType::Dielectric);
    }

    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
}

RENDER_TEST(test_cuda_path_backend_availability_contract) {
    std::string reason;
    if (renderer::cuda_path_backend_available(&reason)) {
        RENDER_CHECK(renderer::cuda_path_backend_compiled());
        RENDER_CHECK(reason.empty());
    } else {
        RENDER_CHECK(!reason.empty());
    }
}

renderer::Color image_mean(const renderer::Image& image) {
    renderer::Color sum = renderer::Color::Zero();
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            sum += image.pixel(x, y);
        }
    }
    return sum / static_cast<float>(image.width() * image.height());
}

RENDER_TEST(test_cuda_pathtracer_is_deterministic_when_available) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }

    const renderer::Scene scene = renderer::make_cornell_box_scene();
    const renderer::Camera camera(
        renderer::Vec3(0.0f, 0.15f, 1.5f),
        renderer::Vec3(0.0f, 0.15f, -2.0f),
        renderer::Vec3::UnitY(),
        45.0f,
        1.0f);
    renderer::RenderSettings settings;
    settings.width = 48;
    settings.height = 48;
    settings.path.samples_per_pixel = 16;
    settings.path.sample_seed_offset = 91;

    const renderer::RenderResult first =
        render_cuda_scene(scene, camera, settings);
    const renderer::RenderResult cuda =
        render_cuda_scene(scene, camera, settings);
    RENDER_CHECK(image_colors_are_finite(cuda.image));
    const renderer::Color first_mean = image_mean(first.image);
    const renderer::Color cuda_mean = image_mean(cuda.image);
    for (int channel = 0; channel < 3; ++channel) {
        RENDER_CHECK(nearly_equal(first_mean[channel], cuda_mean[channel], 1.0e-6f));
    }
}

RENDER_TEST(test_cuda_pathtracer_alpha_texture_and_interactive_reset_when_available) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }

    renderer::Scene scene;
    scene.environment = renderer::Color::Zero();
    scene.textures.emplace_back(
        1,
        1,
        std::vector<renderer::Color>{renderer::Color::Zero()});
    renderer::Material cutout;
    cutout.opacity_texture_id = 0;
    cutout.alpha_cutoff = 0.5f;
    renderer::Material light;
    light.type = renderer::MaterialType::Emissive;
    light.emission = renderer::Color(2.0f, 1.0f, 0.5f);
    scene.materials.push_back(cutout);
    scene.materials.push_back(light);
    scene.triangles.push_back(make_test_triangle_at_z(-1.0f, 0));
    scene.triangles.push_back(make_test_triangle_at_z(-2.0f, 1));

    const renderer::Camera camera(
        renderer::Vec3::Zero(),
        -renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitY(),
        20.0f,
        1.0f);
    renderer::RenderSettings settings;
    settings.width = 16;
    settings.height = 16;
    settings.path.samples_per_pixel = 1;
    const renderer::RenderResult result =
        render_cuda_scene(scene, camera, settings);
    RENDER_CHECK(image_colors_are_finite(result.image));
    RENDER_CHECK(result.image.pixel(8, 8).x() > 1.5f);

    renderer::PathInteractiveSession session;
    renderer::RenderSceneSnapshot snapshot =
        renderer::make_render_scene_snapshot(scene);
    renderer::Framebuffer framebuffer(settings.width, settings.height);
    renderer::InteractiveFrameState frame_state;
    session.reset(snapshot, settings);
    session.render_next_frame(snapshot, camera, settings, frame_state, framebuffer);
    session.render_next_frame(snapshot, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(session.accumulated_samples() == 2);
    RENDER_CHECK(framebuffer_colors_are_finite(framebuffer));
    const renderer::CudaPathStatistics before_camera_reset =
        *session.cuda_statistics();
    frame_state.camera_changed = true;
    session.render_next_frame(snapshot, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(session.accumulated_samples() == 1);
    const renderer::CudaPathStatistics after_camera_reset =
        *session.cuda_statistics();
    RENDER_CHECK(
        after_camera_reset.allocation_generation ==
        before_camera_reset.allocation_generation);
    RENDER_CHECK(
        after_camera_reset.geometry_upload_bytes ==
        before_camera_reset.geometry_upload_bytes);
    RENDER_CHECK(
        after_camera_reset.material_binding_upload_bytes ==
        before_camera_reset.material_binding_upload_bytes);
    RENDER_CHECK(
        after_camera_reset.material_upload_bytes ==
        before_camera_reset.material_upload_bytes);
    RENDER_CHECK(
        after_camera_reset.texture_upload_bytes ==
        before_camera_reset.texture_upload_bytes);
    RENDER_CHECK(
        after_camera_reset.lighting_upload_bytes ==
        before_camera_reset.lighting_upload_bytes);
    RENDER_CHECK(
        after_camera_reset.bvh_upload_bytes ==
        before_camera_reset.bvh_upload_bytes);

    frame_state = renderer::InteractiveFrameState{};
    frame_state.reset_requested = true;
    session.render_next_frame(snapshot, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(session.accumulated_samples() == 1);

    frame_state = renderer::InteractiveFrameState{};
    frame_state.scene_changes = renderer::SceneChange::All;
    session.render_next_frame(snapshot, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(session.accumulated_samples() == 2);

    snapshot.point_lights.push_back(renderer::PointLight{
        renderer::Vec3(0.0f, 0.0f, 1.0f),
        renderer::Color(0.1f, 0.2f, 0.3f),
        0.0f});
    snapshot.revisions.lighting++;
    const renderer::CudaPathStatistics before_lighting =
        *session.cuda_statistics();
    frame_state = renderer::InteractiveFrameState{};
    session.render_next_frame(snapshot, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(session.accumulated_samples() == 1);
    const renderer::CudaPathStatistics after_lighting =
        *session.cuda_statistics();
    RENDER_CHECK(
        after_lighting.lighting_upload_bytes >
        before_lighting.lighting_upload_bytes);
    RENDER_CHECK(
        after_lighting.geometry_upload_bytes ==
        before_lighting.geometry_upload_bytes);
    RENDER_CHECK(
        after_lighting.texture_upload_bytes ==
        before_lighting.texture_upload_bytes);
    RENDER_CHECK(
        after_lighting.bvh_upload_bytes ==
        before_lighting.bvh_upload_bytes);

    snapshot.instances[0].materials[1].emission =
        renderer::Color(1.0f, 0.5f, 0.25f);
    snapshot.revisions.materials++;
    snapshot.revisions.material_bindings++;
    const renderer::CudaPathStatistics before_material =
        *session.cuda_statistics();
    frame_state = renderer::InteractiveFrameState{};
    session.render_next_frame(snapshot, camera, settings, frame_state, framebuffer);
    const renderer::CudaPathStatistics after_material =
        *session.cuda_statistics();
    RENDER_CHECK(
        after_material.material_upload_bytes >
        before_material.material_upload_bytes);
    RENDER_CHECK(
        after_material.material_binding_upload_bytes >
        before_material.material_binding_upload_bytes);
    RENDER_CHECK(
        after_material.bvh_upload_bytes ==
        before_material.bvh_upload_bytes);

    settings.width = 12;
    settings.height = 10;
    frame_state = renderer::InteractiveFrameState{};
    frame_state.framebuffer_resized = true;
    session.render_next_frame(snapshot, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(session.accumulated_samples() == 1);
    RENDER_CHECK(framebuffer.width() == 12);
    RENDER_CHECK(framebuffer.height() == 10);

}

RENDER_TEST(test_cuda_pathtracer_auto_interaction_preview_and_native_tiles_when_available) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }

    renderer::Scene scene = renderer::make_triangle_scene();
    scene.environment = renderer::Color(0.1f, 0.2f, 0.3f);
    const renderer::RenderSceneSnapshot snapshot =
        renderer::make_render_scene_snapshot(scene);
    const renderer::Camera camera(
        renderer::Vec3(0.0f, 0.0f, 1.5f),
        renderer::Vec3::Zero(),
        renderer::Vec3::UnitY(),
        45.0f,
        1.0f / 6.0f);
    renderer::RenderSettings settings;
    settings.width = 16;
    settings.height = 96;
    settings.path.sample_seed_offset = 91;

    renderer::PathInteractiveSession reference;
    renderer::Framebuffer reference_frame(settings.width, settings.height);
    renderer::InteractiveFrameState full_state;
    reference.reset(snapshot, settings);
    reference.render_next_frame(
        snapshot,
        camera,
        settings,
        full_state,
        reference_frame);
    RENDER_CHECK(reference.accumulated_samples() == 1);

    renderer::PathInteractiveSession automatic_first_frame;
    renderer::Framebuffer first_frame(settings.width, settings.height);
    renderer::InteractiveFrameState first_frame_state;
    first_frame_state.automatic_interaction_quality = true;
    automatic_first_frame.reset(snapshot, settings);
    automatic_first_frame.render_next_frame(
        snapshot,
        camera,
        settings,
        first_frame_state,
        first_frame);
    RENDER_CHECK(automatic_first_frame.accumulated_samples() == 0);
    RENDER_CHECK(
        automatic_first_frame.cuda_statistics()->work_mode ==
        renderer::CudaPathWorkMode::InteractionPreview);
    RENDER_CHECK(
        automatic_first_frame.cuda_statistics()->presentation_updated);

    renderer::PathInteractiveSession automatic;
    renderer::Framebuffer automatic_frame(settings.width, settings.height);
    renderer::InteractiveFrameState interaction_state;
    interaction_state.automatic_interaction_quality = true;
    interaction_state.camera_changed = true;
    automatic.reset(snapshot, settings);
    automatic.render_next_frame(
        snapshot,
        camera,
        settings,
        interaction_state,
        automatic_frame);
    RENDER_CHECK(automatic.accumulated_samples() == 0);
    RENDER_CHECK(
        automatic.cuda_statistics()->work_mode ==
        renderer::CudaPathWorkMode::InteractionPreview);
    interaction_state.camera_changed = false;
    for (int frame = 0; frame < 7; ++frame) {
        automatic.render_next_frame(
            snapshot,
            camera,
            settings,
            interaction_state,
            automatic_frame);
    }
    std::vector<renderer::Color> preview_pixels;
    preview_pixels.reserve(
        static_cast<std::size_t>(settings.width) *
        static_cast<std::size_t>(settings.height));
    for (int y = 0; y < settings.height; ++y) {
        for (int x = 0; x < settings.width; ++x) {
            preview_pixels.push_back(
                automatic_frame.pixel(x, y));
        }
    }
    const std::uint64_t preview_downloads =
        automatic.cuda_statistics()->framebuffer_downloads;

    automatic.render_next_frame(
        snapshot,
        camera,
        settings,
        interaction_state,
        automatic_frame);
    RENDER_CHECK(automatic.accumulated_samples() == 0);
    const renderer::CudaPathStatistics partial_tile =
        *automatic.cuda_statistics();
    RENDER_CHECK(
        partial_tile.work_mode == renderer::CudaPathWorkMode::NativeTile);
    RENDER_CHECK(partial_tile.internal_width == settings.width);
    RENDER_CHECK(partial_tile.tile_rows == 64);
    RENDER_CHECK(nearly_equal(
        partial_tile.sweep_progress,
        2.0f / 3.0f));
    RENDER_CHECK(!partial_tile.presentation_updated);
    RENDER_CHECK(
        partial_tile.framebuffer_downloads ==
        preview_downloads);
    for (int y = 0; y < settings.height; ++y) {
        for (int x = 0; x < settings.width; ++x) {
            const std::size_t index =
                static_cast<std::size_t>(y) *
                    static_cast<std::size_t>(settings.width) +
                static_cast<std::size_t>(x);
            RENDER_CHECK(
                (automatic_frame.pixel(x, y) -
                 preview_pixels[index]).norm() <
                1.0e-6f);
        }
    }

    interaction_state.camera_changed = true;
    automatic.render_next_frame(
        snapshot,
        camera,
        settings,
        interaction_state,
        automatic_frame);
    RENDER_CHECK(automatic.accumulated_samples() == 0);
    RENDER_CHECK(
        automatic.cuda_statistics()->work_mode ==
        renderer::CudaPathWorkMode::InteractionPreview);
    RENDER_CHECK(
        automatic.cuda_statistics()->presentation_updated);

    interaction_state.camera_changed = false;
    for (int frame = 0; frame < 8; ++frame) {
        automatic.render_next_frame(
            snapshot,
            camera,
            settings,
            interaction_state,
            automatic_frame);
    }
    RENDER_CHECK(automatic.accumulated_samples() == 0);
    const std::uint64_t reset_preview_downloads =
        automatic.cuda_statistics()->framebuffer_downloads;
    automatic.render_next_frame(
        snapshot,
        camera,
        settings,
        interaction_state,
        automatic_frame);
    RENDER_CHECK(automatic.accumulated_samples() == 1);
    const renderer::CudaPathStatistics completed_tile =
        *automatic.cuda_statistics();
    RENDER_CHECK(completed_tile.sweep_progress == 0.0f);
    RENDER_CHECK(completed_tile.presentation_updated);
    RENDER_CHECK(
        completed_tile.framebuffer_downloads ==
        reset_preview_downloads + 1);
    RENDER_CHECK(framebuffer_colors_are_finite(automatic_frame));
    for (int y = 0; y < settings.height; ++y) {
        for (int x = 0; x < settings.width; ++x) {
            RENDER_CHECK(
                (automatic_frame.pixel(x, y) -
                 reference_frame.pixel(x, y)).norm() <
                1.0e-6f);
        }
    }
}

renderer::Mat4 make_test_instance_matrix(
    const renderer::Vec3& translation,
    float rotation_degrees,
    const renderer::Vec3& scale) {
    renderer::Mat4 result = renderer::Mat4::Identity();
    const renderer::Mat3 rotation =
        Eigen::AngleAxisf(
            rotation_degrees * (3.14159265358979323846f / 180.0f),
            renderer::Vec3::UnitZ())
            .toRotationMatrix();
    result.topLeftCorner<3, 3>() = rotation * scale.asDiagonal();
    result.topRightCorner<3, 1>() = translation;
    return result;
}

void check_framebuffers_near(
    const renderer::Framebuffer& lhs,
    const renderer::Framebuffer& rhs,
    float tolerance) {
    RENDER_CHECK(lhs.width() == rhs.width());
    RENDER_CHECK(lhs.height() == rhs.height());
    for (int y = 0; y < lhs.height(); ++y) {
        for (int x = 0; x < lhs.width(); ++x) {
            RENDER_CHECK(
                (lhs.pixel(x, y) - rhs.pixel(x, y))
                    .cwiseAbs()
                    .maxCoeff() <= tolerance);
        }
    }
}

RENDER_TEST(test_render_scene_snapshot_view_and_cuda_transform_refit_when_available) {
    renderer::Scene local_scene;
    local_scene.environment = renderer::Color(0.01f, 0.02f, 0.03f);
    renderer::Material emissive;
    emissive.type = renderer::MaterialType::Emissive;
    emissive.emission = renderer::Color(2.0f, 1.0f, 0.5f);
    emissive.two_sided = true;
    local_scene.materials.push_back(emissive);
    local_scene.triangles.emplace_back(
        renderer::Vec3(-0.5f, -0.5f, 0.0f),
        renderer::Vec3(0.5f, -0.5f, 0.0f),
        renderer::Vec3(0.5f, 0.5f, 0.0f),
        0);
    local_scene.triangles.emplace_back(
        renderer::Vec3(-0.5f, -0.5f, 0.0f),
        renderer::Vec3(0.5f, 0.5f, 0.0f),
        renderer::Vec3(-0.5f, 0.5f, 0.0f),
        0);

    renderer::SceneDocument document = renderer::SceneDocument::from_scene(
        std::move(local_scene),
        "Instanced emissive quad");
    const renderer::ObjectId first = document.objects().front().id;
    const renderer::ObjectId second = document.duplicate_subtree(first);
    RENDER_CHECK(second != renderer::kInvalidObjectId);
    std::optional<renderer::SceneMaterialOverride> second_material =
        document.material_properties(second, 0);
    RENDER_CHECK(second_material.has_value());
    second_material->emission =
        renderer::Color(0.25f, 3.0f, 0.75f);
    RENDER_CHECK(document.set_material_override(
        second,
        *second_material));
    RENDER_CHECK(document.set_world_matrix(
        first,
        make_test_instance_matrix(
            renderer::Vec3(-0.65f, 0.0f, -2.0f),
            18.0f,
            renderer::Vec3(1.15f, 0.7f, 1.0f))));
    RENDER_CHECK(document.set_world_matrix(
        second,
        make_test_instance_matrix(
            renderer::Vec3(0.65f, 0.0f, -2.0f),
            -12.0f,
            renderer::Vec3(-0.8f, 1.25f, 1.0f))));
    const renderer::ObjectId group =
        document.create_group("Shared parent");
    RENDER_CHECK(document.reparent(first, group));
    RENDER_CHECK(document.reparent(second, group));
    RENDER_CHECK(document.set_world_matrix(
        group,
        make_test_instance_matrix(
            renderer::Vec3(0.0f, 0.05f, 0.0f),
            3.0f,
            renderer::Vec3::Ones())));

    const renderer::RenderSceneSnapshot& initial_instances =
        document.render_scene_snapshot();
    RENDER_CHECK(initial_instances.assets.size() == 1);
    RENDER_CHECK(initial_instances.instances.size() == 2);
    RENDER_CHECK(
        initial_instances.instances[0].asset_index ==
        initial_instances.instances[1].asset_index);
    RENDER_CHECK(
        initial_instances.assets[0].local_scene->triangles.size() == 2);
    RENDER_CHECK(
        initial_instances.instances[0].object_to_world.isApprox(
            document.world_matrix(initial_instances.instances[0].object_id),
            1.0e-5f));

    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }

    renderer::RenderSceneSnapshot invalid_instances =
        initial_instances;
    invalid_instances.instances[0]
        .object_to_world(0, 0) =
        std::numeric_limits<float>::quiet_NaN();
    bool rejected_invalid_instance = false;
    try {
        renderer::CudaPathInteractiveRenderer invalid_renderer;
        renderer::RenderSettings invalid_settings;
        invalid_settings.width = 1;
        invalid_settings.height = 1;
        invalid_renderer.reset(invalid_instances, invalid_settings);
    } catch (const std::runtime_error&) {
        rejected_invalid_instance = true;
    }
    RENDER_CHECK(rejected_invalid_instance);

    renderer::RenderSettings settings;
    settings.width = 32;
    settings.height = 24;
    settings.path.sample_seed_offset = 173;
    const renderer::Camera camera(
        renderer::Vec3::Zero(),
        -renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitY(),
        45.0f,
        static_cast<float>(settings.width) /
            static_cast<float>(settings.height));
    const renderer::InteractiveFrameState frame_state;

    const renderer::Scene initial_flat_scene =
        renderer::flatten_render_scene_snapshot(
            document.render_scene_snapshot());
    const renderer::RenderSceneSnapshot initial_flat_snapshot =
        renderer::make_render_scene_snapshot(initial_flat_scene);
    renderer::PathInteractiveSession flat_session;
    renderer::Framebuffer flat_frame(settings.width, settings.height);
    flat_session.reset(initial_flat_snapshot, settings);
    flat_session.render_next_frame(
        initial_flat_snapshot,
        camera,
        settings,
        frame_state,
        flat_frame);

    renderer::PathInteractiveSession instanced_session;
    renderer::Framebuffer instanced_frame(settings.width, settings.height);
    instanced_session.reset(
        document.render_scene_snapshot(),
        settings);
    instanced_session.render_next_frame(
        document.render_scene_snapshot(),
        camera,
        settings,
        frame_state,
        instanced_frame);
    RENDER_CHECK(framebuffer_colors_are_finite(instanced_frame));
    check_framebuffers_near(flat_frame, instanced_frame, 1.0e-5f);

    const renderer::CudaPathStatistics before_material_update =
        *instanced_session.cuda_statistics();
    RENDER_CHECK(before_material_update.blas_build_count == 1);
    RENDER_CHECK(before_material_update.tlas_build_count == 1);
    second_material = document.material_properties(second, 0);
    RENDER_CHECK(second_material.has_value());
    second_material->emission =
        renderer::Color(0.5f, 2.5f, 1.0f);
    RENDER_CHECK(document.set_material_override(
        second,
        *second_material));
    renderer::InteractiveFrameState material_state;
    material_state.scene_changes =
        renderer::SceneChange::Materials |
        renderer::SceneChange::MaterialBindings;
    instanced_session.render_next_frame(
        document.render_scene_snapshot(),
        camera,
        settings,
        material_state,
        instanced_frame);
    const renderer::CudaPathStatistics before_transform =
        *instanced_session.cuda_statistics();
    RENDER_CHECK(
        before_transform.material_upload_bytes >
        before_material_update.material_upload_bytes);
    RENDER_CHECK(
        before_transform.blas_build_count ==
        before_material_update.blas_build_count);
    RENDER_CHECK(
        before_transform.geometry_upload_bytes ==
        before_material_update.geometry_upload_bytes);
    RENDER_CHECK(
        before_transform.bvh_upload_bytes ==
        before_material_update.bvh_upload_bytes);
    RENDER_CHECK(
        before_transform.tlas_build_count ==
        before_material_update.tlas_build_count);

    RENDER_CHECK(document.set_world_matrix(
        first,
        make_test_instance_matrix(
            renderer::Vec3(-0.35f, 0.15f, -1.7f),
            31.0f,
            renderer::Vec3(-1.15f, 0.7f, 1.0f))));
    renderer::InteractiveFrameState transform_state;
    transform_state.scene_changes =
        renderer::SceneChange::InstanceTransforms;
    instanced_session.render_next_frame(
        document.render_scene_snapshot(),
        camera,
        settings,
        transform_state,
        instanced_frame);
    const renderer::CudaPathStatistics after_transform =
        *instanced_session.cuda_statistics();
    RENDER_CHECK(instanced_session.accumulated_samples() == 1);
    RENDER_CHECK(
        after_transform.instance_upload_bytes >
        before_transform.instance_upload_bytes);
    RENDER_CHECK(
        after_transform.tlas_refit_count ==
        before_transform.tlas_refit_count + 1);
    RENDER_CHECK(
        after_transform.blas_build_count ==
        before_transform.blas_build_count);
    RENDER_CHECK(
        after_transform.tlas_build_count ==
        before_transform.tlas_build_count);
    RENDER_CHECK(
        after_transform.geometry_upload_bytes ==
        before_transform.geometry_upload_bytes);
    RENDER_CHECK(
        after_transform.material_binding_upload_bytes ==
        before_transform.material_binding_upload_bytes);
    RENDER_CHECK(
        after_transform.material_upload_bytes ==
        before_transform.material_upload_bytes);
    RENDER_CHECK(
        after_transform.texture_upload_bytes ==
        before_transform.texture_upload_bytes);
    RENDER_CHECK(
        after_transform.bvh_upload_bytes ==
        before_transform.bvh_upload_bytes);
    RENDER_CHECK(
        after_transform.tlas_upload_bytes ==
        before_transform.tlas_upload_bytes);
    RENDER_CHECK(
        after_transform.lighting_upload_bytes ==
        before_transform.lighting_upload_bytes);

    RENDER_CHECK(document.set_world_matrix(
        first,
        make_test_instance_matrix(
            renderer::Vec3(-0.35f, 0.15f, -1.7f),
            31.0f,
            renderer::Vec3(-1.35f, 0.55f, 1.0f))));
    const renderer::CudaPathStatistics before_scale =
        after_transform;
    instanced_session.render_next_frame(
        document.render_scene_snapshot(),
        camera,
        settings,
        transform_state,
        instanced_frame);
    const renderer::CudaPathStatistics after_scale =
        *instanced_session.cuda_statistics();
    RENDER_CHECK(
        after_scale.lighting_upload_bytes >
        before_scale.lighting_upload_bytes);
    RENDER_CHECK(
        after_scale.blas_build_count ==
        before_scale.blas_build_count);
    RENDER_CHECK(
        after_scale.geometry_upload_bytes ==
        before_scale.geometry_upload_bytes);
    RENDER_CHECK(
        after_scale.tlas_refit_count ==
        before_scale.tlas_refit_count + 1);

    const renderer::Scene transformed_flat_scene =
        renderer::flatten_render_scene_snapshot(
            document.render_scene_snapshot());
    const renderer::RenderSceneSnapshot transformed_flat_snapshot =
        renderer::make_render_scene_snapshot(transformed_flat_scene);
    renderer::PathInteractiveSession transformed_flat_session;
    renderer::Framebuffer transformed_flat_frame(
        settings.width,
        settings.height);
    transformed_flat_session.reset(transformed_flat_snapshot, settings);
    transformed_flat_session.render_next_frame(
        transformed_flat_snapshot,
        camera,
        settings,
        frame_state,
        transformed_flat_frame);
    RENDER_CHECK(framebuffer_colors_are_finite(instanced_frame));
    check_framebuffers_near(
        transformed_flat_frame,
        instanced_frame,
        1.0e-5f);

    const renderer::CudaPathStatistics before_drag =
        *instanced_session.cuda_statistics();
    for (int frame = 0;
         frame < 100;
         ++frame) {
        RENDER_CHECK(document.set_world_matrix(
            first,
            make_test_instance_matrix(
                renderer::Vec3(
                    -0.35f + 0.001f * static_cast<float>(frame),
                    0.15f,
                    -1.7f),
                31.0f,
                renderer::Vec3(-1.35f, 0.55f, 1.0f))));
        instanced_session.render_next_frame(
            document.render_scene_snapshot(),
            camera,
            settings,
            transform_state,
            instanced_frame);
    }
    const renderer::CudaPathStatistics after_drag =
        *instanced_session.cuda_statistics();
    RENDER_CHECK(
        after_drag.tlas_refit_count ==
        before_drag.tlas_refit_count +
            static_cast<std::uint64_t>(
                100));
    RENDER_CHECK(
        after_drag.blas_build_count ==
        before_drag.blas_build_count);
    RENDER_CHECK(
        after_drag.geometry_upload_bytes ==
        before_drag.geometry_upload_bytes);
    RENDER_CHECK(
        after_drag.material_binding_upload_bytes ==
        before_drag.material_binding_upload_bytes);
    RENDER_CHECK(
        after_drag.material_upload_bytes ==
        before_drag.material_upload_bytes);
    RENDER_CHECK(
        after_drag.texture_upload_bytes ==
        before_drag.texture_upload_bytes);
    RENDER_CHECK(
        after_drag.bvh_upload_bytes ==
        before_drag.bvh_upload_bytes);
    RENDER_CHECK(
        after_drag.lighting_upload_bytes ==
        before_drag.lighting_upload_bytes);

    const renderer::ObjectId duplicate =
        document.duplicate_subtree(second);
    RENDER_CHECK(duplicate != renderer::kInvalidObjectId);
    renderer::InteractiveFrameState topology_state;
    topology_state.scene_changes =
        renderer::SceneChange::All;
    instanced_session.render_next_frame(
        document.render_scene_snapshot(),
        camera,
        settings,
        topology_state,
        instanced_frame);
    const renderer::CudaPathStatistics after_duplicate =
        *instanced_session.cuda_statistics();
    RENDER_CHECK(
        after_duplicate.blas_build_count ==
        after_drag.blas_build_count);
    RENDER_CHECK(
        after_duplicate.geometry_upload_bytes ==
        after_drag.geometry_upload_bytes);
    RENDER_CHECK(
        after_duplicate.bvh_upload_bytes ==
        after_drag.bvh_upload_bytes);
    RENDER_CHECK(
        after_duplicate.tlas_build_count ==
        after_drag.tlas_build_count + 1);

    RENDER_CHECK(document.erase_subtree(duplicate));
    instanced_session.render_next_frame(
        document.render_scene_snapshot(),
        camera,
        settings,
        topology_state,
        instanced_frame);
    const renderer::CudaPathStatistics after_delete =
        *instanced_session.cuda_statistics();
    RENDER_CHECK(
        after_delete.blas_build_count ==
        after_duplicate.blas_build_count);
    RENDER_CHECK(
        after_delete.geometry_upload_bytes ==
        after_duplicate.geometry_upload_bytes);
    RENDER_CHECK(
        after_delete.tlas_build_count ==
        after_duplicate.tlas_build_count + 1);

    RENDER_CHECK(document.set_object_visible(first, false));
    RENDER_CHECK(document.set_object_visible(second, false));
    const renderer::RenderSceneSnapshot& hidden_view =
        document.render_scene_snapshot();
    RENDER_CHECK(hidden_view.assets.size() == 1);
    RENDER_CHECK(hidden_view.instances.empty());
    instanced_session.render_next_frame(
        hidden_view,
        camera,
        settings,
        topology_state,
        instanced_frame);
    const renderer::CudaPathStatistics after_hide =
        *instanced_session.cuda_statistics();
    RENDER_CHECK(
        after_hide.blas_build_count ==
        after_delete.blas_build_count);
    RENDER_CHECK(
        after_hide.geometry_upload_bytes ==
        after_delete.geometry_upload_bytes);
    const renderer::Color hidden_environment(
        0.01f,
        0.02f,
        0.03f);
    RENDER_CHECK(
        (instanced_frame.pixel(16, 12) -
         hidden_environment)
            .cwiseAbs()
            .maxCoeff() <
        1.0e-6f);

    RENDER_CHECK(document.set_object_visible(first, true));
    RENDER_CHECK(document.set_object_visible(second, true));
    instanced_session.render_next_frame(
        document.render_scene_snapshot(),
        camera,
        settings,
        topology_state,
        instanced_frame);
    const renderer::CudaPathStatistics after_show =
        *instanced_session.cuda_statistics();
    RENDER_CHECK(
        after_show.blas_build_count ==
        after_hide.blas_build_count);
    RENDER_CHECK(
        after_show.geometry_upload_bytes ==
        after_hide.geometry_upload_bytes);
}

renderer::Scene make_cuda_nee_test_scene(
    bool triangle_light,
    bool faces_receiver = true,
    bool two_sided = false,
    bool alpha_cutout = false,
    bool blocked = false,
    bool degenerate = false) {
    renderer::Scene scene;
    scene.environment = renderer::Color::Zero();

    renderer::Material receiver;
    receiver.base_color = renderer::Color(0.8f, 0.8f, 0.8f);
    renderer::Material light;
    light.type = renderer::MaterialType::Emissive;
    light.emission = renderer::Color(10.0f, 8.0f, 6.0f);
    light.two_sided = two_sided;
    if (alpha_cutout) {
        light.opacity_texture_id = 0;
        light.alpha_cutoff = 0.5f;
        scene.textures.emplace_back(
            1,
            1,
            std::vector<renderer::Color>{renderer::Color::Zero()});
    }
    scene.materials = {receiver, light};
    scene.triangles.emplace_back(
        renderer::Vec3(-10.0f, -10.0f, -1.0f),
        renderer::Vec3(10.0f, -10.0f, -1.0f),
        renderer::Vec3(0.0f, 10.0f, -1.0f),
        0);

    if (triangle_light) {
        const renderer::Vec3 a(-1.0f, 1.0f, 0.5f);
        const renderer::Vec3 b(0.0f, 3.0f, 0.5f);
        const renderer::Vec3 c = degenerate
            ? renderer::Vec3(1.0f, 5.0f, 0.5f)
            : renderer::Vec3(1.0f, 1.0f, 0.5f);
        if (faces_receiver) {
            scene.triangles.emplace_back(a, b, c, 1);
        } else {
            scene.triangles.emplace_back(a, c, b, 1);
        }
    } else {
        scene.spheres.emplace_back(
            renderer::Vec3(0.0f, 2.0f, 0.5f),
            0.75f,
            1);
    }

    if (blocked) {
        scene.triangles.emplace_back(
            renderer::Vec3(-2.0f, 0.25f, -0.25f),
            renderer::Vec3(2.0f, 0.25f, -0.25f),
            renderer::Vec3(2.0f, 2.0f, -0.25f),
            0);
        scene.triangles.emplace_back(
            renderer::Vec3(-2.0f, 0.25f, -0.25f),
            renderer::Vec3(2.0f, 2.0f, -0.25f),
            renderer::Vec3(-2.0f, 2.0f, -0.25f),
            0);
    }
    return scene;
}

renderer::Color render_cuda_nee_test_scene(
    const renderer::Scene& scene,
    int samples_per_pixel,
    std::uint64_t seed_offset = 321) {
    const renderer::Camera camera(
        renderer::Vec3::Zero(),
        -renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitY(),
        20.0f,
        1.0f);
    renderer::RenderSettings settings;
    settings.width = 1;
    settings.height = 1;
    settings.path.samples_per_pixel = samples_per_pixel;
    settings.path.sample_seed_offset = seed_offset;
    return render_cuda_scene(scene, camera, settings)
        .image.pixel(0, 0);
}

renderer::Scene make_cuda_rect_area_light_scene(
    bool faces_receiver = true,
    bool two_sided = false,
    bool blocked = false) {
    renderer::Scene scene = make_cuda_nee_test_scene(
        true,
        true,
        false,
        false,
        blocked);
    scene.triangles.erase(scene.triangles.begin() + 1);
    scene.materials.resize(1);
    renderer::RectAreaLight rectangle;
    rectangle.position = renderer::Vec3(0.0f, 2.0f, 0.5f);
    rectangle.axis_u = renderer::Vec3(1.0f, 0.0f, 0.0f);
    rectangle.axis_v = renderer::Vec3(
        0.0f,
        faces_receiver ? 1.0f : -1.0f,
        0.0f);
    rectangle.radiance = renderer::Color(10.0f, 8.0f, 6.0f);
    rectangle.two_sided = two_sided;
    scene.rect_area_lights.push_back(rectangle);
    return scene;
}

RENDER_TEST(test_cuda_pathtracer_emissive_nee_and_mis_when_available) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }

    const renderer::Scene triangle_scene =
        make_cuda_nee_test_scene(true);
    const renderer::Color triangle =
        render_cuda_nee_test_scene(triangle_scene, 256);
    const renderer::Color repeated =
        render_cuda_nee_test_scene(triangle_scene, 256);
    RENDER_CHECK(triangle.allFinite());
    RENDER_CHECK(triangle.x() > 0.1f);
    RENDER_CHECK((triangle - repeated).cwiseAbs().maxCoeff() < 1e-6f);

    const renderer::Color blocked = render_cuda_nee_test_scene(
        make_cuda_nee_test_scene(true, true, false, false, true),
        256);
    RENDER_CHECK(blocked.allFinite());
    RENDER_CHECK(blocked.maxCoeff() < triangle.maxCoeff() * 0.05f);

    const renderer::Color one_sided_back = render_cuda_nee_test_scene(
        make_cuda_nee_test_scene(true, false, false),
        256);
    const renderer::Color two_sided_back = render_cuda_nee_test_scene(
        make_cuda_nee_test_scene(true, false, true),
        256);
    RENDER_CHECK(one_sided_back.maxCoeff() < 1e-6f);
    RENDER_CHECK(two_sided_back.x() > 0.1f);

    const renderer::Color cutout = render_cuda_nee_test_scene(
        make_cuda_nee_test_scene(true, true, false, true),
        256);
    RENDER_CHECK(cutout.allFinite());
    RENDER_CHECK(cutout.maxCoeff() < 1e-6f);

    const renderer::Color sphere = render_cuda_nee_test_scene(
        make_cuda_nee_test_scene(false, true, true),
        512);
    RENDER_CHECK(sphere.allFinite());
    RENDER_CHECK(sphere.x() > 0.05f);

    const renderer::Color degenerate = render_cuda_nee_test_scene(
        make_cuda_nee_test_scene(true, true, false, false, false, true),
        64);
    RENDER_CHECK(degenerate.allFinite());
    RENDER_CHECK(degenerate.maxCoeff() < 1e-6f);

    const renderer::Color rectangle = render_cuda_nee_test_scene(
        make_cuda_rect_area_light_scene(),
        256);
    RENDER_CHECK(rectangle.allFinite());
    RENDER_CHECK(rectangle.x() > 0.1f);
    const renderer::Color blocked_rectangle = render_cuda_nee_test_scene(
        make_cuda_rect_area_light_scene(true, false, true),
        256);
    RENDER_CHECK(blocked_rectangle.allFinite());
    // The blocker receives light and legitimately returns some of it after a
    // diffuse/microfacet bounce. Energy-preserving GGX makes that indirect
    // component slightly stronger than the old single-scatter model.
    RENDER_CHECK(
        blocked_rectangle.maxCoeff() < rectangle.maxCoeff() * 0.15f);
    const renderer::Color rectangle_back = render_cuda_nee_test_scene(
        make_cuda_rect_area_light_scene(false, false),
        256);
    const renderer::Color rectangle_two_sided = render_cuda_nee_test_scene(
        make_cuda_rect_area_light_scene(false, true),
        256);
    RENDER_CHECK(rectangle_back.maxCoeff() < 1.0e-6f);
    RENDER_CHECK(rectangle_two_sided.x() > 0.1f);
}

renderer::Scene make_cuda_emissive_surface_contract_scene(
    renderer::AlphaMode alpha_mode,
    float vertex_alpha,
    bool use_emissive_uv1_transform) {
    renderer::Scene scene = make_cuda_nee_test_scene(
        true,
        true,
        true);
    renderer::Material& light = scene.materials[1];
    light.type = renderer::MaterialType::Pbr;
    light.alpha_mode = alpha_mode;
    light.opacity = alpha_mode == renderer::AlphaMode::Opaque ? 0.0f : 1.0f;
    light.alpha_cutoff = 0.5f;

    const renderer::Triangle& original = scene.triangles[1];
    std::array<renderer::TriangleVertex, 3> vertices;
    for (int index = 0; index < 3; ++index) {
        vertices[static_cast<std::size_t>(index)].position =
            original.vertex(index).position;
        vertices[static_cast<std::size_t>(index)].uv =
            renderer::Vec2(0.25f, 0.5f);
        vertices[static_cast<std::size_t>(index)].uv1 =
            renderer::Vec2(0.25f, 0.5f);
        vertices[static_cast<std::size_t>(index)].has_uv1 = true;
        vertices[static_cast<std::size_t>(index)].color =
            renderer::Color::Ones();
        vertices[static_cast<std::size_t>(index)].alpha = vertex_alpha;
        vertices[static_cast<std::size_t>(index)].has_color = true;
    }
    scene.triangles[1] = renderer::Triangle(
        vertices[0],
        vertices[1],
        vertices[2],
        1);

    if (alpha_mode == renderer::AlphaMode::Opaque) {
        scene.textures.emplace_back(
            1,
            1,
            std::vector<renderer::Color>{renderer::Color::Zero()});
        light.opacity_texture_id = 0;
    }
    if (use_emissive_uv1_transform) {
        scene.textures.emplace_back(
            2,
            1,
            std::vector<renderer::Color>{
                renderer::Color::Zero(),
                renderer::Color::Ones()});
        scene.textures.back().set_sampler(
            renderer::TextureWrap::ClampToEdge,
            renderer::TextureWrap::ClampToEdge,
            renderer::TextureFilter::Nearest,
            renderer::TextureFilter::Nearest);
        light.emissive_texture_id =
            static_cast<int>(scene.textures.size() - 1);
        light.emissive_texture_transform.texcoord = 1;
        light.emissive_texture_transform.offset =
            renderer::Vec2(0.5f, 0.0f);
    }
    return scene;
}

RENDER_TEST(test_cuda_emissive_nee_surface_contract_when_available) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }

    const renderer::Color opaque = render_cuda_nee_test_scene(
        make_cuda_emissive_surface_contract_scene(
            renderer::AlphaMode::Opaque,
            0.0f,
            false),
        512,
        1201);
    const renderer::Color mask = render_cuda_nee_test_scene(
        make_cuda_emissive_surface_contract_scene(
            renderer::AlphaMode::Mask,
            0.0f,
            false),
        512,
        1201);
    const renderer::Color blend = render_cuda_nee_test_scene(
        make_cuda_emissive_surface_contract_scene(
            renderer::AlphaMode::Blend,
            0.25f,
            false),
        512,
        1201);
    renderer::Scene transformed_uv1_scene =
        make_cuda_emissive_surface_contract_scene(
            renderer::AlphaMode::Opaque,
            1.0f,
            true);
    renderer::Scene untransformed_uv1_scene = transformed_uv1_scene;
    untransformed_uv1_scene.materials[1]
        .emissive_texture_transform.offset = renderer::Vec2::Zero();
    const renderer::Color transformed_uv1 = render_cuda_nee_test_scene(
        transformed_uv1_scene,
        512,
        1301);
    const renderer::Color untransformed_uv1 = render_cuda_nee_test_scene(
        untransformed_uv1_scene,
        512,
        1301);

    RENDER_CHECK(opaque.allFinite());
    RENDER_CHECK(mask.allFinite());
    RENDER_CHECK(blend.allFinite());
    RENDER_CHECK(transformed_uv1.allFinite());
    RENDER_CHECK(untransformed_uv1.allFinite());
    RENDER_CHECK(opaque.x() > 0.1f);
    RENDER_CHECK(mask.maxCoeff() < 1.0e-6f);
    RENDER_CHECK(blend.x() > 0.01f);
    RENDER_CHECK(blend.x() < opaque.x() * 0.5f);
    RENDER_CHECK(transformed_uv1.x() > 0.1f);
    RENDER_CHECK(untransformed_uv1.maxCoeff() < 1.0e-6f);
}

renderer::Color render_instanced_document_pixel(
    renderer::SceneDocument& document,
    const renderer::Camera& camera,
    int samples_per_pixel,
    std::uint64_t seed_offset) {
    renderer::RenderSettings settings;
    settings.width = 1;
    settings.height = 1;
    settings.path.sample_seed_offset = seed_offset;
    renderer::PathInteractiveSession session;
    renderer::Framebuffer framebuffer(1, 1);
    const renderer::InteractiveFrameState frame_state;
    session.reset(
        document.render_scene_snapshot(),
        settings);
    for (int sample = 0;
         sample < samples_per_pixel;
         ++sample) {
        session.render_next_frame(
            document.render_scene_snapshot(),
            camera,
            settings,
            frame_state,
            framebuffer);
    }
    RENDER_CHECK(
        session.accumulated_samples() ==
        samples_per_pixel);
    return framebuffer.pixel(0, 0);
}

RENDER_TEST(test_cuda_instanced_nee_mis_transforms_when_available) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }

    const renderer::Camera camera(
        renderer::Vec3::Zero(),
        -renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitY(),
        20.0f,
        1.0f);
    renderer::SceneDocument triangle_document =
        renderer::SceneDocument::from_scene(
            make_cuda_nee_test_scene(
                true,
                true,
                false),
            "Instanced NEE triangle");
    const renderer::ObjectId triangle_object =
        triangle_document.objects().front().id;
    RENDER_CHECK(triangle_document.set_world_matrix(
        triangle_object,
        make_test_instance_matrix(
            renderer::Vec3::Zero(),
            12.0f,
            renderer::Vec3(1.35f, 0.8f, 1.0f))));
    const renderer::Scene transformed_flat =
        renderer::flatten_render_scene_snapshot(triangle_document.render_scene_snapshot());
    const renderer::Color flat_value =
        render_cuda_nee_test_scene(
            transformed_flat,
            1024,
            719);
    const renderer::Color instanced_value =
        render_instanced_document_pixel(
            triangle_document,
            camera,
            1024,
            719);
    RENDER_CHECK(flat_value.allFinite());
    RENDER_CHECK(instanced_value.allFinite());
    RENDER_CHECK(instanced_value.x() > 0.05f);
    for (int channel = 0; channel < 3; ++channel) {
        const float tolerance = std::max(
            1.0e-4f,
            std::abs(flat_value[channel]) * 0.01f);
        RENDER_CHECK(
            std::abs(
                flat_value[channel] -
                instanced_value[channel]) <=
            tolerance);
    }

    renderer::SceneDocument sphere_document =
        renderer::SceneDocument::from_scene(
            make_cuda_nee_test_scene(
                false,
                true,
                true),
            "Instanced NEE sphere");
    const renderer::ObjectId sphere_object =
        sphere_document.objects().front().id;
    RENDER_CHECK(sphere_document.set_world_matrix(
        sphere_object,
        make_test_instance_matrix(
            renderer::Vec3::Zero(),
            -17.0f,
            renderer::Vec3(1.5f, 0.65f, 1.0f))));
    const renderer::Color sphere_value =
        render_instanced_document_pixel(
            sphere_document,
            camera,
            1024,
            811);
    RENDER_CHECK(sphere_value.allFinite());
    RENDER_CHECK(sphere_value.x() > 0.02f);

    renderer::SceneDocument alpha_document =
        renderer::SceneDocument::from_scene(
            make_cuda_nee_test_scene(
                true,
                true,
                false,
                true),
            "Instanced alpha-cutout light");
    const renderer::Color alpha_value =
        render_instanced_document_pixel(
            alpha_document,
            camera,
            std::min(128, 1024),
            913);
    RENDER_CHECK(alpha_value.allFinite());
    RENDER_CHECK(alpha_value.maxCoeff() < 1.0e-6f);
}

RENDER_TEST(test_cuda_pathtracer_lighting_contracts_when_available) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }

    renderer::Scene environment_scene;
    environment_scene.environment = renderer::Color(0.2f, 0.3f, 0.4f);
    const renderer::Color environment = render_one_path_pixel(environment_scene);
    RENDER_CHECK((environment - environment_scene.environment).cwiseAbs().maxCoeff() < 1e-6f);

    renderer::Scene back_face_scene;
    back_face_scene.environment = renderer::Color::Zero();
    renderer::Material one_sided_light;
    one_sided_light.type = renderer::MaterialType::Emissive;
    one_sided_light.emission = renderer::Color(1.5f, 0.5f, 0.25f);
    one_sided_light.two_sided = false;
    back_face_scene.materials.push_back(one_sided_light);
    back_face_scene.triangles.emplace_back(
        renderer::Vec3(-1.0f, -1.0f, -1.0f),
        renderer::Vec3(0.0f, 1.0f, -1.0f),
        renderer::Vec3(1.0f, -1.0f, -1.0f),
        0);
    const renderer::Color one_sided = render_one_path_pixel(back_face_scene);
    RENDER_CHECK(one_sided.maxCoeff() < 1e-6f);
    back_face_scene.materials[0].two_sided = true;
    const renderer::Color two_sided = render_one_path_pixel(back_face_scene);
    RENDER_CHECK(two_sided.x() > 1.4f);

    renderer::Scene dark = make_path_direct_light_scene();
    const renderer::Color unlit = render_one_path_pixel(dark);
    renderer::Scene directional = dark;
    directional.directional_lights.push_back(renderer::DirectionalLight{
        renderer::Vec3(0.0f, 0.0f, -1.0f),
        renderer::Color(2.0f, 2.0f, 2.0f)});
    const renderer::Color directionally_lit = render_one_path_pixel(directional);
    RENDER_CHECK(directionally_lit.x() > unlit.x() + 0.5f);

    renderer::Scene near_scene = dark;
    near_scene.point_lights.push_back(renderer::PointLight{
        renderer::Vec3(0.0f, 0.0f, 1.0f),
        renderer::Color(8.0f, 8.0f, 8.0f)});
    renderer::Scene far_scene = dark;
    far_scene.point_lights.push_back(renderer::PointLight{
        renderer::Vec3(0.0f, 0.0f, 3.0f),
        renderer::Color(8.0f, 8.0f, 8.0f)});
    const renderer::Color near_value = render_one_path_pixel(near_scene);
    const renderer::Color far_value = render_one_path_pixel(far_scene);
    RENDER_CHECK(near_value.x() > far_value.x() * 3.5f);

    renderer::Scene visible = dark;
    visible.point_lights.push_back(renderer::PointLight{
        renderer::Vec3(2.0f, 0.0f, 0.0f),
        renderer::Color(20.0f, 20.0f, 20.0f)});
    renderer::Scene blocked = visible;
    blocked.triangles.emplace_back(
        renderer::Vec3(1.0f, -10.0f, -2.0f),
        renderer::Vec3(1.0f, 10.0f, -2.0f),
        renderer::Vec3(1.0f, 0.0f, 1.0f),
        0);
    const renderer::Color visible_value = render_one_path_pixel(visible);
    const renderer::Color blocked_value = render_one_path_pixel(blocked);
    RENDER_CHECK(visible_value.x() > blocked_value.x() + 0.05f);

    const renderer::Scene roulette_scene = make_path_roulette_layer_scene();
    const renderer::Camera camera(
        renderer::Vec3::Zero(),
        -renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitY(),
        10.0f,
        1.0f);
    renderer::RenderSettings settings;
    settings.width = 1;
    settings.height = 1;
    settings.path.samples_per_pixel = 2048;
    const renderer::Color roulette_mean =
        render_cuda_scene(roulette_scene, camera, settings).image.pixel(0, 0);
    RENDER_CHECK(roulette_mean.allFinite());
    RENDER_CHECK(roulette_mean.x() > 0.9f);
    RENDER_CHECK(roulette_mean.x() < 1.1f);

    settings.path.samples_per_pixel = 1;
    settings.path.russian_roulette_start_bounce = 64;
    settings.path.max_bounces = 4;
    const renderer::Color truncated =
        render_cuda_scene(roulette_scene, camera, settings).image.pixel(0, 0);
    RENDER_CHECK(truncated.isApprox(renderer::Color::Zero(), 1e-6f));
    settings.path.max_bounces = 5;
    const renderer::Color complete =
        render_cuda_scene(roulette_scene, camera, settings).image.pixel(0, 0);
    RENDER_CHECK(complete.isApprox(renderer::Color::Ones(), 1e-6f));
}

RENDER_TEST(test_cuda_pathtracer_spheres_materials_and_bump_texture_when_available) {
    if (!renderer::cuda_path_backend_available()) {
        RENDER_SKIP("CUDA path backend unavailable");
    }

    renderer::Scene scene;
    scene.environment = renderer::Color(0.08f, 0.12f, 0.18f);
    renderer::Material diffuse;
    diffuse.base_color = renderer::Color(0.8f, 0.8f, 0.8f);
    diffuse.diffuse_texture_id = 0;
    diffuse.bump_texture_id = 1;
    diffuse.bump_scale = 0.4f;
    renderer::Material metal;
    metal.type = renderer::MaterialType::Metal;
    metal.base_color = renderer::Color(0.9f, 0.75f, 0.55f);
    metal.roughness = 0.08f;
    renderer::Material dielectric;
    dielectric.type = renderer::MaterialType::Dielectric;
    dielectric.ior = 1.5f;
    scene.materials = {diffuse, metal, dielectric};
    scene.textures.emplace_back(
        2,
        2,
        std::vector<renderer::Color>{
            renderer::Color(1.0f, 0.2f, 0.2f), renderer::Color(0.2f, 1.0f, 0.2f),
            renderer::Color(0.2f, 0.2f, 1.0f), renderer::Color(1.0f, 1.0f, 0.2f)});
    scene.textures.emplace_back(
        2,
        2,
        std::vector<renderer::Color>{
            renderer::Color::Zero(), renderer::Color::Ones(),
            renderer::Color::Ones(), renderer::Color::Zero()});
    scene.triangles.emplace_back(
        renderer::Vec3(-3.0f, -1.0f, -2.0f),
        renderer::Vec3(3.0f, -1.0f, -2.0f),
        renderer::Vec3(0.0f, 2.5f, -3.0f),
        0,
        renderer::Vec2(0.0f, 0.0f),
        renderer::Vec2(2.0f, 0.0f),
        renderer::Vec2(1.0f, 2.0f));
    scene.spheres.emplace_back(renderer::Vec3(-0.7f, -0.25f, -1.3f), 0.45f, 1);
    scene.spheres.emplace_back(renderer::Vec3(0.7f, -0.25f, -1.3f), 0.45f, 2);
    scene.directional_lights.push_back(renderer::DirectionalLight{
        renderer::Vec3(-0.4f, -0.6f, -1.0f).normalized(),
        renderer::Color(2.0f, 2.0f, 2.0f)});

    const renderer::Camera camera(
        renderer::Vec3(0.0f, 0.0f, 1.2f),
        renderer::Vec3(0.0f, 0.0f, -2.0f),
        renderer::Vec3::UnitY(),
        45.0f,
        1.0f);
    renderer::RenderSettings settings;
    settings.width = 40;
    settings.height = 40;
    settings.path.samples_per_pixel = 32;
    settings.path.sample_seed_offset = 123;
    const renderer::RenderResult first =
        render_cuda_scene(scene, camera, settings);
    const renderer::RenderResult cuda =
        render_cuda_scene(scene, camera, settings);
    RENDER_CHECK(image_colors_are_finite(cuda.image));
    const renderer::Color first_mean = image_mean(first.image);
    const renderer::Color cuda_mean = image_mean(cuda.image);
    for (int channel = 0; channel < 3; ++channel) {
        RENDER_CHECK(nearly_equal(first_mean[channel], cuda_mean[channel], 1.0e-6f));
    }

    for (int cycle = 0; cycle < 3; ++cycle) {
        settings.width = 8;
        settings.height = 8;
        settings.path.samples_per_pixel = 1;
        const renderer::RenderResult recreated =
            render_cuda_scene(scene, camera, settings);
        RENDER_CHECK(image_colors_are_finite(recreated.image));
    }
}

RENDER_TEST(test_scene_document_import_transform_hierarchy_history_and_roundtrip) {
    const std::filesystem::path directory = "test_scene_document_assets";
    const std::filesystem::path nested = directory / "nested";
    const std::filesystem::path scene_path = directory / "test_scene.rscene";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(nested);
    const auto write_triangle = [](const std::filesystem::path& path, float x) {
        std::ofstream obj(path);
        obj << "v " << x - 0.5f << " -0.5 0\n";
        obj << "v " << x + 0.5f << " -0.5 0\n";
        obj << "v " << x << " 0.5 0\n";
        obj << "f 1 2 3\n";
    };
    write_triangle(directory / "first.obj", 0.0f);
    write_triangle(nested / "second.obj", 2.0f);

    renderer::SceneDocument document;
    const std::vector<renderer::ObjectId> imported =
        document.import_path(directory, 64, 64);
    RENDER_CHECK(imported.size() == 2);
    RENDER_CHECK(document.assets().size() == 2);
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(document.render_scene_snapshot()).triangles.size() == 2);
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(document.render_scene_snapshot()).directional_lights.size() == 1);

    const renderer::SceneObject* first = document.find(imported[0]);
    RENDER_CHECK(first != nullptr);
    const renderer::ObjectId first_id = first->id;
    renderer::SceneTrs first_transform;
    first_transform.translation = renderer::Vec3(3.0f, 1.0f, -2.0f);
    RENDER_CHECK(document.set_local_trs(first_id, first_transform));
    document.checkpoint();
    const renderer::Mat4 before_reparent = document.world_matrix(first_id);
    const renderer::ObjectId group = document.create_group("Moved group");
    const renderer::SceneObject* group_object = document.find(group);
    renderer::SceneTrs group_transform;
    group_transform.translation = renderer::Vec3(-4.0f, 2.0f, 0.0f);
    RENDER_CHECK(document.set_local_trs(group_object->id, group_transform));
    document.checkpoint();
    RENDER_CHECK(document.reparent(first_id, group));
    RENDER_CHECK(document.world_matrix(first_id).isApprox(before_reparent, 1.0e-4f));

    const renderer::ObjectId duplicate = document.duplicate_subtree(first_id);
    RENDER_CHECK(duplicate != renderer::kInvalidObjectId);
    RENDER_CHECK(document.assets().size() == 2);
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(document.render_scene_snapshot()).triangles.size() == 3);
    RENDER_CHECK(document.undo());
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(document.render_scene_snapshot()).triangles.size() == 2);
    RENDER_CHECK(document.redo());
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(document.render_scene_snapshot()).triangles.size() == 3);

    const renderer::SceneObject* root = nullptr;
    for (const renderer::SceneObject& object : document.objects()) {
        if (object.type == renderer::SceneObjectType::Group &&
            object.parent_id == renderer::kInvalidObjectId &&
            object.name == directory.filename().string()) {
            root = &object;
            break;
        }
    }
    RENDER_CHECK(root != nullptr);
    RENDER_CHECK(document.set_object_visible(root->id, false));
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(document.render_scene_snapshot()).triangles.size() == 2);
    RENDER_CHECK(document.set_object_visible(root->id, true));
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(document.render_scene_snapshot()).triangles.size() == 3);

    document.save(scene_path);
    RENDER_CHECK(!document.dirty());
    renderer::SceneDocument loaded =
        renderer::SceneDocument::load(scene_path, 64, 64);
    RENDER_CHECK(!loaded.dirty());
    RENDER_CHECK(loaded.assets().size() == 2);
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(loaded.render_scene_snapshot()).triangles.size() == 3);
    RENDER_CHECK(loaded.objects().size() == document.objects().size());

    const renderer::Ray pick_ray(
        renderer::Vec3(3.0f, 1.0f, 2.0f),
        renderer::Vec3(0.0f, 0.0f, -1.0f));
    const auto pick = loaded.pick(pick_ray);
    RENDER_CHECK(pick.has_value());

    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_scene_document_material_overrides_are_per_object_and_roundtrip) {
    const std::filesystem::path directory =
        "test_scene_document_material_overrides";
    const std::filesystem::path obj_path = directory / "model.obj";
    const std::filesystem::path mtl_path = directory / "model.mtl";
    const std::filesystem::path texture_path = directory / "surface.ppm";
    const std::filesystem::path scene_path = directory / "materials.rscene";
    const std::filesystem::path version_one_path =
        directory / "materials-v1.rscene";
    const std::filesystem::path invalid_slot_path =
        directory / "materials-invalid-slot.rscene";
    const std::filesystem::path missing_asset_path =
        directory / "materials-missing-asset.rscene";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    write_single_pixel_ppm(texture_path.string(), 128);
    {
        std::ofstream mtl(mtl_path);
        mtl << "newmtl TexturedSurface\n";
        mtl << "Kd 0.8 0.6 0.4\n";
        mtl << "d 0.9\n";
        mtl << "map_Kd surface.ppm\n";
        mtl << "map_d surface.ppm\n";
        mtl << "bump -bm 0.3 surface.ppm\n";
        mtl << "newmtl PlainSurface\n";
        mtl << "Kd 0.2 0.3 0.4\n";
    }
    {
        std::ofstream obj(obj_path);
        obj << "mtllib model.mtl\n";
        obj << "v -1 -1 0\n";
        obj << "v 0 -1 0\n";
        obj << "v -1 1 0\n";
        obj << "v 0 -1 0\n";
        obj << "v 1 -1 0\n";
        obj << "v 1 1 0\n";
        obj << "vt 0 0\nvt 1 0\nvt 0 1\n";
        obj << "vn 0 0 1\n";
        obj << "usemtl TexturedSurface\n";
        obj << "f 1/1/1 2/2/1 3/3/1\n";
        obj << "usemtl PlainSurface\n";
        obj << "f 4/1/1 5/2/1 6/3/1\n";
    }

    renderer::SceneDocument document;
    const renderer::ObjectId first_id =
        document.import_path(obj_path, 64, 64).front();
    const renderer::ObjectId second_id =
        document.import_path(obj_path, 64, 64).front();
    RENDER_CHECK(document.assets().size() == 1);
    const renderer::SceneMeshAsset* asset =
        document.asset_for_object(first_id);
    RENDER_CHECK(asset != nullptr);
    RENDER_CHECK(asset == document.asset_for_object(second_id));
    RENDER_CHECK(asset->material_names.size() == 3);
    RENDER_CHECK(asset->material_names[0] == "TexturedSurface");
    RENDER_CHECK(asset->material_names[1] == "PlainSurface");
    RENDER_CHECK(asset->material_names[2] == "<default>");

    const auto source_textured = document.material_properties(first_id, 0);
    const auto source_plain = document.material_properties(first_id, 1);
    RENDER_CHECK(source_textured.has_value());
    RENDER_CHECK(source_plain.has_value());
    RENDER_CHECK(source_textured->use_diffuse_texture);
    RENDER_CHECK(source_textured->use_opacity_texture);
    RENDER_CHECK(source_textured->use_bump_texture);
    RENDER_CHECK(!source_plain->use_diffuse_texture);
    RENDER_CHECK(!source_plain->use_opacity_texture);
    RENDER_CHECK(!source_plain->use_bump_texture);

    renderer::SceneMaterialOverride material_override = *source_textured;
    material_override.type = renderer::MaterialType::Metal;
    material_override.base_color = renderer::Color(0.25f, 0.5f, 0.75f);
    material_override.emission = renderer::Color(2.0f, 3.0f, 4.0f);
    material_override.roughness = 0.15f;
    material_override.ior = 1.7f;
    material_override.opacity = 0.55f;
    material_override.alpha_cutoff = 0.4f;
    material_override.bump_scale = 0.7f;
    material_override.two_sided = false;
    RENDER_CHECK(document.set_material_override(first_id, material_override));
    document.checkpoint();
    RENDER_CHECK(document.material_override(first_id, 0) != nullptr);
    RENDER_CHECK(document.material_override(second_id, 0) == nullptr);

    const renderer::Scene& overridden_scene = renderer::flatten_render_scene_snapshot(document.render_scene_snapshot());
    RENDER_CHECK(overridden_scene.triangles.size() == 4);
    const int first_material_id =
        overridden_scene.triangles[0].material_id();
    const int second_material_id =
        overridden_scene.triangles[2].material_id();
    RENDER_CHECK(first_material_id >= 0);
    RENDER_CHECK(second_material_id >= 0);
    const renderer::Material& first_material =
        overridden_scene.materials[static_cast<std::size_t>(first_material_id)];
    const renderer::Material& second_material =
        overridden_scene.materials[static_cast<std::size_t>(second_material_id)];
    RENDER_CHECK(first_material.type == renderer::MaterialType::Metal);
    RENDER_CHECK(second_material.type == renderer::MaterialType::Diffuse);
    RENDER_CHECK(nearly_equal(first_material.base_color.x(), 0.25f));
    RENDER_CHECK(nearly_equal(second_material.base_color.x(), 0.8f));
    RENDER_CHECK(first_material.diffuse_texture_id >= 0);
    RENDER_CHECK(first_material.diffuse_texture_id ==
        second_material.diffuse_texture_id);
    RENDER_CHECK(first_material.opacity_texture_id ==
        second_material.opacity_texture_id);
    RENDER_CHECK(first_material.bump_texture_id ==
        second_material.bump_texture_id);
    RENDER_CHECK(overridden_scene.textures.size() ==
        asset->local_scene.textures.size());
    const renderer::Color texture_sample =
        overridden_scene
            .textures[static_cast<std::size_t>(
                first_material.diffuse_texture_id)]
            .sample(renderer::Vec2::Zero());
    const renderer::Color tinted =
        renderer::sample_material_base_color(
            overridden_scene,
            first_material,
            renderer::Vec2::Zero());
    RENDER_CHECK(nearly_equal(
        tinted.x(),
        texture_sample.x() * material_override.base_color.x()));
    RENDER_CHECK(nearly_equal(
        tinted.y(),
        texture_sample.y() * material_override.base_color.y()));
    RENDER_CHECK(nearly_equal(
        tinted.z(),
        texture_sample.z() * material_override.base_color.z()));

    RENDER_CHECK(document.undo());
    RENDER_CHECK(document.material_override(first_id, 0) == nullptr);
    RENDER_CHECK(document.redo());
    RENDER_CHECK(document.material_override(first_id, 0) != nullptr);

    const renderer::ObjectId duplicate_id =
        document.duplicate_subtree(first_id);
    RENDER_CHECK(duplicate_id != renderer::kInvalidObjectId);
    RENDER_CHECK(document.material_override(duplicate_id, 0) != nullptr);
    renderer::SceneMaterialOverride duplicate_override =
        *document.material_override(duplicate_id, 0);
    duplicate_override.base_color = renderer::Color(0.9f, 0.1f, 0.2f);
    duplicate_override.use_diffuse_texture = false;
    RENDER_CHECK(document.set_material_override(
        duplicate_id,
        duplicate_override));
    document.checkpoint();
    RENDER_CHECK(document.material_override(first_id, 0)->base_color.isApprox(
        material_override.base_color));
    RENDER_CHECK(document.material_override(duplicate_id, 0)->base_color.isApprox(
        duplicate_override.base_color));
    const renderer::Scene& duplicated_scene = renderer::flatten_render_scene_snapshot(document.render_scene_snapshot());
    RENDER_CHECK(
        duplicated_scene
            .materials[static_cast<std::size_t>(
                duplicated_scene.triangles[0].material_id())]
            .diffuse_texture_id >= 0);
    RENDER_CHECK(
        duplicated_scene
            .materials[static_cast<std::size_t>(
                duplicated_scene.triangles[4].material_id())]
            .diffuse_texture_id == -1);

    const renderer::SceneObject* first_object = document.find(first_id);
    RENDER_CHECK(first_object != nullptr);
    RENDER_CHECK(document.set_object_locked(first_id, true));
    RENDER_CHECK(!document.set_material_override(first_id, material_override));
    RENDER_CHECK(!document.clear_material_override(first_id, 0));
    RENDER_CHECK(document.set_object_locked(first_id, false));

    RENDER_CHECK(document.clear_material_override(duplicate_id, 0));
    document.checkpoint();
    RENDER_CHECK(document.material_override(duplicate_id, 0) == nullptr);
    RENDER_CHECK(document.undo());
    RENDER_CHECK(document.material_override(duplicate_id, 0) != nullptr);
    RENDER_CHECK(document.redo());
    RENDER_CHECK(document.material_override(duplicate_id, 0) == nullptr);
    const auto reset_properties =
        document.material_properties(duplicate_id, 0);
    RENDER_CHECK(reset_properties.has_value());
    RENDER_CHECK(reset_properties->base_color.isApprox(
        asset->local_scene.materials[0].base_color));

    document.save(scene_path);
    nlohmann::json saved_json;
    {
        std::ifstream input(scene_path);
        input >> saved_json;
    }
    RENDER_CHECK(saved_json.at("version").get<int>() == 5);
    std::size_t objects_with_overrides = 0;
    for (const auto& object_json : saved_json.at("objects")) {
        if (object_json.contains("material_overrides")) {
            ++objects_with_overrides;
        }
    }
    RENDER_CHECK(objects_with_overrides == 1);

    renderer::SceneDocument loaded =
        renderer::SceneDocument::load(scene_path, 64, 64);
    RENDER_CHECK(!loaded.dirty());
    RENDER_CHECK(loaded.assets().size() == 1);
    RENDER_CHECK(loaded.material_override(first_id, 0) != nullptr);
    RENDER_CHECK(loaded.material_override(second_id, 0) == nullptr);
    RENDER_CHECK(loaded.material_override(duplicate_id, 0) == nullptr);
    RENDER_CHECK(loaded.asset_for_object(first_id)->material_names[0] ==
        "TexturedSurface");
    RENDER_CHECK(loaded.material_override(first_id, 0)->base_color.isApprox(
        material_override.base_color));
    RENDER_CHECK(
        loaded.material_override(first_id, 0)->type ==
        material_override.type);
    RENDER_CHECK(nearly_equal(
        loaded.material_override(first_id, 0)->roughness,
        material_override.roughness));
    RENDER_CHECK(nearly_equal(
        loaded.material_override(first_id, 0)->ior,
        material_override.ior));
    RENDER_CHECK(loaded.material_override(first_id, 0)->emission.isApprox(
        material_override.emission));
    RENDER_CHECK(nearly_equal(
        loaded.material_override(first_id, 0)->opacity,
        material_override.opacity));
    RENDER_CHECK(nearly_equal(
        loaded.material_override(first_id, 0)->alpha_cutoff,
        material_override.alpha_cutoff));
    RENDER_CHECK(nearly_equal(
        loaded.material_override(first_id, 0)->bump_scale,
        material_override.bump_scale));
    RENDER_CHECK(
        loaded.material_override(first_id, 0)->two_sided ==
        material_override.two_sided);
    RENDER_CHECK(loaded.material_override(first_id, 0)->use_diffuse_texture);
    RENDER_CHECK(loaded.material_override(first_id, 0)->use_opacity_texture);
    RENDER_CHECK(loaded.material_override(first_id, 0)->use_bump_texture);

    nlohmann::json version_one_json = saved_json;
    version_one_json["version"] = 1;
    version_one_json["environment"] = saved_json["environment"]["color"];
    for (auto& asset_json : version_one_json["assets"]) {
        asset_json.erase("kind");
        asset_json.erase("mesh_index");
    }
    for (auto& object_json : version_one_json["objects"]) {
        object_json.erase("material_overrides");
        object_json.erase("local_matrix");
        object_json["translation"] = nlohmann::json::array({0.0f, 0.0f, 0.0f});
        object_json["rotation_degrees"] = nlohmann::json::array({0.0f, 0.0f, 0.0f});
        object_json["scale"] = nlohmann::json::array({1.0f, 1.0f, 1.0f});
    }
    {
        std::ofstream output(version_one_path);
        output << version_one_json.dump(2) << '\n';
    }
    renderer::SceneDocument version_one =
        renderer::SceneDocument::load(version_one_path, 64, 64);
    for (const renderer::SceneObject& object : version_one.objects()) {
        RENDER_CHECK(object.material_overrides.empty());
    }

    nlohmann::json invalid_slot_json = saved_json;
    bool changed_slot = false;
    for (auto& object_json : invalid_slot_json["objects"]) {
        if (object_json.contains("material_overrides")) {
            object_json["material_overrides"][0]["slot"] = 999;
            changed_slot = true;
            break;
        }
    }
    RENDER_CHECK(changed_slot);
    {
        std::ofstream output(invalid_slot_path);
        output << invalid_slot_json.dump(2) << '\n';
    }
    renderer::SceneDocument invalid_slot =
        renderer::SceneDocument::load(invalid_slot_path, 64, 64);
    RENDER_CHECK(invalid_slot.material_override(first_id, 999) != nullptr);
    RENDER_CHECK(!invalid_slot.material_properties(first_id, 999).has_value());
    const bool saw_slot_warning = std::any_of(
        invalid_slot.warnings().begin(),
        invalid_slot.warnings().end(),
        [](const std::string& warning) {
            return warning.find("material override slot") != std::string::npos;
        });
    RENDER_CHECK(saw_slot_warning);
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(invalid_slot.render_scene_snapshot()).triangles.size() == 6);

    nlohmann::json missing_asset_json = saved_json;
    missing_asset_json["assets"][0]["path"] = "missing.obj";
    {
        std::ofstream output(missing_asset_path);
        output << missing_asset_json.dump(2) << '\n';
    }
    renderer::SceneDocument missing_asset =
        renderer::SceneDocument::load(missing_asset_path, 64, 64);
    RENDER_CHECK(missing_asset.material_override(first_id, 0) != nullptr);
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(missing_asset.render_scene_snapshot()).triangles.empty());
    const bool saw_missing_warning = std::any_of(
        missing_asset.warnings().begin(),
        missing_asset.warnings().end(),
        [](const std::string& warning) {
            return warning.find("missing asset") != std::string::npos;
        });
    RENDER_CHECK(saw_missing_warning);

    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_viewer_session_roundtrip_and_partial_asset_recovery) {
    const std::filesystem::path directory = "test_viewer_session";
    const std::filesystem::path obj_path = directory / "model.obj";
    const std::filesystem::path session_path = directory / "last-session.json";
    const std::filesystem::path original_scene_path =
        directory / "original.rscene";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    {
        std::ofstream obj(obj_path);
        obj << "v -1 -1 0\n";
        obj << "v 1 -1 0\n";
        obj << "v 0 1 0\n";
        obj << "f 1 2 3\n";
    }

    renderer::SceneDocument document;
    const renderer::ObjectId imported =
        document.import_path(obj_path, 64, 64).front();
    const renderer::SceneObject* imported_object = document.find(imported);
    RENDER_CHECK(imported_object != nullptr);
    renderer::SceneTrs imported_transform;
    imported_transform.translation = renderer::Vec3(2.0f, 3.0f, 4.0f);
    RENDER_CHECK(document.set_local_trs(imported, imported_transform));
    const auto source_material =
        document.material_properties(imported, 0);
    RENDER_CHECK(source_material.has_value());
    renderer::SceneMaterialOverride material_override = *source_material;
    material_override.base_color = renderer::Color(0.2f, 0.4f, 0.8f);
    RENDER_CHECK(document.set_material_override(imported, material_override));
    document.checkpoint();
    document.restore_file_state(original_scene_path, true);
    const std::filesystem::path file_path_before = document.file_path();
    RENDER_CHECK(document.dirty());

    renderer::ViewerSessionState state;
    state.document_path = document.file_path();
    state.document_dirty = document.dirty();
    state.window_width = 1400;
    state.window_height = 900;
    state.ui.mode = renderer::InteractiveRenderMode::Path;
    state.ui.camera_mode = renderer::ViewerCameraMode::Free;
    state.ui.render_scale = 0.75f;
    state.ui.ui_font_scale = 1.25f;
    state.ui.automatic_interaction_quality = false;
    state.ui.path_accumulation_paused = true;
    state.render_settings.realtime.internal_scale=.5f;
    state.render_settings.realtime.diffuse_history=48;
    state.render_settings.realtime.denoiser=renderer::RealtimeDenoiser::Optix;
    state.render_settings.realtime.debug_view=renderer::RealtimeDebugView::Variance;
    state.ui.show_point_light_markers = false;
    state.ui.panel_visible = false;
    state.ui.scene_panel_visible = false;
    state.ui.inspector_panel_visible = true;
    state.ui.rendering_panel_visible = false;
    state.ui.camera_lighting_panel_visible = true;
    state.ui.techniques_panel_visible = false;
    state.ui.display.exposure_ev = 1.5f;
    state.ui.display.tone_mapper = renderer::ToneMapper::Aces;
    state.ui.selected_objects = {imported};
    state.ui.active_object = imported;
    state.ui.material_editor_object = imported;
    state.ui.selected_material_slot = 0;
    state.ui.gizmo_operation = 2;
    state.ui.gizmo_local = true;
    state.render_settings.path.cuda_device = 0;
    state.render_settings.path.max_bounces = 12;
    state.render_settings.opengl.npr.style = renderer::OpenGlRenderStyle::Sketch;
    state.render_settings.opengl.npr.toon_levels = 5;
    state.render_settings.opengl.npr.outline_width = 2.5f;
    state.render_settings.opengl.npr.outline_strength = 0.7f;
    state.render_settings.opengl.npr.sketch_scale = 12.0f;
    state.render_settings.opengl.npr.sketch_tone = 1.4f;
    state.render_settings.opengl.npr.sketch_use_uv = true;
    state.render_settings.path.russian_roulette_start_bounce = 5;
    state.render_settings.path.russian_roulette_min_probability = 0.10f;
    state.render_settings.path.russian_roulette_max_probability = 0.90f;
    state.render_settings.opengl.ibl_enabled = false;
    state.render_settings.opengl.ltc_area_lights_enabled = false;
    state.render_settings.opengl.shadow_map.enabled = false;
    state.render_settings.opengl.shadow_map.resolution = 2048;
    state.render_settings.opengl.shadow_map.max_shadow_lights = 5;
    state.render_settings.opengl.shadow_map.constant_bias = 0.001f;
    state.render_settings.opengl.shadow_map.slope_bias = 0.004f;
    state.render_settings.opengl.shadow_map.projection_padding = 0.1f;
    state.render_settings.opengl.shadow_map.debug_view =
        renderer::OpenGlShadowDebugView::PenumbraRadius;
    state.render_settings.opengl.shadow_map.debug_shadow_slot = 3;
    state.render_settings.opengl.pcss.enabled = false;
    state.render_settings.opengl.pcss.blocker_samples = 12;
    state.render_settings.opengl.pcss.filter_samples = 24;
    state.render_settings.opengl.pcss.max_penumbra_texels = 48.0f;
    state.render_settings.opengl.pcss.light_size_scale = 1.5f;
    state.render_settings.opengl.dominant_light.enabled = false;
    state.render_settings.opengl.dominant_light.peak_threshold_ev = 4.0f;
    state.render_settings.opengl.dominant_light.minimum_energy_fraction = 0.02f;
    state.render_settings.opengl.dominant_light.intensity_scale = 1.25f;
    state.render_settings.opengl.ambient_occlusion.mode =
        renderer::OpenGlAmbientOcclusionMode::Ssao;
    state.render_settings.opengl.ambient_occlusion.ssao.sample_count = 48;
    state.render_settings.opengl.ambient_occlusion.ssao.radius_scale = 0.15f;
    state.render_settings.opengl.ambient_occlusion.ssao.depth_bias_fraction = 0.04f;
    state.render_settings.opengl.ambient_occlusion.ssao.intensity = 1.5f;
    state.render_settings.opengl.ambient_occlusion.gtao.slice_count = 5;
    state.render_settings.opengl.ambient_occlusion.gtao.samples_per_side = 4;
    state.render_settings.opengl.ambient_occlusion.gtao.radius_scale = 0.2f;
    state.render_settings.opengl.ambient_occlusion.gtao.falloff_fraction = 0.7f;
    state.render_settings.opengl.ambient_occlusion.gtao.thickness_fraction = 0.3f;
    state.render_settings.opengl.ambient_occlusion.gtao.intensity = 1.25f;
    state.render_settings.opengl.ambient_occlusion.gtao.bent_normals_enabled = false;
    state.render_settings.opengl.ambient_occlusion.denoise.enabled = false;
    state.render_settings.opengl.ambient_occlusion.denoise.kernel_radius = 3;
    state.render_settings.opengl.ambient_occlusion.denoise.depth_sigma_fraction = 0.2f;
    state.render_settings.opengl.ambient_occlusion.denoise.normal_power = 12.0f;
    state.render_settings.opengl.ssr.enabled = false;
    state.render_settings.opengl.ssr.rays_per_pixel = 7;
    state.render_settings.opengl.ssr.max_steps = 144;
    state.render_settings.opengl.ssr.max_distance_scale = 2.25f;
    state.render_settings.opengl.ssr.thickness_scale = 0.03f;
    state.render_settings.opengl.ssr.edge_fade = 0.2f;
    state.render_settings.opengl.ssr.max_history_frames = 48;
    state.render_settings.opengl.ssr.denoise_passes = 4;
    state.render_settings.opengl.ssr.denoise_depth_sigma_fraction = 0.08f;
    state.render_settings.opengl.ssr.denoise_normal_power = 24.0f;
    state.render_settings.opengl.ssr.debug_view =
        renderer::OpenGlSsrDebugView::Final;
    state.render_settings.opengl.ddgi.auto_fit = false;
    state.render_settings.opengl.ddgi.origin = {-4, -2, -8};
    state.render_settings.opengl.ddgi.extent = {8, 4, 6};
    state.render_settings.opengl.ddgi.probe_counts = {8, 6, 10};
    state.render_settings.opengl.ddgi.rays_per_probe = 192;
    state.render_settings.opengl.ddgi.probes_per_frame = 120;
    state.render_settings.opengl.ddgi.hysteresis = 0.9f;
    state.render_settings.opengl.ddgi.paused = true;
    state.render_settings.opengl.ddgi.show_probes = true;
    state.camera.eye = renderer::Vec3(4.0f, 5.0f, 6.0f);
    state.camera.forward =
        renderer::Vec3(-1.0f, -0.5f, -2.0f).normalized();
    state.camera.up = renderer::Vec3::UnitY();
    state.camera.vertical_fov_degrees = 52.0f;
    state.camera.orbit_distance = 7.5f;
    state.camera.free_movement_speed = 2.25f;

    renderer::ViewerSessionStore::save(session_path, document, state);
    RENDER_CHECK(document.file_path() == file_path_before);
    RENDER_CHECK(document.dirty());
    RENDER_CHECK(!std::filesystem::exists(session_path.string() + ".tmp"));

    nlohmann::json saved_json;
    {
        std::ifstream input(session_path);
        input >> saved_json;
    }
    RENDER_CHECK(saved_json.at("version").get<int>() == 7);
    const auto& source =
        saved_json.at("document").at("snapshot").at("assets").at(0).at("source");
    RENDER_CHECK(source.at("kind").get<std::string>() == "obj");
    RENDER_CHECK(std::filesystem::path(
        source.at("path").get<std::string>()).is_absolute());

    renderer::ViewerSessionState loaded =
        renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(loaded.window_width == 1400);
    RENDER_CHECK(loaded.render_settings.realtime == state.render_settings.realtime);
    RENDER_CHECK(loaded.render_settings.opengl.ddgi == state.render_settings.opengl.ddgi);
    RENDER_CHECK(!saved_json.at("view").contains("path_accumulation_paused"));
    RENDER_CHECK(loaded.render_settings.opengl.npr == state.render_settings.opengl.npr);
    RENDER_CHECK(loaded.window_height == 900);
    RENDER_CHECK(loaded.document.file_path() == file_path_before);
    RENDER_CHECK(loaded.document.dirty());
    RENDER_CHECK(!loaded.document.can_undo());
    RENDER_CHECK(!loaded.document.can_redo());
    RENDER_CHECK(loaded.document.objects().size() == 2);
    RENDER_CHECK(loaded.document.find(imported) != nullptr);
    RENDER_CHECK(loaded.document.local_trs(imported).has_value());
    RENDER_CHECK(loaded.document.local_trs(imported)->translation.isApprox(
        renderer::Vec3(2.0f, 3.0f, 4.0f)));
    RENDER_CHECK(loaded.document.material_override(imported, 0) != nullptr);
    RENDER_CHECK(
        loaded.document.material_override(imported, 0)->base_color.isApprox(
            material_override.base_color));
    if (renderer::cuda_path_backend_available()) {
        RENDER_CHECK(loaded.ui.mode == renderer::InteractiveRenderMode::Path);
        RENDER_CHECK(loaded.migration_warning.empty());
    } else {
        RENDER_CHECK(loaded.ui.mode == renderer::InteractiveRenderMode::OpenGl);
        RENDER_CHECK(!loaded.migration_warning.empty());
    }
    RENDER_CHECK(loaded.ui.camera_mode == renderer::ViewerCameraMode::Free);
    RENDER_CHECK(nearly_equal(loaded.ui.render_scale, 0.75f));
    RENDER_CHECK(nearly_equal(loaded.ui.ui_font_scale, 1.25f));
    RENDER_CHECK(!loaded.ui.automatic_interaction_quality);
    RENDER_CHECK(!loaded.ui.path_accumulation_paused);
    RENDER_CHECK(!loaded.ui.show_point_light_markers);
    RENDER_CHECK(!loaded.ui.panel_visible);
    RENDER_CHECK(!loaded.ui.scene_panel_visible);
    RENDER_CHECK(loaded.ui.inspector_panel_visible);
    RENDER_CHECK(!loaded.ui.rendering_panel_visible);
    RENDER_CHECK(loaded.ui.camera_lighting_panel_visible);
    RENDER_CHECK(!loaded.ui.techniques_panel_visible);
    RENDER_CHECK(nearly_equal(loaded.ui.display.exposure_ev, 1.5f));
    RENDER_CHECK(loaded.ui.display.tone_mapper == renderer::ToneMapper::Aces);
    RENDER_CHECK(loaded.ui.selected_objects == std::vector<renderer::ObjectId>{imported});
    RENDER_CHECK(loaded.ui.active_object == imported);
    RENDER_CHECK(loaded.ui.material_editor_object == imported);
    RENDER_CHECK(loaded.ui.gizmo_operation == 2);
    RENDER_CHECK(loaded.ui.gizmo_local);
    RENDER_CHECK(loaded.render_settings.path.cuda_device == 0);
    RENDER_CHECK(loaded.render_settings.path.max_bounces == 12);
    RENDER_CHECK(
        loaded.render_settings.path.russian_roulette_start_bounce == 5);
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.path.russian_roulette_min_probability,
        0.10f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.path.russian_roulette_max_probability,
        0.90f));
    RENDER_CHECK(!loaded.render_settings.opengl.ibl_enabled);
    RENDER_CHECK(!loaded.render_settings.opengl.ltc_area_lights_enabled);
    RENDER_CHECK(!loaded.render_settings.opengl.shadow_map.enabled);
    RENDER_CHECK(loaded.render_settings.opengl.shadow_map.resolution == 2048);
    RENDER_CHECK(loaded.render_settings.opengl.shadow_map.max_shadow_lights == 5);
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.shadow_map.constant_bias, 0.001f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.shadow_map.slope_bias, 0.004f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.shadow_map.projection_padding, 0.1f));
    RENDER_CHECK(
        loaded.render_settings.opengl.shadow_map.debug_view ==
        renderer::OpenGlShadowDebugView::PenumbraRadius);
    RENDER_CHECK(loaded.render_settings.opengl.shadow_map.debug_shadow_slot == 3);
    RENDER_CHECK(!loaded.render_settings.opengl.pcss.enabled);
    RENDER_CHECK(loaded.render_settings.opengl.pcss.blocker_samples == 12);
    RENDER_CHECK(loaded.render_settings.opengl.pcss.filter_samples == 24);
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.pcss.max_penumbra_texels, 48.0f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.pcss.light_size_scale, 1.5f));
    RENDER_CHECK(!loaded.render_settings.opengl.dominant_light.enabled);
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.dominant_light.peak_threshold_ev, 4.0f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.dominant_light.minimum_energy_fraction,
        0.02f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.dominant_light.intensity_scale, 1.25f));
    RENDER_CHECK(
        loaded.render_settings.opengl.ambient_occlusion.mode ==
        renderer::OpenGlAmbientOcclusionMode::Ssao);
    RENDER_CHECK(
        loaded.render_settings.opengl.ambient_occlusion.ssao.sample_count == 48);
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ambient_occlusion.ssao.radius_scale,
        0.15f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ambient_occlusion.ssao.depth_bias_fraction,
        0.04f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ambient_occlusion.ssao.intensity,
        1.5f));
    RENDER_CHECK(
        loaded.render_settings.opengl.ambient_occlusion.gtao.slice_count == 5);
    RENDER_CHECK(
        loaded.render_settings.opengl.ambient_occlusion.gtao.samples_per_side == 4);
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ambient_occlusion.gtao.radius_scale,
        0.2f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ambient_occlusion.gtao.falloff_fraction,
        0.7f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ambient_occlusion.gtao.thickness_fraction,
        0.3f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ambient_occlusion.gtao.intensity,
        1.25f));
    RENDER_CHECK(
        !loaded.render_settings.opengl.ambient_occlusion.gtao.bent_normals_enabled);
    RENDER_CHECK(!loaded.render_settings.opengl.ambient_occlusion.denoise.enabled);
    RENDER_CHECK(
        loaded.render_settings.opengl.ambient_occlusion.denoise.kernel_radius == 3);
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ambient_occlusion.denoise.depth_sigma_fraction,
        0.2f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ambient_occlusion.denoise.normal_power,
        12.0f));
    RENDER_CHECK(!loaded.render_settings.opengl.ssr.enabled);
    RENDER_CHECK(loaded.render_settings.opengl.ssr.rays_per_pixel == 7);
    RENDER_CHECK(loaded.render_settings.opengl.ssr.max_steps == 144);
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ssr.max_distance_scale, 2.25f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ssr.thickness_scale, 0.03f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ssr.edge_fade, 0.2f));
    RENDER_CHECK(
        loaded.render_settings.opengl.ssr.max_history_frames == 48);
    RENDER_CHECK(loaded.render_settings.opengl.ssr.denoise_passes == 4);
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ssr.denoise_depth_sigma_fraction,
        0.08f));
    RENDER_CHECK(nearly_equal(
        loaded.render_settings.opengl.ssr.denoise_normal_power, 24.0f));
    RENDER_CHECK(
        loaded.render_settings.opengl.ssr.debug_view ==
        renderer::OpenGlSsrDebugView::Final);
    RENDER_CHECK(loaded.camera.eye.isApprox(state.camera.eye));
    RENDER_CHECK(loaded.camera.forward.isApprox(state.camera.forward));
    RENDER_CHECK(nearly_equal(
        loaded.camera.vertical_fov_degrees,
        state.camera.vertical_fov_degrees));
    RENDER_CHECK(nearly_equal(
        loaded.camera.orbit_distance,
        state.camera.orbit_distance));
    RENDER_CHECK(nearly_equal(
        loaded.camera.free_movement_speed,
        state.camera.free_movement_speed));

    nlohmann::json ignored_legacy_field_json = saved_json;
    ignored_legacy_field_json["render"]["max_depth"] = 99;
    {
        std::ofstream output(session_path);
        output << ignored_legacy_field_json.dump(2) << '\n';
    }
    const renderer::ViewerSessionState ignored_legacy_field =
        renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(
        ignored_legacy_field.ui.mode ==
        (renderer::cuda_path_backend_available()
            ? renderer::InteractiveRenderMode::Path
            : renderer::InteractiveRenderMode::OpenGl));

    nlohmann::json old_session_json = saved_json;
    old_session_json["view"].erase("automatic_interaction_quality");
    old_session_json["render"].erase("max_bounces");
    old_session_json["render"].erase("rr_start_bounce");
    old_session_json["render"].erase("rr_min_probability");
    old_session_json["render"].erase("rr_max_probability");
    {
        std::ofstream output(session_path);
        output << old_session_json.dump(2) << '\n';
    }
    const renderer::ViewerSessionState old_session =
        renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(!old_session.ui.automatic_interaction_quality);
    RENDER_CHECK(old_session.render_settings.path.max_bounces == 64);
    RENDER_CHECK(
        old_session.render_settings.path.russian_roulette_start_bounce == 3);
    RENDER_CHECK(nearly_equal(
        old_session.render_settings.path.russian_roulette_min_probability,
        0.05f));
    RENDER_CHECK(nearly_equal(
        old_session.render_settings.path.russian_roulette_max_probability,
        0.95f));

    for (const char* removed_mode : {"raster", "ray"}) {
        nlohmann::json removed_mode_json = saved_json;
        removed_mode_json["view"]["mode"] = removed_mode;
        {
            std::ofstream output(session_path);
            output << removed_mode_json.dump(2) << '\n';
        }
        bool rejected = false;
        try {
            (void)renderer::ViewerSessionStore::load(session_path);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        RENDER_CHECK(rejected);
    }

    state.ui.mode = renderer::InteractiveRenderMode::OpenGl;
    renderer::ViewerSessionStore::save(session_path, document, state);
    RENDER_CHECK(
        renderer::ViewerSessionStore::load(session_path).ui.mode ==
        renderer::InteractiveRenderMode::OpenGl);
    state.ui.mode = renderer::InteractiveRenderMode::Path;

    auto version_six_json = saved_json;
    version_six_json["version"] = 6;
    version_six_json["render"]["opengl"].erase("ddgi");
    {
        std::ofstream output(session_path);
        output << version_six_json.dump(2) << '\n';
    }
    const auto version_six = renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(!version_six.render_settings.opengl.ddgi.enabled);
    RENDER_CHECK(version_six.render_settings.opengl.ssr == state.render_settings.opengl.ssr);
    RENDER_CHECK(version_six.render_settings.opengl.ibl_enabled == state.render_settings.opengl.ibl_enabled);

    nlohmann::json version_three_json = saved_json;
    version_three_json["version"] = 3;
    version_three_json["render"]["opengl"].erase("ambient_occlusion");
    version_three_json["render"]["opengl"].erase("npr");
    {
        std::ofstream output(session_path);
        output << version_three_json.dump(2) << '\n';
    }
    const renderer::ViewerSessionState version_three =
        renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(version_three.render_settings.opengl.npr == renderer::NprRenderSettings{});
    RENDER_CHECK(
        version_three.render_settings.opengl.ambient_occlusion ==
        renderer::AmbientOcclusionRenderSettings{});

    RENDER_CHECK(!saved_json.at("render").at("opengl").contains("ssgi"));
    nlohmann::json legacy_transport_json = saved_json;
    legacy_transport_json["version"] = 5;
    legacy_transport_json["render"]["opengl"]["ssgi"] = saved_json["render"]["opengl"]["ssr"];
    legacy_transport_json["render"]["opengl"]["ssr"] = {
        {"enabled", false}, {"max_steps", 96}, {"intensity", 4.0f}, {"debug_view", 0}};
    for (bool old_gi_enabled : {false, true}) {
        legacy_transport_json["render"]["opengl"]["ssgi"]["enabled"] = old_gi_enabled;
        {
            std::ofstream output(session_path);
            output << legacy_transport_json.dump(2) << '\n';
        }
        const auto migrated = renderer::ViewerSessionStore::load(session_path);
        RENDER_CHECK(migrated.render_settings.opengl.ssr.enabled == old_gi_enabled);
        RENDER_CHECK(migrated.render_settings.opengl.ssr.max_steps == 144);
        RENDER_CHECK(migrated.render_settings.opengl.ssr.rays_per_pixel == 7);
    }
    legacy_transport_json["version"] = 4;
    {
        std::ofstream output(session_path);
        output << legacy_transport_json.dump(2) << '\n';
    }
    const auto old_reflections = renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(old_reflections.render_settings.opengl.ssr.max_steps == 96);
    RENDER_CHECK(old_reflections.render_settings.opengl.ssr.rays_per_pixel == 2);

    nlohmann::json clamped_ssr_json = saved_json;
    auto& clamped_ssr = clamped_ssr_json["render"]["opengl"]["ssr"];
    clamped_ssr["rays_per_pixel"] = 99;
    clamped_ssr["max_steps"] = -1;
    clamped_ssr["max_distance_scale"] = nullptr;
    clamped_ssr["thickness_scale"] = 2.0f;
    clamped_ssr["edge_fade"] = -1.0f;
    clamped_ssr["max_history_frames"] = 0;
    clamped_ssr["denoise_passes"] = 99;
    clamped_ssr["denoise_depth_sigma_fraction"] = 2.0f;
    clamped_ssr["denoise_normal_power"] = 0.0f;
    clamped_ssr["debug_view"] = 99;
    clamped_ssr_json["render"]["opengl"]["shadow_map"]["debug_view"] = 2;
    clamped_ssr_json["render"]["opengl"]["ambient_occlusion"]["debug_view"] = 1;
    {
        std::ofstream output(session_path);
        output << clamped_ssr_json.dump(2) << '\n';
    }
    const renderer::ViewerSessionState clamped_ssr_state =
        renderer::ViewerSessionStore::load(session_path);
    const auto& loaded_ssr = clamped_ssr_state.render_settings.opengl.ssr;
    RENDER_CHECK(loaded_ssr.rays_per_pixel == 8);
    RENDER_CHECK(loaded_ssr.max_steps == 8);
    RENDER_CHECK(nearly_equal(loaded_ssr.max_distance_scale, 1.0f));
    RENDER_CHECK(nearly_equal(loaded_ssr.thickness_scale, 0.1f));
    RENDER_CHECK(nearly_equal(loaded_ssr.edge_fade, 0.0f));
    RENDER_CHECK(loaded_ssr.max_history_frames == 1);
    RENDER_CHECK(loaded_ssr.denoise_passes == 4);
    RENDER_CHECK(nearly_equal(
        loaded_ssr.denoise_depth_sigma_fraction, 0.5f));
    RENDER_CHECK(nearly_equal(loaded_ssr.denoise_normal_power, 1.0f));
    RENDER_CHECK(
        loaded_ssr.debug_view ==
        renderer::OpenGlSsrDebugView::HistoryLength);
    RENDER_CHECK(
        clamped_ssr_state.render_settings.opengl.shadow_map.debug_view ==
        renderer::OpenGlShadowDebugView::Final);
    RENDER_CHECK(
        clamped_ssr_state.render_settings.opengl.ambient_occlusion.debug_view ==
        renderer::OpenGlAmbientOcclusionDebugView::Final);

    nlohmann::json clamped_ao_json = saved_json;
    auto& clamped_ao =
        clamped_ao_json["render"]["opengl"]["ambient_occlusion"];
    clamped_ao["mode"] = 99;
    clamped_ao["debug_view"] = 99;
    clamped_ao["ssao"]["sample_count"] = 999;
    clamped_ao["ssao"]["radius_scale"] = nullptr;
    clamped_ao["gtao"]["slice_count"] = -10;
    clamped_ao["denoise"]["kernel_radius"] = 99;
    clamped_ao_json["render"]["opengl"]["shadow_map"]["debug_view"] = 2;
    {
        std::ofstream output(session_path);
        output << clamped_ao_json.dump(2) << '\n';
    }
    const renderer::ViewerSessionState clamped_ao_state =
        renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(
        clamped_ao_state.render_settings.opengl.ambient_occlusion.mode ==
        renderer::OpenGlAmbientOcclusionMode::Gtao);
    RENDER_CHECK(
        clamped_ao_state.render_settings.opengl.ambient_occlusion.debug_view ==
        renderer::OpenGlAmbientOcclusionDebugView::LinearDepth);
    RENDER_CHECK(
        clamped_ao_state.render_settings.opengl.ambient_occlusion.ssao.sample_count ==
        64);
    RENDER_CHECK(nearly_equal(
        clamped_ao_state.render_settings.opengl.ambient_occlusion.ssao.radius_scale,
        0.10f));
    RENDER_CHECK(
        clamped_ao_state.render_settings.opengl.ambient_occlusion.gtao.slice_count ==
        1);
    RENDER_CHECK(
        clamped_ao_state.render_settings.opengl.ambient_occlusion.denoise.kernel_radius ==
        4);
    RENDER_CHECK(
        clamped_ao_state.render_settings.opengl.shadow_map.debug_view ==
        renderer::OpenGlShadowDebugView::Final);

    nlohmann::json legacy_json = saved_json;
    legacy_json["version"] = 2;
    auto& legacy_view = legacy_json["view"];
    legacy_view.erase("scene_panel_visible");
    legacy_view.erase("inspector_panel_visible");
    legacy_view.erase("rendering_panel_visible");
    legacy_view.erase("camera_lighting_panel_visible");
    legacy_view.erase("camera_panel_visible");
    legacy_view.erase("techniques_panel_visible");
    legacy_json["render"].erase("opengl");
    {
        std::ofstream output(session_path);
        output << legacy_json.dump(2) << '\n';
    }
    const renderer::ViewerSessionState legacy =
        renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(legacy.ui.scene_panel_visible);
    RENDER_CHECK(legacy.ui.inspector_panel_visible);
    RENDER_CHECK(legacy.ui.rendering_panel_visible);
    RENDER_CHECK(legacy.ui.camera_lighting_panel_visible);
    RENDER_CHECK(legacy.ui.techniques_panel_visible);
    renderer::OpenGlRenderSettings legacy_defaults;
    legacy_defaults.ddgi.enabled = false;
    RENDER_CHECK(legacy.render_settings.opengl == legacy_defaults);

    nlohmann::json version_one_session_json = saved_json;
    version_one_session_json["version"] = 1;
    version_one_session_json["render"]["path_backend"] = "cpu";
    version_one_session_json["render"].erase("opengl");
    version_one_session_json["view"].erase("camera_panel_visible");
    version_one_session_json["view"].erase("techniques_panel_visible");
    {
        std::ofstream output(session_path);
        output << version_one_session_json.dump(2) << '\n';
    }
    const renderer::ViewerSessionState version_one_session =
        renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(version_one_session.ui.camera_lighting_panel_visible);
    RENDER_CHECK(version_one_session.ui.techniques_panel_visible);
    RENDER_CHECK(
        version_one_session.render_settings.opengl == legacy_defaults);
    if (renderer::cuda_path_backend_available()) {
        RENDER_CHECK(
            version_one_session.ui.mode ==
            renderer::InteractiveRenderMode::Path);
        RENDER_CHECK(
            version_one_session.migration_warning.find("Migrated v1 cpu") !=
            std::string::npos);
    } else {
        RENDER_CHECK(
            version_one_session.ui.mode ==
            renderer::InteractiveRenderMode::OpenGl);
        RENDER_CHECK(!version_one_session.migration_warning.empty());
    }

    renderer::SceneDocument mixed = renderer::SceneDocument::from_scene(
        renderer::make_cornell_box_scene(),
        "Builtin Cornell Box",
        "cornell_box");
    mixed.import_path(obj_path, 64, 64);
    state.document_path.clear();
    state.document_dirty = true;
    renderer::ViewerSessionStore::save(session_path, mixed, state);
    renderer::ViewerSessionState restored_mixed =
        renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(restored_mixed.document.assets().size() == 2);
    RENDER_CHECK(restored_mixed.document.objects().size() == 3);

    {
        std::ifstream input(session_path);
        input >> saved_json;
    }
    for (auto& asset_json :
         saved_json["document"]["snapshot"]["assets"]) {
        auto& asset_source = asset_json["source"];
        if (asset_source.at("kind").get<std::string>() == "obj") {
            asset_source["path"] =
                std::filesystem::absolute(directory / "missing.obj").generic_string();
        }
    }
    {
        std::ofstream output(session_path);
        output << saved_json.dump(2) << '\n';
    }
    renderer::ViewerSessionState partial =
        renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(partial.document.assets().size() == 1);
    RENDER_CHECK(partial.document.objects().size() == 2);
    const bool saw_missing_warning = std::any_of(
        partial.document.warnings().begin(),
        partial.document.warnings().end(),
        [](const std::string& warning) {
            return warning.find("missing session asset") != std::string::npos;
        });
    RENDER_CHECK(saw_missing_warning);

    saved_json["version"] = 999;
    {
        std::ofstream output(session_path);
        output << saved_json.dump(2) << '\n';
    }
    bool rejected_version = false;
    try {
        static_cast<void>(renderer::ViewerSessionStore::load(session_path));
    } catch (const std::runtime_error&) {
        rejected_version = true;
    }
    RENDER_CHECK(rejected_version);

    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_viewer_quality_defaults_survive_session_roundtrip) {
    const std::filesystem::path path="test_viewer_quality_defaults.json";
    renderer::ViewerSessionState defaults;
    defaults.document.create_point_light("Test light",renderer::Vec3::Zero(),renderer::Color::Ones());
    renderer::ViewerSessionStore::save(path,defaults.document,defaults);
    const auto loaded=renderer::ViewerSessionStore::load(path);
    RENDER_CHECK(loaded.render_settings.realtime==defaults.render_settings.realtime);
    RENDER_CHECK(loaded.render_settings.opengl==defaults.render_settings.opengl);
    RENDER_CHECK(loaded.ui.display.tone_mapper==renderer::ToneMapper::Aces);
    RENDER_CHECK(loaded.ui.display.exposure_ev==0 && loaded.ui.render_scale==1);
    // Optional canonical export for repeatable viewer profile validation.
    if(const char* output=std::getenv("VIEWER_DEFAULTS_EXPORT"))
        renderer::ViewerSessionStore::save(output,defaults.document,defaults);
    std::filesystem::remove(path);
}

RENDER_TEST(test_ddgi_reset_repeatedly_invalidates_probe_history_and_fit) {
    renderer::DdgiSettings settings;
    settings.reset_generation=7;settings.fit_generation=11;
    settings.enabled=false;settings.paused=true;settings.auto_fit=false;
    settings.intensity=4;settings.show_probes=true;
    settings.origin={20,30,40};settings.probe_counts={3,3,3};
    for(int click=1;click<=2;++click) {
        renderer::reset_ddgi_settings(settings);
        auto expected=renderer::DdgiSettings{};
        expected.reset_generation=7+click;expected.fit_generation=11+click;
        RENDER_CHECK(settings==expected);
    }
}

RENDER_TEST(test_viewer_session_omits_and_skips_unreferenced_assets) {
    const std::filesystem::path directory =
        "test_viewer_session_unreferenced_assets";
    const std::filesystem::path live_obj = directory / "live.obj";
    const std::filesystem::path orphan_obj = directory / "orphan.obj";
    const std::filesystem::path session_path =
        directory / "last-session.json";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const auto write_triangle = [](const std::filesystem::path& path) {
        std::ofstream obj(path);
        obj << "v -1 -1 0\n";
        obj << "v 1 -1 0\n";
        obj << "v 0 1 0\n";
        obj << "f 1 2 3\n";
    };
    write_triangle(live_obj);
    write_triangle(orphan_obj);

    renderer::SceneDocument document;
    const renderer::ObjectId live_id =
        document.import_path(live_obj, 64, 64).front();
    const renderer::ObjectId orphan_id =
        document.import_path(orphan_obj, 64, 64).front();
    RENDER_CHECK(document.assets().size() == 2);
    RENDER_CHECK(document.erase_subtree(orphan_id));
    RENDER_CHECK(document.find(orphan_id) == nullptr);
    RENDER_CHECK(document.assets().size() == 2);

    RENDER_CHECK(document.undo());
    RENDER_CHECK(document.find(orphan_id) != nullptr);
    RENDER_CHECK(document.redo());
    RENDER_CHECK(document.find(orphan_id) == nullptr);

    const nlohmann::json snapshot = document.session_snapshot();
    RENDER_CHECK(snapshot.at("assets").size() == 1);
    RENDER_CHECK(
        snapshot.at("assets").at(0).at("source").at("path")
            .get<std::string>().find("orphan.obj") ==
        std::string::npos);

    renderer::ViewerSessionState state;
    state.window_width = 640;
    state.window_height = 480;
    renderer::ViewerSessionStore::save(
        session_path,
        document,
        state);

    nlohmann::json legacy_dirty_session;
    {
        std::ifstream input(session_path);
        input >> legacy_dirty_session;
    }
    auto& saved_assets =
        legacy_dirty_session["document"]["snapshot"]["assets"];
    RENDER_CHECK(saved_assets.size() == 1);
    saved_assets.push_back({
        {"id", 999},
        {"source", {
            {"kind", "obj"},
            {"path", std::filesystem::absolute(orphan_obj).generic_string()},
        }},
    });
    {
        std::ofstream output(session_path);
        output << legacy_dirty_session.dump(2) << '\n';
    }

    const renderer::ViewerSessionState restored =
        renderer::ViewerSessionStore::load(session_path);
    RENDER_CHECK(restored.document.assets().size() == 1);
    RENDER_CHECK(restored.document.find(live_id) != nullptr);
    const bool saw_orphan_warning = std::any_of(
        restored.document.warnings().begin(),
        restored.document.warnings().end(),
        [](const std::string& warning) {
            return warning.find("orphan.obj") != std::string::npos;
        });
    RENDER_CHECK(!saw_orphan_warning);

    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_environment_map_sampling_sh_and_document_roundtrip) {
    const renderer::Color constant(1.5f, 0.75f, 0.25f);
    std::vector<renderer::Color> pixels(64U * 32U, constant);
    const auto environment = std::make_shared<const renderer::EnvironmentMap>(
        64,
        32,
        std::move(pixels));

    const std::array<renderer::Vec3, 4> directions{
        renderer::Vec3(1.0f, 0.2f, 0.3f).normalized(),
        renderer::Vec3(-0.4f, 0.8f, 0.1f).normalized(),
        renderer::Vec3(0.1f, -0.7f, -0.9f).normalized(),
        renderer::Vec3(-0.6f, -0.2f, 0.75f).normalized()};
    for (const renderer::Vec3& direction : directions) {
        const renderer::Vec2 uv = renderer::EnvironmentMap::direction_to_uv(direction);
        const renderer::Vec3 reconstructed =
            renderer::EnvironmentMap::uv_to_direction(uv);
        RENDER_CHECK(direction.dot(reconstructed) > 0.99999f);
        RENDER_CHECK(environment->sample_direction(direction).isApprox(constant, 1.0e-5f));
        RENDER_CHECK(nearly_equal(
            environment->direction_pdf(direction),
            1.0f / (4.0f * 3.14159265358979323846f),
            2.0e-5f));
    }
    const renderer::EnvironmentMapSample sampled = environment->sample(0.37f, 0.2f, 0.8f);
    RENDER_CHECK(sampled.direction.allFinite());
    RENDER_CHECK(sampled.radiance.isApprox(constant, 1.0e-5f));
    RENDER_CHECK(sampled.pdf > 0.0f);
    const renderer::Color irradiance =
        environment->diffuse_irradiance(renderer::Vec3::UnitY());
    RENDER_CHECK((irradiance - constant * 3.14159265358979323846f)
        .cwiseAbs().maxCoeff() < 0.03f);
    const renderer::Color rotated = renderer::environment_radiance(
        renderer::Color::Ones(),
        environment,
        2.0f,
        90.0f,
        renderer::Vec3::UnitX());
    RENDER_CHECK(rotated.isApprox(constant * 2.0f, 1.0e-5f));

    bool rejected_layout = false;
    try {
        static_cast<void>(renderer::EnvironmentMap(
            3,
            2,
            std::vector<renderer::Color>(6, renderer::Color::Ones())));
    } catch (const std::invalid_argument&) {
        rejected_layout = true;
    }
    RENDER_CHECK(rejected_layout);

    const std::filesystem::path directory = "test_environment_document";
    const std::filesystem::path image_path = directory / "environment.ppm";
    const std::filesystem::path scene_path = directory / "environment.rscene";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    {
        std::ofstream output(image_path, std::ios::binary);
        output << "P6\n4 2\n255\n";
        const std::array<unsigned char, 24> image_pixels{
            255, 128, 64, 255, 128, 64, 255, 128, 64, 255, 128, 64,
            255, 128, 64, 255, 128, 64, 255, 128, 64, 255, 128, 64};
        output.write(
            reinterpret_cast<const char*>(image_pixels.data()),
            static_cast<std::streamsize>(image_pixels.size()));
    }
    renderer::SceneDocument document;
    document.set_environment_map(image_path);
    const renderer::Color decoded =
        document.environment_map()->sample_direction(renderer::Vec3::UnitX());
    RENDER_CHECK(nearly_equal(decoded.x(), 1.0f, 1.0e-6f));
    RENDER_CHECK(nearly_equal(decoded.y(), 0.2158605f, 1.0e-5f));
    RENDER_CHECK(nearly_equal(decoded.z(), 0.0512695f, 1.0e-5f));
    document.set_environment_intensity(1.75f);
    document.set_environment_rotation_degrees(-35.0f);
    document.set_environment_background_visible(false);
    document.save(scene_path);
    renderer::SceneDocument restored =
        renderer::SceneDocument::load(scene_path, 64, 64);
    RENDER_CHECK(restored.environment_map() != nullptr);
    RENDER_CHECK(nearly_equal(restored.environment_intensity(), 1.75f));
    RENDER_CHECK(nearly_equal(restored.environment_rotation_degrees(), -35.0f));
    RENDER_CHECK(!restored.environment_background_visible());
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(restored.render_scene_snapshot()).environment_map != nullptr);
    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_environment_importance_pdf_integrates_and_histogram_matches_pmf) {
    constexpr int width = 8;
    constexpr int height = 4;
    constexpr float pi = 3.14159265358979323846f;
    std::vector<renderer::Color> pixels;
    pixels.reserve(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float value = 0.1f + static_cast<float>(1 + x + 3 * y);
            pixels.emplace_back(value, value * 0.7f, value * 0.3f);
        }
    }
    const renderer::EnvironmentMap environment(
        width,
        height,
        std::move(pixels));

    float integrated_pdf = 0.0f;
    for (int y = 0; y < height; ++y) {
        const float theta0 = pi * static_cast<float>(y) /
            static_cast<float>(height);
        const float theta1 = pi * static_cast<float>(y + 1) /
            static_cast<float>(height);
        const float texel_solid_angle =
            (2.0f * pi / static_cast<float>(width)) *
            (std::cos(theta0) - std::cos(theta1));
        for (int x = 0; x < width; ++x) {
            const renderer::Vec2 uv(
                (static_cast<float>(x) + 0.5f) /
                    static_cast<float>(width),
                (static_cast<float>(y) + 0.5f) /
                    static_cast<float>(height));
            integrated_pdf += environment.direction_pdf(
                renderer::EnvironmentMap::uv_to_direction(uv)) *
                texel_solid_angle;
        }
    }
    RENDER_CHECK(nearly_equal(integrated_pdf, 1.0f, 2.0e-5f));

    constexpr int sample_count = 32768;
    std::vector<int> counts(width * height, 0);
    for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
        const float select =
            (static_cast<float>(sample_index) + 0.5f) /
            static_cast<float>(sample_count);
        const renderer::EnvironmentMapSample sample =
            environment.sample(select, 0.375f, 0.625f);
        const renderer::Vec2 uv =
            renderer::EnvironmentMap::direction_to_uv(sample.direction);
        const int x = std::clamp(
            static_cast<int>(uv.x() * static_cast<float>(width)),
            0,
            width - 1);
        const int y = std::clamp(
            static_cast<int>(uv.y() * static_cast<float>(height)),
            0,
            height - 1);
        ++counts[static_cast<std::size_t>(y * width + x)];
    }
    const auto& pmf = environment.importance_pmf();
    for (std::size_t index = 0; index < pmf.size(); ++index) {
        const float measured = static_cast<float>(counts[index]) /
            static_cast<float>(sample_count);
        RENDER_CHECK(std::abs(measured - pmf[index]) < 1.0e-4f);
    }
}

RENDER_TEST(test_pbr_sampling_pdf_and_texture_sampler_contracts) {
    const renderer::Color base_color(0.8f, 0.35f, 0.12f);
    constexpr float metallic = 0.4f;
    const renderer::PbrSurface surface{
        base_color * (1.0f - metallic),
        renderer::Color::Constant(0.04f) * (1.0f - metallic) +
            base_color * metallic,
        renderer::Color::Ones(),
        renderer::Color::Constant(0.04f),
        renderer::Color::Ones(),
        0.45f,
        true};
    const renderer::PbrSample sampled = renderer::sample_pbr(
        surface,
        renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitZ(),
        0.1f,
        0.37f,
        0.81f);
    RENDER_CHECK(sampled.valid);
    RENDER_CHECK(sampled.direction.allFinite());
    RENDER_CHECK(sampled.weight.allFinite());
    RENDER_CHECK(sampled.pdf > 0.0f);
    const renderer::PbrEvaluation evaluated = renderer::evaluate_pbr(
        surface,
        renderer::Vec3::UnitZ(),
        renderer::Vec3::UnitZ(),
        sampled.direction);
    RENDER_CHECK(evaluated.brdf.allFinite());
    RENDER_CHECK(nearly_equal(evaluated.pdf, sampled.pdf, 1.0e-5f));

    renderer::ImageTexture texture(
        2,
        1,
        std::vector<renderer::Color>{
            renderer::Color(1.0f, 0.0f, 0.0f),
            renderer::Color(0.0f, 1.0f, 0.0f)});
    texture.set_sampler(
        renderer::TextureWrap::ClampToEdge,
        renderer::TextureWrap::ClampToEdge,
        renderer::TextureFilter::Nearest,
        renderer::TextureFilter::Nearest);
    RENDER_CHECK(texture.sample(renderer::Vec2(-0.25f, 0.5f)).x() > 0.99f);
    RENDER_CHECK(texture.sample(renderer::Vec2(1.25f, 0.5f)).y() > 0.99f);
    texture.set_sampler(
        renderer::TextureWrap::MirroredRepeat,
        renderer::TextureWrap::ClampToEdge,
        renderer::TextureFilter::Nearest,
        renderer::TextureFilter::Nearest);
    RENDER_CHECK(texture.sample(renderer::Vec2(1.25f, 0.5f)).y() > 0.99f);

    renderer::Scene alpha_scene;
    renderer::Material alpha_material;
    alpha_material.type = renderer::MaterialType::Pbr;
    alpha_material.alpha_mode = renderer::AlphaMode::Mask;
    alpha_material.opacity = 0.8f;
    renderer::HitRecord alpha_hit;
    alpha_hit.vertex_alpha = 0.25f;
    RENDER_CHECK(nearly_equal(
        renderer::sample_material_opacity(alpha_scene, alpha_material, alpha_hit),
        0.2f));
    alpha_material.alpha_mode = renderer::AlphaMode::Opaque;
    RENDER_CHECK(nearly_equal(
        renderer::sample_material_opacity(alpha_scene, alpha_material, alpha_hit),
        1.0f));
}

RENDER_TEST(test_smooth_ggx_peak_preserves_normalization) {
    renderer::PbrSurface surface;surface.diffuse_color=renderer::Color::Zero();
    surface.specular_f0=surface.specular_f90=renderer::Color::Ones();
    const auto normal=renderer::Vec3::UnitZ();
    for(float roughness:{.02f,.025f,.04f,.1f}) {
        surface.roughness=roughness;
        const auto value=renderer::evaluate_pbr(surface,normal,normal,normal);
        const float expected=1/(4*3.14159265358979323846f*std::pow(roughness,4));
        RENDER_CHECK(value.brdf.allFinite());
        RENDER_CHECK(std::abs(value.brdf.x()/expected-1)<.002f);
    }
}

RENDER_TEST(test_kulla_conty_ggx_white_furnace_and_reciprocity) {
    const renderer::Vec3 normal = renderer::Vec3::UnitZ();
    const auto integrate_white_furnace = [&](float roughness, float n_dot_v) {
        renderer::PbrSurface surface;
        surface.diffuse_color = renderer::Color::Zero();
        surface.specular_f0 = renderer::Color::Ones();
        surface.specular_f90 = renderer::Color::Ones();
        surface.roughness = roughness;
        const renderer::Vec3 outgoing(
            std::sqrt(std::max(0.0f, 1.0f - n_dot_v * n_dot_v)),
            0.0f,
            n_dot_v);
        constexpr int cosine_samples = 192;
        constexpr int azimuth_samples = 384;
        constexpr float two_pi = 6.28318530717958647692f;
        renderer::Color integrated = renderer::Color::Zero();
        for (int cosine_index = 0; cosine_index < cosine_samples; ++cosine_index) {
            const float n_dot_l =
                (static_cast<float>(cosine_index) + 0.5f) /
                static_cast<float>(cosine_samples);
            const float sine = std::sqrt(std::max(0.0f, 1.0f - n_dot_l * n_dot_l));
            for (int azimuth_index = 0; azimuth_index < azimuth_samples;
                 ++azimuth_index) {
                const float azimuth = two_pi *
                    (static_cast<float>(azimuth_index) + 0.5f) /
                    static_cast<float>(azimuth_samples);
                const renderer::Vec3 incoming(
                    sine * std::cos(azimuth),
                    sine * std::sin(azimuth),
                    n_dot_l);
                integrated += renderer::evaluate_pbr(
                                  surface,
                                  normal,
                                  outgoing,
                                  incoming)
                                  .brdf *
                    n_dot_l;
            }
        }
        return integrated *
            (two_pi /
             static_cast<float>(cosine_samples * azimuth_samples));
    };

    RENDER_CHECK(nearly_equal(
        renderer::ggx_directional_albedo(1.0f, 1.0f),
        1.0f - std::log(2.0f),
        2.0e-3f));
    for (const std::pair<float, float> parameters : {
             std::pair{1.0f, 1.0f},
             std::pair{0.75f, 1.0f},
             std::pair{1.0f, 0.4f}}) {
        const renderer::Color reflected = integrate_white_furnace(
            parameters.first,
            parameters.second);
        RENDER_CHECK((reflected - renderer::Color::Ones()).cwiseAbs().maxCoeff() <
            1.5e-2f);
    }

    renderer::PbrSurface sampled_surface;
    sampled_surface.diffuse_color = renderer::Color::Zero();
    sampled_surface.specular_f0 = renderer::Color::Ones();
    sampled_surface.specular_f90 = renderer::Color::Ones();
    sampled_surface.roughness = 1.0f;
    renderer::PcgRandom rng(1729);
    renderer::Color sampled_reflectance = renderer::Color::Zero();
    constexpr int bsdf_sample_count = 65536;
    for (int sample_index = 0; sample_index < bsdf_sample_count; ++sample_index) {
        const renderer::PbrSample sample = renderer::sample_pbr(
            sampled_surface,
            normal,
            normal,
            rng.next_float(),
            rng.next_float(),
            rng.next_float());
        if (sample.valid) {
            sampled_reflectance += sample.weight;
        }
    }
    sampled_reflectance /= static_cast<float>(bsdf_sample_count);
    RENDER_CHECK(
        (sampled_reflectance - renderer::Color::Ones()).cwiseAbs().maxCoeff() <
        2.0e-2f);

    renderer::PbrSurface reciprocal_surface;
    reciprocal_surface.diffuse_color = renderer::Color(0.3f, 0.15f, 0.05f);
    reciprocal_surface.specular_f0 = renderer::Color(0.7f, 0.45f, 0.2f);
    reciprocal_surface.specular_f90 = renderer::Color::Ones();
    reciprocal_surface.roughness = 0.82f;
    const renderer::Vec3 outgoing = renderer::Vec3(0.6f, 0.0f, 0.8f).normalized();
    const renderer::Vec3 incoming = renderer::Vec3(-0.2f, 0.7f, 0.68f).normalized();
    const renderer::Color forward = renderer::evaluate_pbr(
        reciprocal_surface, normal, outgoing, incoming).brdf;
    const renderer::Color reverse = renderer::evaluate_pbr(
        reciprocal_surface, normal, incoming, outgoing).brdf;
    RENDER_CHECK((forward - reverse).cwiseAbs().maxCoeff() < 1.0e-6f);
}

RENDER_TEST(test_gltf_static_scene_import_and_flattening) {
    const std::filesystem::path directory = "test_gltf_static_scene";
    const std::filesystem::path gltf_path = directory / "scene.gltf";
    const std::filesystem::path binary_path = directory / "mesh.bin";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const std::array<float, 9> positions{
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f};
    const std::array<float, 9> normals{
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f};
    const std::array<float, 6> uvs{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    {
        std::ofstream output(binary_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(positions.data()), sizeof(positions));
        output.write(reinterpret_cast<const char*>(normals.data()), sizeof(normals));
        output.write(reinterpret_cast<const char*>(uvs.data()), sizeof(uvs));
        output.write(reinterpret_cast<const char*>(indices.data()), sizeof(indices));
    }
    nlohmann::json gltf{
        {"asset", {{"version", "2.0"}}},
        {"extensionsUsed", {"KHR_lights_punctual"}},
        {"extensions", {{"KHR_lights_punctual", {{"lights", {
            {{"type", "spot"}, {"color", {1.0, 0.5, 0.25}}, {"intensity", 12.0},
             {"range", 5.0},
             {"spot", {{"innerConeAngle", 0.1}, {"outerConeAngle", 0.5}}}}
        }}}}}},
        {"scene", 0},
        {"scenes", {{{"nodes", {0}}}}},
        {"nodes", {
            {{"name", "Root"}, {"children", {1, 2, 3, 4}}},
            {{"name", "Left"}, {"mesh", 0}, {"translation", {-1.0, 0.0, 0.0}}},
            {{"name", "Right"}, {"mesh", 0}, {"translation", {1.0, 0.0, 0.0}}},
            {{"name", "CameraLight"}, {"camera", 0},
             {"translation", {0.0, 0.0, 3.0}},
             {"extensions", {{"KHR_lights_punctual", {{"light", 0}}}}}},
            {{"name", "MillimeterRoot"}, {"children", {5}},
             {"scale", {0.001, 0.001, 0.001}}},
            {{"name", "MillimeterMesh"}, {"mesh", 0}}
        }},
        {"cameras", {{{"name", "MainCamera"}, {"type", "perspective"},
            {"perspective", {{"yfov", 0.7}, {"znear", 0.1}}}}}},
        {"materials", {{{"name", "PBR"}, {"doubleSided", false},
            {"alphaMode", "MASK"}, {"alphaCutoff", 0.35},
            {"pbrMetallicRoughness", {
                {"baseColorFactor", {0.8, 0.4, 0.2, 0.9}},
                {"metallicFactor", 0.7}, {"roughnessFactor", 0.25}}}}}},
        {"meshes", {{{"name", "SharedTriangle"}, {"primitives", {{
            {"attributes", {{"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}},
            {"indices", 3}, {"material", 0}
        }}}}}},
        {"buffers", {{{"uri", "mesh.bin"}, {"byteLength", 102}}}},
        {"bufferViews", {
            {{"buffer", 0}, {"byteOffset", 0}, {"byteLength", 36}},
            {{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 36}},
            {{"buffer", 0}, {"byteOffset", 72}, {"byteLength", 24}},
            {{"buffer", 0}, {"byteOffset", 96}, {"byteLength", 6}}
        }},
        {"accessors", {
            {{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"},
             {"min", {0.0, 0.0, 0.0}}, {"max", {1.0, 1.0, 0.0}}},
            {{"bufferView", 1}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 2}, {"componentType", 5126}, {"count", 3}, {"type", "VEC2"}},
            {{"bufferView", 3}, {"componentType", 5123}, {"count", 3}, {"type", "SCALAR"}}
        }}
    };
    {
        std::ofstream output(gltf_path);
        output << gltf.dump(2) << '\n';
    }

    const renderer::LoadedGltfScene loaded =
        renderer::load_gltf_scene(gltf_path, 320, 200);
    RENDER_CHECK(loaded.meshes.size() == 1);
    RENDER_CHECK(loaded.nodes.size() == 6);
    RENDER_CHECK(loaded.cameras.size() == 1);
    RENDER_CHECK(loaded.nodes[1].mesh_index == 0);
    RENDER_CHECK(loaded.nodes[2].mesh_index == 0);
    RENDER_CHECK(loaded.nodes[3].light_type == renderer::GltfNodeAsset::LightType::Spot);
    renderer::SceneTransform millimeter_node_transform;
    millimeter_node_transform.local_matrix = loaded.nodes[4].local_transform;
    RENDER_CHECK(millimeter_node_transform.valid());
    const renderer::Material& material = loaded.meshes[0].scene.materials[0];
    RENDER_CHECK(material.type == renderer::MaterialType::Pbr);
    RENDER_CHECK(nearly_equal(material.metallic, 0.7f));
    RENDER_CHECK(nearly_equal(material.roughness, 0.25f));
    RENDER_CHECK(material.alpha_mode == renderer::AlphaMode::Mask);
    RENDER_CHECK(!material.two_sided);

    const renderer::LoadedScene flattened =
        renderer::load_scene_asset(gltf_path.string(), 320, 200);
    RENDER_CHECK(flattened.scene.triangles.size() == 3);
    RENDER_CHECK(flattened.scene.spot_lights.size() == 1);
    RENDER_CHECK(nearly_equal(flattened.scene.spot_lights[0].range, 5.0f));
    RENDER_CHECK(flattened.bounds.min.x() < -0.99f);
    RENDER_CHECK(flattened.bounds.max.x() > 1.99f);
    RENDER_CHECK(flattened.camera.eye().z() > 2.9f);

    renderer::SceneDocument document;
    const std::vector<renderer::ObjectId> imported =
        document.import_path(gltf_path, 320, 200);
    RENDER_CHECK(!imported.empty());
    RENDER_CHECK(document.assets().size() == 1);
    const std::size_t camera_count = static_cast<std::size_t>(std::count_if(
        document.objects().begin(),
        document.objects().end(),
        [](const renderer::SceneObject& object) {
            return object.type == renderer::SceneObjectType::Camera;
        }));
    const std::size_t spot_count = static_cast<std::size_t>(std::count_if(
        document.objects().begin(),
        document.objects().end(),
        [](const renderer::SceneObject& object) {
            return object.type == renderer::SceneObjectType::SpotLight;
        }));
    RENDER_CHECK(camera_count == 1);
    RENDER_CHECK(spot_count == 1);
    const auto imported_spot = std::find_if(
        document.objects().begin(),
        document.objects().end(),
        [](const renderer::SceneObject& object) {
            return object.type == renderer::SceneObjectType::SpotLight;
        });
    RENDER_CHECK(imported_spot != document.objects().end());
    RENDER_CHECK(nearly_equal(imported_spot->light_range, 5.0f));
    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_gltf_texture_origin_sharing_and_material_extensions) {
    const std::filesystem::path directory = "test_gltf_pbr_extensions";
    const std::filesystem::path gltf_path = directory / "scene.gltf";
    const std::filesystem::path required_transmission_path =
        directory / "required_transmission.gltf";
    const std::filesystem::path binary_path = directory / "mesh.bin";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    renderer::ImageTexture bottom_left_texture(
        2,
        2,
        std::vector<renderer::Color>{
            renderer::Color(1.0f, 0.0f, 0.0f),
            renderer::Color(0.0f, 1.0f, 0.0f),
            renderer::Color(0.0f, 0.0f, 1.0f),
            renderer::Color(1.0f, 1.0f, 1.0f)},
        renderer::TextureUvOrigin::BottomLeft);
    bottom_left_texture.set_sampler(
        renderer::TextureWrap::ClampToEdge,
        renderer::TextureWrap::ClampToEdge,
        renderer::TextureFilter::Nearest,
        renderer::TextureFilter::Nearest);
    RENDER_CHECK(
        bottom_left_texture.sample(renderer::Vec2(0.0f, 0.0f)).z() > 0.99f);

    renderer::Scene normal_scene;
    normal_scene.textures.emplace_back(
        2,
        2,
        std::vector<renderer::Color>{
            renderer::Color(0.5f, 0.5f, 1.0f),
            renderer::Color(1.0f, 0.5f, 0.5f),
            renderer::Color(0.5f, 1.0f, 0.5f),
            renderer::Color(0.5f, 0.5f, 1.0f)},
        renderer::TextureUvOrigin::TopLeft);
    normal_scene.textures[0].set_sampler(
        renderer::TextureWrap::ClampToEdge,
        renderer::TextureWrap::ClampToEdge,
        renderer::TextureFilter::Nearest,
        renderer::TextureFilter::Nearest);
    renderer::Material normal_material;
    normal_material.type = renderer::MaterialType::Pbr;
    normal_material.normal_texture_id = 0;
    normal_material.normal_texture_transform.offset = renderer::Vec2(0.5f, 0.0f);
    renderer::HitRecord normal_hit;
    normal_hit.uv = renderer::Vec2::Zero();
    normal_hit.vertex_color = renderer::Color::Ones();
    normal_hit.vertex_alpha = 1.0f;
    normal_hit.geometric_normal = renderer::Vec3::UnitZ();
    normal_hit.shading_normal = renderer::Vec3::UnitZ();
    normal_hit.tangent = renderer::Vec3::UnitX();
    normal_hit.bitangent = renderer::Vec3::UnitY();
    normal_hit.has_valid_uv_basis = true;
    const renderer::SurfaceMaterialSample normal_surface =
        renderer::evaluate_surface_material(
            normal_scene,
            normal_material,
            normal_hit);
    RENDER_CHECK(normal_surface.shading_normal.x() > 0.99f);

    const std::array<float, 9> positions{
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f};
    const std::array<float, 9> normals{
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f};
    const std::array<float, 6> uvs{0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f};
    const std::array<std::uint16_t, 3> indices{0, 1, 2};
    {
        std::ofstream output(binary_path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(positions.data()), sizeof(positions));
        output.write(reinterpret_cast<const char*>(normals.data()), sizeof(normals));
        output.write(reinterpret_cast<const char*>(uvs.data()), sizeof(uvs));
        output.write(reinterpret_cast<const char*>(indices.data()), sizeof(indices));
    }

    constexpr const char* image_uri =
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAAAXNSR0IArs4c6QAAAARnQU1B"
        "AACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAAbSURBVBhXY/jPwODA8J+hgYGB4f+B"
        "/////wcAOSUIeujznm4AAAAASUVORK5CYII=";
    nlohmann::json gltf{
        {"asset", {{"version", "2.0"}}},
        {"extensionsUsed", {
            "KHR_texture_transform",
            "KHR_materials_pbrSpecularGlossiness",
            "KHR_materials_specular",
            "KHR_materials_ior",
            "KHR_materials_emissive_strength",
            "KHR_materials_transmission"}},
        {"extensionsRequired", {"KHR_materials_pbrSpecularGlossiness"}},
        {"scene", 0},
        {"scenes", {{{"nodes", {0, 1}}}}},
        {"nodes", {
            {{"name", "SpecGloss"}, {"mesh", 0}},
            {{"name", "MetalRough"}, {"mesh", 1}, {"translation", {1.5, 0.0, 0.0}}}}},
        {"images", {{{"uri", image_uri}}}},
        {"samplers", {{{"magFilter", 9728}, {"minFilter", 9728},
            {"wrapS", 33071}, {"wrapT", 33071}}}},
        {"textures", {
            {{"sampler", 0}, {"source", 0}},
            {{"sampler", 0}, {"source", 0}}}},
        {"materials", {
            {{"name", "SpecGloss"},
             {"alphaMode", "BLEND"},
             {"extensions", {
                 {"KHR_materials_pbrSpecularGlossiness", {
                     {"diffuseFactor", {0.5, 0.5, 0.5, 0.8}},
                     {"diffuseTexture", {
                         {"index", 0},
                         {"extensions", {{"KHR_texture_transform", {
                             {"offset", {0.5, 0.0}}, {"scale", {0.5, 1.0}}}}}}}},
                     {"specularFactor", {0.2, 0.3, 0.4}},
                     {"glossinessFactor", 0.7},
                     {"specularGlossinessTexture", {{"index", 1}}}}}}}},
            {{"name", "MetalRough"},
             {"emissiveFactor", {0.1, 0.2, 0.3}},
             {"pbrMetallicRoughness", {
                 {"baseColorFactor", {0.8, 0.6, 0.4, 1.0}},
                 {"metallicFactor", 0.0},
                 {"roughnessFactor", 0.4}}},
             {"extensions", {
                 {"KHR_materials_specular", {
                     {"specularFactor", 0.5},
                     {"specularColorFactor", {0.5, 1.0, 0.25}}}},
                 {"KHR_materials_ior", {{"ior", 2.0}}},
                 {"KHR_materials_emissive_strength", {{"emissiveStrength", 3.0}}},
                 {"KHR_materials_transmission", {{"transmissionFactor", 0.75}}}}}}}},
        {"meshes", {
            {{"primitives", {{{"attributes", {
                {"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}},
                {"indices", 3}, {"material", 0}}}}},
            {{"primitives", {{{"attributes", {
                {"POSITION", 0}, {"NORMAL", 1}, {"TEXCOORD_0", 2}}},
                {"indices", 3}, {"material", 1}}}}}}},
        {"buffers", {{{"uri", "mesh.bin"}, {"byteLength", 102}}}},
        {"bufferViews", {
            {{"buffer", 0}, {"byteOffset", 0}, {"byteLength", 36}},
            {{"buffer", 0}, {"byteOffset", 36}, {"byteLength", 36}},
            {{"buffer", 0}, {"byteOffset", 72}, {"byteLength", 24}},
            {{"buffer", 0}, {"byteOffset", 96}, {"byteLength", 6}}}},
        {"accessors", {
            {{"bufferView", 0}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"},
             {"min", {0.0, 0.0, 0.0}}, {"max", {1.0, 1.0, 0.0}}},
            {{"bufferView", 1}, {"componentType", 5126}, {"count", 3}, {"type", "VEC3"}},
            {{"bufferView", 2}, {"componentType", 5126}, {"count", 3}, {"type", "VEC2"}},
            {{"bufferView", 3}, {"componentType", 5123}, {"count", 3}, {"type", "SCALAR"}}}}
    };
    {
        std::ofstream output(gltf_path);
        output << gltf.dump(2) << '\n';
    }

    const renderer::LoadedGltfScene loaded =
        renderer::load_gltf_scene(gltf_path, 64, 64);
    RENDER_CHECK(loaded.meshes.size() == 2);
    RENDER_CHECK(loaded.meshes[0].scene.textures.size() == 2);
    const renderer::ImageTexture& first_texture = loaded.meshes[0].scene.textures[0];
    const renderer::ImageTexture& second_texture = loaded.meshes[0].scene.textures[1];
    RENDER_CHECK(first_texture.uv_origin() == renderer::TextureUvOrigin::TopLeft);
    RENDER_CHECK(first_texture.shares_pixel_storage_with(second_texture));
    RENDER_CHECK(first_texture.same_resource_view(second_texture));
    RENDER_CHECK(first_texture.sample(renderer::Vec2(0.0f, 0.0f)).x() > 0.99f);
    RENDER_CHECK(nearly_equal(
        first_texture.sample_alpha(renderer::Vec2(0.0f, 0.0f)),
        64.0f / 255.0f,
        1.0e-5f));

    const renderer::Material& spec_gloss = loaded.meshes[0].scene.materials[0];
    RENDER_CHECK(spec_gloss.pbr_workflow == renderer::PbrWorkflow::SpecularGlossiness);
    RENDER_CHECK(nearly_equal(spec_gloss.roughness, 0.3f));
    renderer::HitRecord hit;
    hit.uv = renderer::Vec2::Zero();
    hit.vertex_color = renderer::Color::Ones();
    hit.vertex_alpha = 1.0f;
    hit.geometric_normal = renderer::Vec3::UnitZ();
    hit.shading_normal = renderer::Vec3::UnitZ();
    const renderer::SurfaceMaterialSample spec_gloss_surface =
        renderer::evaluate_surface_material(
            loaded.meshes[0].scene,
            spec_gloss,
            hit);
    RENDER_CHECK(spec_gloss_surface.base_color.y() > 0.49f);
    RENDER_CHECK(spec_gloss_surface.base_color.x() < 1.0e-5f);
    RENDER_CHECK(nearly_equal(
        spec_gloss_surface.opacity,
        0.8f * (128.0f / 255.0f),
        1.0e-4f));
    RENDER_CHECK(nearly_equal(spec_gloss_surface.specular_f0.x(), 0.2f, 1.0e-5f));
    RENDER_CHECK(nearly_equal(spec_gloss_surface.specular_f0.y(), 0.0f, 1.0e-5f));
    RENDER_CHECK(nearly_equal(
        spec_gloss_surface.roughness,
        1.0f - 0.7f * (64.0f / 255.0f),
        1.0e-4f));

    const renderer::Material& metal_rough = loaded.meshes[1].scene.materials[1];
    RENDER_CHECK(metal_rough.pbr_workflow == renderer::PbrWorkflow::MetallicRoughness);
    RENDER_CHECK(nearly_equal(metal_rough.ior, 2.0f));
    RENDER_CHECK(nearly_equal(metal_rough.specular_factor, 0.5f));
    RENDER_CHECK(nearly_equal(metal_rough.emission.z(), 0.9f));
    const renderer::SurfaceMaterialSample metal_rough_surface =
        renderer::evaluate_surface_material(
            loaded.meshes[1].scene,
            metal_rough,
            hit);
    RENDER_CHECK(nearly_equal(
        metal_rough_surface.specular_f0.x(),
        (1.0f / 9.0f) * 0.5f * 0.5f,
        1.0e-5f));
    RENDER_CHECK(std::count_if(
        loaded.warnings.begin(),
        loaded.warnings.end(),
        [](const std::string& warning) {
            return warning.find("KHR_materials_transmission") != std::string::npos;
        }) == 1);

    renderer::SceneDocument document;
    document.import_path(gltf_path, 64, 64);
    RENDER_CHECK(document.assets().size() == 2);
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(document.render_scene_snapshot()).textures.size() == 1);
    RENDER_CHECK(document.render_scene_snapshot().textures.size() == 1);

    const std::filesystem::path scene_path = directory / "shared-textures.rscene";
    document.save(scene_path);
    renderer::SceneDocument restored =
        renderer::SceneDocument::load(scene_path, 64, 64);
    RENDER_CHECK(restored.assets().size() == 2);
    RENDER_CHECK(restored.assets()[0]->source_mesh_index !=
                 restored.assets()[1]->source_mesh_index);
    RENDER_CHECK(!restored.assets()[0]->local_scene.textures.empty());
    RENDER_CHECK(!restored.assets()[1]->local_scene.textures.empty());
    RENDER_CHECK(
        restored.assets()[0]->local_scene.textures[0].shares_pixel_storage_with(
            restored.assets()[1]->local_scene.textures[0]));
    RENDER_CHECK(renderer::flatten_render_scene_snapshot(restored.render_scene_snapshot()).textures.size() == 1);
    RENDER_CHECK(restored.render_scene_snapshot().textures.size() == 1);

    nlohmann::json required_transmission = gltf;
    required_transmission["extensionsRequired"] = {
        "KHR_materials_pbrSpecularGlossiness",
        "KHR_materials_transmission"};
    {
        std::ofstream output(required_transmission_path);
        output << required_transmission.dump(2) << '\n';
    }
    bool rejected_required_transmission = false;
    try {
        (void)renderer::load_gltf_scene(required_transmission_path, 64, 64);
    } catch (const std::runtime_error& error) {
        rejected_required_transmission =
            std::string(error.what()).find("KHR_materials_transmission") != std::string::npos;
    }
    RENDER_CHECK(rejected_required_transmission);
    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_render_scene_snapshot_validates_optional_material_slots) {
    renderer::Scene missing_scene;
    missing_scene.materials.push_back(renderer::Material{});
    missing_scene.triangles.emplace_back(
        renderer::Vec3(-1.0f, -1.0f, 0.0f),
        renderer::Vec3(1.0f, -1.0f, 0.0f),
        renderer::Vec3(0.0f, 1.0f, 0.0f),
        -1);
    renderer::SceneDocument document = renderer::SceneDocument::from_scene(
        std::move(missing_scene),
        "Missing material");
    const renderer::ObjectId first = document.objects().front().id;
    const renderer::ObjectId second = document.duplicate_subtree(first);
    RENDER_CHECK(second != renderer::kInvalidObjectId);

    const renderer::RenderSceneSnapshot& snapshot =
        document.render_scene_snapshot();
    RENDER_CHECK(snapshot.assets.size() == 1);
    RENDER_CHECK(snapshot.instances.size() == 2);
    RENDER_CHECK(snapshot.assets[0].triangle_material_slots.size() == 1);
    RENDER_CHECK(
        !snapshot.assets[0].triangle_material_slots[0].has_value());

    const renderer::Scene& flattened = renderer::flatten_render_scene_snapshot(document.render_scene_snapshot());
    RENDER_CHECK(flattened.triangles.size() == 2);
    for (const renderer::Triangle& triangle : flattened.triangles) {
        const int material_id = triangle.material_id();
        RENDER_CHECK(material_id >= 0);
        RENDER_CHECK(
            static_cast<std::size_t>(material_id) <
            flattened.materials.size());
        RENDER_CHECK(flattened.materials[material_id].base_color.isApprox(
            renderer::Color(1.0f, 0.0f, 1.0f)));
    }

    renderer::Scene invalid_scene;
    invalid_scene.materials.push_back(renderer::Material{});
    invalid_scene.triangles.emplace_back(
        renderer::Vec3(-1.0f, -1.0f, 0.0f),
        renderer::Vec3(1.0f, -1.0f, 0.0f),
        renderer::Vec3(0.0f, 1.0f, 0.0f),
        1);
    renderer::SceneDocument invalid_document =
        renderer::SceneDocument::from_scene(
            std::move(invalid_scene),
            "Invalid material");
    bool rejected = false;
    try {
        (void)invalid_document.render_scene_snapshot();
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("out of range") !=
            std::string::npos;
    }
    RENDER_CHECK(rejected);
}

RENDER_TEST(test_scene_revisions_and_mergeable_edit_transactions) {
    renderer::SceneDocument document = renderer::SceneDocument::from_scene(
        renderer::make_triangle_scene(),
        "Revision scene");
    const renderer::ObjectId object_id = document.objects().front().id;
    const renderer::SceneRevisions initial = document.revisions();

    RENDER_CHECK(document.set_object_name(object_id, "Renamed"));
    RENDER_CHECK(document.revisions() == initial);
    RENDER_CHECK(document.dirty());
    document.checkpoint();

    renderer::Mat4 translated = renderer::Mat4::Identity();
    translated(0, 3) = 1.0f;
    {
        renderer::SceneEditTransaction edit =
            document.begin_edit("inspector-transform");
        RENDER_CHECK(document.set_world_matrix(object_id, translated));
        const renderer::SceneRevisions preview = document.revisions();
        RENDER_CHECK(preview.transforms > initial.transforms);
        RENDER_CHECK(preview.lighting > initial.lighting);
        RENDER_CHECK(document.dirty());
        edit.commit();
    }
    translated(0, 3) = 2.0f;
    {
        renderer::SceneEditTransaction edit =
            document.begin_edit("inspector-transform");
        RENDER_CHECK(document.set_world_matrix(object_id, translated));
        edit.commit();
    }
    RENDER_CHECK(document.world_matrix(object_id).isApprox(translated));
    RENDER_CHECK(document.undo());
    RENDER_CHECK(nearly_equal(
        document.world_matrix(object_id)(0, 3),
        0.0f));
    RENDER_CHECK(document.redo());
    RENDER_CHECK(nearly_equal(
        document.world_matrix(object_id)(0, 3),
        2.0f));

    const renderer::Mat4 before_cancel = document.world_matrix(object_id);
    const renderer::SceneRevisions before_cancel_revisions =
        document.revisions();
    {
        renderer::SceneEditTransaction edit = document.begin_edit();
        renderer::Mat4 temporary = before_cancel;
        temporary(1, 3) = 7.0f;
        RENDER_CHECK(document.set_world_matrix(object_id, temporary));
        edit.cancel();
    }
    RENDER_CHECK(document.world_matrix(object_id).isApprox(before_cancel));
    RENDER_CHECK(
        document.revisions().transforms >
        before_cancel_revisions.transforms);

    const renderer::SceneRevisions before_environment =
        document.revisions();
    document.set_environment_intensity(2.0f);
    RENDER_CHECK(
        document.revisions().environment ==
        before_environment.environment + 1);
    RENDER_CHECK(
        document.revisions().geometry ==
        before_environment.geometry);
    RENDER_CHECK(
        document.render_scene_snapshot().revisions ==
        document.revisions());
}

RENDER_TEST(test_shear_matrix_reparent_undo_and_v5_roundtrip) {
    const std::filesystem::path directory = "test_scene_matrix_v5";
    const std::filesystem::path obj_path = directory / "triangle.obj";
    const std::filesystem::path scene_path = directory / "matrix.rscene";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    {
        std::ofstream obj(obj_path);
        obj << "v -1 -1 0\n";
        obj << "v 1 -1 0\n";
        obj << "v 0 1 0\n";
        obj << "f 1 2 3\n";
    }

    renderer::SceneDocument document;
    const std::vector<renderer::ObjectId> imported =
        document.import_path(obj_path, 64, 64);
    RENDER_CHECK(imported.size() == 1);
    const renderer::ObjectId mesh_id = imported.front();
    const renderer::ObjectId group_id = document.create_group("Parent");
    renderer::Mat4 parent_world = renderer::Mat4::Identity();
    parent_world(0, 3) = -2.0f;
    parent_world(1, 3) = 1.0f;
    RENDER_CHECK(document.set_world_matrix(group_id, parent_world));
    document.checkpoint();

    renderer::Mat4 shear_world = renderer::Mat4::Identity();
    shear_world(0, 1) = 0.35f;
    shear_world(1, 2) = -0.2f;
    shear_world(0, 3) = 3.0f;
    shear_world(2, 3) = -4.0f;
    RENDER_CHECK(document.set_world_matrix(mesh_id, shear_world));
    document.checkpoint();
    RENDER_CHECK(!document.local_trs(mesh_id).has_value());

    const renderer::Mat4 before_reparent = document.world_matrix(mesh_id);
    RENDER_CHECK(document.reparent(mesh_id, group_id));
    RENDER_CHECK(
        (document.world_matrix(mesh_id) - before_reparent)
            .cwiseAbs()
            .maxCoeff() < 1.0e-5f);
    RENDER_CHECK(!document.local_trs(mesh_id).has_value());
    RENDER_CHECK(document.undo());
    RENDER_CHECK(
        (document.world_matrix(mesh_id) - before_reparent)
            .cwiseAbs()
            .maxCoeff() < 1.0e-5f);
    RENDER_CHECK(document.redo());
    RENDER_CHECK(
        (document.world_matrix(mesh_id) - before_reparent)
            .cwiseAbs()
            .maxCoeff() < 1.0e-5f);

    const renderer::Mat4 accepted_local =
        document.find(mesh_id)->transform.local_matrix;
    renderer::Mat4 non_affine = before_reparent;
    non_affine(3, 0) = 0.1f;
    RENDER_CHECK(!document.set_world_matrix(mesh_id, non_affine));
    RENDER_CHECK(document.find(mesh_id)->transform.local_matrix.isApprox(
        accepted_local));
    renderer::Mat4 singular = before_reparent;
    singular.row(1).setZero();
    RENDER_CHECK(!document.set_world_matrix(mesh_id, singular));
    RENDER_CHECK(document.find(mesh_id)->transform.local_matrix.isApprox(
        accepted_local));

    document.save(scene_path);
    nlohmann::json saved;
    {
        std::ifstream input(scene_path);
        input >> saved;
    }
    RENDER_CHECK(saved.at("version").get<int>() == 5);
    for (const auto& object : saved.at("objects")) {
        RENDER_CHECK(object.at("local_matrix").size() == 4);
        for (const auto& row : object.at("local_matrix")) {
            RENDER_CHECK(row.size() == 4);
        }
    }
    renderer::SceneDocument loaded =
        renderer::SceneDocument::load(scene_path, 64, 64);
    RENDER_CHECK(
        (loaded.world_matrix(mesh_id) - before_reparent)
            .cwiseAbs()
            .maxCoeff() < 1.0e-5f);
    RENDER_CHECK(!loaded.local_trs(mesh_id).has_value());
    std::filesystem::remove_all(directory);
}

RENDER_TEST(test_document_sphere_mesh_negative_nonuniform_scale_and_pick) {
    renderer::Scene scene;
    scene.materials.push_back(renderer::Material{});
    scene.spheres.emplace_back(
        renderer::Vec3::Zero(),
        1.0f,
        0);
    renderer::SceneDocument document = renderer::SceneDocument::from_scene(
        std::move(scene),
        "Sphere");
    const renderer::ObjectId sphere_id = document.objects().front().id;
    renderer::SceneTrs trs;
    trs.scale = renderer::Vec3(-2.0f, 1.0f, 0.5f);
    RENDER_CHECK(document.set_local_trs(sphere_id, trs));
    document.checkpoint();

    const renderer::RenderSceneSnapshot& snapshot =
        document.render_scene_snapshot();
    RENDER_CHECK(snapshot.assets.size() == 1);
    RENDER_CHECK(snapshot.instances.size() == 1);
    RENDER_CHECK(snapshot.assets[0].local_scene != nullptr);
    RENDER_CHECK(snapshot.assets[0].local_scene->spheres.empty());
    RENDER_CHECK(
        snapshot.assets[0].local_scene->triangles.size() == 3968);
    for (const renderer::Triangle& triangle :
         snapshot.assets[0].local_scene->triangles) {
        for (int vertex = 0; vertex < 3; ++vertex) {
            RENDER_CHECK(triangle.vertex(vertex).has_normal);
            RENDER_CHECK(triangle.vertex(vertex).uv.allFinite());
        }
    }
    const renderer::Bounds3 bounds = document.world_bounds(sphere_id);
    RENDER_CHECK(bounds.min.isApprox(renderer::Vec3(-2.0f, -1.0f, -0.5f), 1.0e-4f));
    RENDER_CHECK(bounds.max.isApprox(renderer::Vec3(2.0f, 1.0f, 0.5f), 1.0e-4f));
    const renderer::Ray ray(
        renderer::Vec3(0.2f, 0.1f, 3.0f),
        renderer::Vec3(0.0f, 0.0f, -1.0f));
    const auto picked = document.pick(ray);
    RENDER_CHECK(picked.has_value());
    RENDER_CHECK(picked->object_id == sphere_id);
    RENDER_CHECK(
        renderer::flatten_render_scene_snapshot(document.render_scene_snapshot()).triangles.size() ==
        snapshot.assets[0].local_scene->triangles.size());
}
