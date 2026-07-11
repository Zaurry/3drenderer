#include "test_framework.h"

#include "acceleration/bvh.h"
#include "core/color.h"
#include "core/image.h"
#include "core/math/bounds.h"
#include "core/math/mat4.h"
#include "core/math/ray.h"
#include "core/math/types.h"
#include "core/math/vec2.h"
#include "core/math/vec3.h"
#include "core/math/vec4.h"
#include "core/random.h"
#include "core/timer.h"
#include "interactive/frame_rate_counter.h"
#include "interactive/orbit_camera_controller.h"
#include "render/renderer.h"
#include "render/render_settings.h"
#include "render/depth_buffer.h"
#include "render/framebuffer.h"
#include "render/interactive/interactive_render_session.h"
#include "render/interactive/path_interactive_session.h"
#include "render/interactive/raster_interactive_session.h"
#include "render/interactive/ray_interactive_session.h"
#include "render/pathtracer/pathtracer_renderer.h"
#include "render/rasterizer/raster_geometry.h"
#include "render/rasterizer/rasterizer_renderer.h"
#include "render/raytracer/raytracer_renderer.h"
#include "render/scene_intersector.h"
#include "sampling/sampler.h"
#include "scene/camera.h"
#include "scene/material.h"
#include "scene/material_evaluator.h"
#include "scene/obj_loader.h"
#include "scene/primitive.h"
#include "scene/scene_asset_loader.h"
#include "scene/scene.h"
#include "scene/texture.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
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

void test_vec3_arithmetic() {
    renderer::Vec3 a(1.0, 2.0, 3.0);
    renderer::Vec3 b(4.0, -2.0, 0.5);

    renderer::Vec3 sum = a + b;
    RENDER_CHECK(nearly_equal(sum.x(), 5.0));
    RENDER_CHECK(nearly_equal(sum.y(), 0.0));
    RENDER_CHECK(nearly_equal(sum.z(), 3.5));

    renderer::Vec3 difference = a - b;
    RENDER_CHECK(nearly_equal(difference.x(), -3.0));
    RENDER_CHECK(nearly_equal(difference.y(), 4.0));
    RENDER_CHECK(nearly_equal(difference.z(), 2.5));

    renderer::Vec3 scaled_right = a * 2.0f;
    RENDER_CHECK(nearly_equal(scaled_right.x(), 2.0));
    RENDER_CHECK(nearly_equal(scaled_right.y(), 4.0));
    RENDER_CHECK(nearly_equal(scaled_right.z(), 6.0));

    renderer::Vec3 scaled_left = 0.5f * b;
    RENDER_CHECK(nearly_equal(scaled_left.x(), 2.0));
    RENDER_CHECK(nearly_equal(scaled_left.y(), -1.0));
    RENDER_CHECK(nearly_equal(scaled_left.z(), 0.25));

    renderer::Vec3 divided = a / 2.0f;
    RENDER_CHECK(nearly_equal(divided.x(), 0.5));
    RENDER_CHECK(nearly_equal(divided.y(), 1.0));
    RENDER_CHECK(nearly_equal(divided.z(), 1.5));

    renderer::Vec3 negated = -a;
    RENDER_CHECK(nearly_equal(negated.x(), -1.0));
    RENDER_CHECK(nearly_equal(negated.y(), -2.0));
    RENDER_CHECK(nearly_equal(negated.z(), -3.0));

    renderer::Vec3 min_v = renderer::min_components(a, b);
    RENDER_CHECK(nearly_equal(min_v.x(), 1.0));
    RENDER_CHECK(nearly_equal(min_v.y(), -2.0));
    RENDER_CHECK(nearly_equal(min_v.z(), 0.5));

    renderer::Vec3 max_v = renderer::max_components(a, b);
    RENDER_CHECK(nearly_equal(max_v.x(), 4.0));
    RENDER_CHECK(nearly_equal(max_v.y(), 2.0));
    RENDER_CHECK(nearly_equal(max_v.z(), 3.0));

    RENDER_CHECK(nearly_equal(renderer::dot(a, b), 1.0 * 4.0 + 2.0 * -2.0 + 3.0 * 0.5));

    renderer::Vec3 c = renderer::cross(renderer::Vec3(1, 0, 0), renderer::Vec3(0, 1, 0));
    RENDER_CHECK(nearly_equal(c.x(), 0.0));
    RENDER_CHECK(nearly_equal(c.y(), 0.0));
    RENDER_CHECK(nearly_equal(c.z(), 1.0));

    renderer::Vec3 n = renderer::normalize(renderer::Vec3(0, 3, 4));
    RENDER_CHECK(nearly_equal(renderer::length(n), 1.0));
    RENDER_CHECK(nearly_equal(n.y(), 0.6f, 1e-6f));
    RENDER_CHECK(nearly_equal(n.z(), 0.8f, 1e-6f));
}

void test_mat4_composition_order() {
    const renderer::Mat4 transform =
        (Eigen::Translation3f(renderer::Vec3(1.0f, 2.0f, 3.0f)) *
         Eigen::Scaling(2.0f, 3.0f, 4.0f))
            .matrix();
    const renderer::Vec4 p = transform * renderer::Vec4(1.0f, 1.0f, 1.0f, 1.0f);
    RENDER_CHECK(nearly_equal(p.x(), 3.0));
    RENDER_CHECK(nearly_equal(p.y(), 5.0));
    RENDER_CHECK(nearly_equal(p.z(), 7.0));
    RENDER_CHECK(nearly_equal(p.w(), 1.0));
}

void test_mat4_perspective_uses_degrees_and_ndc_depth() {
    const renderer::Mat4 p = renderer::make_perspective_matrix(90.0f, 1.0f, 1.0f, 10.0f);
    RENDER_CHECK(nearly_equal(p(0, 0), 1.0));
    RENDER_CHECK(nearly_equal(p(1, 1), 1.0));
    RENDER_CHECK(nearly_equal(p(2, 2), -11.0f / 9.0f));
    RENDER_CHECK(nearly_equal(p(2, 3), -20.0f / 9.0f));
    RENDER_CHECK(nearly_equal(p(3, 2), -1.0));

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

void test_mat4_perspective_invalid_inputs_throw() {
    check_perspective_invalid_input_throws(0.0, 1.0, 1.0, 10.0);
    check_perspective_invalid_input_throws(-1.0, 1.0, 1.0, 10.0);
    check_perspective_invalid_input_throws(180.0, 1.0, 1.0, 10.0);
    check_perspective_invalid_input_throws(181.0, 1.0, 1.0, 10.0);
    check_perspective_invalid_input_throws(90.0, 0.0, 1.0, 10.0);
    check_perspective_invalid_input_throws(90.0, -1.0, 1.0, 10.0);
    check_perspective_invalid_input_throws(90.0, 1.0, 0.0, 10.0);
    check_perspective_invalid_input_throws(90.0, 1.0, -1.0, 10.0);
    check_perspective_invalid_input_throws(90.0, 1.0, 1.0, 1.0);
    check_perspective_invalid_input_throws(90.0, 1.0, 10.0, 1.0);
    check_perspective_invalid_input_throws(
        std::numeric_limits<float>::quiet_NaN(), 1.0f, 1.0f, 10.0f);
    check_perspective_invalid_input_throws(
        90.0f, std::numeric_limits<float>::infinity(), 1.0f, 10.0f);
    check_perspective_invalid_input_throws(
        90.0f, 1.0f, std::numeric_limits<float>::quiet_NaN(), 10.0f);
    check_perspective_invalid_input_throws(
        90.0f, 1.0f, 1.0f, -std::numeric_limits<float>::infinity());
}

void test_mat4_look_at() {
    const renderer::Mat4 view = renderer::make_look_at_matrix(
        renderer::Vec3(0.0f, 0.0f, 0.0f),
        renderer::Vec3(0.0f, 0.0f, -1.0f),
        renderer::Vec3(0.0f, 1.0f, 0.0f));
    RENDER_CHECK(nearly_equal(view(0, 0), 1.0));
    RENDER_CHECK(nearly_equal(view(1, 1), 1.0));
    RENDER_CHECK(nearly_equal(view(2, 2), 1.0));
    RENDER_CHECK(nearly_equal(view(3, 3), 1.0));
    const renderer::Vec4 p = view * renderer::Vec4(0.0f, 0.0f, -1.0f, 1.0f);
    RENDER_CHECK(nearly_equal(p.z(), -1.0));
}

void test_mat4_look_at_invalid_inputs_throw() {
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

void test_camera_center_ray_points_forward() {
    renderer::Camera camera(
        renderer::Vec3(0, 0, 0),
        renderer::Vec3(0, 0, -1),
        renderer::Vec3(0, 1, 0),
        60.0,
        1.0);

    renderer::Ray ray = camera.generate_ray(0.5, 0.5);
    RENDER_CHECK(nearly_equal(ray.direction.x(), 0.0f, 1e-6f));
    RENDER_CHECK(nearly_equal(ray.direction.y(), 0.0f, 1e-6f));
    RENDER_CHECK(ray.direction.z() < -0.999);
}

void test_camera_rejects_non_finite_screen_coordinates() {
    renderer::Camera camera(
        renderer::Vec3(0, 0, 0),
        renderer::Vec3(0, 0, -1),
        renderer::Vec3(0, 1, 0),
        60.0,
        1.0);

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

void test_ray_and_bounds_intersection() {
    renderer::Ray ray(renderer::Vec3(0, 0, -5), renderer::Vec3(0, 0, 1));
    renderer::Bounds3 box(renderer::Vec3(-1, -1, -1), renderer::Vec3(1, 1, 1));
    RENDER_CHECK(box.intersect(ray, 0.001f, 1000.0f));

    renderer::Ray miss(renderer::Vec3(5, 5, -5), renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!box.intersect(miss, 0.001f, 1000.0f));
}

void test_bounds_intersection_counts_corner_touch_as_hit() {
    renderer::Bounds3 box(renderer::Vec3(-1, -1, -1), renderer::Vec3(1, 1, 1));
    renderer::Ray corner_touch(renderer::Vec3(-2, -2, 1), renderer::Vec3(1, 1, 0));
    RENDER_CHECK(box.intersect(corner_touch, 0.001f, 1000.0f));
}

void test_image_invalid_dimensions_throw_invalid_argument() {
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

void test_image_stores_gamma_corrected_pixels() {
    renderer::Image image(2, 1);
    image.set_pixel(0, 0, renderer::Color(1.0, 0.25, 0.0));
    renderer::Rgb8 pixel = image.pixel_rgb8(0, 0);
    RENDER_CHECK(pixel.r == 255);
    RENDER_CHECK(pixel.g >= 135 && pixel.g <= 137);
    RENDER_CHECK(pixel.b == 0);
}

void test_to_rgb8_uses_standard_srgb_transfer_curve() {
    RENDER_CHECK(renderer::channel_to_rgb8(0.0031308f) == 10);
    RENDER_CHECK(renderer::channel_to_rgb8(0.5f) == 188);
}

void test_to_rgb8_sanitizes_non_finite_channels() {
    const renderer::Rgb8 nan_pixel = renderer::to_rgb8(
        renderer::Color(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f));
    RENDER_CHECK(nan_pixel.r == 0);

    const renderer::Rgb8 infinity_pixel = renderer::to_rgb8(
        renderer::Color(std::numeric_limits<float>::infinity(), 0.0f, 0.0f));
    RENDER_CHECK(infinity_pixel.r == 255);
}

void test_orbit_camera_controller_zoom_and_orbit_change_camera() {
    renderer::Bounds3 bounds(renderer::Vec3(-1, 0, -1), renderer::Vec3(1, 2, 1));
    renderer::OrbitCameraController controller(bounds, 1.0f);
    renderer::Camera before = controller.camera();

    controller.orbit(0.5f, 0.25f);
    controller.zoom(-1.0f);
    renderer::Camera after = controller.camera();

    RENDER_CHECK(renderer::length(after.eye() - before.eye()) > 0.001);
    RENDER_CHECK(after.viewport_width() > 0.0);
}

void test_orbit_camera_controller_horizontal_drag_tracks_scene_direction() {
    renderer::Bounds3 bounds(renderer::Vec3(-1, 0, -1), renderer::Vec3(1, 2, 1));
    renderer::OrbitCameraController controller(bounds, 1.0f);
    const float before_x = controller.camera().eye().x();

    controller.orbit(25.0f, 0.0f);
    const float after_x = controller.camera().eye().x();

    RENDER_CHECK(after_x < before_x);
}

void test_frame_rate_counter_reports_window_average() {
    renderer::FrameRateCounter counter(0.25f);
    RENDER_CHECK(!counter.snapshot().valid);
    RENDER_CHECK(!counter.tick(0.10f));

    RENDER_CHECK(counter.tick(0.15f));
    const renderer::FrameRateSnapshot snapshot = counter.snapshot();
    RENDER_CHECK(snapshot.valid);
    RENDER_CHECK(snapshot.frames == 2);
    RENDER_CHECK(nearly_equal(snapshot.frames_per_second, 8.0));
    RENDER_CHECK(nearly_equal(snapshot.milliseconds_per_frame, 125.0));

    counter.reset();
    RENDER_CHECK(!counter.snapshot().valid);
}

void test_viewer_title_format_includes_fps_and_path_samples() {
    renderer::FrameRateSnapshot warming_up;
    const std::string raster_warming_title = renderer::format_viewer_title(
        renderer::InteractiveRenderMode::Raster,
        warming_up,
        0);
    RENDER_CHECK(raster_warming_title.find("raster") != std::string::npos);
    RENDER_CHECK(raster_warming_title.find("FPS --") != std::string::npos);
    RENDER_CHECK(raster_warming_title.find("spp") == std::string::npos);

    renderer::FrameRateSnapshot snapshot;
    snapshot.valid = true;
    snapshot.frames = 3;
    snapshot.frames_per_second = 60.0;
    snapshot.milliseconds_per_frame = 16.666f;

    const std::string path_title = renderer::format_viewer_title(
        renderer::InteractiveRenderMode::Path,
        snapshot,
        12);
    RENDER_CHECK(path_title.find("path") != std::string::npos);
    RENDER_CHECK(path_title.find("60.0 FPS") != std::string::npos);
    RENDER_CHECK(path_title.find("16.7 ms") != std::string::npos);
    RENDER_CHECK(path_title.find("12 spp") != std::string::npos);
}

void test_framebuffer_clear_set_and_rgba8_conversion() {
    renderer::Framebuffer framebuffer(2, 1);
    framebuffer.clear(renderer::Color(0.25, 0.0, 1.0));
    framebuffer.set_pixel(1, 0, renderer::Color(1.0, 0.25, 0.0));

    RENDER_CHECK(framebuffer.width() == 2);
    RENDER_CHECK(framebuffer.height() == 1);
    RENDER_CHECK(nearly_equal(framebuffer.pixel(0, 0).z(), 1.0));

    const std::vector<std::uint8_t> rgba = framebuffer.to_rgba8();
    RENDER_CHECK(rgba.size() == 8);
    RENDER_CHECK(rgba[3] == 255);
    RENDER_CHECK(rgba[4] == 255);
    RENDER_CHECK(rgba[7] == 255);
}

void test_depth_buffer_clear_resize_and_access() {
    renderer::DepthBuffer depth(2, 2);
    depth.clear(42.0);
    depth.set(1, 0, 0.5);

    RENDER_CHECK(depth.width() == 2);
    RENDER_CHECK(depth.height() == 2);
    RENDER_CHECK(nearly_equal(depth.get(0, 0), 42.0));
    RENDER_CHECK(nearly_equal(depth.get(1, 0), 0.5));

    depth.resize(1, 1);
    depth.clear(7.0);
    RENDER_CHECK(depth.width() == 1);
    RENDER_CHECK(depth.height() == 1);
    RENDER_CHECK(nearly_equal(depth.get(0, 0), 7.0));
}

void test_sphere_intersection() {
    renderer::Material material;
    material.base_color = renderer::Color(1, 0, 0);
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0, 0);

    renderer::Ray ray(renderer::Vec3(0, 0, -5), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(sphere.intersect(ray, 0.001, 1000.0, hit));
    RENDER_CHECK(nearly_equal(static_cast<float>(hit.t), 4.0f));
    RENDER_CHECK(nearly_equal(hit.position.z(), -1.0));
    RENDER_CHECK(nearly_equal(renderer::length(hit.shading_normal), 1.0));
    RENDER_CHECK(hit.material_id == 0);
}

void test_sphere_rejects_zero_direction_ray() {
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0, 0);
    renderer::Ray ray(renderer::Vec3(0, 0, -5), renderer::Vec3(0, 0, 0));
    renderer::HitRecord hit;
    RENDER_CHECK(!sphere.intersect(ray, 0.001, 1000.0, hit));
}

void test_sphere_invalid_radius_throws() {
    bool threw_zero_radius = false;
    try {
        renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 0.0, 0);
    } catch (const std::invalid_argument&) {
        threw_zero_radius = true;
    }
    RENDER_CHECK(threw_zero_radius);

    bool threw_negative_radius = false;
    try {
        renderer::Sphere sphere(renderer::Vec3(0, 0, 0), -1.0, 0);
    } catch (const std::invalid_argument&) {
        threw_negative_radius = true;
    }
    RENDER_CHECK(threw_negative_radius);

    bool threw_nan_radius = false;
    try {
        renderer::Sphere sphere(renderer::Vec3(0, 0, 0), std::numeric_limits<double>::quiet_NaN(), 0);
    } catch (const std::invalid_argument&) {
        threw_nan_radius = true;
    }
    RENDER_CHECK(threw_nan_radius);

    bool threw_infinite_radius = false;
    try {
        renderer::Sphere sphere(renderer::Vec3(0, 0, 0), std::numeric_limits<double>::infinity(), 0);
    } catch (const std::invalid_argument&) {
        threw_infinite_radius = true;
    }
    RENDER_CHECK(threw_infinite_radius);
}

void test_sphere_invalid_center_throws() {
    bool threw_nan_center = false;
    try {
        renderer::Sphere sphere(
            renderer::Vec3(std::numeric_limits<float>::quiet_NaN(), 0, 0),
            1.0,
            0);
    } catch (const std::invalid_argument&) {
        threw_nan_center = true;
    }
    RENDER_CHECK(threw_nan_center);

    bool threw_infinite_center = false;
    try {
        renderer::Sphere sphere(
            renderer::Vec3(std::numeric_limits<float>::infinity(), 0, 0),
            1.0,
            0);
    } catch (const std::invalid_argument&) {
        threw_infinite_center = true;
    }
    RENDER_CHECK(threw_infinite_center);
}

void test_sphere_rejects_non_finite_direction_rays() {
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0, 0);

    renderer::HitRecord nan_hit;
    renderer::Ray nan_ray(
        renderer::Vec3(0, 0, -5),
        renderer::Vec3(0, 0, std::numeric_limits<float>::quiet_NaN()));
    RENDER_CHECK(!sphere.intersect(nan_ray, 0.001, 1000.0, nan_hit));

    renderer::HitRecord infinite_hit;
    renderer::Ray infinite_ray(
        renderer::Vec3(0, 0, -5),
        renderer::Vec3(0, 0, std::numeric_limits<float>::infinity()));
    RENDER_CHECK(!sphere.intersect(infinite_ray, 0.001, 1000.0, infinite_hit));
}

void test_sphere_rejects_non_finite_origin_rays() {
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0, 0);

    renderer::HitRecord nan_hit;
    renderer::Ray nan_ray(
        renderer::Vec3(std::numeric_limits<float>::quiet_NaN(), 0, -5),
        renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!sphere.intersect(nan_ray, 0.001, 1000.0, nan_hit));

    renderer::HitRecord infinite_hit;
    renderer::Ray infinite_ray(
        renderer::Vec3(std::numeric_limits<float>::infinity(), 0, -5),
        renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!sphere.intersect(infinite_ray, 0.001, 1000.0, infinite_hit));
}

void test_sphere_inside_ray_reports_back_face() {
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0, 1);
    renderer::Ray ray(renderer::Vec3(0, 0, 0), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(sphere.intersect(ray, 0.001, 1000.0, hit));
    RENDER_CHECK(!hit.front_face);
    RENDER_CHECK(nearly_equal(static_cast<float>(hit.t), 1.0f));
    RENDER_CHECK(nearly_equal(hit.shading_normal.z(), -1.0));
}

void test_sphere_bounds_include_center_and_radius() {
    renderer::Sphere sphere(renderer::Vec3(1, 2, 3), 2.0, 0);
    const renderer::Bounds3 bounds = sphere.bounds();
    RENDER_CHECK(nearly_equal(bounds.min.x(), -1.0));
    RENDER_CHECK(nearly_equal(bounds.min.y(), 0.0));
    RENDER_CHECK(nearly_equal(bounds.min.z(), 1.0));
    RENDER_CHECK(nearly_equal(bounds.max.x(), 3.0));
    RENDER_CHECK(nearly_equal(bounds.max.y(), 4.0));
    RENDER_CHECK(nearly_equal(bounds.max.z(), 5.0));
}

void test_triangle_intersection() {
    renderer::Triangle tri(
        renderer::Vec3(-1, 0, 0),
        renderer::Vec3(1, 0, 0),
        renderer::Vec3(0, 1, 0),
        2);

    renderer::Ray ray(renderer::Vec3(0, 0.25, -2), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(tri.intersect(ray, 0.001, 1000.0, hit));
    RENDER_CHECK(nearly_equal(hit.position.x(), 0.0));
    RENDER_CHECK(nearly_equal(hit.position.y(), 0.25));
    RENDER_CHECK(hit.material_id == 2);
}

void test_triangle_interpolates_shading_normal_separately_from_geometry() {
    const renderer::Triangle triangle(
        renderer::TriangleVertex{
            renderer::Vec3(-1.0, -1.0, -1.0),
            renderer::Vec2(0.0, 0.0),
            renderer::normalize(renderer::Vec3(0.0, 1.0, 1.0)),
            true},
        renderer::TriangleVertex{
            renderer::Vec3(1.0, -1.0, -1.0),
            renderer::Vec2(1.0, 0.0),
            renderer::normalize(renderer::Vec3(1.0, 0.0, 1.0)),
            true},
        renderer::TriangleVertex{
            renderer::Vec3(0.0, 1.0, -1.0),
            renderer::Vec2(0.5, 1.0),
            renderer::Vec3(0.0, 0.0, 1.0),
            true},
        0);

    renderer::HitRecord hit;
    const renderer::Ray ray(renderer::Vec3(0.0, 0.0, 0.0), renderer::Vec3(0.0, 0.0, -1.0));
    RENDER_CHECK(triangle.intersect(ray, 1e-6, 10.0, hit));
    RENDER_CHECK(renderer::dot(hit.geometric_normal, renderer::Vec3(0.0, 0.0, 1.0)) > 0.999);
    RENDER_CHECK(renderer::dot(hit.shading_normal, hit.geometric_normal) > 0.0);
    RENDER_CHECK(renderer::length(hit.shading_normal - hit.geometric_normal) > 0.01);
}

void test_triangle_invalid_vertices_throw() {
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

void test_triangle_rejects_non_finite_rays() {
    renderer::Triangle tri(
        renderer::Vec3(-1, 0, 0),
        renderer::Vec3(1, 0, 0),
        renderer::Vec3(0, 1, 0),
        2);

    renderer::HitRecord nan_origin_hit;
    renderer::Ray nan_origin_ray(
        renderer::Vec3(std::numeric_limits<float>::quiet_NaN(), 0.25f, -2.0f),
        renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!tri.intersect(nan_origin_ray, 0.001, 1000.0, nan_origin_hit));

    renderer::HitRecord infinite_direction_hit;
    renderer::Ray infinite_direction_ray(
        renderer::Vec3(0, 0.25, -2),
        renderer::Vec3(0, 0, std::numeric_limits<float>::infinity()));
    RENDER_CHECK(!tri.intersect(infinite_direction_ray, 0.001, 1000.0, infinite_direction_hit));
}

void test_triangle_back_side_hit_reports_back_face() {
    renderer::Triangle tri(
        renderer::Vec3(-1, 0, 0),
        renderer::Vec3(1, 0, 0),
        renderer::Vec3(0, 1, 0),
        2);

    renderer::Ray ray(renderer::Vec3(0, 0.25, -2), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(tri.intersect(ray, 0.001, 1000.0, hit));
    RENDER_CHECK(!hit.front_face);
    RENDER_CHECK(renderer::dot(hit.shading_normal, ray.direction) < 0.0);
}

void test_triangle_boundary_hits_succeed() {
    renderer::Triangle tri(
        renderer::Vec3(-1, 0, 0),
        renderer::Vec3(1, 0, 0),
        renderer::Vec3(0, 1, 0),
        2);

    renderer::HitRecord vertex_hit;
    renderer::Ray vertex_ray(renderer::Vec3(-1, 0, -2), renderer::Vec3(0, 0, 1));
    RENDER_CHECK(tri.intersect(vertex_ray, 0.001, 1000.0, vertex_hit));
    RENDER_CHECK(nearly_equal(vertex_hit.position.x(), -1.0));
    RENDER_CHECK(nearly_equal(vertex_hit.position.y(), 0.0));

    renderer::HitRecord edge_hit;
    renderer::Ray edge_ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 0, 1));
    RENDER_CHECK(tri.intersect(edge_ray, 0.001, 1000.0, edge_hit));
    RENDER_CHECK(nearly_equal(edge_hit.position.x(), 0.0));
    RENDER_CHECK(nearly_equal(edge_hit.position.y(), 0.0));
}

void test_degenerate_triangle_misses() {
    renderer::Triangle tri(
        renderer::Vec3(0, 0, 0),
        renderer::Vec3(1, 1, 1),
        renderer::Vec3(2, 2, 2),
        2);

    renderer::Ray ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(!tri.intersect(ray, 0.001, 1000.0, hit));
}

bool brute_force_triangle_intersect(
    const std::vector<renderer::Triangle>& tris,
    const renderer::Ray& ray,
    double t_min,
    double t_max,
    renderer::HitRecord& closest_hit) {
    bool hit_anything = false;
    double closest_t = t_max;
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
    const bool brute_force_found = brute_force_triangle_intersect(tris, ray, 0.001, 1000.0, brute_force_hit);
    const bool bvh_found = bvh.intersect(ray, 0.001, 1000.0, bvh_hit);
    RENDER_CHECK(bvh_found == brute_force_found);
    if (!brute_force_found) {
        return;
    }

    RENDER_CHECK(nearly_equal(static_cast<float>(bvh_hit.t), static_cast<float>(brute_force_hit.t)));
    RENDER_CHECK(bvh_hit.material_id == brute_force_hit.material_id);
    RENDER_CHECK(nearly_equal(bvh_hit.position.x(), brute_force_hit.position.x()));
    RENDER_CHECK(nearly_equal(bvh_hit.position.y(), brute_force_hit.position.y()));
    RENDER_CHECK(nearly_equal(bvh_hit.position.z(), brute_force_hit.position.z()));
    RENDER_CHECK(renderer::dot(bvh_hit.shading_normal, brute_force_hit.shading_normal) > 0.999);
}

void test_empty_bvh_has_no_nodes_or_hits() {
    std::vector<renderer::Triangle> tris;

    renderer::Bvh bvh;
    bvh.build(tris);

    RENDER_CHECK(bvh.nodes().empty());
    RENDER_CHECK(bvh.primitive_indices().empty());

    renderer::HitRecord hit;
    RENDER_CHECK(!bvh.intersect(
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 0, 1)),
        0.001,
        1000.0,
        hit));
}

void test_bvh_matches_bruteforce_triangle_hit() {
    std::vector<renderer::Triangle> tris;
    tris.emplace_back(renderer::Vec3(-1, 0, 0), renderer::Vec3(1, 0, 0), renderer::Vec3(0, 1, 0), 0);
    tris.emplace_back(renderer::Vec3(-1, 0, 5), renderer::Vec3(1, 0, 5), renderer::Vec3(0, 1, 5), 0);

    renderer::Bvh bvh;
    bvh.build(tris);

    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(0, 0.25, -2), renderer::Vec3(0, 0, 1)));
    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(3, 3, -2), renderer::Vec3(0, 0, 1)));
}

void test_bvh_splits_and_traverses_interior_nodes() {
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

    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(0, 0.25, -2), renderer::Vec3(0, 0, 1)));
    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(6, 0.25, -2), renderer::Vec3(0, 0, 1)));
    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(15, 0.25, -2), renderer::Vec3(0, 0, 1)));
    check_bvh_matches_bruteforce(bvh, tris, renderer::Ray(renderer::Vec3(1.5, 0.25, -2), renderer::Vec3(0, 0, 1)));
}

void test_checker_texture_is_deterministic_for_positive_and_negative_coordinates() {
    renderer::CheckerTexture texture;
    texture.even = renderer::Color(1, 0, 0);
    texture.odd = renderer::Color(0, 1, 0);
    texture.scale = 1.0;

    const renderer::Color positive =
        texture.sample(renderer::Vec2::Zero(), renderer::Vec3(0.25, 0.25, 0.25));
    RENDER_CHECK(nearly_equal(positive.x(), 1.0));
    RENDER_CHECK(nearly_equal(positive.y(), 0.0));
    RENDER_CHECK(nearly_equal(positive.z(), 0.0));

    const renderer::Color negative =
        texture.sample(renderer::Vec2::Zero(), renderer::Vec3(-0.25, 0.25, 0.25));
    RENDER_CHECK(nearly_equal(negative.x(), 0.0));
    RENDER_CHECK(nearly_equal(negative.y(), 1.0));
    RENDER_CHECK(nearly_equal(negative.z(), 0.0));
}

void test_builtin_scene_contains_renderable_geometry() {
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
                   material.base_color.x() > 0.5 &&
                   material.base_color.y() > 0.5 &&
                   material.base_color.z() > 0.5;
        });
}

bool has_nonzero_emissive_material(const renderer::Scene& scene) {
    return std::any_of(
        scene.materials.begin(),
        scene.materials.end(),
        [](const renderer::Material& material) {
            return material.type == renderer::MaterialType::Emissive &&
                   renderer::length_squared(material.emission) > 0.0;
        });
}

bool material_id_in_range(const renderer::Scene& scene, int material_id) {
    return material_id >= 0 &&
           static_cast<std::size_t>(material_id) < scene.materials.size();
}

bool intersect_closest_sphere(const renderer::Scene& scene, const renderer::Ray& ray, renderer::HitRecord& closest_hit) {
    bool hit_anything = false;
    double closest_t = 1000.0;
    for (const renderer::Sphere& sphere : scene.spheres) {
        renderer::HitRecord hit;
        if (sphere.intersect(ray, 0.001, closest_t, hit)) {
            hit_anything = true;
            closest_t = hit.t;
            closest_hit = hit;
        }
    }
    return hit_anything;
}

bool intersect_closest_triangle(const renderer::Scene& scene, const renderer::Ray& ray, renderer::HitRecord& closest_hit) {
    bool hit_anything = false;
    double closest_t = 1000.0;
    for (const renderer::Triangle& triangle : scene.triangles) {
        renderer::HitRecord hit;
        if (triangle.intersect(ray, 0.001, closest_t, hit)) {
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

void test_raster_triangle_scene_contains_triangle_and_light() {
    renderer::Scene scene = renderer::make_raster_triangle_scene();
    RENDER_CHECK(!scene.materials.empty());
    RENDER_CHECK(!scene.triangles.empty());
    RENDER_CHECK(!scene.point_lights.empty() || !scene.directional_lights.empty());
}

void test_mirror_spheres_scene_contains_metal_sphere_and_point_light() {
    renderer::Scene scene = renderer::make_mirror_spheres_scene();
    RENDER_CHECK(scene.spheres.size() >= 2);
    RENDER_CHECK(has_material_type(scene, renderer::MaterialType::Metal));
    RENDER_CHECK(!scene.point_lights.empty());
}

void test_cornell_box_scene_contains_walls_and_expected_materials() {
    renderer::Scene scene = renderer::make_cornell_box_scene();
    RENDER_CHECK(scene.triangles.size() >= 12);
    RENDER_CHECK(has_red_like_material(scene));
    RENDER_CHECK(has_green_like_material(scene));
    RENDER_CHECK(has_white_diffuse_material(scene));
    RENDER_CHECK(has_nonzero_emissive_material(scene));
}

void test_builtin_scene_probe_material_ids_are_in_range() {
    renderer::Scene gradient_scene = renderer::make_gradient_sphere_scene();
    check_sphere_hit_material_in_range(
        gradient_scene,
        renderer::Ray(renderer::Vec3(0, 0, 0), renderer::Vec3(0, 0, -1)));

    renderer::Scene raster_scene = renderer::make_raster_triangle_scene();
    check_triangle_hit_material_in_range(
        raster_scene,
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
           material.base_color.x() > 0.5 &&
           material.base_color.y() > 0.5 &&
           material.base_color.z() > 0.5;
}

void test_cornell_box_wall_normals_face_inward() {
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

void test_cornell_box_light_uses_emissive_material_and_faces_downward() {
    renderer::Scene scene = renderer::make_cornell_box_scene();
    renderer::HitRecord hit;
    RENDER_CHECK(intersect_closest_triangle(
        scene,
        renderer::Ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 1, 0)),
        hit));
    RENDER_CHECK(material_id_in_range(scene, hit.material_id));
    RENDER_CHECK(scene.materials[hit.material_id].type == renderer::MaterialType::Emissive);
    RENDER_CHECK(renderer::length_squared(scene.materials[hit.material_id].emission) > 0.0);
    RENDER_CHECK(hit.front_face);
    RENDER_CHECK(hit.shading_normal.y() < -0.999);
}

void test_cosine_sample_is_in_upper_hemisphere() {
    renderer::PcgRandom rng(42);
    for (int i = 0; i < 100; ++i) {
        renderer::Vec3 d = renderer::cosine_weighted_hemisphere(rng);
        RENDER_CHECK(d.z() >= -1e-9);
        RENDER_CHECK(nearly_equal(renderer::length(d), 1.0f, 1e-6f));
    }
}

void test_render_settings_defaults_are_useful() {
    renderer::RenderSettings settings;
    RENDER_CHECK(settings.width == 512);
    RENDER_CHECK(settings.height == 512);
    RENDER_CHECK(settings.samples_per_pixel == 1);
    RENDER_CHECK(settings.max_depth == 5);
    RENDER_CHECK(settings.sample_seed_offset == 0);
}

void test_raytracer_renders_visible_sphere() {
    renderer::Scene scene = renderer::make_gradient_sphere_scene();
    renderer::Camera camera(
        renderer::Vec3(0, 0, 2),
        renderer::Vec3(0, 0, -1),
        renderer::Vec3(0, 1, 0),
        45.0,
        1.0);

    renderer::RenderSettings settings;
    settings.width = 32;
    settings.height = 32;
    settings.max_depth = 3;

    renderer::RayTracerRenderer renderer_instance;
    renderer::RenderResult result = renderer_instance.render(scene, camera, settings);
    renderer::Color center = result.image.pixel(16, 16);
    RENDER_CHECK(center.x() > 0.05 || center.y() > 0.05 || center.z() > 0.05);
}

void test_raytracer_renders_triangle_scene_with_direct_light() {
    renderer::Scene scene = renderer::make_raster_triangle_scene();
    renderer::Camera camera(
        renderer::Vec3(0, 0, 2),
        renderer::Vec3(0, 0, -1),
        renderer::Vec3(0, 1, 0),
        45.0,
        1.0);

    renderer::RenderSettings settings;
    settings.width = 32;
    settings.height = 32;
    settings.max_depth = 2;

    renderer::RayTracerRenderer renderer_instance;
    renderer::RenderResult result = renderer_instance.render(scene, camera, settings);
    renderer::Color center = result.image.pixel(16, 16);
    RENDER_CHECK(center.y() > 0.1);
    RENDER_CHECK(center.z() > 0.1);
}

void test_pathtracer_renders_emissive_scene() {
    renderer::Scene scene = renderer::make_cornell_box_scene();
    renderer::Camera camera(
        renderer::Vec3(0, 1, 4),
        renderer::Vec3(0, 1, 0),
        renderer::Vec3(0, 1, 0),
        40.0,
        1.0);

    renderer::RenderSettings settings;
    settings.width = 16;
    settings.height = 16;
    settings.samples_per_pixel = 2;
    settings.max_depth = 3;
    settings.thread_count = 1;

    renderer::PathTracerRenderer renderer_instance;
    renderer::RenderResult result = renderer_instance.render(scene, camera, settings);

    double luminance_sum = 0.0;
    for (int y = 0; y < result.image.height(); ++y) {
        for (int x = 0; x < result.image.width(); ++x) {
            renderer::Color c = result.image.pixel(x, y);
            luminance_sum += c.x() + c.y() + c.z();
        }
    }
    RENDER_CHECK(luminance_sum > 0.1);
}

renderer::Scene make_path_direct_light_scene() {
    renderer::Scene scene;
    scene.environment = renderer::Color::Zero();
    renderer::Material material;
    material.type = renderer::MaterialType::Diffuse;
    material.base_color = renderer::Color(1.0, 1.0, 1.0);
    scene.materials.push_back(material);
    scene.triangles.emplace_back(
        renderer::Vec3(-10.0, -10.0, -1.0),
        renderer::Vec3(10.0, -10.0, -1.0),
        renderer::Vec3(0.0, 10.0, -1.0),
        0);
    return scene;
}

renderer::Color render_one_path_pixel(const renderer::Scene& scene) {
    const renderer::Camera camera(
        renderer::Vec3(0.0, 0.0, 0.0),
        renderer::Vec3(0.0, 0.0, -1.0),
        renderer::Vec3(0.0, 1.0, 0.0),
        20.0,
        1.0);
    renderer::RenderSettings settings;
    settings.width = 1;
    settings.height = 1;
    settings.samples_per_pixel = 1;
    settings.max_depth = 1;
    settings.thread_count = 1;
    return renderer::PathTracerRenderer().render(scene, camera, settings).image.pixel(0, 0);
}

void test_pathtracer_receives_directional_light() {
    renderer::Scene dark = make_path_direct_light_scene();
    const renderer::Color unlit = render_one_path_pixel(dark);

    renderer::Scene lit = dark;
    lit.directional_lights.push_back(renderer::DirectionalLight{
        renderer::Vec3(0.0, 0.0, -1.0),
        renderer::Color(2.0, 2.0, 2.0)});
    const renderer::Color illuminated = render_one_path_pixel(lit);

    RENDER_CHECK(illuminated.x() > unlit.x() + 0.5);
}

void test_pathtracer_point_light_uses_inverse_square_falloff() {
    renderer::Scene near_scene = make_path_direct_light_scene();
    near_scene.point_lights.push_back(renderer::PointLight{
        renderer::Vec3(0.0, 0.0, 1.0),
        renderer::Color(8.0, 8.0, 8.0)});
    renderer::Scene far_scene = make_path_direct_light_scene();
    far_scene.point_lights.push_back(renderer::PointLight{
        renderer::Vec3(0.0, 0.0, 3.0),
        renderer::Color(8.0, 8.0, 8.0)});

    const renderer::Color near_value = render_one_path_pixel(near_scene);
    const renderer::Color far_value = render_one_path_pixel(far_scene);
    RENDER_CHECK(near_value.x() > far_value.x() * 3.5);
}

void test_pathtracer_direct_light_respects_shadow_blockers() {
    renderer::Scene visible = make_path_direct_light_scene();
    visible.point_lights.push_back(renderer::PointLight{
        renderer::Vec3(2.0, 0.0, 0.0),
        renderer::Color(20.0, 20.0, 20.0)});
    renderer::Scene blocked = visible;
    blocked.triangles.emplace_back(
        renderer::Vec3(1.0, -10.0, -2.0),
        renderer::Vec3(1.0, 10.0, -2.0),
        renderer::Vec3(1.0, 0.0, 1.0),
        0);

    const renderer::Color visible_value = render_one_path_pixel(visible);
    const renderer::Color blocked_value = render_one_path_pixel(blocked);
    RENDER_CHECK(visible_value.x() > blocked_value.x() + 0.05);
}

void test_rasterizer_draws_triangle() {
    renderer::Scene scene = renderer::make_raster_triangle_scene();
    renderer::Camera camera(
        renderer::Vec3(0, 0, 2),
        renderer::Vec3(0, 0, 0),
        renderer::Vec3(0, 1, 0),
        45.0,
        1.0);

    renderer::RenderSettings settings;
    settings.width = 64;
    settings.height = 64;

    renderer::RasterizerRenderer renderer_instance;
    renderer::RenderResult result = renderer_instance.render(scene, camera, settings);

    int lit_pixels = 0;
    for (int y = 0; y < result.image.height(); ++y) {
        for (int x = 0; x < result.image.width(); ++x) {
            renderer::Color c = result.image.pixel(x, y);
            if (c.x() + c.y() + c.z() > 0.05) {
                ++lit_pixels;
            }
        }
    }
    RENDER_CHECK(lit_pixels > 20);
}

int count_lit_pixels(const renderer::Image& image) {
    int lit_pixels = 0;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const renderer::Color color = image.pixel(x, y);
            if (renderer::length_squared(color) > 1e-8) {
                ++lit_pixels;
            }
        }
    }
    return lit_pixels;
}

void test_perspective_correct_weights_favor_near_vertex() {
    const renderer::Vec3 corrected = renderer::perspective_correct_weights(
        renderer::Vec3(1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f),
        renderer::Vec3(1.0f, 2.0f, 4.0f));
    RENDER_CHECK(corrected.x() > corrected.y());
    RENDER_CHECK(corrected.y() > corrected.z());
    RENDER_CHECK(nearly_equal(corrected.x() + corrected.y() + corrected.z(), 1.0f, 1e-6f));
}

void test_near_plane_clipping_keeps_visible_triangle_portion() {
    const std::array<renderer::RasterVertex, 3> vertices{
        renderer::RasterVertex{renderer::Vec3(-1.0f, -1.0f, 1.0f)},
        renderer::RasterVertex{renderer::Vec3(1.0f, -1.0f, 1.0f)},
        renderer::RasterVertex{renderer::Vec3(0.0f, 1.0f, -0.1f)}};
    const std::vector<renderer::RasterVertex> clipped =
        renderer::clip_triangle_to_near_plane(vertices, 1e-4);
    RENDER_CHECK(clipped.size() == 4);
    for (const renderer::RasterVertex& vertex : clipped) {
        RENDER_CHECK(vertex.view.z() >= 1e-4f);
    }
}

renderer::RenderResult render_test_raster_triangle(
    bool two_sided,
    double opacity,
    bool reverse_winding,
    bool crosses_near_plane) {
    renderer::Scene scene;
    scene.environment = renderer::Color::Zero();
    renderer::Material material;
    material.type = renderer::MaterialType::Emissive;
    material.emission = renderer::Color(1.0, 1.0, 1.0);
    material.two_sided = two_sided;
    material.opacity = opacity;
    scene.materials.push_back(material);

    const renderer::Vec3 a(-1.0f, -1.0f, -1.0f);
    const renderer::Vec3 b(1.0f, -1.0f, -1.0f);
    const renderer::Vec3 c(0.0f, 1.0f, crosses_near_plane ? 0.1f : -1.0f);
    if (reverse_winding) {
        scene.triangles.emplace_back(a, c, b, 0);
    } else {
        scene.triangles.emplace_back(a, b, c, 0);
    }

    const renderer::Camera camera(
        renderer::Vec3(0.0, 0.0, 0.0),
        renderer::Vec3(0.0, 0.0, -1.0),
        renderer::Vec3(0.0, 1.0, 0.0),
        45.0,
        1.0);
    renderer::RenderSettings settings;
    settings.width = 32;
    settings.height = 32;
    return renderer::RasterizerRenderer().render(scene, camera, settings);
}

void test_rasterizer_clips_triangles_crossing_near_plane() {
    const renderer::RenderResult result = render_test_raster_triangle(true, 1.0, false, true);
    RENDER_CHECK(count_lit_pixels(result.image) > 0);
}

void test_rasterizer_applies_alpha_cutout_before_depth_write() {
    const renderer::RenderResult result = render_test_raster_triangle(true, 0.0, false, false);
    RENDER_CHECK(count_lit_pixels(result.image) == 0);
}

void test_rasterizer_respects_single_and_two_sided_materials() {
    const renderer::RenderResult single_sided =
        render_test_raster_triangle(false, 1.0, true, false);
    const renderer::RenderResult two_sided =
        render_test_raster_triangle(true, 1.0, true, false);
    RENDER_CHECK(count_lit_pixels(single_sided.image) == 0);
    RENDER_CHECK(count_lit_pixels(two_sided.image) > 0);
}

int count_lit_pixels(const renderer::Framebuffer& framebuffer) {
    int lit_pixels = 0;
    for (int y = 0; y < framebuffer.height(); ++y) {
        for (int x = 0; x < framebuffer.width(); ++x) {
            const renderer::Color c = framebuffer.pixel(x, y);
            if (c.x() + c.y() + c.z() > 0.05) {
                ++lit_pixels;
            }
        }
    }
    return lit_pixels;
}

bool colors_differ(const renderer::Color& a, const renderer::Color& b, double eps = 1e-12) {
    return std::abs(a.x() - b.x()) > eps || std::abs(a.y() - b.y()) > eps ||
        std::abs(a.z() - b.z()) > eps;
}

bool framebuffers_differ(const renderer::Framebuffer& a, const renderer::Framebuffer& b) {
    if (a.width() != b.width() || a.height() != b.height()) {
        return true;
    }

    for (int y = 0; y < a.height(); ++y) {
        for (int x = 0; x < a.width(); ++x) {
            if (colors_differ(a.pixel(x, y), b.pixel(x, y))) {
                return true;
            }
        }
    }
    return false;
}

renderer::Triangle make_test_triangle_at_z(double z, int material_id) {
    const float depth = static_cast<float>(z);
    return renderer::Triangle(
        renderer::Vec3(-1.0f, -1.0f, depth),
        renderer::Vec3(1.0f, -1.0f, depth),
        renderer::Vec3(0.0f, 1.0f, depth),
        material_id);
}

void test_scene_intersector_skips_alpha_cutout_hits() {
    renderer::Scene scene;
    renderer::Material transparent;
    transparent.opacity = 0.0;
    transparent.alpha_cutoff = 0.5;
    scene.materials.push_back(transparent);
    scene.materials.push_back(renderer::Material());
    scene.triangles.push_back(make_test_triangle_at_z(-1.0, 0));
    scene.triangles.push_back(make_test_triangle_at_z(-2.0, 1));

    const renderer::SceneIntersector intersector(scene);
    renderer::HitRecord hit;
    const renderer::Ray ray(renderer::Vec3(0.0, 0.0, 0.0), renderer::Vec3(0.0, 0.0, -1.0));
    RENDER_CHECK(intersector.intersect(ray, 0.0, 100.0, hit));
    RENDER_CHECK(hit.material_id == 1);
    RENDER_CHECK(hit.position.z() < -1.5);
}

void test_scene_intersector_respects_single_and_two_sided_materials() {
    renderer::Scene scene;
    renderer::Material material;
    material.two_sided = false;
    scene.materials.push_back(material);
    scene.triangles.push_back(make_test_triangle_at_z(-1.0, 0));
    const renderer::Ray back_ray(
        renderer::Vec3(0.0, 0.0, -2.0),
        renderer::Vec3(0.0, 0.0, 1.0));

    renderer::HitRecord hit;
    RENDER_CHECK(!renderer::SceneIntersector(scene).intersect(back_ray, 0.0, 100.0, hit));

    scene.materials[0].two_sided = true;
    RENDER_CHECK(renderer::SceneIntersector(scene).intersect(back_ray, 0.0, 100.0, hit));
    RENDER_CHECK(renderer::dot(hit.geometric_normal, back_ray.direction) < 0.0);
    RENDER_CHECK(renderer::dot(hit.shading_normal, hit.geometric_normal) > 0.0);
}

void test_offset_ray_origin_moves_to_outgoing_side() {
    const renderer::Vec3 position(1000.0, 0.0, 0.0);
    const renderer::Vec3 normal(1.0, 0.0, 0.0);
    RENDER_CHECK(renderer::offset_ray_origin(position, normal, normal).x() > position.x());
    RENDER_CHECK(renderer::offset_ray_origin(position, normal, -normal).x() < position.x());
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
    scene.environment = renderer::Color(0.0, 0.0, 0.0);

    renderer::Material light;
    light.type = renderer::MaterialType::Emissive;
    light.base_color = renderer::Color(1.0, 1.0, 1.0);
    light.emission = renderer::Color(6.0, 6.0, 6.0);
    scene.materials.push_back(light);
    scene.spheres.emplace_back(renderer::Vec3(0.0, 0.0, -1.0), 0.55, 0);

    return scene;
}

void test_interactive_sessions_render_visible_pixels() {
    renderer::RenderSettings settings;
    settings.width = 32;
    settings.height = 32;
    settings.max_depth = 2;
    renderer::Framebuffer framebuffer(32, 32);
    renderer::InteractiveFrameState frame_state;

    renderer::RasterInteractiveSession raster;
    renderer::Scene raster_scene = renderer::make_raster_triangle_scene();
    renderer::Camera raster_camera(
        renderer::Vec3(0, 0, 2),
        renderer::Vec3(0, 0, 0),
        renderer::Vec3(0, 1, 0),
        45.0,
        1.0);
    raster.reset(raster_scene, settings);
    raster.render_next_frame(raster_scene, raster_camera, settings, frame_state, framebuffer);
    RENDER_CHECK(count_lit_pixels(framebuffer) > 0);

    renderer::RayInteractiveSession ray;
    renderer::Scene ray_scene = renderer::make_raster_triangle_scene();
    renderer::Camera ray_camera(
        renderer::Vec3(0, 0, 2),
        renderer::Vec3(0, 0, -1),
        renderer::Vec3(0, 1, 0),
        45.0,
        1.0);
    ray.reset(ray_scene, settings);
    ray.render_next_frame(ray_scene, ray_camera, settings, frame_state, framebuffer);
    RENDER_CHECK(count_lit_pixels(framebuffer) > 0);
}

void test_path_interactive_session_accumulates_and_resets() {
    renderer::Scene scene = renderer::make_cornell_box_scene();
    renderer::Camera camera(
        renderer::Vec3(0.0f, 0.15f, 1.5f),
        renderer::Vec3(0.0f, 0.15f, -2.0f),
        renderer::Vec3(0.0f, 1.0f, 0.0f),
        45.0,
        1.0);
    renderer::RenderSettings settings;
    settings.width = 8;
    settings.height = 8;
    settings.samples_per_pixel = 1;
    settings.max_depth = 3;
    settings.thread_count = 1;
    renderer::Framebuffer framebuffer(8, 8);
    renderer::InteractiveFrameState frame_state;

    renderer::PathInteractiveSession path;
    path.reset(scene, settings);
    path.render_next_frame(scene, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(path.accumulated_samples() == 1);

    path.render_next_frame(scene, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(path.accumulated_samples() == 2);

    frame_state.camera_changed = true;
    path.render_next_frame(scene, camera, settings, frame_state, framebuffer);
    RENDER_CHECK(path.accumulated_samples() == 1);
}

void test_path_interactive_session_accumulates_distinct_samples() {
    renderer::Scene scene = make_emissive_silhouette_scene();
    renderer::Camera camera(
        renderer::Vec3(0.0, 0.0, 2.0),
        renderer::Vec3(0.0, 0.0, -1.0),
        renderer::Vec3(0.0, 1.0, 0.0),
        45.0,
        1.0);
    renderer::RenderSettings settings;
    settings.width = 32;
    settings.height = 32;
    settings.samples_per_pixel = 1;
    settings.max_depth = 1;
    settings.thread_count = 1;
    renderer::Framebuffer framebuffer(32, 32);
    renderer::InteractiveFrameState frame_state;

    renderer::PathInteractiveSession path;
    path.reset(scene, settings);
    path.render_next_frame(scene, camera, settings, frame_state, framebuffer);
    const renderer::Framebuffer first_frame = framebuffer;

    path.render_next_frame(scene, camera, settings, frame_state, framebuffer);

    RENDER_CHECK(path.accumulated_samples() == 2);
    RENDER_CHECK(framebuffers_differ(first_frame, framebuffer));
}

void test_obj_loader_reads_single_triangle() {
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

void test_scene_asset_loader_preserves_obj_vertex_normals() {
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
    const renderer::Ray ray(renderer::Vec3(0.0, 0.0, 0.0), renderer::Vec3(0.0, 0.0, -1.0));
    RENDER_CHECK(loaded.scene.triangles[0].intersect(ray, 1e-6, 10.0, hit));
    RENDER_CHECK(renderer::length(hit.shading_normal - hit.geometric_normal) > 0.01);
    RENDER_CHECK(renderer::dot(hit.shading_normal, hit.geometric_normal) > 0.0);

    std::remove(obj_path.c_str());
}

void test_image_texture_samples_obj_uv_space() {
    const std::string texture_path = "test_map_kd.ppm";
    write_test_ppm_texture(texture_path);

    const renderer::ImageTexture texture = renderer::ImageTexture::load(texture_path);

    const renderer::Color top_left = texture.sample(renderer::Vec2(0.25, 0.75));
    const renderer::Color bottom_left = texture.sample(renderer::Vec2(0.25, 0.25));
    RENDER_CHECK(top_left.x() > 0.9);
    RENDER_CHECK(top_left.y() < 0.1);
    RENDER_CHECK(bottom_left.z() > 0.9);

    std::remove(texture_path.c_str());
}

void test_texture_encoding_distinguishes_srgb_from_linear() {
    const std::string texture_path = "test_texture_encoding.ppm";
    write_single_pixel_ppm(texture_path, 128);

    const renderer::ImageTexture srgb = renderer::ImageTexture::load(
        texture_path,
        renderer::TextureEncoding::Srgb);
    const renderer::ImageTexture linear = renderer::ImageTexture::load(
        texture_path,
        renderer::TextureEncoding::Linear);

    RENDER_CHECK(srgb.sample(renderer::Vec2::Zero()).x() < 0.25);
    RENDER_CHECK(linear.sample(renderer::Vec2::Zero()).x() > 0.49);
    RENDER_CHECK(linear.sample(renderer::Vec2::Zero()).x() < 0.51);

    std::remove(texture_path.c_str());
}

void test_material_evaluator_combines_opacity_and_perturbs_bump_normal() {
    renderer::Scene scene;
    scene.textures.emplace_back(
        4,
        2,
        std::vector<renderer::Color>{
            renderer::Color(0.0, 0.0, 0.0),
            renderer::Color(0.25, 0.25, 0.25),
            renderer::Color(0.5, 0.5, 0.5),
            renderer::Color(1.0, 1.0, 1.0),
            renderer::Color(0.0, 0.0, 0.0),
            renderer::Color(0.25, 0.25, 0.25),
            renderer::Color(0.5, 0.5, 0.5),
            renderer::Color(1.0, 1.0, 1.0)});

    renderer::Material material;
    material.opacity = 0.8;
    material.opacity_texture_id = 0;
    material.bump_texture_id = 0;
    material.bump_scale = 1.0;

    renderer::HitRecord hit;
    hit.uv = renderer::Vec2(0.375, 0.25);
    hit.geometric_normal = renderer::Vec3(0.0, 0.0, 1.0);
    hit.shading_normal = renderer::Vec3(0.0, 0.0, 1.0);
    hit.tangent = renderer::Vec3(1.0, 0.0, 0.0);
    hit.bitangent = renderer::Vec3(0.0, 1.0, 0.0);
    hit.has_valid_uv_basis = true;

    const renderer::SurfaceMaterialSample sample =
        renderer::evaluate_surface_material(scene, material, hit);
    RENDER_CHECK(sample.opacity < material.opacity);
    RENDER_CHECK(renderer::length(sample.shading_normal - hit.shading_normal) > 0.01);
    RENDER_CHECK(renderer::dot(sample.shading_normal, hit.geometric_normal) > 0.0);
    RENDER_CHECK(renderer::dot(sample.shading_normal, hit.shading_normal) > 0.9);

    hit.has_valid_uv_basis = false;
    const renderer::SurfaceMaterialSample fallback =
        renderer::evaluate_surface_material(scene, material, hit);
    RENDER_CHECK(renderer::length(fallback.shading_normal - hit.shading_normal) < 1e-12);
}

void test_scene_asset_loader_loads_map_kd_and_triangle_uvs() {
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
        renderer::Ray(renderer::Vec3(-0.5, 0.5, 0.0), renderer::Vec3(0.0, 0.0, -1.0)),
        0.001,
        10.0,
        hit);
    RENDER_CHECK(did_hit);
    RENDER_CHECK(nearly_equal(hit.uv.x(), 0.25));
    RENDER_CHECK(nearly_equal(hit.uv.y(), 0.75));

    const renderer::Color textured_color = renderer::sample_material_base_color(
        loaded.scene,
        loaded.scene.materials[0],
        hit.uv);
    RENDER_CHECK(textured_color.x() > 0.9);
    RENDER_CHECK(textured_color.y() < 0.1);
    RENDER_CHECK(textured_color.z() < 0.1);

    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
    std::remove(texture_path.c_str());
}

void test_scene_asset_loader_imports_alpha_and_bump_maps() {
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
            .sample_scalar(renderer::Vec2::Zero()) > 0.49);

    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
    std::remove(texture_path.c_str());
}

void test_scene_asset_loader_warns_for_missing_optional_maps() {
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

void test_scene_asset_loader_preserves_obj_mtl_materials() {
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
    RENDER_CHECK(has_red_like_material(loaded.scene));
    RENDER_CHECK(has_nonzero_emissive_material(loaded.scene));
    RENDER_CHECK(has_material_type(loaded.scene, renderer::MaterialType::Metal));
    RENDER_CHECK(has_material_type(loaded.scene, renderer::MaterialType::Dielectric));
    RENDER_CHECK(loaded.camera.viewport_width() > 0.0);

    std::remove(obj_path.c_str());
    std::remove(mtl_path.c_str());
}

int main() {
    RENDER_CHECK(1 + 1 == 2);
    test_vec3_arithmetic();
    test_mat4_composition_order();
    test_mat4_perspective_uses_degrees_and_ndc_depth();
    test_mat4_perspective_invalid_inputs_throw();
    test_mat4_look_at();
    test_mat4_look_at_invalid_inputs_throw();
    test_camera_center_ray_points_forward();
    test_camera_rejects_non_finite_screen_coordinates();
    test_orbit_camera_controller_zoom_and_orbit_change_camera();
    test_orbit_camera_controller_horizontal_drag_tracks_scene_direction();
    test_frame_rate_counter_reports_window_average();
    test_viewer_title_format_includes_fps_and_path_samples();
    test_ray_and_bounds_intersection();
    test_bounds_intersection_counts_corner_touch_as_hit();
    test_image_invalid_dimensions_throw_invalid_argument();
    test_image_stores_gamma_corrected_pixels();
    test_to_rgb8_uses_standard_srgb_transfer_curve();
    test_to_rgb8_sanitizes_non_finite_channels();
    test_framebuffer_clear_set_and_rgba8_conversion();
    test_depth_buffer_clear_resize_and_access();
    test_sphere_intersection();
    test_sphere_rejects_zero_direction_ray();
    test_sphere_invalid_radius_throws();
    test_sphere_invalid_center_throws();
    test_sphere_rejects_non_finite_direction_rays();
    test_sphere_rejects_non_finite_origin_rays();
    test_sphere_inside_ray_reports_back_face();
    test_sphere_bounds_include_center_and_radius();
    test_triangle_intersection();
    test_triangle_interpolates_shading_normal_separately_from_geometry();
    test_triangle_invalid_vertices_throw();
    test_triangle_rejects_non_finite_rays();
    test_triangle_back_side_hit_reports_back_face();
    test_triangle_boundary_hits_succeed();
    test_degenerate_triangle_misses();
    test_empty_bvh_has_no_nodes_or_hits();
    test_bvh_matches_bruteforce_triangle_hit();
    test_bvh_splits_and_traverses_interior_nodes();
    test_scene_intersector_skips_alpha_cutout_hits();
    test_scene_intersector_respects_single_and_two_sided_materials();
    test_offset_ray_origin_moves_to_outgoing_side();
    test_checker_texture_is_deterministic_for_positive_and_negative_coordinates();
    test_builtin_scene_contains_renderable_geometry();
    test_raster_triangle_scene_contains_triangle_and_light();
    test_mirror_spheres_scene_contains_metal_sphere_and_point_light();
    test_cornell_box_scene_contains_walls_and_expected_materials();
    test_builtin_scene_probe_material_ids_are_in_range();
    test_cornell_box_wall_normals_face_inward();
    test_cornell_box_light_uses_emissive_material_and_faces_downward();
    test_cosine_sample_is_in_upper_hemisphere();
    test_render_settings_defaults_are_useful();
    test_raytracer_renders_visible_sphere();
    test_raytracer_renders_triangle_scene_with_direct_light();
    test_pathtracer_renders_emissive_scene();
    test_pathtracer_receives_directional_light();
    test_pathtracer_point_light_uses_inverse_square_falloff();
    test_pathtracer_direct_light_respects_shadow_blockers();
    test_rasterizer_draws_triangle();
    test_perspective_correct_weights_favor_near_vertex();
    test_near_plane_clipping_keeps_visible_triangle_portion();
    test_rasterizer_clips_triangles_crossing_near_plane();
    test_rasterizer_applies_alpha_cutout_before_depth_write();
    test_rasterizer_respects_single_and_two_sided_materials();
    test_interactive_sessions_render_visible_pixels();
    test_path_interactive_session_accumulates_and_resets();
    test_path_interactive_session_accumulates_distinct_samples();
    test_obj_loader_reads_single_triangle();
    test_scene_asset_loader_preserves_obj_vertex_normals();
    test_image_texture_samples_obj_uv_space();
    test_texture_encoding_distinguishes_srgb_from_linear();
    test_material_evaluator_combines_opacity_and_perturbs_bump_normal();
    test_scene_asset_loader_loads_map_kd_and_triangle_uvs();
    test_scene_asset_loader_imports_alpha_and_bump_maps();
    test_scene_asset_loader_warns_for_missing_optional_maps();
    test_scene_asset_loader_preserves_obj_mtl_materials();
    std::cout << "renderer_tests: all tests passed\n";
    return 0;
}
