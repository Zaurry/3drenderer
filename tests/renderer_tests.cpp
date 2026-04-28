#include "test_framework.h"

#include "core/color.h"
#include "core/image.h"
#include "core/math/bounds.h"
#include "core/math/mat4.h"
#include "core/math/ray.h"
#include "core/math/vec2.h"
#include "core/math/vec3.h"
#include "core/math/vec4.h"
#include "scene/material.h"
#include "scene/primitive.h"
#include "scene/texture.h"

#include <iostream>
#include <limits>
#include <stdexcept>

void test_vec3_arithmetic() {
    renderer::Vec3 a(1.0, 2.0, 3.0);
    renderer::Vec3 b(4.0, -2.0, 0.5);

    renderer::Vec3 sum = a + b;
    RENDER_CHECK(nearly_equal(sum.x, 5.0));
    RENDER_CHECK(nearly_equal(sum.y, 0.0));
    RENDER_CHECK(nearly_equal(sum.z, 3.5));

    renderer::Vec3 difference = a - b;
    RENDER_CHECK(nearly_equal(difference.x, -3.0));
    RENDER_CHECK(nearly_equal(difference.y, 4.0));
    RENDER_CHECK(nearly_equal(difference.z, 2.5));

    renderer::Vec3 scaled_right = a * 2.0;
    RENDER_CHECK(nearly_equal(scaled_right.x, 2.0));
    RENDER_CHECK(nearly_equal(scaled_right.y, 4.0));
    RENDER_CHECK(nearly_equal(scaled_right.z, 6.0));

    renderer::Vec3 scaled_left = 0.5 * b;
    RENDER_CHECK(nearly_equal(scaled_left.x, 2.0));
    RENDER_CHECK(nearly_equal(scaled_left.y, -1.0));
    RENDER_CHECK(nearly_equal(scaled_left.z, 0.25));

    renderer::Vec3 divided = a / 2.0;
    RENDER_CHECK(nearly_equal(divided.x, 0.5));
    RENDER_CHECK(nearly_equal(divided.y, 1.0));
    RENDER_CHECK(nearly_equal(divided.z, 1.5));

    renderer::Vec3 negated = -a;
    RENDER_CHECK(nearly_equal(negated.x, -1.0));
    RENDER_CHECK(nearly_equal(negated.y, -2.0));
    RENDER_CHECK(nearly_equal(negated.z, -3.0));

    renderer::Vec3 min_v = renderer::min_components(a, b);
    RENDER_CHECK(nearly_equal(min_v.x, 1.0));
    RENDER_CHECK(nearly_equal(min_v.y, -2.0));
    RENDER_CHECK(nearly_equal(min_v.z, 0.5));

    renderer::Vec3 max_v = renderer::max_components(a, b);
    RENDER_CHECK(nearly_equal(max_v.x, 4.0));
    RENDER_CHECK(nearly_equal(max_v.y, 2.0));
    RENDER_CHECK(nearly_equal(max_v.z, 3.0));

    RENDER_CHECK(nearly_equal(renderer::dot(a, b), 1.0 * 4.0 + 2.0 * -2.0 + 3.0 * 0.5));

    renderer::Vec3 c = renderer::cross(renderer::Vec3(1, 0, 0), renderer::Vec3(0, 1, 0));
    RENDER_CHECK(nearly_equal(c.x, 0.0));
    RENDER_CHECK(nearly_equal(c.y, 0.0));
    RENDER_CHECK(nearly_equal(c.z, 1.0));

    renderer::Vec3 n = renderer::normalize(renderer::Vec3(0, 3, 4));
    RENDER_CHECK(nearly_equal(renderer::length(n), 1.0));
    RENDER_CHECK(nearly_equal(n.y, 0.6));
    RENDER_CHECK(nearly_equal(n.z, 0.8));
}

void test_mat4_translation_and_perspective_divide() {
    renderer::Mat4 t = renderer::Mat4::translation(renderer::Vec3(2, 3, 4));
    renderer::Vec4 p = t * renderer::Vec4(1, 1, 1, 1);
    RENDER_CHECK(nearly_equal(p.x, 3.0));
    RENDER_CHECK(nearly_equal(p.y, 4.0));
    RENDER_CHECK(nearly_equal(p.z, 5.0));
    RENDER_CHECK(nearly_equal(p.w, 1.0));
}

void test_mat4_composition_order() {
    renderer::Mat4 transform =
        renderer::Mat4::translation(renderer::Vec3(1, 2, 3)) *
        renderer::Mat4::scale(renderer::Vec3(2, 3, 4));
    renderer::Vec4 p = transform * renderer::Vec4(1, 1, 1, 1);
    RENDER_CHECK(nearly_equal(p.x, 3.0));
    RENDER_CHECK(nearly_equal(p.y, 5.0));
    RENDER_CHECK(nearly_equal(p.z, 7.0));
    RENDER_CHECK(nearly_equal(p.w, 1.0));
}

void test_mat4_perspective_uses_degrees_and_ndc_depth() {
    renderer::Mat4 p = renderer::Mat4::perspective(90.0, 1.0, 1.0, 10.0);
    RENDER_CHECK(nearly_equal(p.m[1][1], 1.0));

    renderer::Vec4 near_clip = p * renderer::Vec4(0, 0, -1, 1);
    RENDER_CHECK(nearly_equal(near_clip.z / near_clip.w, -1.0));

    renderer::Vec4 far_clip = p * renderer::Vec4(0, 0, -10, 1);
    RENDER_CHECK(nearly_equal(far_clip.z / far_clip.w, 1.0));
}

void check_perspective_invalid_input_throws(
    double vertical_fov_degrees,
    double aspect,
    double near_z,
    double far_z) {
    bool threw = false;
    try {
        renderer::Mat4::perspective(vertical_fov_degrees, aspect, near_z, far_z);
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
}

void test_mat4_look_at() {
    renderer::Mat4 view = renderer::Mat4::look_at(
        renderer::Vec3(0, 0, 0),
        renderer::Vec3(0, 0, -1),
        renderer::Vec3(0, 1, 0));
    renderer::Vec4 p = view * renderer::Vec4(0, 0, -1, 1);
    RENDER_CHECK(nearly_equal(p.z, -1.0));
}

void test_mat4_look_at_invalid_inputs_throw() {
    bool threw_eye_equals_target = false;
    try {
        renderer::Mat4::look_at(
            renderer::Vec3(0, 0, 0),
            renderer::Vec3(0, 0, 0),
            renderer::Vec3(0, 1, 0));
    } catch (const std::invalid_argument&) {
        threw_eye_equals_target = true;
    }
    RENDER_CHECK(threw_eye_equals_target);

    bool threw_zero_up = false;
    try {
        renderer::Mat4::look_at(
            renderer::Vec3(0, 0, 0),
            renderer::Vec3(0, 0, -1),
            renderer::Vec3(0, 0, 0));
    } catch (const std::invalid_argument&) {
        threw_zero_up = true;
    }
    RENDER_CHECK(threw_zero_up);

    bool threw_parallel_up = false;
    try {
        renderer::Mat4::look_at(
            renderer::Vec3(0, 0, 0),
            renderer::Vec3(0, 0, -1),
            renderer::Vec3(0, 0, 1));
    } catch (const std::invalid_argument&) {
        threw_parallel_up = true;
    }
    RENDER_CHECK(threw_parallel_up);
}

void test_ray_and_bounds_intersection() {
    renderer::Ray ray(renderer::Vec3(0, 0, -5), renderer::Vec3(0, 0, 1));
    renderer::Bounds3 box(renderer::Vec3(-1, -1, -1), renderer::Vec3(1, 1, 1));
    RENDER_CHECK(box.intersect(ray, 0.001, 1000.0));

    renderer::Ray miss(renderer::Vec3(5, 5, -5), renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!box.intersect(miss, 0.001, 1000.0));
}

void test_bounds_intersection_counts_corner_touch_as_hit() {
    renderer::Bounds3 box(renderer::Vec3(-1, -1, -1), renderer::Vec3(1, 1, 1));
    renderer::Ray corner_touch(renderer::Vec3(-2, -2, 1), renderer::Vec3(1, 1, 0));
    RENDER_CHECK(box.intersect(corner_touch, 0.001, 1000.0));
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

void test_to_rgb8_sanitizes_non_finite_channels() {
    const renderer::Rgb8 nan_pixel = renderer::to_rgb8(
        renderer::Color(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0));
    RENDER_CHECK(nan_pixel.r == 0);

    const renderer::Rgb8 infinity_pixel = renderer::to_rgb8(
        renderer::Color(std::numeric_limits<double>::infinity(), 0.0, 0.0));
    RENDER_CHECK(infinity_pixel.r == 255);
}

void test_sphere_intersection() {
    renderer::Material material;
    material.base_color = renderer::Color(1, 0, 0);
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0, 0);

    renderer::Ray ray(renderer::Vec3(0, 0, -5), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(sphere.intersect(ray, 0.001, 1000.0, hit));
    RENDER_CHECK(nearly_equal(hit.t, 4.0));
    RENDER_CHECK(nearly_equal(hit.position.z, -1.0));
    RENDER_CHECK(nearly_equal(renderer::length(hit.normal), 1.0));
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
            renderer::Vec3(std::numeric_limits<double>::quiet_NaN(), 0, 0),
            1.0,
            0);
    } catch (const std::invalid_argument&) {
        threw_nan_center = true;
    }
    RENDER_CHECK(threw_nan_center);

    bool threw_infinite_center = false;
    try {
        renderer::Sphere sphere(
            renderer::Vec3(std::numeric_limits<double>::infinity(), 0, 0),
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
        renderer::Vec3(0, 0, std::numeric_limits<double>::quiet_NaN()));
    RENDER_CHECK(!sphere.intersect(nan_ray, 0.001, 1000.0, nan_hit));

    renderer::HitRecord infinite_hit;
    renderer::Ray infinite_ray(
        renderer::Vec3(0, 0, -5),
        renderer::Vec3(0, 0, std::numeric_limits<double>::infinity()));
    RENDER_CHECK(!sphere.intersect(infinite_ray, 0.001, 1000.0, infinite_hit));
}

void test_sphere_rejects_non_finite_origin_rays() {
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0, 0);

    renderer::HitRecord nan_hit;
    renderer::Ray nan_ray(
        renderer::Vec3(std::numeric_limits<double>::quiet_NaN(), 0, -5),
        renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!sphere.intersect(nan_ray, 0.001, 1000.0, nan_hit));

    renderer::HitRecord infinite_hit;
    renderer::Ray infinite_ray(
        renderer::Vec3(std::numeric_limits<double>::infinity(), 0, -5),
        renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!sphere.intersect(infinite_ray, 0.001, 1000.0, infinite_hit));
}

void test_sphere_inside_ray_reports_back_face() {
    renderer::Sphere sphere(renderer::Vec3(0, 0, 0), 1.0, 1);
    renderer::Ray ray(renderer::Vec3(0, 0, 0), renderer::Vec3(0, 0, 1));
    renderer::HitRecord hit;
    RENDER_CHECK(sphere.intersect(ray, 0.001, 1000.0, hit));
    RENDER_CHECK(!hit.front_face);
    RENDER_CHECK(nearly_equal(hit.t, 1.0));
    RENDER_CHECK(nearly_equal(hit.normal.z, -1.0));
}

void test_sphere_bounds_include_center_and_radius() {
    renderer::Sphere sphere(renderer::Vec3(1, 2, 3), 2.0, 0);
    const renderer::Bounds3 bounds = sphere.bounds();
    RENDER_CHECK(nearly_equal(bounds.min.x, -1.0));
    RENDER_CHECK(nearly_equal(bounds.min.y, 0.0));
    RENDER_CHECK(nearly_equal(bounds.min.z, 1.0));
    RENDER_CHECK(nearly_equal(bounds.max.x, 3.0));
    RENDER_CHECK(nearly_equal(bounds.max.y, 4.0));
    RENDER_CHECK(nearly_equal(bounds.max.z, 5.0));
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
    RENDER_CHECK(nearly_equal(hit.position.x, 0.0));
    RENDER_CHECK(nearly_equal(hit.position.y, 0.25));
    RENDER_CHECK(hit.material_id == 2);
}

void test_triangle_invalid_vertices_throw() {
    bool threw_nan_vertex = false;
    try {
        renderer::Triangle tri(
            renderer::Vec3(std::numeric_limits<double>::quiet_NaN(), 0, 0),
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
            renderer::Vec3(std::numeric_limits<double>::infinity(), 0, 0),
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
        renderer::Vec3(std::numeric_limits<double>::quiet_NaN(), 0.25, -2),
        renderer::Vec3(0, 0, 1));
    RENDER_CHECK(!tri.intersect(nan_origin_ray, 0.001, 1000.0, nan_origin_hit));

    renderer::HitRecord infinite_direction_hit;
    renderer::Ray infinite_direction_ray(
        renderer::Vec3(0, 0.25, -2),
        renderer::Vec3(0, 0, std::numeric_limits<double>::infinity()));
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
    RENDER_CHECK(renderer::dot(hit.normal, ray.direction) < 0.0);
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
    RENDER_CHECK(nearly_equal(vertex_hit.position.x, -1.0));
    RENDER_CHECK(nearly_equal(vertex_hit.position.y, 0.0));

    renderer::HitRecord edge_hit;
    renderer::Ray edge_ray(renderer::Vec3(0, 0, -2), renderer::Vec3(0, 0, 1));
    RENDER_CHECK(tri.intersect(edge_ray, 0.001, 1000.0, edge_hit));
    RENDER_CHECK(nearly_equal(edge_hit.position.x, 0.0));
    RENDER_CHECK(nearly_equal(edge_hit.position.y, 0.0));
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

void test_checker_texture_is_deterministic_for_positive_and_negative_coordinates() {
    renderer::CheckerTexture texture;
    texture.even = renderer::Color(1, 0, 0);
    texture.odd = renderer::Color(0, 1, 0);
    texture.scale = 1.0;

    const renderer::Color positive = texture.sample(renderer::Vec2(), renderer::Vec3(0.25, 0.25, 0.25));
    RENDER_CHECK(nearly_equal(positive.x, 1.0));
    RENDER_CHECK(nearly_equal(positive.y, 0.0));
    RENDER_CHECK(nearly_equal(positive.z, 0.0));

    const renderer::Color negative = texture.sample(renderer::Vec2(), renderer::Vec3(-0.25, 0.25, 0.25));
    RENDER_CHECK(nearly_equal(negative.x, 0.0));
    RENDER_CHECK(nearly_equal(negative.y, 1.0));
    RENDER_CHECK(nearly_equal(negative.z, 0.0));
}

int main() {
    RENDER_CHECK(1 + 1 == 2);
    test_vec3_arithmetic();
    test_mat4_translation_and_perspective_divide();
    test_mat4_composition_order();
    test_mat4_perspective_uses_degrees_and_ndc_depth();
    test_mat4_perspective_invalid_inputs_throw();
    test_mat4_look_at();
    test_mat4_look_at_invalid_inputs_throw();
    test_ray_and_bounds_intersection();
    test_bounds_intersection_counts_corner_touch_as_hit();
    test_image_invalid_dimensions_throw_invalid_argument();
    test_image_stores_gamma_corrected_pixels();
    test_to_rgb8_sanitizes_non_finite_channels();
    test_sphere_intersection();
    test_sphere_rejects_zero_direction_ray();
    test_sphere_invalid_radius_throws();
    test_sphere_invalid_center_throws();
    test_sphere_rejects_non_finite_direction_rays();
    test_sphere_rejects_non_finite_origin_rays();
    test_sphere_inside_ray_reports_back_face();
    test_sphere_bounds_include_center_and_radius();
    test_triangle_intersection();
    test_triangle_invalid_vertices_throw();
    test_triangle_rejects_non_finite_rays();
    test_triangle_back_side_hit_reports_back_face();
    test_triangle_boundary_hits_succeed();
    test_degenerate_triangle_misses();
    test_checker_texture_is_deterministic_for_positive_and_negative_coordinates();
    std::cout << "renderer_tests: all tests passed\n";
    return 0;
}
