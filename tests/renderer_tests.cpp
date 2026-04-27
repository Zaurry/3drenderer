#include "test_framework.h"

#include "core/color.h"
#include "core/image.h"
#include "core/math/bounds.h"
#include "core/math/mat4.h"
#include "core/math/ray.h"
#include "core/math/vec2.h"
#include "core/math/vec3.h"
#include "core/math/vec4.h"

#include <iostream>
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

void test_image_stores_gamma_corrected_pixels() {
    renderer::Image image(2, 1);
    image.set_pixel(0, 0, renderer::Color(1.0, 0.25, 0.0));
    renderer::Rgb8 pixel = image.pixel_rgb8(0, 0);
    RENDER_CHECK(pixel.r == 255);
    RENDER_CHECK(pixel.g >= 126 && pixel.g <= 128);
    RENDER_CHECK(pixel.b == 0);
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
    test_image_stores_gamma_corrected_pixels();
    std::cout << "renderer_tests: all tests passed\n";
    return 0;
}
