#pragma once

#include "core/math/types.h"

#include <cmath>
#include <stdexcept>

namespace renderer {

inline Mat4 make_perspective_matrix(
    float vertical_fov_degrees,
    float aspect,
    float near_z,
    float far_z) {
    if (!std::isfinite(vertical_fov_degrees) || !std::isfinite(aspect) ||
        !std::isfinite(near_z) || !std::isfinite(far_z)) {
        throw std::invalid_argument("perspective requires finite inputs");
    }
    if (vertical_fov_degrees <= 0.0f || vertical_fov_degrees >= 180.0f) {
        throw std::invalid_argument("perspective requires vertical_fov_degrees in the range (0, 180)");
    }
    if (aspect <= 0.0f) {
        throw std::invalid_argument("perspective requires a positive aspect ratio");
    }
    if (near_z <= 0.0f) {
        throw std::invalid_argument("perspective requires a positive near plane");
    }
    if (far_z <= near_z) {
        throw std::invalid_argument("perspective requires far_z to be greater than near_z");
    }

    constexpr float pi = 3.14159265358979323846f;
    const float fovy_radians = vertical_fov_degrees * pi / 180.0f;
    const float f = 1.0f / std::tan(fovy_radians * 0.5f);

    Mat4 result = Mat4::Zero();
    result(0, 0) = f / aspect;
    result(1, 1) = f;
    result(2, 2) = -(far_z + near_z) / (far_z - near_z);
    result(2, 3) = -(2.0f * far_z * near_z) / (far_z - near_z);
    result(3, 2) = -1.0f;
    return result;
}

inline Mat4 make_look_at_matrix(const Vec3& eye, const Vec3& target, const Vec3& up) {
    constexpr float epsilon = 1e-12f;
    if (!eye.allFinite() || !target.allFinite() || !up.allFinite()) {
        throw std::invalid_argument("look_at requires finite eye, target, and up vectors");
    }

    const Vec3 view_direction = target - eye;
    const float view_direction_length_squared = view_direction.squaredNorm();
    if (!std::isfinite(view_direction_length_squared)) {
        throw std::invalid_argument("look_at requires a finite view direction");
    }
    if (view_direction_length_squared <= epsilon) {
        throw std::invalid_argument("look_at requires eye and target to be different");
    }

    const float up_length_squared = up.squaredNorm();
    if (!std::isfinite(up_length_squared)) {
        throw std::invalid_argument("look_at requires a finite up vector");
    }
    if (up_length_squared <= epsilon) {
        throw std::invalid_argument("look_at requires a non-zero up vector");
    }

    const Vec3 forward = -view_direction.normalized();
    const Vec3 normalized_up = up.normalized();
    const Vec3 right_candidate = normalized_up.cross(forward);
    const float right_candidate_length_squared = right_candidate.squaredNorm();
    if (!std::isfinite(right_candidate_length_squared) || right_candidate_length_squared <= epsilon) {
        throw std::invalid_argument("look_at up vector must not be parallel to the view direction");
    }

    const Vec3 right = right_candidate.normalized();
    const Vec3 camera_up = forward.cross(right);

    Mat4 result = Mat4::Identity();
    result(0, 0) = right.x();
    result(0, 1) = right.y();
    result(0, 2) = right.z();
    result(0, 3) = -right.dot(eye);
    result(1, 0) = camera_up.x();
    result(1, 1) = camera_up.y();
    result(1, 2) = camera_up.z();
    result(1, 3) = -camera_up.dot(eye);
    result(2, 0) = forward.x();
    result(2, 1) = forward.y();
    result(2, 2) = forward.z();
    result(2, 3) = -forward.dot(eye);
    return result;
}

}  // namespace renderer
