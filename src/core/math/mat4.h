#pragma once

#include "core/math/vec3.h"
#include "core/math/vec4.h"

#include <cmath>
#include <stdexcept>

namespace renderer {

struct Mat4 {
    double m[4][4];

    constexpr Mat4()
        : m{{0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0},
            {0.0, 0.0, 0.0, 0.0}} {}

    // 矩阵在内存中按行主序保存；计算时把 Vec4 当作列向量，使用 M * v。
    // 透视矩阵的垂直视场角使用“度”，裁剪空间深度采用教学中常见的 [-1, 1]。
    static Mat4 identity();
    static Mat4 translation(const Vec3& offset);
    static Mat4 scale(const Vec3& factors);
    static Mat4 perspective(double vertical_fov_degrees, double aspect, double near_z, double far_z);
    static Mat4 look_at(const Vec3& eye, const Vec3& target, const Vec3& up);
};

inline Mat4 Mat4::identity() {
    Mat4 result;
    result.m[0][0] = 1.0;
    result.m[1][1] = 1.0;
    result.m[2][2] = 1.0;
    result.m[3][3] = 1.0;
    return result;
}

inline Mat4 Mat4::translation(const Vec3& offset) {
    Mat4 result = identity();
    result.m[0][3] = offset.x;
    result.m[1][3] = offset.y;
    result.m[2][3] = offset.z;
    return result;
}

inline Mat4 Mat4::scale(const Vec3& factors) {
    Mat4 result;
    result.m[0][0] = factors.x;
    result.m[1][1] = factors.y;
    result.m[2][2] = factors.z;
    result.m[3][3] = 1.0;
    return result;
}

inline Mat4 Mat4::perspective(double vertical_fov_degrees, double aspect, double near_z, double far_z) {
    constexpr double pi = 3.14159265358979323846;
    const double fovy_radians = vertical_fov_degrees * pi / 180.0;
    const double f = 1.0 / std::tan(fovy_radians * 0.5);

    Mat4 result;
    result.m[0][0] = f / aspect;
    result.m[1][1] = f;
    result.m[2][2] = -(far_z + near_z) / (far_z - near_z);
    result.m[2][3] = -(2.0 * far_z * near_z) / (far_z - near_z);
    result.m[3][2] = -1.0;
    return result;
}

inline Mat4 Mat4::look_at(const Vec3& eye, const Vec3& target, const Vec3& up) {
    constexpr double epsilon = 1e-12;
    const Vec3 view_direction = target - eye;
    if (length_squared(view_direction) <= epsilon) {
        throw std::invalid_argument("look_at requires eye and target to be different");
    }
    if (length_squared(up) <= epsilon) {
        throw std::invalid_argument("look_at requires a non-zero up vector");
    }

    // 右手坐标系视图矩阵：相机看向 -Z，适合和上面的透视矩阵配套教学。
    const Vec3 forward = normalize(-view_direction);
    const Vec3 normalized_up = normalize(up);
    const Vec3 right_candidate = cross(normalized_up, forward);
    if (length_squared(right_candidate) <= epsilon) {
        throw std::invalid_argument("look_at up vector must not be parallel to the view direction");
    }

    const Vec3 right = normalize(right_candidate);
    const Vec3 camera_up = cross(forward, right);

    Mat4 result = identity();
    result.m[0][0] = right.x;
    result.m[0][1] = right.y;
    result.m[0][2] = right.z;
    result.m[0][3] = -dot(right, eye);
    result.m[1][0] = camera_up.x;
    result.m[1][1] = camera_up.y;
    result.m[1][2] = camera_up.z;
    result.m[1][3] = -dot(camera_up, eye);
    result.m[2][0] = forward.x;
    result.m[2][1] = forward.y;
    result.m[2][2] = forward.z;
    result.m[2][3] = -dot(forward, eye);
    return result;
}

inline Vec4 operator*(const Mat4& matrix, const Vec4& v) {
    return Vec4(
        matrix.m[0][0] * v.x + matrix.m[0][1] * v.y + matrix.m[0][2] * v.z + matrix.m[0][3] * v.w,
        matrix.m[1][0] * v.x + matrix.m[1][1] * v.y + matrix.m[1][2] * v.z + matrix.m[1][3] * v.w,
        matrix.m[2][0] * v.x + matrix.m[2][1] * v.y + matrix.m[2][2] * v.z + matrix.m[2][3] * v.w,
        matrix.m[3][0] * v.x + matrix.m[3][1] * v.y + matrix.m[3][2] * v.z + matrix.m[3][3] * v.w);
}

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 result;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            for (int k = 0; k < 4; ++k) {
                result.m[row][col] += a.m[row][k] * b.m[k][col];
            }
        }
    }
    return result;
}

}  // namespace renderer
