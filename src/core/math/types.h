#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace renderer {

using Scalar = float;
using Vec2 = Eigen::Vector2f;
using Vec3 = Eigen::Vector3f;
using Vec4 = Eigen::Vector4f;
using Mat3 = Eigen::Matrix3f;
using Mat4 = Eigen::Matrix4f;
using Color = Eigen::Vector3f;

inline float dot(const Vec3& a, const Vec3& b) { return a.dot(b); }
inline Vec3 cross(const Vec3& a, const Vec3& b) { return a.cross(b); }
inline float length(const Vec3& v) { return v.norm(); }
inline float length_squared(const Vec3& v) { return v.squaredNorm(); }
inline Vec3 normalize(const Vec3& v) { return v.isZero() ? Vec3::Zero() : v.normalized(); }
inline Vec3 min_components(const Vec3& a, const Vec3& b) { return a.cwiseMin(b); }
inline Vec3 max_components(const Vec3& a, const Vec3& b) { return a.cwiseMax(b); }

}  // namespace renderer
