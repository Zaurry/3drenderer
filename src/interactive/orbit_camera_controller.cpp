#include "interactive/orbit_camera_controller.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace renderer {

namespace {

constexpr double kMinPitch = -1.5;
constexpr double kMaxPitch = 1.5;
constexpr double kMinDistance = 0.05;

}  // namespace

Eigen::Vector3d to_eigen(const Vec3& value) {
    return Eigen::Vector3d(value.x(), value.y(), value.z());
}

Vec3 to_vec3(const Eigen::Vector3d& value) {
    return Vec3(
        static_cast<float>(value.x()),
        static_cast<float>(value.y()),
        static_cast<float>(value.z()));
}

OrbitCameraController::OrbitCameraController(const Bounds3& bounds, double aspect_ratio)
    : target_(to_eigen((bounds.min + bounds.max) * 0.5f)), aspect_ratio_(aspect_ratio) {
    if (!std::isfinite(aspect_ratio) || aspect_ratio <= 0.0) {
        throw std::invalid_argument("OrbitCameraController aspect ratio must be positive");
    }

    const double radius = std::max(0.5, static_cast<double>(length(bounds.max - bounds.min)) * 0.5);
    distance_ = radius * 2.6;
    pitch_ = 0.12;
}

void OrbitCameraController::set_aspect_ratio(double aspect_ratio) {
    if (!std::isfinite(aspect_ratio) || aspect_ratio <= 0.0) {
        throw std::invalid_argument("OrbitCameraController aspect ratio must be positive");
    }
    aspect_ratio_ = aspect_ratio;
}

void OrbitCameraController::orbit(double delta_x, double delta_y) {
    constexpr double sensitivity = 0.01;
    yaw_ -= delta_x * sensitivity;
    pitch_ = std::clamp(pitch_ + delta_y * sensitivity, kMinPitch, kMaxPitch);
}

void OrbitCameraController::zoom(double wheel_delta) {
    distance_ *= std::exp(-wheel_delta * 0.12);
    distance_ = std::max(distance_, kMinDistance);
}

Camera OrbitCameraController::camera() const {
    return Camera(
        to_vec3(eye()),
        to_vec3(target_),
        Vec3(0.0, 1.0, 0.0),
        vertical_fov_degrees_,
        aspect_ratio_);
}

Eigen::Vector3d OrbitCameraController::eye() const {
    const double cos_pitch = std::cos(pitch_);
    const Eigen::Vector3d offset(
        std::sin(yaw_) * cos_pitch * distance_,
        std::sin(pitch_) * distance_,
        std::cos(yaw_) * cos_pitch * distance_);
    return target_ + offset;
}

}  // namespace renderer
