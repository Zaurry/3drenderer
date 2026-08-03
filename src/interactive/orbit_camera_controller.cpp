#include "interactive/orbit_camera_controller.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace renderer {

namespace {

constexpr float kMinPitch = -1.5f;
constexpr float kMaxPitch = 1.5f;
constexpr float kMinDistance = 0.05f;

}  // namespace

OrbitCameraController::OrbitCameraController(const Bounds3& bounds, float aspect_ratio)
    : target_((bounds.min + bounds.max) * 0.5f), aspect_ratio_(aspect_ratio) {
    if (!std::isfinite(aspect_ratio) || aspect_ratio <= 0.0f) {
        throw std::invalid_argument("OrbitCameraController aspect ratio must be positive");
    }

    const float radius = std::max(0.5f, (bounds.max - bounds.min).norm() * 0.5f);
    distance_ = radius * 2.6f;
    pitch_ = 0.12f;
}

void OrbitCameraController::set_camera(const Camera& camera) {
    target_ = camera.eye() + camera.forward() * distance_;
    const Vec3 offset = -camera.forward();
    pitch_ = std::asin(std::clamp(offset.y(), -1.0f, 1.0f));
    pitch_ = std::clamp(pitch_, kMinPitch, kMaxPitch);
    yaw_ = std::atan2(offset.x(), offset.z());
}

void OrbitCameraController::set_aspect_ratio(float aspect_ratio) {
    if (!std::isfinite(aspect_ratio) || aspect_ratio <= 0.0f) {
        throw std::invalid_argument("OrbitCameraController aspect ratio must be positive");
    }
    aspect_ratio_ = aspect_ratio;
}

void OrbitCameraController::set_vertical_fov_degrees(float vertical_fov_degrees) {
    if (!std::isfinite(vertical_fov_degrees) ||
        vertical_fov_degrees <= 0.0f ||
        vertical_fov_degrees >= 180.0f) {
        throw std::invalid_argument("OrbitCameraController vertical FOV must be in the range (0, 180)");
    }
    vertical_fov_degrees_ = vertical_fov_degrees;
}

float OrbitCameraController::vertical_fov_degrees() const {
    return vertical_fov_degrees_;
}

void OrbitCameraController::set_distance(float distance) {
    if (!std::isfinite(distance) || distance < kMinDistance) {
        throw std::invalid_argument("OrbitCameraController distance must be finite and positive");
    }
    distance_ = distance;
}

float OrbitCameraController::distance() const {
    return distance_;
}

void OrbitCameraController::orbit(float delta_x, float delta_y) {
    constexpr float sensitivity = 0.01f;
    yaw_ -= delta_x * sensitivity;
    pitch_ = std::clamp(pitch_ + delta_y * sensitivity, kMinPitch, kMaxPitch);
}

void OrbitCameraController::pan(
    float delta_x,
    float delta_y,
    float viewport_height_pixels) {
    if (!std::isfinite(delta_x) ||
        !std::isfinite(delta_y) ||
        !std::isfinite(viewport_height_pixels) ||
        viewport_height_pixels <= 0.0f) {
        throw std::invalid_argument(
            "OrbitCameraController pan inputs must be finite and viewport height must be positive");
    }

    const Camera current_camera = camera();
    const float world_units_per_pixel =
        distance_ * current_camera.viewport_height() / viewport_height_pixels;
    target_ +=
        (-delta_x * current_camera.right() + delta_y * current_camera.up()) *
        world_units_per_pixel;
}

void OrbitCameraController::zoom(float wheel_delta) {
    distance_ *= std::exp(-wheel_delta * 0.12f);
    distance_ = std::max(distance_, kMinDistance);
}

Camera OrbitCameraController::camera() const {
    return Camera(
        eye(),
        target_,
        Vec3(0.0f, 1.0f, 0.0f),
        vertical_fov_degrees_,
        aspect_ratio_);
}

Vec3 OrbitCameraController::eye() const {
    const float cos_pitch = std::cos(pitch_);
    const Vec3 offset(
        std::sin(yaw_) * cos_pitch * distance_,
        std::sin(pitch_) * distance_,
        std::cos(yaw_) * cos_pitch * distance_);
    return target_ + offset;
}

}  // namespace renderer
