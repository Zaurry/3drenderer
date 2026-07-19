#include "interactive/free_camera_controller.h"

#include "core/math/constants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace renderer {

namespace {

constexpr float kMinPitch = -1.5f;
constexpr float kMaxPitch = 1.5f;
constexpr float kLookSensitivity = 0.01f;

}  // namespace

FreeCameraController::FreeCameraController(
    const Camera& camera,
    float aspect_ratio,
    float movement_speed)
    : aspect_ratio_(aspect_ratio), movement_speed_(movement_speed) {
    if (!std::isfinite(aspect_ratio) || aspect_ratio <= 0.0f) {
        throw std::invalid_argument("FreeCameraController aspect ratio must be positive");
    }
    if (!std::isfinite(movement_speed) || movement_speed <= 0.0f) {
        throw std::invalid_argument("FreeCameraController movement speed must be positive");
    }
    set_camera(camera);
}

void FreeCameraController::set_camera(const Camera& camera) {
    eye_ = camera.eye();
    const Vec3 camera_forward = camera.forward();
    pitch_ = std::asin(std::clamp(camera_forward.y(), -1.0f, 1.0f));
    pitch_ = std::clamp(pitch_, kMinPitch, kMaxPitch);
    yaw_ = std::atan2(camera_forward.x(), -camera_forward.z());
}

void FreeCameraController::set_aspect_ratio(float aspect_ratio) {
    if (!std::isfinite(aspect_ratio) || aspect_ratio <= 0.0f) {
        throw std::invalid_argument("FreeCameraController aspect ratio must be positive");
    }
    aspect_ratio_ = aspect_ratio;
}

void FreeCameraController::set_vertical_fov_degrees(float vertical_fov_degrees) {
    if (!std::isfinite(vertical_fov_degrees) ||
        vertical_fov_degrees <= 0.0f ||
        vertical_fov_degrees >= 180.0f) {
        throw std::invalid_argument("FreeCameraController vertical FOV must be in the range (0, 180)");
    }
    vertical_fov_degrees_ = vertical_fov_degrees;
}

float FreeCameraController::vertical_fov_degrees() const {
    return vertical_fov_degrees_;
}

void FreeCameraController::set_movement_speed(float movement_speed) {
    if (!std::isfinite(movement_speed) || movement_speed <= 0.0f) {
        throw std::invalid_argument("FreeCameraController movement speed must be positive");
    }
    movement_speed_ = movement_speed;
}

float FreeCameraController::movement_speed() const {
    return movement_speed_;
}

void FreeCameraController::look(float delta_x, float delta_y) {
    if (!std::isfinite(delta_x) || !std::isfinite(delta_y)) {
        throw std::invalid_argument("FreeCameraController mouse delta must be finite");
    }
    yaw_ += delta_x * kLookSensitivity;
    pitch_ = std::clamp(pitch_ - delta_y * kLookSensitivity, kMinPitch, kMaxPitch);
}

bool FreeCameraController::move(
    float forward_axis,
    float right_axis,
    float up_axis,
    float delta_seconds) {
    if (!std::isfinite(forward_axis) ||
        !std::isfinite(right_axis) ||
        !std::isfinite(up_axis) ||
        !std::isfinite(delta_seconds) ||
        delta_seconds < 0.0f) {
        throw std::invalid_argument("FreeCameraController movement inputs must be finite and time must be non-negative");
    }

    Vec3 direction =
        forward() * forward_axis +
        right() * right_axis +
        Vec3::UnitY() * up_axis;
    if (direction.squaredNorm() <= kDirectionEpsilonSquared || delta_seconds == 0.0f) {
        return false;
    }

    direction.normalize();
    eye_ += direction * (movement_speed_ * delta_seconds);
    return true;
}

Camera FreeCameraController::camera() const {
    return Camera(
        eye_,
        eye_ + forward(),
        Vec3::UnitY(),
        vertical_fov_degrees_,
        aspect_ratio_);
}

Vec3 FreeCameraController::forward() const {
    const float cos_pitch = std::cos(pitch_);
    return Vec3(
        std::sin(yaw_) * cos_pitch,
        std::sin(pitch_),
        -std::cos(yaw_) * cos_pitch);
}

Vec3 FreeCameraController::right() const {
    return forward().cross(Vec3::UnitY()).normalized();
}

}  // namespace renderer
