#include "scene/camera.h"

#include <cmath>
#include <stdexcept>

namespace renderer {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-12;

bool is_finite(const Vec3& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

}  // namespace

Camera::Camera(
    const Vec3& eye,
    const Vec3& target,
    const Vec3& up,
    double vertical_fov_degrees,
    double aspect_ratio)
    : eye_(eye) {
    if (!is_finite(eye) || !is_finite(target) || !is_finite(up)) {
        throw std::invalid_argument("Camera vectors must be finite");
    }
    if (!std::isfinite(vertical_fov_degrees) ||
        vertical_fov_degrees <= 0.0 ||
        vertical_fov_degrees >= 180.0) {
        throw std::invalid_argument("Camera vertical_fov_degrees must be finite and in the range (0, 180)");
    }
    if (!std::isfinite(aspect_ratio) || aspect_ratio <= 0.0) {
        throw std::invalid_argument("Camera aspect_ratio must be finite and positive");
    }

    const Vec3 view_direction = target - eye;
    if (length_squared(view_direction) <= kEpsilon) {
        throw std::invalid_argument("Camera eye and target must be different");
    }
    if (length_squared(up) <= kEpsilon) {
        throw std::invalid_argument("Camera up vector must be non-zero");
    }

    // forward：相机从 eye 指向 target 的观察方向；标准相机看向世界 -Z。
    forward_ = normalize(view_direction);
    // right：右手坐标系中的相机右方向，由观察方向和输入 up 叉乘得到。
    const Vec3 right_candidate = cross(forward_, normalize(up));
    if (length_squared(right_candidate) <= kEpsilon) {
        throw std::invalid_argument("Camera up vector must not be parallel to the view direction");
    }
    right_ = normalize(right_candidate);
    // true_up：与 forward/right 正交的实际上方向，修正输入 up 的微小偏斜。
    true_up_ = cross(right_, forward_);

    const double fov_radians = vertical_fov_degrees * kPi / 180.0;
    viewport_height_ = 2.0 * std::tan(fov_radians * 0.5);
    viewport_width_ = viewport_height_ * aspect_ratio;
}

Ray Camera::generate_ray(double u, double v) const {
    if (!std::isfinite(u) || !std::isfinite(v)) {
        throw std::invalid_argument("Camera screen coordinates must be finite");
    }

    const Vec3 direction =
        forward_ +
        ((u - 0.5) * viewport_width_) * right_ +
        ((v - 0.5) * viewport_height_) * true_up_;
    return Ray(eye_, normalize(direction));
}

const Vec3& Camera::eye() const {
    return eye_;
}

const Vec3& Camera::forward() const {
    return forward_;
}

const Vec3& Camera::right() const {
    return right_;
}

const Vec3& Camera::up() const {
    return true_up_;
}

double Camera::viewport_width() const {
    return viewport_width_;
}

double Camera::viewport_height() const {
    return viewport_height_;
}

}  // namespace renderer
