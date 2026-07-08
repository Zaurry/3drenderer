#pragma once

#include "core/math/bounds.h"
#include "core/math/vec3.h"
#include "scene/camera.h"

#include <Eigen/Dense>

namespace renderer {

Eigen::Vector3d to_eigen(const Vec3& value);
Vec3 to_vec3(const Eigen::Vector3d& value);

class OrbitCameraController {
public:
    OrbitCameraController(const Bounds3& bounds, double aspect_ratio);

    void set_aspect_ratio(double aspect_ratio);
    void orbit(double delta_x, double delta_y);
    void zoom(double wheel_delta);
    Camera camera() const;

private:
    Eigen::Vector3d target_;
    double distance_ = 1.0;
    double yaw_ = 0.0;
    double pitch_ = 0.0;
    double aspect_ratio_ = 1.0;
    double vertical_fov_degrees_ = 45.0;

    Eigen::Vector3d eye() const;
};

}  // namespace renderer
