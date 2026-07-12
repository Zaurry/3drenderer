#pragma once

#include "core/math/bounds.h"
#include "core/math/types.h"
#include "scene/camera.h"

namespace renderer {

class OrbitCameraController {
public:
    OrbitCameraController(const Bounds3& bounds, float aspect_ratio);

    void set_aspect_ratio(float aspect_ratio);
    void orbit(float delta_x, float delta_y);
    void zoom(float wheel_delta);
    Camera camera() const;

private:
    Vec3 target_ = Vec3::Zero();
    float distance_ = 1.0f;
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
    float aspect_ratio_ = 1.0f;
    float vertical_fov_degrees_ = 45.0f;

    Vec3 eye() const;
};

}  // namespace renderer
