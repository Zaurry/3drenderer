#pragma once

#include "core/math/bounds.h"
#include "core/math/types.h"
#include "scene/camera.h"

namespace renderer {

class OrbitCameraController {
public:
    OrbitCameraController(const Bounds3& bounds, float aspect_ratio);

    void set_camera(const Camera& camera);
    void set_aspect_ratio(float aspect_ratio);
    void set_vertical_fov_degrees(float vertical_fov_degrees);
    float vertical_fov_degrees() const;
    void set_distance(float distance);
    float distance() const;
    void orbit(float delta_x, float delta_y);
    void pan(float delta_x, float delta_y, float viewport_height_pixels);
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
