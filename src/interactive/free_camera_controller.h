#pragma once

#include "core/math/types.h"
#include "scene/camera.h"

namespace renderer {

class FreeCameraController {
public:
    FreeCameraController(
        const Camera& camera,
        float aspect_ratio,
        float movement_speed);

    void set_camera(const Camera& camera);
    void set_aspect_ratio(float aspect_ratio);
    void set_vertical_fov_degrees(float vertical_fov_degrees);
    float vertical_fov_degrees() const;
    void set_movement_speed(float movement_speed);
    float movement_speed() const;
    void look(float delta_x, float delta_y);
    bool move(
        float forward_axis,
        float right_axis,
        float up_axis,
        float delta_seconds);
    Camera camera() const;

private:
    Vec3 eye_ = Vec3::Zero();
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
    float aspect_ratio_ = 1.0f;
    float vertical_fov_degrees_ = 45.0f;
    float movement_speed_ = 1.0f;

    Vec3 forward() const;
    Vec3 right() const;
};

}  // namespace renderer
