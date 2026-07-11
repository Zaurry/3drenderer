#pragma once

#include "core/math/ray.h"
#include "core/math/vec3.h"

namespace renderer {

class Camera {
public:
    Camera(
        const Vec3& eye,
        const Vec3& target,
        const Vec3& up,
        float vertical_fov_degrees,
        float aspect_ratio);

    Ray generate_ray(float u, float v) const;
    const Vec3& eye() const;
    const Vec3& forward() const;
    const Vec3& right() const;
    const Vec3& up() const;
    float viewport_width() const;
    float viewport_height() const;

private:
    Vec3 eye_;
    Vec3 forward_;
    Vec3 right_;
    Vec3 true_up_;
    float viewport_width_ = 0.0f;
    float viewport_height_ = 0.0f;
};

}  // namespace renderer
