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
        double vertical_fov_degrees,
        double aspect_ratio);

    Ray generate_ray(double u, double v) const;
    const Vec3& eye() const;
    const Vec3& forward() const;
    const Vec3& right() const;
    const Vec3& up() const;
    double viewport_width() const;
    double viewport_height() const;

private:
    Vec3 eye_;
    Vec3 forward_;
    Vec3 right_;
    Vec3 true_up_;
    double viewport_width_;
    double viewport_height_;
};

}  // namespace renderer
