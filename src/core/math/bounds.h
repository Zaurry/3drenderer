#pragma once

#include "core/math/ray.h"
#include "core/math/vec3.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace renderer {

struct Bounds3 {
    Vec3 min;
    Vec3 max;

    Bounds3()
        : min(
              std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity(),
              std::numeric_limits<double>::infinity()),
          max(
              -std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity(),
              -std::numeric_limits<double>::infinity()) {}

    constexpr Bounds3(const Vec3& min_point, const Vec3& max_point)
        : min(min_point), max(max_point) {}

    void expand(const Vec3& p) {
        min = min_components(min, p);
        max = max_components(max, p);
    }

    void expand(const Bounds3& bounds) {
        expand(bounds.min);
        expand(bounds.max);
    }

    constexpr Vec3 extent() const {
        return max - min;
    }

    int longest_axis() const {
        const Vec3 size = extent();
        if (size.x >= size.y && size.x >= size.z) {
            return 0;
        }
        if (size.y >= size.z) {
            return 1;
        }
        return 2;
    }

    bool intersect(const Ray& ray, double t_min, double t_max) const {
        const double origins[3] = {ray.origin.x, ray.origin.y, ray.origin.z};
        const double directions[3] = {ray.direction.x, ray.direction.y, ray.direction.z};
        const double bounds_min[3] = {min.x, min.y, min.z};
        const double bounds_max[3] = {max.x, max.y, max.z};
        constexpr double parallel_epsilon = 1e-12;

        // Slab 法：每个坐标轴给出一段可命中的 t 区间，三个区间的交集非空才算射中 AABB。
        for (int axis = 0; axis < 3; ++axis) {
            if (std::abs(directions[axis]) < parallel_epsilon) {
                // 射线几乎平行于该轴的 slab；如果起点不在 slab 内，就永远无法进入盒子。
                if (origins[axis] < bounds_min[axis] || origins[axis] > bounds_max[axis]) {
                    return false;
                }
                continue;
            }

            const double inv_direction = 1.0 / directions[axis];
            double near_t = (bounds_min[axis] - origins[axis]) * inv_direction;
            double far_t = (bounds_max[axis] - origins[axis]) * inv_direction;
            if (near_t > far_t) {
                std::swap(near_t, far_t);
            }

            t_min = std::max(t_min, near_t);
            t_max = std::min(t_max, far_t);
            if (t_max <= t_min) {
                return false;
            }
        }

        return true;
    }
};

}  // namespace renderer
