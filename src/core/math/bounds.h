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
              std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::infinity(),
              std::numeric_limits<float>::infinity()),
          max(
              -std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity(),
              -std::numeric_limits<float>::infinity()) {}

    Bounds3(const Vec3& min_point, const Vec3& max_point)
        : min(min_point), max(max_point) {}

    void expand(const Vec3& p) {
        min = min.cwiseMin(p);
        max = max.cwiseMax(p);
    }

    void expand(const Bounds3& bounds) {
        expand(bounds.min);
        expand(bounds.max);
    }

    Vec3 extent() const {
        return max - min;
    }

    int longest_axis() const {
        const Vec3 size = extent();
        if (size.x() >= size.y() && size.x() >= size.z()) {
            return 0;
        }
        if (size.y() >= size.z()) {
            return 1;
        }
        return 2;
    }

    bool intersect(const Ray& ray, float t_min, float t_max) const {
        constexpr float kParallelDirectionEpsilon = 1e-12f;

        // Slab 法：每个坐标轴给出一段可命中的 t 区间，三个区间的交集非空才算射中 AABB。
        for (int axis = 0; axis < 3; ++axis) {
            const float direction = ray.direction[axis];
            if (std::abs(direction) < kParallelDirectionEpsilon) {
                // 射线几乎平行于该轴的 slab；如果起点不在 slab 内，就永远无法进入盒子。
                if (ray.origin[axis] < min[axis] || ray.origin[axis] > max[axis]) {
                    return false;
                }
                continue;
            }

            const float safe_direction = std::copysign(
                std::max(std::abs(direction), kParallelDirectionEpsilon),
                direction);
            const float inv_direction = 1.0f / safe_direction;
            float near_t = (min[axis] - ray.origin[axis]) * inv_direction;
            float far_t = (max[axis] - ray.origin[axis]) * inv_direction;
            if (near_t > far_t) {
                std::swap(near_t, far_t);
            }

            t_min = std::max(t_min, near_t);
            t_max = std::min(t_max, far_t);
            if (t_max < t_min) {
                return false;
            }
        }

        return true;
    }
};

}  // namespace renderer
