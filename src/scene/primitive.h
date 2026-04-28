#pragma once

#include "core/math/bounds.h"
#include "core/math/ray.h"
#include "core/math/vec2.h"
#include "core/math/vec3.h"

#include <cmath>
#include <stdexcept>

namespace renderer {

inline bool all_components_finite(const Vec3& v) {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

struct HitRecord {
    double t = 0.0;
    Vec3 position;
    Vec3 normal;
    Vec2 uv;
    int material_id = -1;
    bool front_face = true;

    void set_face_normal(const Ray& ray, const Vec3& outward_normal) {
        front_face = dot(ray.direction, outward_normal) < 0.0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Sphere {
public:
    Sphere(const Vec3& center, double radius, int material_id)
        : center_(center), radius_(radius), material_id_(material_id) {
        if (!all_components_finite(center)) {
            throw std::invalid_argument("Sphere center must be finite");
        }
        if (!std::isfinite(radius) || radius <= 0.0) {
            throw std::invalid_argument("Sphere radius must be finite and positive");
        }
    }

    bool intersect(const Ray& ray, double t_min, double t_max, HitRecord& hit) const {
        if (!all_components_finite(ray.origin)) {
            return false;
        }
        if (!all_components_finite(ray.direction)) {
            return false;
        }

        const Vec3 oc = ray.origin - center_;
        const double a = length_squared(ray.direction);
        constexpr double direction_epsilon = 1e-24;
        if (a <= direction_epsilon) {
            return false;
        }

        const double half_b = dot(oc, ray.direction);
        const double c = length_squared(oc) - radius_ * radius_;
        const double discriminant = half_b * half_b - a * c;
        if (discriminant < 0.0) {
            return false;
        }

        const double sqrt_discriminant = std::sqrt(discriminant);
        double root = (-half_b - sqrt_discriminant) / a;
        // t_min 用来避开自相交痤疮：阴影光线或反弹光线从表面出发时，不能再次命中刚离开的同一表面。
        if (root < t_min || root > t_max) {
            root = (-half_b + sqrt_discriminant) / a;
            if (root < t_min || root > t_max) {
                return false;
            }
        }

        hit.t = root;
        hit.position = ray.at(root);
        const Vec3 outward_normal = (hit.position - center_) / radius_;
        hit.set_face_normal(ray, outward_normal);
        hit.uv = Vec2();
        hit.material_id = material_id_;
        return true;
    }

    Bounds3 bounds() const {
        const Vec3 radius_vec(radius_, radius_, radius_);
        return Bounds3(center_ - radius_vec, center_ + radius_vec);
    }

private:
    Vec3 center_;
    double radius_;
    int material_id_;
};

class Triangle {
public:
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c, int material_id)
        : a_(a), b_(b), c_(c), material_id_(material_id) {
        if (!all_components_finite(a) || !all_components_finite(b) || !all_components_finite(c)) {
            throw std::invalid_argument("Triangle vertices must be finite");
        }
    }

    bool intersect(const Ray& ray, double t_min, double t_max, HitRecord& hit) const {
        if (!all_components_finite(ray.origin) || !all_components_finite(ray.direction)) {
            return false;
        }

        constexpr double epsilon = 1e-12;
        const Vec3 edge1 = b_ - a_;
        const Vec3 edge2 = c_ - a_;
        const Vec3 h = cross(ray.direction, edge2);
        const double determinant = dot(edge1, h);
        if (std::abs(determinant) < epsilon) {
            return false;
        }

        const double inv_determinant = 1.0 / determinant;
        const Vec3 s = ray.origin - a_;
        const double u = inv_determinant * dot(s, h);
        if (u < 0.0 || u > 1.0) {
            return false;
        }

        const Vec3 q = cross(s, edge1);
        const double v = inv_determinant * dot(ray.direction, q);
        if (v < 0.0 || u + v > 1.0) {
            return false;
        }

        const double t = inv_determinant * dot(edge2, q);
        // t_min 用来避免自相交痤疮：从表面发出的阴影光线或反弹光线，需要跳过起点处的微小重复命中。
        if (t < t_min || t > t_max) {
            return false;
        }

        hit.t = t;
        hit.position = ray.at(t);
        hit.set_face_normal(ray, normalize(cross(edge1, edge2)));
        hit.uv = Vec2(u, v);
        hit.material_id = material_id_;
        return true;
    }

    Bounds3 bounds() const {
        Bounds3 box;
        box.expand(a_);
        box.expand(b_);
        box.expand(c_);
        return box;
    }

    Vec3 centroid() const {
        return (a_ + b_ + c_) / 3.0;
    }

private:
    Vec3 a_;
    Vec3 b_;
    Vec3 c_;
    int material_id_;
};

}  // namespace renderer
