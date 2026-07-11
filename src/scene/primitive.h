#pragma once

#include "core/math/bounds.h"
#include "core/math/ray.h"
#include "core/math/vec2.h"
#include "core/math/vec3.h"

#include <cmath>
#include <stdexcept>

namespace renderer {

inline bool all_components_finite(const Vec3& v) {
    return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
}

inline bool usable_direction(const Vec3& v) {
    return all_components_finite(v) && length_squared(v) > 1e-24;
}

inline void make_orthonormal_basis(const Vec3& normal, Vec3& tangent, Vec3& bitangent) {
    const Vec3 helper = std::abs(normal.x()) > 0.9
        ? Vec3(0.0, 1.0, 0.0)
        : Vec3(1.0, 0.0, 0.0);
    tangent = normalize(cross(helper, normal));
    bitangent = normalize(cross(normal, tangent));
}

struct TriangleVertex {
    Vec3 position = Vec3::Zero();
    Vec2 uv = Vec2::Zero();
    Vec3 normal = Vec3::Zero();
    bool has_normal = false;
};

struct HitRecord {
    double t = 0.0;
    Vec3 position = Vec3::Zero();
    Vec2 uv = Vec2::Zero();
    Vec3 geometric_normal = Vec3::Zero();
    Vec3 shading_normal = Vec3::Zero();
    Vec3 tangent = Vec3::Zero();
    Vec3 bitangent = Vec3::Zero();
    bool has_valid_uv_basis = false;
    int material_id = -1;
    bool front_face = true;

    void set_normals(const Ray& ray, const Vec3& outward_geometric, const Vec3& outward_shading) {
        const Vec3 unit_geometric = normalize(outward_geometric);
        Vec3 unit_shading = usable_direction(outward_shading)
            ? normalize(outward_shading)
            : unit_geometric;
        if (dot(unit_shading, unit_geometric) < 0.0) {
            unit_shading = -unit_shading;
        }

        front_face = dot(ray.direction, unit_geometric) < 0.0;
        geometric_normal = front_face ? unit_geometric : -unit_geometric;
        shading_normal = front_face ? unit_shading : -unit_shading;
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
        hit.set_normals(ray, outward_normal, outward_normal);
        make_orthonormal_basis(hit.shading_normal, hit.tangent, hit.bitangent);
        hit.uv = Vec2::Zero();
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
        : Triangle(a, b, c, material_id, Vec2::Zero(), Vec2::Zero(), Vec2::Zero()) {}

    Triangle(
        const Vec3& a,
        const Vec3& b,
        const Vec3& c,
        int material_id,
        const Vec2& uv0,
        const Vec2& uv1,
        const Vec2& uv2)
        : Triangle(
            TriangleVertex{a, uv0, Vec3::Zero(), false},
            TriangleVertex{b, uv1, Vec3::Zero(), false},
            TriangleVertex{c, uv2, Vec3::Zero(), false},
            material_id) {}

    Triangle(
        const TriangleVertex& v0,
        const TriangleVertex& v1,
        const TriangleVertex& v2,
        int material_id)
        : vertices_{v0, v1, v2}, material_id_(material_id) {
        if (!all_components_finite(v0.position) ||
            !all_components_finite(v1.position) ||
            !all_components_finite(v2.position)) {
            throw std::invalid_argument("Triangle vertices must be finite");
        }
        for (TriangleVertex& vertex : vertices_) {
            vertex.has_normal = vertex.has_normal && usable_direction(vertex.normal);
            if (vertex.has_normal) {
                vertex.normal = normalize(vertex.normal);
            }
        }
    }

    bool intersect(const Ray& ray, double t_min, double t_max, HitRecord& hit) const {
        if (!all_components_finite(ray.origin) || !all_components_finite(ray.direction)) {
            return false;
        }

        constexpr double epsilon = 1e-12;
        const Vec3 edge1 = b() - a();
        const Vec3 edge2 = c() - a();
        const Vec3 h = cross(ray.direction, edge2);
        const double determinant = dot(edge1, h);
        if (std::abs(determinant) < epsilon) {
            return false;
        }

        const double inv_determinant = 1.0 / determinant;
        const Vec3 s = ray.origin - a();
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
        const double w0 = 1.0 - u - v;
        const Vec3 outward_geometric = normalize(cross(edge1, edge2));
        const Vec3 outward_shading = interpolate_shading_normal(w0, u, v);
        hit.set_normals(ray, outward_geometric, outward_shading);
        tangent_basis(hit.shading_normal, hit.tangent, hit.bitangent);
        hit.has_valid_uv_basis = has_valid_uv_basis();
        hit.uv = interpolate_uv(w0, u, v);
        hit.material_id = material_id_;
        return true;
    }

    Bounds3 bounds() const {
        Bounds3 box;
        box.expand(a());
        box.expand(b());
        box.expand(c());
        return box;
    }

    Vec3 centroid() const {
        return (a() + b() + c()) / 3.0;
    }

    const Vec3& a() const {
        return vertices_[0].position;
    }

    const Vec3& b() const {
        return vertices_[1].position;
    }

    const Vec3& c() const {
        return vertices_[2].position;
    }

    const TriangleVertex& vertex(int index) const {
        if (index < 0 || index > 2) {
            throw std::out_of_range("Triangle vertex index is out of range");
        }
        return vertices_[index];
    }

    int material_id() const {
        return material_id_;
    }

    Vec2 interpolate_uv(double w0, double w1, double w2) const {
        return Vec2(
            vertices_[0].uv.x() * w0 + vertices_[1].uv.x() * w1 + vertices_[2].uv.x() * w2,
            vertices_[0].uv.y() * w0 + vertices_[1].uv.y() * w1 + vertices_[2].uv.y() * w2);
    }

    Vec3 geometric_normal() const {
        return normalize(cross(b() - a(), c() - a()));
    }

    Vec3 interpolate_shading_normal(double w0, double w1, double w2) const {
        const Vec3 face_normal = geometric_normal();
        if (!vertices_[0].has_normal || !vertices_[1].has_normal || !vertices_[2].has_normal) {
            return face_normal;
        }

        Vec3 interpolated =
            vertices_[0].normal * w0 +
            vertices_[1].normal * w1 +
            vertices_[2].normal * w2;
        if (!usable_direction(interpolated)) {
            return face_normal;
        }
        interpolated = normalize(interpolated);
        return dot(interpolated, face_normal) < 0.0 ? -interpolated : interpolated;
    }

    void tangent_basis(const Vec3& shading_normal, Vec3& tangent, Vec3& bitangent) const {
        const Vec3 edge1 = b() - a();
        const Vec3 edge2 = c() - a();
        const double du1 = vertices_[1].uv.x() - vertices_[0].uv.x();
        const double dv1 = vertices_[1].uv.y() - vertices_[0].uv.y();
        const double du2 = vertices_[2].uv.x() - vertices_[0].uv.x();
        const double dv2 = vertices_[2].uv.y() - vertices_[0].uv.y();
        const double determinant = du1 * dv2 - dv1 * du2;
        if (std::abs(determinant) <= 1e-12) {
            make_orthonormal_basis(shading_normal, tangent, bitangent);
            return;
        }

        const double inverse = 1.0 / determinant;
        const Vec3 raw_tangent = (edge1 * dv2 - edge2 * dv1) * inverse;
        const Vec3 raw_bitangent = (edge2 * du1 - edge1 * du2) * inverse;
        tangent = raw_tangent - shading_normal * dot(raw_tangent, shading_normal);
        if (!usable_direction(tangent)) {
            make_orthonormal_basis(shading_normal, tangent, bitangent);
            return;
        }
        tangent = normalize(tangent);
        bitangent = normalize(cross(shading_normal, tangent));
        if (usable_direction(raw_bitangent) && dot(bitangent, raw_bitangent) < 0.0) {
            bitangent = -bitangent;
        }
    }

    bool has_valid_uv_basis() const {
        const double du1 = vertices_[1].uv.x() - vertices_[0].uv.x();
        const double dv1 = vertices_[1].uv.y() - vertices_[0].uv.y();
        const double du2 = vertices_[2].uv.x() - vertices_[0].uv.x();
        const double dv2 = vertices_[2].uv.y() - vertices_[0].uv.y();
        return std::isfinite(du1) && std::isfinite(dv1) &&
            std::isfinite(du2) && std::isfinite(dv2) &&
            std::abs(du1 * dv2 - dv1 * du2) > 1e-12;
    }

private:
    TriangleVertex vertices_[3];
    int material_id_;
};

}  // namespace renderer
