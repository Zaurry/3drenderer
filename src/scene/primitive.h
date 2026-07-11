#pragma once

#include "core/math/bounds.h"
#include "core/math/ray.h"
#include "core/math/vec2.h"
#include "core/math/vec3.h"

#include <cmath>
#include <stdexcept>

namespace renderer {

inline bool all_components_finite(const Vec3& v) {
    return v.allFinite();
}

inline bool usable_direction(const Vec3& v) {
    return v.allFinite() && v.squaredNorm() > 1e-24f;
}

inline void make_orthonormal_basis(const Vec3& normal, Vec3& tangent, Vec3& bitangent) {
    const Vec3 helper = std::abs(normal.x()) > 0.9f
        ? Vec3(0.0f, 1.0f, 0.0f)
        : Vec3(1.0f, 0.0f, 0.0f);
    tangent = helper.cross(normal).normalized();
    bitangent = normal.cross(tangent).normalized();
}

struct TriangleVertex {
    Vec3 position = Vec3::Zero();
    Vec2 uv = Vec2::Zero();
    Vec3 normal = Vec3::Zero();
    bool has_normal = false;
};

struct HitRecord {
    float t = 0.0f;
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
        const Vec3 unit_geometric = outward_geometric.normalized();
        Vec3 unit_shading = usable_direction(outward_shading)
            ? outward_shading.normalized()
            : unit_geometric;
        if (unit_shading.dot(unit_geometric) < 0.0f) {
            unit_shading = -unit_shading;
        }

        front_face = ray.direction.dot(unit_geometric) < 0.0f;
        geometric_normal = front_face ? unit_geometric : -unit_geometric;
        shading_normal = front_face ? unit_shading : -unit_shading;
    }
};

class Sphere {
public:
    Sphere(const Vec3& center, float radius, int material_id)
        : center_(center), radius_(radius), material_id_(material_id) {
        if (!all_components_finite(center)) {
            throw std::invalid_argument("Sphere center must be finite");
        }
        if (!std::isfinite(radius) || radius <= 0.0f) {
            throw std::invalid_argument("Sphere radius must be finite and positive");
        }
    }

    bool intersect(const Ray& ray, float t_min, float t_max, HitRecord& hit) const {
        if (!all_components_finite(ray.origin)) {
            return false;
        }
        if (!all_components_finite(ray.direction)) {
            return false;
        }

        const Vec3 oc = ray.origin - center_;
        const float a = ray.direction.squaredNorm();
        constexpr float direction_epsilon = 1e-24f;
        if (a <= direction_epsilon) {
            return false;
        }

        const float half_b = oc.dot(ray.direction);
        const float c = oc.squaredNorm() - radius_ * radius_;
        const float discriminant = half_b * half_b - a * c;
        if (discriminant < 0.0f) {
            return false;
        }

        const float sqrt_discriminant = std::sqrt(discriminant);
        float root = (-half_b - sqrt_discriminant) / a;
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
    float radius_;
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
                vertex.normal.normalize();
            }
        }
    }

    bool intersect(const Ray& ray, float t_min, float t_max, HitRecord& hit) const {
        if (!all_components_finite(ray.origin) || !all_components_finite(ray.direction)) {
            return false;
        }

        constexpr float parallel_epsilon = 1e-12f;
        const Vec3 edge1 = b() - a();
        const Vec3 edge2 = c() - a();
        const Vec3 h = ray.direction.cross(edge2);
        const float determinant = edge1.dot(h);
        if (std::abs(determinant) < parallel_epsilon) {
            return false;
        }

        const float inv_determinant = 1.0f / determinant;
        const Vec3 s = ray.origin - a();
        const float u = inv_determinant * s.dot(h);
        if (u < 0.0f || u > 1.0f) {
            return false;
        }

        const Vec3 q = s.cross(edge1);
        const float v = inv_determinant * ray.direction.dot(q);
        if (v < 0.0f || u + v > 1.0f) {
            return false;
        }

        const float t = inv_determinant * edge2.dot(q);
        // t_min 用来避免自相交痤疮：从表面发出的阴影光线或反弹光线，需要跳过起点处的微小重复命中。
        if (t < t_min || t > t_max) {
            return false;
        }

        hit.t = t;
        hit.position = ray.at(t);
        const float w0 = 1.0f - u - v;
        const Vec3 outward_geometric = edge1.cross(edge2).normalized();
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
        return (a() + b() + c()) / 3.0f;
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

    Vec2 interpolate_uv(float w0, float w1, float w2) const {
        return vertices_[0].uv * w0 + vertices_[1].uv * w1 + vertices_[2].uv * w2;
    }

    Vec3 geometric_normal() const {
        return (b() - a()).cross(c() - a()).normalized();
    }

    Vec3 interpolate_shading_normal(float w0, float w1, float w2) const {
        const Vec3 face_normal = geometric_normal();
        if (!vertices_[0].has_normal || !vertices_[1].has_normal || !vertices_[2].has_normal) {
            return face_normal;
        }

        Vec3 interpolated =
            vertices_[0].normal * w0 + vertices_[1].normal * w1 + vertices_[2].normal * w2;
        if (!usable_direction(interpolated)) {
            return face_normal;
        }
        interpolated.normalize();
        return interpolated.dot(face_normal) < 0.0f ? -interpolated : interpolated;
    }

    void tangent_basis(const Vec3& shading_normal, Vec3& tangent, Vec3& bitangent) const {
        const Vec3 edge1 = b() - a();
        const Vec3 edge2 = c() - a();
        const float du1 = vertices_[1].uv.x() - vertices_[0].uv.x();
        const float dv1 = vertices_[1].uv.y() - vertices_[0].uv.y();
        const float du2 = vertices_[2].uv.x() - vertices_[0].uv.x();
        const float dv2 = vertices_[2].uv.y() - vertices_[0].uv.y();
        const float determinant = du1 * dv2 - dv1 * du2;
        constexpr float uv_determinant_epsilon = 1e-12f;
        if (std::abs(determinant) <= uv_determinant_epsilon) {
            make_orthonormal_basis(shading_normal, tangent, bitangent);
            return;
        }

        const float inverse = 1.0f / determinant;
        const Vec3 raw_tangent = (edge1 * dv2 - edge2 * dv1) * inverse;
        const Vec3 raw_bitangent = (edge2 * du1 - edge1 * du2) * inverse;
        tangent = raw_tangent - shading_normal * raw_tangent.dot(shading_normal);
        if (!usable_direction(tangent)) {
            make_orthonormal_basis(shading_normal, tangent, bitangent);
            return;
        }
        tangent.normalize();
        bitangent = shading_normal.cross(tangent).normalized();
        if (usable_direction(raw_bitangent) && bitangent.dot(raw_bitangent) < 0.0f) {
            bitangent = -bitangent;
        }
    }

    bool has_valid_uv_basis() const {
        const float du1 = vertices_[1].uv.x() - vertices_[0].uv.x();
        const float dv1 = vertices_[1].uv.y() - vertices_[0].uv.y();
        const float du2 = vertices_[2].uv.x() - vertices_[0].uv.x();
        const float dv2 = vertices_[2].uv.y() - vertices_[0].uv.y();
        constexpr float uv_determinant_epsilon = 1e-12f;
        return std::isfinite(du1) && std::isfinite(dv1) &&
            std::isfinite(du2) && std::isfinite(dv2) &&
            std::abs(du1 * dv2 - dv1 * du2) > uv_determinant_epsilon;
    }

private:
    TriangleVertex vertices_[3];
    int material_id_;
};

}  // namespace renderer
