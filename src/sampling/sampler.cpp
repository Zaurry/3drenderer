#include "sampling/sampler.h"

#include <algorithm>
#include <cmath>

namespace renderer {

namespace {

constexpr float kPi = 3.14159265358979323846f;

}

Vec3 cosine_weighted_hemisphere(PcgRandom& rng) {
    const float r1 = rng.next_float();
    const float r2 = rng.next_float();
    const float phi = 2.0f * kPi * r1;
    const float radius = std::sqrt(r2);
    const float x = radius * std::cos(phi);
    const float y = radius * std::sin(phi);
    const float z = std::sqrt(std::max(0.0f, 1.0f - r2));
    return Vec3(x, y, z).normalized();
}

Vec3 random_in_unit_sphere(PcgRandom& rng) {
    for (int attempt = 0; attempt < 1024; ++attempt) {
        const Vec3 p(
            2.0f * rng.next_float() - 1.0f,
            2.0f * rng.next_float() - 1.0f,
            2.0f * rng.next_float() - 1.0f);
        if (p.squaredNorm() < 1.0f) {
            return p;
        }
    }
    return Vec3::Zero();
}

Vec3 reflect(const Vec3& v, const Vec3& normal) {
    return v - (2.0f * v.dot(normal)) * normal;
}

bool refract(const Vec3& unit_direction, const Vec3& normal, float eta_ratio, Vec3& refracted) {
    const float cos_theta = std::min((-unit_direction).dot(normal), 1.0f);
    const Vec3 r_out_perp = eta_ratio * (unit_direction + cos_theta * normal);
    const float k = 1.0f - r_out_perp.squaredNorm();
    if (k < 0.0f) {
        return false;
    }
    refracted = r_out_perp - std::sqrt(k) * normal;
    return true;
}

}  // namespace renderer
