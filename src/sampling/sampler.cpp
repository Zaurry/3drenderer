#include "sampling/sampler.h"

#include <algorithm>
#include <cmath>

namespace renderer {

namespace {

constexpr double kPi = 3.14159265358979323846;

}

Vec3 cosine_weighted_hemisphere(PcgRandom& rng) {
    const double r1 = rng.next_double();
    const double r2 = rng.next_double();
    const double phi = 2.0 * kPi * r1;
    const double radius = std::sqrt(r2);
    const double x = radius * std::cos(phi);
    const double y = radius * std::sin(phi);
    const double z = std::sqrt(std::max(0.0, 1.0 - r2));
    return normalize(Vec3(
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z)));
}

Vec3 random_in_unit_sphere(PcgRandom& rng) {
    for (int attempt = 0; attempt < 1024; ++attempt) {
        const Vec3 p(
            static_cast<float>(2.0 * rng.next_double() - 1.0),
            static_cast<float>(2.0 * rng.next_double() - 1.0),
            static_cast<float>(2.0 * rng.next_double() - 1.0));
        if (length_squared(p) < 1.0f) {
            return p;
        }
    }
    return Vec3::Zero();
}

Vec3 reflect(const Vec3& v, const Vec3& normal) {
    return v - (2.0f * dot(v, normal)) * normal;
}

bool refract(const Vec3& unit_direction, const Vec3& normal, double eta_ratio, Vec3& refracted) {
    const float cos_theta = std::min(dot(-unit_direction, normal), 1.0f);
    const Vec3 r_out_perp =
        static_cast<float>(eta_ratio) * (unit_direction + cos_theta * normal);
    const float k = 1.0f - length_squared(r_out_perp);
    if (k < 0.0f) {
        return false;
    }
    refracted = r_out_perp - std::sqrt(k) * normal;
    return true;
}

}  // namespace renderer
