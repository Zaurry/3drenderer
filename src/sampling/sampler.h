#pragma once

#include "core/math/vec3.h"
#include "core/random.h"

namespace renderer {

Vec3 cosine_weighted_hemisphere(PcgRandom& rng);
Vec3 random_in_unit_sphere(PcgRandom& rng);
Vec3 reflect(const Vec3& v, const Vec3& normal);
bool refract(const Vec3& unit_direction, const Vec3& normal, float eta_ratio, Vec3& refracted);

}  // namespace renderer
