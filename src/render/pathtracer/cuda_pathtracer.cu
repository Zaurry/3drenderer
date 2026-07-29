#define EIGEN_NO_CUDA
#include "render/pathtracer/cuda_pathtracer.h"

#include "acceleration/bvh.h"
#include "core/timer.h"
#include "scene/material.h"
#include "scene/texture.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace renderer {
namespace {

constexpr int kThreadsPerBlock = 64;
constexpr int kMaxPathBounces = 64;
constexpr int kRussianRouletteStartBounce = 3;
constexpr int kMaxTransparentLayers = 64;
constexpr int kBvhStackCapacity = 64;
constexpr float kPi = 3.14159265358979323846f;

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(result));
    }
}

struct DVec2 {
    float x;
    float y;
};

struct DVec3 {
    float x;
    float y;
    float z;
};

struct DRay {
    DVec3 origin;
    DVec3 direction;
};

struct DVertex {
    DVec3 position;
    DVec2 uv;
    DVec3 normal;
    int has_normal;
};

struct DTriangle {
    DVertex vertices[3];
};

struct DSphere {
    DVec3 center;
    float radius;
};

struct DMaterial {
    int type;
    DVec3 base_color;
    DVec3 emission;
    float roughness;
    float ior;
    float opacity;
    float alpha_cutoff;
    float bump_scale;
    int two_sided;
    int diffuse_texture_id;
    int opacity_texture_id;
    int bump_texture_id;
};

struct DTexture {
    int width;
    int height;
    int first_pixel;
};

struct DBvhNode {
    DVec3 bounds_min;
    DVec3 bounds_max;
    int left;
    int right;
    int first;
    int count;
};

struct DPointLight {
    DVec3 position;
    DVec3 intensity;
};

struct DDirectionalLight {
    DVec3 direction;
    DVec3 radiance;
};

struct DCamera {
    DVec3 eye;
    DVec3 forward;
    DVec3 right;
    DVec3 up;
    float viewport_width;
    float viewport_height;
};

struct DScene {
    const DMaterial* materials;
    int material_count;
    const DTexture* textures;
    int texture_count;
    const DVec3* texels;
    const DSphere* spheres;
    const int* sphere_material_ids;
    int sphere_count;
    const DTriangle* triangles;
    const int* triangle_material_ids;
    int triangle_count;
    const DBvhNode* bvh_nodes;
    int bvh_node_count;
    const int* primitive_indices;
    const DPointLight* point_lights;
    int point_light_count;
    const DDirectionalLight* directional_lights;
    int directional_light_count;
    DVec3 environment;
};

struct DHit {
    float t;
    DVec3 position;
    DVec2 uv;
    DVec3 geometric_normal;
    DVec3 shading_normal;
    int material_id;
    int front_face;
    int primitive_kind;
    int primitive_index;
    float barycentric_u;
    float barycentric_v;
};

struct DCompactHit {
    float t;
    float u;
    float v;
    int primitive_kind;
    int primitive_index;
};

struct DSurface {
    DVec3 base_color;
    float opacity;
    DVec3 shading_normal;
};

struct DPcgState {
    unsigned long long state;
    unsigned long long increment;
};

__host__ DVec3 to_device(const Vec3& value) {
    return DVec3{value.x(), value.y(), value.z()};
}

__host__ DVec2 to_device(const Vec2& value) {
    return DVec2{value.x(), value.y()};
}

__device__ DVec3 v3(float x, float y, float z) {
    return DVec3{x, y, z};
}

__device__ DVec3 add(DVec3 a, DVec3 b) {
    return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ DVec3 sub(DVec3 a, DVec3 b) {
    return v3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ DVec3 mul(DVec3 a, float value) {
    return v3(a.x * value, a.y * value, a.z * value);
}

__device__ DVec3 divv(DVec3 a, float value) {
    return mul(a, 1.0f / value);
}

__device__ DVec3 product(DVec3 a, DVec3 b) {
    return v3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float dot(DVec3 a, DVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ DVec3 cross(DVec3 a, DVec3 b) {
    return v3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ float length_squared(DVec3 value) {
    return dot(value, value);
}

__device__ bool finite(DVec3 value) {
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}

__device__ bool usable(DVec3 value) {
    return finite(value) && length_squared(value) > 1.0e-24f;
}

__device__ DVec3 normalize(DVec3 value) {
    const float squared = length_squared(value);
    return squared > 1.0e-24f ? divv(value, sqrtf(squared)) : v3(0.0f, 0.0f, 0.0f);
}

__device__ float component(DVec3 value, int axis) {
    return axis == 0 ? value.x : (axis == 1 ? value.y : value.z);
}

__device__ float max_component(DVec3 value) {
    return fmaxf(value.x, fmaxf(value.y, value.z));
}

__device__ DVec3 reflect_vector(DVec3 value, DVec3 normal) {
    return sub(value, mul(normal, 2.0f * dot(value, normal)));
}

__device__ bool refract_vector(
    DVec3 unit_direction,
    DVec3 normal,
    float eta_ratio,
    DVec3& refracted) {
    const float cos_theta = fminf(dot(mul(unit_direction, -1.0f), normal), 1.0f);
    const DVec3 perpendicular = mul(add(unit_direction, mul(normal, cos_theta)), eta_ratio);
    const float k = 1.0f - length_squared(perpendicular);
    if (k < 0.0f) {
        return false;
    }
    refracted = sub(perpendicular, mul(normal, sqrtf(k)));
    return true;
}

__device__ void set_normals(
    DHit& hit,
    const DRay& ray,
    DVec3 outward_geometric,
    DVec3 outward_shading) {
    const DVec3 unit_geometric = normalize(outward_geometric);
    DVec3 unit_shading = usable(outward_shading)
        ? normalize(outward_shading)
        : unit_geometric;
    if (dot(unit_shading, unit_geometric) < 0.0f) {
        unit_shading = mul(unit_shading, -1.0f);
    }
    hit.front_face = dot(ray.direction, unit_geometric) < 0.0f;
    hit.geometric_normal = hit.front_face ? unit_geometric : mul(unit_geometric, -1.0f);
    hit.shading_normal = hit.front_face ? unit_shading : mul(unit_shading, -1.0f);
}

__device__ unsigned int pcg_next(DPcgState& rng) {
    const unsigned long long old_state = rng.state;
    rng.state = old_state * 6364136223846793005ULL + rng.increment;
    const unsigned int xorshifted = static_cast<unsigned int>(((old_state >> 18U) ^ old_state) >> 27U);
    const unsigned int rotation = static_cast<unsigned int>(old_state >> 59U);
    return (xorshifted >> rotation) | (xorshifted << ((0U - rotation) & 31U));
}

__device__ void pcg_seed(DPcgState& rng, unsigned long long seed) {
    rng.state = 0;
    rng.increment = (seed << 1U) | 1U;
    pcg_next(rng);
    rng.state += seed;
    pcg_next(rng);
}

__device__ float random_float(DPcgState& rng) {
    return static_cast<float>(pcg_next(rng) >> 8U) * (1.0f / 16777216.0f);
}

__device__ unsigned long long pixel_seed(
    int x,
    int y,
    int width,
    unsigned long long sample_seed_offset) {
    unsigned long long seed = 1469598103934665603ULL;
    seed ^= static_cast<unsigned long long>(x + 1);
    seed *= 1099511628211ULL;
    seed ^= static_cast<unsigned long long>((y + 1) * 65537);
    seed *= 1099511628211ULL;
    seed ^= static_cast<unsigned long long>(width + 1);
    if (sample_seed_offset != 0) {
        seed *= 1099511628211ULL;
        seed ^= sample_seed_offset;
    }
    return seed;
}

__device__ DVec3 random_in_unit_sphere(DPcgState& rng) {
    for (int attempt = 0; attempt < 1024; ++attempt) {
        const DVec3 value = v3(
            2.0f * random_float(rng) - 1.0f,
            2.0f * random_float(rng) - 1.0f,
            2.0f * random_float(rng) - 1.0f);
        if (length_squared(value) < 1.0f) {
            return value;
        }
    }
    return v3(0.0f, 0.0f, 0.0f);
}

__device__ DVec3 cosine_weighted_hemisphere(DPcgState& rng) {
    const float r1 = random_float(rng);
    const float r2 = random_float(rng);
    const float phi = 2.0f * kPi * r1;
    const float radius = sqrtf(r2);
    return normalize(v3(
        radius * cosf(phi),
        radius * sinf(phi),
        sqrtf(fmaxf(0.0f, 1.0f - r2))));
}

__device__ bool intersect_bounds(
    const DBvhNode& node,
    const DRay& ray,
    float t_min,
    float t_max) {
    constexpr float epsilon = 1.0e-12f;
    for (int axis = 0; axis < 3; ++axis) {
        const float direction = component(ray.direction, axis);
        const float origin = component(ray.origin, axis);
        const float minimum = component(node.bounds_min, axis);
        const float maximum = component(node.bounds_max, axis);
        if (fabsf(direction) < epsilon) {
            if (origin < minimum || origin > maximum) {
                return false;
            }
            continue;
        }
        const float safe_direction = copysignf(fmaxf(fabsf(direction), epsilon), direction);
        float near_t = (minimum - origin) / safe_direction;
        float far_t = (maximum - origin) / safe_direction;
        if (near_t > far_t) {
            const float temporary = near_t;
            near_t = far_t;
            far_t = temporary;
        }
        t_min = fmaxf(t_min, near_t);
        t_max = fminf(t_max, far_t);
        if (t_max < t_min) {
            return false;
        }
    }
    return true;
}

__device__ bool intersect_sphere_geometry(
    const DSphere& sphere,
    const DRay& ray,
    float t_min,
    float t_max,
    float& hit_t) {
    if (!finite(ray.origin) || !finite(ray.direction)) {
        return false;
    }
    const DVec3 oc = sub(ray.origin, sphere.center);
    const float a = length_squared(ray.direction);
    if (a <= 1.0e-24f) {
        return false;
    }
    const float half_b = dot(oc, ray.direction);
    const float c = length_squared(oc) - sphere.radius * sphere.radius;
    const float discriminant = half_b * half_b - a * c;
    if (discriminant < 0.0f) {
        return false;
    }
    const float root_delta = sqrtf(discriminant);
    float root = (-half_b - root_delta) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + root_delta) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }
    hit_t = root;
    return true;
}

__device__ bool intersect_triangle_geometry(
    const DTriangle& triangle,
    const DRay& ray,
    float t_min,
    float t_max,
    float& hit_t,
    float& hit_u,
    float& hit_v) {
    const DVec3 a = triangle.vertices[0].position;
    const DVec3 edge1 = sub(triangle.vertices[1].position, a);
    const DVec3 edge2 = sub(triangle.vertices[2].position, a);
    const DVec3 h = cross(ray.direction, edge2);
    const float determinant = dot(edge1, h);
    if (fabsf(determinant) < 1.0e-12f) {
        return false;
    }
    const float inverse = 1.0f / determinant;
    const DVec3 s = sub(ray.origin, a);
    const float u = inverse * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const DVec3 q = cross(s, edge1);
    const float v = inverse * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    const float t = inverse * dot(edge2, q);
    if (t < t_min || t > t_max) {
        return false;
    }

    hit_t = t;
    hit_u = u;
    hit_v = v;
    return true;
}

__device__ bool nearest_hit(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max,
    DCompactHit& hit) {
    bool found = false;
    float closest = t_max;
    for (int sphere_index = 0; sphere_index < scene.sphere_count; ++sphere_index) {
        float candidate_t = 0.0f;
        if (intersect_sphere_geometry(
                scene.spheres[sphere_index], ray, t_min, closest, candidate_t)) {
            found = true;
            closest = candidate_t;
            hit = DCompactHit{candidate_t, 0.0f, 0.0f, 0, sphere_index};
        }
    }

    if (scene.bvh_node_count <= 0) {
        return found;
    }
    int stack[kBvhStackCapacity];
    int stack_size = 1;
    stack[0] = 0;
    while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        const DBvhNode& node = scene.bvh_nodes[node_index];
        if (!intersect_bounds(node, ray, t_min, closest)) {
            continue;
        }
        if (node.count > 0) {
            for (int offset = 0; offset < node.count; ++offset) {
                const int primitive_index = scene.primitive_indices[node.first + offset];
                float candidate_t = 0.0f;
                float candidate_u = 0.0f;
                float candidate_v = 0.0f;
                if (intersect_triangle_geometry(
                        scene.triangles[primitive_index],
                        ray,
                        t_min,
                        closest,
                        candidate_t,
                        candidate_u,
                        candidate_v)) {
                    found = true;
                    closest = candidate_t;
                    hit = DCompactHit{
                        candidate_t,
                        candidate_u,
                        candidate_v,
                        1,
                        primitive_index};
                }
            }
            continue;
        }
        if (node.left >= 0) {
            if (stack_size >= kBvhStackCapacity) {
                return found;
            }
            stack[stack_size++] = node.left;
        }
        if (node.right >= 0) {
            if (stack_size >= kBvhStackCapacity) {
                return found;
            }
            stack[stack_size++] = node.right;
        }
    }
    return found;
}

__device__ void reconstruct_hit(
    const DScene& scene,
    const DRay& ray,
    const DCompactHit& compact,
    bool reconstruct_shading,
    DHit& hit) {
    hit.t = compact.t;
    hit.position = add(ray.origin, mul(ray.direction, compact.t));
    hit.primitive_kind = compact.primitive_kind;
    hit.primitive_index = compact.primitive_index;
    hit.barycentric_u = compact.u;
    hit.barycentric_v = compact.v;
    if (compact.primitive_kind == 0) {
        const DSphere& sphere = scene.spheres[compact.primitive_index];
        const DVec3 outward = divv(sub(hit.position, sphere.center), sphere.radius);
        set_normals(hit, ray, outward, outward);
        hit.uv = DVec2{0.0f, 0.0f};
        hit.material_id = scene.sphere_material_ids[compact.primitive_index];
        return;
    }

    const DTriangle& triangle = scene.triangles[compact.primitive_index];
    const float w = 1.0f - compact.u - compact.v;
    const DVec3 edge1 =
        sub(triangle.vertices[1].position, triangle.vertices[0].position);
    const DVec3 edge2 =
        sub(triangle.vertices[2].position, triangle.vertices[0].position);
    const DVec3 geometric = normalize(cross(edge1, edge2));
    DVec3 shading = geometric;
    if (reconstruct_shading &&
        triangle.vertices[0].has_normal &&
        triangle.vertices[1].has_normal &&
        triangle.vertices[2].has_normal) {
        shading = add(
            add(
                mul(triangle.vertices[0].normal, w),
                mul(triangle.vertices[1].normal, compact.u)),
            mul(triangle.vertices[2].normal, compact.v));
        if (!usable(shading)) {
            shading = geometric;
        } else {
            shading = normalize(shading);
            if (dot(shading, geometric) < 0.0f) {
                shading = mul(shading, -1.0f);
            }
        }
    }
    hit.uv = DVec2{
        triangle.vertices[0].uv.x * w +
            triangle.vertices[1].uv.x * compact.u +
            triangle.vertices[2].uv.x * compact.v,
        triangle.vertices[0].uv.y * w +
            triangle.vertices[1].uv.y * compact.u +
            triangle.vertices[2].uv.y * compact.v};
    set_normals(hit, ray, geometric, shading);
    hit.material_id = scene.triangle_material_ids[compact.primitive_index];
}

__device__ int wrap_index(int value, int size) {
    const int wrapped = value % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

__device__ DVec3 sample_texture(const DScene& scene, int texture_id, DVec2 uv) {
    if (texture_id < 0 || texture_id >= scene.texture_count) {
        return v3(1.0f, 0.0f, 1.0f);
    }
    const DTexture texture = scene.textures[texture_id];
    if (texture.width <= 0 || texture.height <= 0) {
        return v3(1.0f, 0.0f, 1.0f);
    }
    const float u_floor = floorf(uv.x);
    const float v_floor = floorf(uv.y);
    const float u = uv.x - u_floor;
    const float v = uv.y - v_floor;
    const float x = u * static_cast<float>(texture.width) - 0.5f;
    const float y = (1.0f - v) * static_cast<float>(texture.height) - 0.5f;
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const int ix0 = wrap_index(x0, texture.width);
    const int ix1 = wrap_index(x0 + 1, texture.width);
    const int iy0 = wrap_index(y0, texture.height);
    const int iy1 = wrap_index(y0 + 1, texture.height);
    const DVec3 c00 = scene.texels[texture.first_pixel + iy0 * texture.width + ix0];
    const DVec3 c10 = scene.texels[texture.first_pixel + iy0 * texture.width + ix1];
    const DVec3 c01 = scene.texels[texture.first_pixel + iy1 * texture.width + ix0];
    const DVec3 c11 = scene.texels[texture.first_pixel + iy1 * texture.width + ix1];
    const DVec3 top = add(mul(c00, 1.0f - tx), mul(c10, tx));
    const DVec3 bottom = add(mul(c01, 1.0f - tx), mul(c11, tx));
    return add(mul(top, 1.0f - ty), mul(bottom, ty));
}

__device__ float sample_scalar(const DScene& scene, int texture_id, DVec2 uv) {
    const DVec3 color = sample_texture(scene, texture_id, uv);
    return color.x * 0.2126f + color.y * 0.7152f + color.z * 0.0722f;
}

__device__ float material_opacity(const DScene& scene, const DMaterial& material, DVec2 uv) {
    float opacity = fminf(fmaxf(material.opacity, 0.0f), 1.0f);
    if (material.opacity_texture_id >= 0 && material.opacity_texture_id < scene.texture_count) {
        opacity *= sample_scalar(scene, material.opacity_texture_id, uv);
    }
    return fminf(fmaxf(opacity, 0.0f), 1.0f);
}

__device__ bool intersect_scene(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max,
    DHit& hit,
    bool reconstruct_shading) {
    float search_min = t_min;
    for (int layer = 0; layer < kMaxTransparentLayers; ++layer) {
        DCompactHit compact{};
        if (!nearest_hit(scene, ray, search_min, t_max, compact)) {
            return false;
        }
        DHit candidate{};
        reconstruct_hit(scene, ray, compact, false, candidate);
        if (candidate.material_id < 0 || candidate.material_id >= scene.material_count) {
            hit = candidate;
            return true;
        }
        const DMaterial material = scene.materials[candidate.material_id];
        const bool visible_side = material.two_sided || candidate.front_face;
        if (visible_side && material_opacity(scene, material, candidate.uv) >= material.alpha_cutoff) {
            if (reconstruct_shading) {
                reconstruct_hit(scene, ray, compact, true, candidate);
            }
            hit = candidate;
            return true;
        }
        const float advanced = nextafterf(candidate.t, t_max);
        if (!(advanced > search_min)) {
            return false;
        }
        search_min = advanced;
    }
    return false;
}

__device__ DVec3 bumped_normal(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit) {
    if (material.bump_texture_id < 0 || material.bump_texture_id >= scene.texture_count ||
        hit.primitive_kind != 1 || !usable(hit.shading_normal)) {
        return hit.shading_normal;
    }
    const DTexture texture = scene.textures[material.bump_texture_id];
    if (texture.width <= 0 || texture.height <= 0) {
        return hit.shading_normal;
    }
    const DTriangle& triangle = scene.triangles[hit.primitive_index];
    const DVec3 edge1 =
        sub(triangle.vertices[1].position, triangle.vertices[0].position);
    const DVec3 edge2 =
        sub(triangle.vertices[2].position, triangle.vertices[0].position);
    const float du1 = triangle.vertices[1].uv.x - triangle.vertices[0].uv.x;
    const float dv1 = triangle.vertices[1].uv.y - triangle.vertices[0].uv.y;
    const float du2 = triangle.vertices[2].uv.x - triangle.vertices[0].uv.x;
    const float dv2 = triangle.vertices[2].uv.y - triangle.vertices[0].uv.y;
    const float uv_determinant = du1 * dv2 - dv1 * du2;
    if (!isfinite(uv_determinant) || fabsf(uv_determinant) <= 1.0e-12f) {
        return hit.shading_normal;
    }
    const float uv_inverse = 1.0f / uv_determinant;
    const DVec3 raw_tangent =
        mul(sub(mul(edge1, dv2), mul(edge2, dv1)), uv_inverse);
    const DVec3 raw_bitangent =
        mul(sub(mul(edge2, du1), mul(edge1, du2)), uv_inverse);
    DVec3 tangent =
        sub(raw_tangent, mul(hit.shading_normal, dot(raw_tangent, hit.shading_normal)));
    if (!usable(tangent)) {
        return hit.shading_normal;
    }
    tangent = normalize(tangent);
    DVec3 bitangent = normalize(cross(hit.shading_normal, tangent));
    if (!usable(bitangent)) {
        return hit.shading_normal;
    }
    if (usable(raw_bitangent) && dot(bitangent, raw_bitangent) < 0.0f) {
        bitangent = mul(bitangent, -1.0f);
    }
    const float du = 1.0f / static_cast<float>(texture.width);
    const float dv = 1.0f / static_cast<float>(texture.height);
    const float left = sample_scalar(scene, material.bump_texture_id, DVec2{hit.uv.x - du, hit.uv.y});
    const float right = sample_scalar(scene, material.bump_texture_id, DVec2{hit.uv.x + du, hit.uv.y});
    const float down = sample_scalar(scene, material.bump_texture_id, DVec2{hit.uv.x, hit.uv.y - dv});
    const float up = sample_scalar(scene, material.bump_texture_id, DVec2{hit.uv.x, hit.uv.y + dv});
    const DVec3 gradient = add(
        mul(tangent, (right - left) * 0.5f),
        mul(bitangent, (up - down) * 0.5f));
    const DVec3 candidate = sub(hit.shading_normal, mul(gradient, material.bump_scale));
    if (!usable(candidate)) {
        return hit.shading_normal;
    }
    DVec3 result = normalize(candidate);
    if (dot(result, hit.geometric_normal) < 0.0f) {
        result = mul(result, -1.0f);
    }
    return result;
}

__device__ DSurface evaluate_surface(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit) {
    DSurface surface{};
    surface.base_color = material.base_color;
    if (material.diffuse_texture_id >= 0 && material.diffuse_texture_id < scene.texture_count) {
        surface.base_color = product(
            surface.base_color,
            sample_texture(scene, material.diffuse_texture_id, hit.uv));
    }
    surface.opacity = material_opacity(scene, material, hit.uv);
    surface.shading_normal = bumped_normal(scene, material, hit);
    return surface;
}

__device__ DVec3 offset_origin(DVec3 position, DVec3 normal, DVec3 direction) {
    if (!finite(normal) || length_squared(normal) <= 1.0e-24f) {
        return position;
    }
    const float scale = fmaxf(1.0f, fmaxf(fabsf(position.x), fmaxf(fabsf(position.y), fabsf(position.z))));
    const float sign = dot(direction, normal) >= 0.0f ? 1.0f : -1.0f;
    DVec3 result = add(position, mul(normal, sign * scale * (32.0f * FLT_EPSILON)));
    float* values = &result.x;
    const float normal_values[3] = {sign * normal.x, sign * normal.y, sign * normal.z};
    const float position_values[3] = {position.x, position.y, position.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (normal_values[axis] != 0.0f && values[axis] == position_values[axis]) {
            values[axis] = nextafterf(
                position_values[axis],
                normal_values[axis] > 0.0f ? FLT_MAX : -FLT_MAX);
        }
    }
    return result;
}

__device__ float reflectance(float cosine, float refraction_index) {
    float r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
    r0 *= r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

__device__ DVec3 tangent_to_world(DVec3 local, DVec3 normal) {
    const DVec3 w = normalize(normal);
    const DVec3 helper = fabsf(w.x) > 0.9f
        ? v3(0.0f, 1.0f, 0.0f)
        : v3(1.0f, 0.0f, 0.0f);
    const DVec3 v = normalize(cross(w, helper));
    const DVec3 u = cross(v, w);
    return normalize(add(add(mul(u, local.x), mul(v, local.y)), mul(w, local.z)));
}

__device__ bool scatter(
    const DRay& ray,
    const DHit& hit,
    const DMaterial& material,
    const DSurface& surface,
    DPcgState& rng,
    DVec3& attenuation,
    DRay& scattered) {
    if (material.type == static_cast<int>(MaterialType::Diffuse)) {
        const DVec3 direction = tangent_to_world(cosine_weighted_hemisphere(rng), surface.shading_normal);
        attenuation = surface.base_color;
        scattered = DRay{offset_origin(hit.position, hit.geometric_normal, direction), direction};
        return true;
    }
    if (material.type == static_cast<int>(MaterialType::Metal)) {
        DVec3 direction = reflect_vector(normalize(ray.direction), surface.shading_normal);
        if (material.roughness > 0.0f) {
            direction = add(direction, mul(random_in_unit_sphere(rng), fmaxf(0.0f, material.roughness)));
        }
        if (!usable(direction)) {
            return false;
        }
        direction = normalize(direction);
        if (dot(direction, surface.shading_normal) <= 0.0f) {
            return false;
        }
        attenuation = surface.base_color;
        scattered = DRay{offset_origin(hit.position, hit.geometric_normal, direction), direction};
        return true;
    }
    if (material.type == static_cast<int>(MaterialType::Dielectric)) {
        const float ratio = hit.front_face ? 1.0f / material.ior : material.ior;
        const DVec3 unit_direction = normalize(ray.direction);
        const float cosine = fminf(dot(mul(unit_direction, -1.0f), surface.shading_normal), 1.0f);
        DVec3 refracted{};
        const bool can_refract = refract_vector(unit_direction, surface.shading_normal, ratio, refracted);
        const bool choose_reflection = !can_refract || reflectance(cosine, ratio) > random_float(rng);
        const DVec3 direction = normalize(choose_reflection
            ? reflect_vector(unit_direction, surface.shading_normal)
            : refracted);
        attenuation = v3(1.0f, 1.0f, 1.0f);
        scattered = DRay{offset_origin(hit.position, hit.geometric_normal, direction), direction};
        return true;
    }
    return false;
}

__device__ DVec3 direct_lighting(
    const DScene& scene,
    const DHit& hit,
    const DSurface& surface) {
    DVec3 direct = v3(0.0f, 0.0f, 0.0f);
    constexpr float inverse_pi = 0.31830988618379067154f;
    for (int index = 0; index < scene.directional_light_count; ++index) {
        const DDirectionalLight light = scene.directional_lights[index];
        if (!usable(light.direction)) {
            continue;
        }
        const DVec3 light_direction = normalize(mul(light.direction, -1.0f));
        const float cosine = fmaxf(0.0f, dot(surface.shading_normal, light_direction));
        if (cosine <= 0.0f) {
            continue;
        }
        const DRay shadow{offset_origin(hit.position, hit.geometric_normal, light_direction), light_direction};
        DHit blocker{};
        if (!intersect_scene(scene, shadow, 0.0f, 1.0e30f, blocker, false)) {
            direct = add(direct, mul(product(surface.base_color, light.radiance), cosine * inverse_pi));
        }
    }
    for (int index = 0; index < scene.point_light_count; ++index) {
        const DPointLight light = scene.point_lights[index];
        const DVec3 to_light = sub(light.position, hit.position);
        const float distance_squared = length_squared(to_light);
        if (distance_squared <= 1.0e-12f) {
            continue;
        }
        const float distance = sqrtf(distance_squared);
        const DVec3 light_direction = divv(to_light, distance);
        const float cosine = fmaxf(0.0f, dot(surface.shading_normal, light_direction));
        if (cosine <= 0.0f) {
            continue;
        }
        const DRay shadow{offset_origin(hit.position, hit.geometric_normal, light_direction), light_direction};
        DHit blocker{};
        if (!intersect_scene(
                scene,
                shadow,
                0.0f,
                distance - 1.0e-7f,
                blocker,
                false)) {
            direct = add(
                direct,
                mul(product(surface.base_color, divv(light.intensity, distance_squared)), cosine * inverse_pi));
        }
    }
    return direct;
}

__device__ DVec3 trace_path(
    const DScene& scene,
    DRay ray,
    DPcgState& rng) {
    DVec3 radiance = v3(0.0f, 0.0f, 0.0f);
    DVec3 throughput = v3(1.0f, 1.0f, 1.0f);
    for (int bounce = 0; bounce < kMaxPathBounces; ++bounce) {
        DHit hit{};
        if (!intersect_scene(scene, ray, 0.0f, 1.0e30f, hit, true)) {
            radiance = add(radiance, product(throughput, scene.environment));
            break;
        }
        if (hit.material_id < 0 || hit.material_id >= scene.material_count) {
            radiance = add(radiance, product(throughput, v3(1.0f, 0.0f, 1.0f)));
            break;
        }
        const DMaterial material = scene.materials[hit.material_id];
        const DSurface surface = evaluate_surface(scene, material, hit);
        if (material.type == static_cast<int>(MaterialType::Emissive)) {
            radiance = add(radiance, product(throughput, material.emission));
            break;
        }
        DVec3 attenuation{};
        DRay scattered{};
        if (!scatter(ray, hit, material, surface, rng, attenuation, scattered)) {
            radiance = add(radiance, product(throughput, material.emission));
            break;
        }
        const DVec3 direct = material.type == static_cast<int>(MaterialType::Diffuse)
            ? direct_lighting(scene, hit, surface)
            : v3(0.0f, 0.0f, 0.0f);
        radiance = add(radiance, product(throughput, add(material.emission, direct)));
        throughput = product(throughput, attenuation);
        if (!finite(throughput) || max_component(throughput) <= 0.0f) {
            break;
        }
        if (bounce + 1 >= kRussianRouletteStartBounce) {
            const float probability = fminf(fmaxf(max_component(throughput), 0.05f), 0.95f);
            if (random_float(rng) >= probability) {
                break;
            }
            throughput = divv(throughput, probability);
        }
        ray = scattered;
    }
    return radiance;
}

__global__ void initialize_frame_kernel(
    DVec3* accumulation,
    DPcgState* random_states,
    int width,
    int height,
    unsigned long long seed_offset) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = width * height;
    if (index >= pixel_count) {
        return;
    }
    accumulation[index] = v3(0.0f, 0.0f, 0.0f);
    const int x = index % width;
    const int y = index / width;
    pcg_seed(random_states[index], pixel_seed(x, y, width, seed_offset));
}

__global__ void render_sample_kernel(
    DScene scene,
    DCamera camera,
    DVec3* accumulation,
    DVec3* resolved,
    DPcgState* random_states,
    int width,
    int height,
    int completed_samples,
    cudaSurfaceObject_t output_surface) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = width * height;
    if (index >= pixel_count) {
        return;
    }
    const int x = index % width;
    const int y = index / width;
    DPcgState rng = random_states[index];
    const float u = (static_cast<float>(x) + random_float(rng)) / static_cast<float>(width);
    const float v = 1.0f -
        (static_cast<float>(y) + random_float(rng)) / static_cast<float>(height);
    const DVec3 direction = normalize(add(
        add(
            camera.forward,
            mul(camera.right, (u - 0.5f) * camera.viewport_width)),
        mul(camera.up, (v - 0.5f) * camera.viewport_height)));
    const DVec3 sample = trace_path(scene, DRay{camera.eye, direction}, rng);
    const DVec3 sum = add(accumulation[index], sample);
    accumulation[index] = sum;
    const DVec3 color = divv(sum, static_cast<float>(completed_samples + 1));
    if (resolved != nullptr) {
        resolved[index] = color;
    }
    if (output_surface != 0) {
        surf2Dwrite(
            make_float4(color.x, color.y, color.z, 1.0f),
            output_surface,
            x * static_cast<int>(sizeof(float4)),
            y);
    }
    random_states[index] = rng;
}

__global__ void resolve_frame_kernel(
    const DVec3* accumulation,
    DVec3* resolved,
    int pixel_count,
    int sample_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    resolved[index] = sample_count > 0
        ? divv(accumulation[index], static_cast<float>(sample_count))
        : v3(0.0f, 0.0f, 0.0f);
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() {
        if (data_) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void resize(std::size_t count, CudaPathStatistics& statistics) {
        count_ = count;
        if (count <= capacity_) {
            return;
        }
        std::size_t new_capacity = std::max(count, capacity_ + capacity_ / 2);
        new_capacity = std::max<std::size_t>(new_capacity, 1);
        T* replacement = nullptr;
        check_cuda(
            cudaMalloc(
                reinterpret_cast<void**>(&replacement),
                new_capacity * sizeof(T)),
            "cudaMalloc");
        if (data_) {
            check_cuda(cudaFree(data_), "cudaFree");
        }
        data_ = replacement;
        capacity_ = new_capacity;
        ++statistics.allocation_generation;
    }

    std::uint64_t upload(
        const std::vector<T>& values,
        cudaStream_t stream,
        CudaPathStatistics& statistics) {
        resize(values.size(), statistics);
        if (!values.empty()) {
            check_cuda(
                cudaMemcpyAsync(
                    data_,
                    values.data(),
                    values.size() * sizeof(T),
                    cudaMemcpyHostToDevice,
                    stream),
                "cudaMemcpyAsync host to device");
        }
        return static_cast<std::uint64_t>(values.size() * sizeof(T));
    }

    void download(T* values, cudaStream_t stream) const {
        if (count_ > 0) {
            check_cuda(
                cudaMemcpyAsync(
                    values,
                    data_,
                    count_ * sizeof(T),
                    cudaMemcpyDeviceToHost,
                    stream),
                "cudaMemcpyAsync device to host");
        }
    }

    T* get() const { return data_; }
    std::size_t size() const { return count_; }
    std::size_t capacity() const { return capacity_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
    std::size_t capacity_ = 0;
};

template <typename T>
class PinnedHostBuffer {
public:
    ~PinnedHostBuffer() {
        if (data_) {
            cudaFreeHost(data_);
        }
    }

    PinnedHostBuffer(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;
    PinnedHostBuffer() = default;

    void resize(std::size_t count, CudaPathStatistics& statistics) {
        count_ = count;
        if (count <= capacity_) {
            return;
        }
        std::size_t new_capacity = std::max(count, capacity_ + capacity_ / 2);
        new_capacity = std::max<std::size_t>(new_capacity, 1);
        T* replacement = nullptr;
        check_cuda(
            cudaHostAlloc(
                reinterpret_cast<void**>(&replacement),
                new_capacity * sizeof(T),
                cudaHostAllocDefault),
            "cudaHostAlloc");
        if (data_) {
            check_cuda(cudaFreeHost(data_), "cudaFreeHost");
        }
        data_ = replacement;
        capacity_ = new_capacity;
        ++statistics.allocation_generation;
    }

    T* get() const { return data_; }
    std::size_t size() const { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
    std::size_t capacity_ = 0;
};

class CudaEventTimer {
public:
    CudaEventTimer() = default;

    ~CudaEventTimer() {
        if (stop_) {
            cudaEventDestroy(stop_);
        }
        if (start_) {
            cudaEventDestroy(start_);
        }
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void begin(cudaStream_t stream) {
        ensure_created();
        check_cuda(cudaEventRecord(start_, stream), "cudaEventRecord start");
        pending_ = true;
    }

    void end(cudaStream_t stream) {
        check_cuda(cudaEventRecord(stop_, stream), "cudaEventRecord stop");
    }

    bool update(float& milliseconds) {
        if (!pending_) {
            return false;
        }
        const cudaError_t query = cudaEventQuery(stop_);
        if (query == cudaErrorNotReady) {
            return false;
        }
        check_cuda(query, "cudaEventQuery");
        check_cuda(
            cudaEventElapsedTime(&milliseconds, start_, stop_),
            "cudaEventElapsedTime");
        pending_ = false;
        return true;
    }

    bool pending() const {
        return pending_;
    }

    void finish(float& milliseconds) {
        if (!pending_) {
            return;
        }
        check_cuda(cudaEventSynchronize(stop_), "cudaEventSynchronize");
        check_cuda(
            cudaEventElapsedTime(&milliseconds, start_, stop_),
            "cudaEventElapsedTime");
        pending_ = false;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    bool pending_ = false;

    void ensure_created() {
        if (start_) {
            return;
        }
        check_cuda(cudaEventCreate(&start_), "cudaEventCreate start");
        check_cuda(cudaEventCreate(&stop_), "cudaEventCreate stop");
    }
};

class CudaSceneStorage {
public:
    CudaSceneStorage(
        const Scene& scene,
        cudaStream_t stream,
        CudaPathStatistics& statistics)
        : stream_(stream), statistics_(statistics) {
        sync(scene, SceneChange::All);
    }

    DScene view() const { return view_; }

    void sync(const Scene& scene, SceneChangeSet changes) {
        upload_timer_.finish(statistics_.upload_milliseconds);
        upload_timer_.begin(stream_);
        if (has_scene_change(changes, SceneChange::Materials)) {
            materials_host_.clear();
            materials_host_.reserve(scene.materials.size());
            for (const Material& material : scene.materials) {
                materials_host_.push_back(DMaterial{
                static_cast<int>(material.type),
                to_device(material.base_color),
                to_device(material.emission),
                material.roughness,
                material.ior,
                material.opacity,
                material.alpha_cutoff,
                material.bump_scale,
                material.two_sided ? 1 : 0,
                material.diffuse_texture_id,
                material.opacity_texture_id,
                material.bump_texture_id});
            }
            statistics_.material_upload_bytes +=
                materials_.upload(materials_host_, stream_, statistics_);
        }

        if (has_scene_change(changes, SceneChange::Textures)) {
            textures_host_.clear();
            texels_host_.clear();
            textures_host_.reserve(scene.textures.size());
            std::size_t texel_count = 0;
            for (const ImageTexture& texture : scene.textures) {
                texel_count += texture.pixels().size();
            }
            texels_host_.reserve(texel_count);
            for (const ImageTexture& texture : scene.textures) {
                const int first = static_cast<int>(texels_host_.size());
                textures_host_.push_back(
                    DTexture{texture.width(), texture.height(), first});
                for (const Color& color : texture.pixels()) {
                    texels_host_.push_back(to_device(color));
                }
            }
            statistics_.texture_upload_bytes +=
                textures_.upload(textures_host_, stream_, statistics_);
            statistics_.texture_upload_bytes +=
                texels_.upload(texels_host_, stream_, statistics_);
        }

        if (has_scene_change(changes, SceneChange::Geometry)) {
            spheres_host_.clear();
            spheres_host_.reserve(scene.spheres.size());
            const bool pack_bindings =
                has_scene_change(changes, SceneChange::MaterialBindings);
            if (pack_bindings) {
                sphere_material_ids_host_.clear();
                sphere_material_ids_host_.reserve(scene.spheres.size());
            }
            for (const Sphere& sphere : scene.spheres) {
                spheres_host_.push_back(
                    DSphere{to_device(sphere.center()), sphere.radius()});
                if (pack_bindings) {
                    sphere_material_ids_host_.push_back(sphere.material_id());
                }
            }
            statistics_.geometry_upload_bytes +=
                spheres_.upload(spheres_host_, stream_, statistics_);

            triangles_host_.clear();
            triangles_host_.reserve(scene.triangles.size());
            if (pack_bindings) {
                triangle_material_ids_host_.clear();
                triangle_material_ids_host_.reserve(scene.triangles.size());
            }
            for (const Triangle& triangle : scene.triangles) {
                DTriangle packed{};
                for (int vertex_index = 0; vertex_index < 3; ++vertex_index) {
                    const TriangleVertex& vertex = triangle.vertex(vertex_index);
                    packed.vertices[vertex_index] = DVertex{
                        to_device(vertex.position),
                        to_device(vertex.uv),
                        to_device(vertex.normal),
                        vertex.has_normal ? 1 : 0};
                }
                triangles_host_.push_back(packed);
                if (pack_bindings) {
                    triangle_material_ids_host_.push_back(
                        triangle.material_id());
                }
            }
            statistics_.geometry_upload_bytes +=
                triangles_.upload(triangles_host_, stream_, statistics_);

            Bvh bvh;
            bvh.build_layout(scene.triangles);
            if (bvh.maximum_depth() + 1 >= kBvhStackCapacity) {
                throw std::runtime_error(
                    "host BVH exceeds the CUDA traversal stack capacity");
            }
            bvh_nodes_host_.clear();
            bvh_nodes_host_.reserve(bvh.nodes().size());
            for (const BvhNode& node : bvh.nodes()) {
                bvh_nodes_host_.push_back(DBvhNode{
                    to_device(node.bounds.min),
                    to_device(node.bounds.max),
                    node.left,
                    node.right,
                    node.first,
                    node.count});
            }
            statistics_.bvh_upload_bytes +=
                bvh_nodes_.upload(bvh_nodes_host_, stream_, statistics_);
            primitive_indices_host_ = bvh.primitive_indices();
            statistics_.bvh_upload_bytes +=
                primitive_indices_.upload(
                    primitive_indices_host_,
                    stream_,
                    statistics_);
        }

        if (has_scene_change(changes, SceneChange::MaterialBindings)) {
            if (!has_scene_change(changes, SceneChange::Geometry)) {
                sphere_material_ids_host_.clear();
                sphere_material_ids_host_.reserve(scene.spheres.size());
                for (const Sphere& sphere : scene.spheres) {
                    sphere_material_ids_host_.push_back(sphere.material_id());
                }
                triangle_material_ids_host_.clear();
                triangle_material_ids_host_.reserve(scene.triangles.size());
                for (const Triangle& triangle : scene.triangles) {
                    triangle_material_ids_host_.push_back(
                        triangle.material_id());
                }
            }
            statistics_.material_binding_upload_bytes +=
                sphere_material_ids_.upload(
                    sphere_material_ids_host_,
                    stream_,
                    statistics_);
            statistics_.material_binding_upload_bytes +=
                triangle_material_ids_.upload(
                    triangle_material_ids_host_,
                    stream_,
                    statistics_);
        }

        if (has_scene_change(changes, SceneChange::Lighting)) {
            point_lights_host_.clear();
            point_lights_host_.reserve(scene.point_lights.size());
            for (const PointLight& light : scene.point_lights) {
                point_lights_host_.push_back(
                    DPointLight{
                        to_device(light.position),
                        to_device(light.intensity)});
            }
            directional_lights_host_.clear();
            directional_lights_host_.reserve(scene.directional_lights.size());
            for (const DirectionalLight& light : scene.directional_lights) {
                directional_lights_host_.push_back(
                    DDirectionalLight{
                        to_device(light.direction),
                        to_device(light.radiance)});
            }
            statistics_.lighting_upload_bytes +=
                point_lights_.upload(
                    point_lights_host_,
                    stream_,
                    statistics_);
            statistics_.lighting_upload_bytes +=
                directional_lights_.upload(
                    directional_lights_host_,
                    stream_,
                    statistics_);
            statistics_.lighting_upload_bytes += sizeof(DVec3);
        }

        view_.materials = materials_.get();
        view_.material_count = static_cast<int>(materials_.size());
        view_.textures = textures_.get();
        view_.texture_count = static_cast<int>(textures_.size());
        view_.texels = texels_.get();
        view_.spheres = spheres_.get();
        view_.sphere_material_ids = sphere_material_ids_.get();
        view_.sphere_count = static_cast<int>(spheres_.size());
        view_.triangles = triangles_.get();
        view_.triangle_material_ids = triangle_material_ids_.get();
        view_.triangle_count = static_cast<int>(triangles_.size());
        view_.bvh_nodes = bvh_nodes_.get();
        view_.bvh_node_count = static_cast<int>(bvh_nodes_.size());
        view_.primitive_indices = primitive_indices_.get();
        view_.point_lights = point_lights_.get();
        view_.point_light_count = static_cast<int>(point_lights_.size());
        view_.directional_lights = directional_lights_.get();
        view_.directional_light_count = static_cast<int>(directional_lights_.size());
        view_.environment = to_device(scene.environment);
        upload_timer_.end(stream_);
    }

    void update_timing() {
        upload_timer_.update(statistics_.upload_milliseconds);
    }

private:
    cudaStream_t stream_ = nullptr;
    CudaPathStatistics& statistics_;
    CudaEventTimer upload_timer_;
    DeviceBuffer<DMaterial> materials_;
    DeviceBuffer<DTexture> textures_;
    DeviceBuffer<DVec3> texels_;
    DeviceBuffer<DSphere> spheres_;
    DeviceBuffer<int> sphere_material_ids_;
    DeviceBuffer<DTriangle> triangles_;
    DeviceBuffer<int> triangle_material_ids_;
    DeviceBuffer<DBvhNode> bvh_nodes_;
    DeviceBuffer<int> primitive_indices_;
    DeviceBuffer<DPointLight> point_lights_;
    DeviceBuffer<DDirectionalLight> directional_lights_;
    std::vector<DMaterial> materials_host_;
    std::vector<DTexture> textures_host_;
    std::vector<DVec3> texels_host_;
    std::vector<DSphere> spheres_host_;
    std::vector<int> sphere_material_ids_host_;
    std::vector<DTriangle> triangles_host_;
    std::vector<int> triangle_material_ids_host_;
    std::vector<DBvhNode> bvh_nodes_host_;
    std::vector<int> primitive_indices_host_;
    std::vector<DPointLight> point_lights_host_;
    std::vector<DDirectionalLight> directional_lights_host_;
    DScene view_{};
};

class CudaFrameStorage {
public:
    explicit CudaFrameStorage(CudaPathStatistics& statistics)
        : statistics_(statistics) {
        check_cuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    ~CudaFrameStorage() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
        }
    }

    void reset(int width, int height, std::uint64_t seed_offset) {
        if (width <= 0 || height <= 0) {
            throw std::invalid_argument("CUDA framebuffer dimensions must be positive");
        }
        const std::size_t count =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        if (count > static_cast<std::size_t>(INT_MAX)) {
            throw std::invalid_argument("CUDA framebuffer contains too many pixels");
        }
        width_ = width;
        height_ = height;
        reset_timer_.update(statistics_.reset_milliseconds);
        const bool record_timing = !reset_timer_.pending();
        if (record_timing) {
            reset_timer_.begin(stream_);
        }
        accumulation_.resize(count, statistics_);
        random_states_.resize(count, statistics_);
        const int block_count = static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        initialize_frame_kernel<<<block_count, kThreadsPerBlock, 0, stream_>>>(
            accumulation_.get(),
            random_states_.get(),
            width,
            height,
            seed_offset);
        check_cuda(cudaGetLastError(), "initialize_frame_kernel launch");
        if (record_timing) {
            reset_timer_.end(stream_);
        }
        samples_ = 0;
    }

    void render_sample(
        const DScene& scene,
        const Camera& camera,
        cudaSurfaceObject_t output_surface = 0) {
        const DCamera packed_camera{
            to_device(camera.eye()),
            to_device(camera.forward()),
            to_device(camera.right()),
            to_device(camera.up()),
            camera.viewport_width(),
            camera.viewport_height()};
        const std::size_t count = accumulation_.size();
        DVec3* resolved = nullptr;
        if (output_surface == 0) {
            resolved_.resize(count, statistics_);
            resolved = resolved_.get();
        }
        trace_timer_.update(statistics_.trace_milliseconds);
        const bool record_timing = !trace_timer_.pending();
        if (record_timing) {
            trace_timer_.begin(stream_);
        }
        const int block_count = static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        render_sample_kernel<<<block_count, kThreadsPerBlock, 0, stream_>>>(
            scene,
            packed_camera,
            accumulation_.get(),
            resolved,
            random_states_.get(),
            width_,
            height_,
            samples_,
            output_surface);
        check_cuda(cudaGetLastError(), "render_sample_kernel launch");
        if (record_timing) {
            trace_timer_.end(stream_);
        }
        ++samples_;
    }

    void synchronize_and_check_errors() {
        check_cuda(cudaStreamSynchronize(stream_), "path rendering stream synchronize");
        update_timings();
    }

    std::vector<Color> download_pixels(bool use_pinned_staging = true) {
        const std::size_t count = accumulation_.size();
        resolved_.resize(count, statistics_);
        std::vector<DVec3> pageable_pixels;
        DVec3* packed_pixels = nullptr;
        if (use_pinned_staging) {
            pinned_pixels_.resize(count, statistics_);
            packed_pixels = pinned_pixels_.get();
        } else {
            pageable_pixels.resize(count);
            packed_pixels = pageable_pixels.data();
        }
        const int block_count =
            static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        resolve_frame_kernel<<<block_count, kThreadsPerBlock, 0, stream_>>>(
            accumulation_.get(),
            resolved_.get(),
            static_cast<int>(count),
            samples_);
        check_cuda(cudaGetLastError(), "resolve_frame_kernel launch");
        resolved_.download(packed_pixels, stream_);
        ++statistics_.framebuffer_downloads;
        synchronize_and_check_errors();
        std::vector<Color> pixels;
        pixels.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const DVec3 value = packed_pixels[index];
            pixels.emplace_back(value.x, value.y, value.z);
        }
        return pixels;
    }

    cudaStream_t stream() const { return stream_; }
    int width() const { return width_; }
    int height() const { return height_; }
    int samples() const { return samples_; }

    void update_timings() {
        reset_timer_.update(statistics_.reset_milliseconds);
        trace_timer_.update(statistics_.trace_milliseconds);
    }

private:
    CudaPathStatistics& statistics_;
    cudaStream_t stream_ = nullptr;
    CudaEventTimer reset_timer_;
    CudaEventTimer trace_timer_;
    DeviceBuffer<DVec3> accumulation_;
    DeviceBuffer<DVec3> resolved_;
    DeviceBuffer<DPcgState> random_states_;
    PinnedHostBuffer<DVec3> pinned_pixels_;
    int width_ = 0;
    int height_ = 0;
    int samples_ = 0;
};

}  // namespace

bool cuda_path_backend_compiled() {
    return true;
}

bool cuda_path_backend_available(std::string* reason) {
    int device_count = 0;
    const cudaError_t result = cudaGetDeviceCount(&device_count);
    if (result != cudaSuccess) {
        if (reason) {
            *reason = cudaGetErrorString(result);
        }
        cudaGetLastError();
        return false;
    }
    if (device_count <= 0) {
        if (reason) {
            *reason = "no CUDA-capable device was found";
        }
        return false;
    }
    if (reason) {
        reason->clear();
    }
    return true;
}

RenderResult render_cuda_path(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings) {
    Timer timer;
    CudaPathStatistics statistics;
    CudaFrameStorage frame(statistics);
    CudaSceneStorage device_scene(scene, frame.stream(), statistics);
    frame.reset(settings.width, settings.height, settings.path.sample_seed_offset);
    const int sample_count = std::max(1, settings.path.samples_per_pixel);
    for (int sample = 0; sample < sample_count; ++sample) {
        frame.render_sample(device_scene.view(), camera);
    }
    Image image(settings.width, settings.height);
    image.set_pixels(frame.download_pixels(false));
    return RenderResult{std::move(image), timer.elapsed_seconds(), ExecutionBackend::Cuda};
}

class CudaPathInteractiveRenderer::Impl {
public:
    Impl()
        : frame_(statistics_) {}

    void reset(const Scene& scene, const RenderSettings& settings) {
        if (!scene_) {
            scene_ = std::make_unique<CudaSceneStorage>(
                scene,
                frame_.stream(),
                statistics_);
        } else {
            scene_->sync(scene, SceneChange::All);
        }
        frame_.reset(settings.width, settings.height, settings.path.sample_seed_offset);
    }

    void render_next_frame(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target) {
        render_next_frame_to_surface(scene, camera, settings, frame_state, 0);
        download_current_frame(target);
    }

    void render_next_frame_to_surface(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        CudaSurfaceHandle surface) {
        if (!scene_) {
            scene_ = std::make_unique<CudaSceneStorage>(
                scene,
                frame_.stream(),
                statistics_);
        } else if (frame_state.scene_changes != SceneChange::None) {
            scene_->sync(scene, frame_state.scene_changes);
        }
        const bool reset_accumulation =
            frame_.width() != settings.width ||
            frame_.height() != settings.height ||
            frame_state.camera_changed ||
            frame_state.scene_changes != SceneChange::None ||
            frame_state.framebuffer_resized ||
            frame_state.reset_requested;
        if (reset_accumulation) {
            frame_.reset(settings.width, settings.height, settings.path.sample_seed_offset);
        }
        frame_.render_sample(
            scene_->view(),
            camera,
            static_cast<cudaSurfaceObject_t>(surface));
        scene_->update_timing();
    }

    void download_current_frame(Framebuffer& target) {
        if (target.width() != frame_.width() || target.height() != frame_.height()) {
            target.resize(frame_.width(), frame_.height());
        }
        target.set_pixels(frame_.download_pixels());
    }

    int accumulated_samples() const {
        return frame_.samples();
    }

    CudaStreamHandle stream_handle() const {
        return reinterpret_cast<CudaStreamHandle>(frame_.stream());
    }

    const CudaPathStatistics& statistics() {
        frame_.update_timings();
        if (scene_) {
            scene_->update_timing();
        }
        return statistics_;
    }

    void set_presentation_state(bool interop_active, bool fallback_active) {
        statistics_.interop_active = interop_active;
        statistics_.fallback_active = fallback_active;
    }

private:
    CudaPathStatistics statistics_;
    CudaFrameStorage frame_;
    std::unique_ptr<CudaSceneStorage> scene_;
};

CudaPathInteractiveRenderer::CudaPathInteractiveRenderer()
    : impl_(std::make_unique<Impl>()) {}
CudaPathInteractiveRenderer::~CudaPathInteractiveRenderer() = default;
CudaPathInteractiveRenderer::CudaPathInteractiveRenderer(CudaPathInteractiveRenderer&&) noexcept = default;
CudaPathInteractiveRenderer& CudaPathInteractiveRenderer::operator=(CudaPathInteractiveRenderer&&) noexcept = default;

void CudaPathInteractiveRenderer::reset(const Scene& scene, const RenderSettings& settings) {
    impl_->reset(scene, settings);
}

void CudaPathInteractiveRenderer::render_next_frame(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    Framebuffer& target) {
    impl_->render_next_frame(scene, camera, settings, frame_state, target);
}

void CudaPathInteractiveRenderer::render_next_frame_to_surface(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    CudaSurfaceHandle surface) {
    if (surface == 0) {
        throw std::invalid_argument("CUDA surface output requires a non-zero surface handle");
    }
    impl_->render_next_frame_to_surface(scene, camera, settings, frame_state, surface);
}

void CudaPathInteractiveRenderer::download_current_frame(Framebuffer& target) {
    impl_->download_current_frame(target);
}

int CudaPathInteractiveRenderer::accumulated_samples() const {
    return impl_->accumulated_samples();
}

CudaStreamHandle CudaPathInteractiveRenderer::stream_handle() const {
    return impl_->stream_handle();
}

const CudaPathStatistics& CudaPathInteractiveRenderer::statistics() const {
    return impl_->statistics();
}

void CudaPathInteractiveRenderer::set_presentation_state(
    bool interop_active,
    bool fallback_active) {
    impl_->set_presentation_state(interop_active, fallback_active);
}

}  // namespace renderer
