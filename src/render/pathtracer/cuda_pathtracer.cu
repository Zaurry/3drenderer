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

constexpr int kThreadsPerBlock = 128;
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

struct DEmissiveLight {
    int primitive_kind;
    int primitive_index;
    int material_id;
    float area;
    float selection_pdf;
    float cumulative_probability;
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
    const DEmissiveLight* emissive_lights;
    int emissive_light_count;
    const int* sphere_light_indices;
    const int* triangle_light_indices;
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

struct DPathState {
    DRay ray;
    DVec3 throughput;
    float previous_bsdf_pdf;
    int previous_was_delta;
    int pixel_index;
};

struct DWavefrontHit {
    DCompactHit hit;
    int found;
};

struct DShadingRecord {
    DVec3 position;
    DVec3 geometric_normal;
    DVec3 shading_normal;
    DVec3 base_color;
    DVec3 throughput;
    int pixel_index;
    int valid;
};

struct DShadowTask {
    DRay ray;
    DVec3 contribution;
    float t_max;
    int pixel_index;
};

struct DFrameParameters {
    DScene scene;
    DCamera camera;
    DVec3* accumulation;
    DVec3* sample_radiance;
    DPcgState* random_states;
    int width;
    int height;
    int completed_samples;
    int batch_sample_count;
    int* batch_sample_index;
    int path_capacity;
    int* error_code;
    cudaSurfaceObject_t output_surface;
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

__device__ float power_heuristic(float first_pdf, float second_pdf) {
    if (!(first_pdf > 0.0f) || !isfinite(first_pdf)) {
        return 0.0f;
    }
    if (!(second_pdf > 0.0f) || !isfinite(second_pdf)) {
        return 1.0f;
    }
    const float first_squared = first_pdf * first_pdf;
    const float second_squared = second_pdf * second_pdf;
    return first_squared / (first_squared + second_squared);
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

__device__ bool intersect_scene_compact(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max,
    DCompactHit& hit) {
    float search_min = t_min;
    for (int layer = 0; layer < kMaxTransparentLayers; ++layer) {
        DCompactHit compact{};
        if (!nearest_hit(scene, ray, search_min, t_max, compact)) {
            return false;
        }
        DHit candidate{};
        reconstruct_hit(scene, ray, compact, false, candidate);
        if (candidate.material_id < 0 || candidate.material_id >= scene.material_count) {
            hit = compact;
            return true;
        }
        const DMaterial material = scene.materials[candidate.material_id];
        const bool visible_side = material.two_sided || candidate.front_face;
        if (visible_side && material_opacity(scene, material, candidate.uv) >= material.alpha_cutoff) {
            hit = compact;
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

__device__ bool occluded_scene(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max) {
    DCompactHit compact{};
    return intersect_scene_compact(scene, ray, t_min, t_max, compact);
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
    DRay& scattered,
    float& bsdf_pdf,
    int& was_delta) {
    if (material.type == static_cast<int>(MaterialType::Diffuse)) {
        const DVec3 direction = tangent_to_world(cosine_weighted_hemisphere(rng), surface.shading_normal);
        attenuation = surface.base_color;
        scattered = DRay{offset_origin(hit.position, hit.geometric_normal, direction), direction};
        bsdf_pdf = fmaxf(0.0f, dot(surface.shading_normal, direction)) / kPi;
        was_delta = 0;
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
        bsdf_pdf = 0.0f;
        was_delta = 1;
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
        bsdf_pdf = 0.0f;
        was_delta = 1;
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
        if (!occluded_scene(scene, shadow, 0.0f, 1.0e30f)) {
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
        if (!occluded_scene(
                scene,
                shadow,
                0.0f,
                distance - 1.0e-7f)) {
            direct = add(
                direct,
                mul(product(surface.base_color, divv(light.intensity, distance_squared)), cosine * inverse_pi));
        }
    }
    return direct;
}

__device__ int emissive_light_index(const DScene& scene, const DHit& hit) {
    if (hit.primitive_kind == 0) {
        return hit.primitive_index >= 0 &&
                hit.primitive_index < scene.sphere_count &&
                scene.sphere_light_indices != nullptr
            ? scene.sphere_light_indices[hit.primitive_index]
            : -1;
    }
    return hit.primitive_kind == 1 &&
            hit.primitive_index >= 0 &&
            hit.primitive_index < scene.triangle_count &&
            scene.triangle_light_indices != nullptr
        ? scene.triangle_light_indices[hit.primitive_index]
        : -1;
}

__device__ float emissive_light_pdf_for_hit(
    const DScene& scene,
    DVec3 previous_position,
    const DHit& hit) {
    const int light_index = emissive_light_index(scene, hit);
    if (light_index < 0 || light_index >= scene.emissive_light_count) {
        return 0.0f;
    }
    const DEmissiveLight light = scene.emissive_lights[light_index];
    if (!(light.area > 0.0f) || !(light.selection_pdf > 0.0f)) {
        return 0.0f;
    }
    const DVec3 to_previous = sub(previous_position, hit.position);
    const float distance_squared = length_squared(to_previous);
    if (!(distance_squared > 1.0e-12f)) {
        return 0.0f;
    }
    const DVec3 direction_to_previous = divv(to_previous, sqrtf(distance_squared));
    const float light_cosine = fmaxf(
        0.0f,
        dot(hit.geometric_normal, direction_to_previous));
    if (!(light_cosine > 0.0f)) {
        return 0.0f;
    }
    return light.selection_pdf * distance_squared / (light.area * light_cosine);
}

__device__ bool sample_emissive_shadow_task(
    const DScene& scene,
    const DHit& hit,
    const DSurface& surface,
    DVec3 throughput,
    int pixel_index,
    DPcgState& rng,
    DShadowTask& task) {
    if (scene.emissive_light_count <= 0 || scene.emissive_lights == nullptr) {
        return false;
    }

    const float selection_sample = random_float(rng);
    int low = 0;
    int high = scene.emissive_light_count - 1;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (selection_sample <=
            scene.emissive_lights[middle].cumulative_probability) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    const DEmissiveLight light = scene.emissive_lights[low];
    if (light.material_id < 0 || light.material_id >= scene.material_count ||
        !(light.area > 0.0f) || !(light.selection_pdf > 0.0f)) {
        return false;
    }

    DVec3 light_position{};
    DVec3 outward_normal{};
    DVec2 light_uv{0.0f, 0.0f};
    if (light.primitive_kind == 0) {
        if (light.primitive_index < 0 ||
            light.primitive_index >= scene.sphere_count) {
            return false;
        }
        const DSphere sphere = scene.spheres[light.primitive_index];
        const float z = 1.0f - 2.0f * random_float(rng);
        const float radial = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        const float phi = 2.0f * kPi * random_float(rng);
        outward_normal = v3(radial * cosf(phi), radial * sinf(phi), z);
        light_position = add(sphere.center, mul(outward_normal, sphere.radius));
    } else if (light.primitive_kind == 1) {
        if (light.primitive_index < 0 ||
            light.primitive_index >= scene.triangle_count) {
            return false;
        }
        const DTriangle triangle = scene.triangles[light.primitive_index];
        const float root = sqrtf(random_float(rng));
        const float w0 = 1.0f - root;
        const float w1 = root * (1.0f - random_float(rng));
        const float w2 = 1.0f - w0 - w1;
        light_position = add(
            add(
                mul(triangle.vertices[0].position, w0),
                mul(triangle.vertices[1].position, w1)),
            mul(triangle.vertices[2].position, w2));
        outward_normal = normalize(cross(
            sub(
                triangle.vertices[1].position,
                triangle.vertices[0].position),
            sub(
                triangle.vertices[2].position,
                triangle.vertices[0].position)));
        light_uv = DVec2{
            triangle.vertices[0].uv.x * w0 +
                triangle.vertices[1].uv.x * w1 +
                triangle.vertices[2].uv.x * w2,
            triangle.vertices[0].uv.y * w0 +
                triangle.vertices[1].uv.y * w1 +
                triangle.vertices[2].uv.y * w2};
    } else {
        return false;
    }

    const DMaterial light_material = scene.materials[light.material_id];
    if (material_opacity(scene, light_material, light_uv) <
        light_material.alpha_cutoff) {
        return false;
    }

    const DVec3 to_light = sub(light_position, hit.position);
    const float distance_squared = length_squared(to_light);
    if (!(distance_squared > 1.0e-12f)) {
        return false;
    }
    const float distance = sqrtf(distance_squared);
    const DVec3 light_direction = divv(to_light, distance);
    const float surface_cosine = fmaxf(
        0.0f,
        dot(surface.shading_normal, light_direction));
    const float raw_light_cosine = dot(
        outward_normal,
        mul(light_direction, -1.0f));
    const float light_cosine = light_material.two_sided
        ? fabsf(raw_light_cosine)
        : fmaxf(0.0f, raw_light_cosine);
    if (!(surface_cosine > 0.0f) || !(light_cosine > 0.0f)) {
        return false;
    }

    const float light_pdf =
        light.selection_pdf * distance_squared / (light.area * light_cosine);
    const float bsdf_pdf = surface_cosine / kPi;
    if (!(light_pdf > 0.0f) || !isfinite(light_pdf)) {
        return false;
    }
    const float mis_weight = power_heuristic(light_pdf, bsdf_pdf);
    const DVec3 contribution = mul(
        product(
            product(throughput, surface.base_color),
            light_material.emission),
        surface_cosine * mis_weight / (kPi * light_pdf));
    if (!finite(contribution) || max_component(contribution) <= 0.0f) {
        return false;
    }

    const DVec3 shadow_origin = offset_origin(
        hit.position,
        hit.geometric_normal,
        light_direction);
    const DVec3 shadow_vector = sub(light_position, shadow_origin);
    const float shadow_distance_squared = length_squared(shadow_vector);
    if (!(shadow_distance_squared > 1.0e-12f)) {
        return false;
    }
    const float shadow_distance = sqrtf(shadow_distance_squared);
    task = DShadowTask{
        DRay{shadow_origin, divv(shadow_vector, shadow_distance)},
        contribution,
        shadow_distance * (1.0f - 1.0e-5f),
        pixel_index};
    return task.t_max > 0.0f;
}

__global__ void initialize_frame_kernel(
    DVec3* accumulation,
    DPcgState* random_states,
    int width,
    int height,
    unsigned long long seed_offset,
    int* error_code) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = width * height;
    if (index >= pixel_count) {
        return;
    }
    accumulation[index] = v3(0.0f, 0.0f, 0.0f);
    const int x = index % width;
    const int y = index / width;
    pcg_seed(random_states[index], pixel_seed(x, y, width, seed_offset));
    if (index == 0) {
        *error_code = 0;
    }
}

__global__ void prepare_sample_kernel(
    const DFrameParameters* parameters,
    DPathState* primary_paths,
    int* primary_count,
    int* secondary_count,
    int* bounce_index,
    cudaGraphConditionalHandle wavefront_handle,
    cudaGraphConditionalHandle emissive_handle) {
    const DFrameParameters& frame = *parameters;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = frame.width * frame.height;
    if (index >= pixel_count) {
        return;
    }
    const int x = index % frame.width;
    const int y = index / frame.width;
    DPcgState rng = frame.random_states[index];
    const float u =
        (static_cast<float>(x) + random_float(rng)) /
        static_cast<float>(frame.width);
    const float v = 1.0f -
        (static_cast<float>(y) + random_float(rng)) /
            static_cast<float>(frame.height);
    const DVec3 direction = normalize(add(
        add(
            frame.camera.forward,
            mul(
                frame.camera.right,
                (u - 0.5f) * frame.camera.viewport_width)),
        mul(
            frame.camera.up,
            (v - 0.5f) * frame.camera.viewport_height)));
    primary_paths[index] = DPathState{
        DRay{frame.camera.eye, direction},
        v3(1.0f, 1.0f, 1.0f),
        0.0f,
        1,
        index};
    frame.sample_radiance[index] = v3(0.0f, 0.0f, 0.0f);
    frame.random_states[index] = rng;
    if (index == 0) {
        *primary_count = pixel_count;
        *secondary_count = 0;
        *bounce_index = 0;
        if (wavefront_handle != 0) {
            cudaGraphSetConditional(wavefront_handle, 1U);
        }
        if (emissive_handle != 0) {
            cudaGraphSetConditional(
                emissive_handle,
                frame.scene.emissive_light_count > 0 ? 1U : 0U);
        }
    }
}

__global__ void intersect_wavefront_kernel(
    const DFrameParameters* parameters,
    const DPathState* first_paths,
    const DPathState* second_paths,
    int* path_counts,
    const int* bounce_index,
    DWavefrontHit* hits) {
    const DFrameParameters& frame = *parameters;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int active_index = *bounce_index & 1;
    const DPathState* active_paths =
        active_index == 0 ? first_paths : second_paths;
    int count = path_counts[active_index];
    if (index == 0) {
        path_counts[1 - active_index] = 0;
    }
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = max(0, min(count, frame.path_capacity));
    }
    if (index >= count) {
        return;
    }
    DCompactHit hit{};
    const bool found = intersect_scene_compact(
        frame.scene,
        active_paths[index].ray,
        0.0f,
        1.0e30f,
        hit);
    hits[index] = DWavefrontHit{hit, found ? 1 : 0};
}

__global__ void shade_wavefront_kernel(
    const DFrameParameters* parameters,
    DPathState* first_paths,
    DPathState* second_paths,
    int* path_counts,
    const int* bounce_index,
    const DWavefrontHit* hits,
    DShadingRecord* shading_records) {
    const DFrameParameters& frame = *parameters;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int bounce = *bounce_index;
    const int active_index = bounce & 1;
    const int next_index = 1 - active_index;
    const DPathState* active_paths =
        active_index == 0 ? first_paths : second_paths;
    DPathState* next_paths =
        active_index == 0
        ? second_paths
        : first_paths;
    int count = path_counts[active_index];
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = max(0, min(count, frame.path_capacity));
    }
    if (index >= count) {
        return;
    }

    const DPathState path = active_paths[index];
    const DWavefrontHit wavefront_hit = hits[index];
    shading_records[index].valid = 0;
    if (!wavefront_hit.found) {
        frame.sample_radiance[path.pixel_index] = add(
            frame.sample_radiance[path.pixel_index],
            product(path.throughput, frame.scene.environment));
        return;
    }

    DHit hit{};
    reconstruct_hit(
        frame.scene,
        path.ray,
        wavefront_hit.hit,
        true,
        hit);
    if (hit.material_id < 0 || hit.material_id >= frame.scene.material_count) {
        frame.sample_radiance[path.pixel_index] = add(
            frame.sample_radiance[path.pixel_index],
            product(path.throughput, v3(1.0f, 0.0f, 1.0f)));
        return;
    }
    const DMaterial material = frame.scene.materials[hit.material_id];
    const DSurface surface = evaluate_surface(frame.scene, material, hit);

    if (max_component(material.emission) > 0.0f) {
        float emission_weight = 1.0f;
        if (!path.previous_was_delta && path.previous_bsdf_pdf > 0.0f) {
            const float light_pdf = emissive_light_pdf_for_hit(
                frame.scene,
                path.ray.origin,
                hit);
            emission_weight = power_heuristic(
                path.previous_bsdf_pdf,
                light_pdf);
        }
        frame.sample_radiance[path.pixel_index] = add(
            frame.sample_radiance[path.pixel_index],
            mul(
                product(path.throughput, material.emission),
                emission_weight));
    }
    if (material.type == static_cast<int>(MaterialType::Emissive)) {
        return;
    }

    DPcgState rng = frame.random_states[path.pixel_index];
    if (material.type == static_cast<int>(MaterialType::Diffuse)) {
        shading_records[index] = DShadingRecord{
            hit.position,
            hit.geometric_normal,
            surface.shading_normal,
            surface.base_color,
            path.throughput,
            path.pixel_index,
            1};
    }

    DVec3 attenuation{};
    DRay scattered{};
    float bsdf_pdf = 0.0f;
    int was_delta = 1;
    if (!scatter(
            path.ray,
            hit,
            material,
            surface,
            rng,
            attenuation,
            scattered,
            bsdf_pdf,
            was_delta)) {
        frame.random_states[path.pixel_index] = rng;
        return;
    }

    DVec3 throughput = product(path.throughput, attenuation);
    if (!finite(throughput) || max_component(throughput) <= 0.0f ||
        bounce + 1 >= kMaxPathBounces) {
        frame.random_states[path.pixel_index] = rng;
        return;
    }
    if (bounce + 1 >= kRussianRouletteStartBounce) {
        const float probability = fminf(
            fmaxf(max_component(throughput), 0.05f),
            0.95f);
        if (random_float(rng) >= probability) {
            frame.random_states[path.pixel_index] = rng;
            return;
        }
        throughput = divv(throughput, probability);
    }

    const unsigned int active_mask = __activemask();
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int leader = __ffs(active_mask) - 1;
    int output_base = 0;
    if (lane == leader) {
        output_base = atomicAdd(
            path_counts + next_index,
            __popc(active_mask));
    }
    output_base = __shfl_sync(
        active_mask,
        output_base,
        leader);
    const unsigned int lower_lanes =
        lane == 0 ? 0U : ((1U << lane) - 1U);
    const int output_index =
        output_base + __popc(active_mask & lower_lanes);
    if (output_index < frame.path_capacity) {
        next_paths[output_index] = DPathState{
            scattered,
            throughput,
            bsdf_pdf,
            was_delta,
            path.pixel_index};
    } else {
        atomicCAS(frame.error_code, 0, 1);
    }
    frame.random_states[path.pixel_index] = rng;
}

__global__ void sample_emissive_lights_kernel(
    const DFrameParameters* parameters,
    const DShadingRecord* shading_records,
    DShadowTask* shadow_tasks,
    const int* path_counts,
    const int* bounce_index) {
    const DFrameParameters& frame = *parameters;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = path_counts[*bounce_index & 1];
    count = max(0, min(count, frame.path_capacity));
    if (index >= count) {
        return;
    }
    shadow_tasks[index].t_max = 0.0f;
    if (frame.scene.emissive_light_count <= 0) {
        return;
    }
    const DShadingRecord record = shading_records[index];
    if (!record.valid) {
        return;
    }
    DHit hit{};
    hit.position = record.position;
    hit.geometric_normal = record.geometric_normal;
    DSurface surface{};
    surface.base_color = record.base_color;
    surface.shading_normal = record.shading_normal;
    DPcgState rng = frame.random_states[record.pixel_index];
    DShadowTask shadow_task{};
    if (sample_emissive_shadow_task(
            frame.scene,
            hit,
            surface,
            record.throughput,
            record.pixel_index,
            rng,
            shadow_task)) {
        shadow_tasks[index] = shadow_task;
    }
    frame.random_states[record.pixel_index] = rng;
}

__global__ void direct_visibility_kernel(
    const DFrameParameters* parameters,
    const DShadingRecord* shading_records,
    const DShadowTask* shadow_tasks,
    const int* path_counts,
    const int* bounce_index) {
    const DFrameParameters& frame = *parameters;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    int count = path_counts[*bounce_index & 1];
    count = max(0, min(count, frame.path_capacity));
    if (index >= count) {
        return;
    }
    const DShadingRecord record = shading_records[index];
    if (!record.valid) {
        return;
    }
    DHit hit{};
    hit.position = record.position;
    hit.geometric_normal = record.geometric_normal;
    DSurface surface{};
    surface.base_color = record.base_color;
    surface.shading_normal = record.shading_normal;
    DVec3 contribution = product(
        record.throughput,
        direct_lighting(frame.scene, hit, surface));
    if (frame.scene.emissive_light_count > 0) {
        const DShadowTask task = shadow_tasks[index];
        if (task.t_max > 0.0f) {
            if (!occluded_scene(
                    frame.scene,
                    task.ray,
                    0.0f,
                    task.t_max)) {
                contribution = add(contribution, task.contribution);
            }
        }
    }
    frame.sample_radiance[record.pixel_index] = add(
        frame.sample_radiance[record.pixel_index],
        contribution);
}

__global__ void advance_wavefront_kernel(
    const DFrameParameters* parameters,
    const int* path_counts,
    int* bounce_index,
    cudaGraphConditionalHandle conditional_handle) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const DFrameParameters& frame = *parameters;
    const int next_bounce = *bounce_index + 1;
    *bounce_index = next_bounce;
    int count = next_bounce < kMaxPathBounces
        ? path_counts[next_bounce & 1]
        : 0;
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = 0;
    }
    if (conditional_handle != 0) {
        cudaGraphSetConditional(
            conditional_handle,
            count > 0 && next_bounce < kMaxPathBounces ? 1U : 0U);
    }
}

__global__ void finalize_sample_kernel(
    const DFrameParameters* parameters) {
    const DFrameParameters& frame = *parameters;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int pixel_count = frame.width * frame.height;
    if (index >= pixel_count) {
        return;
    }
    const DVec3 sum = add(
        frame.accumulation[index],
        frame.sample_radiance[index]);
    frame.accumulation[index] = sum;
    if (frame.output_surface != 0) {
        const int completed_samples =
            frame.completed_samples + *frame.batch_sample_index;
        const DVec3 color = divv(
            sum,
            static_cast<float>(completed_samples + 1));
        const int x = index % frame.width;
        const int y = index / frame.width;
        surf2Dwrite(
            make_float4(color.x, color.y, color.z, 1.0f),
            frame.output_surface,
            x * static_cast<int>(sizeof(float4)),
            y);
    }
}

__global__ void prepare_sample_batch_kernel(int* batch_sample_index) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *batch_sample_index = 0;
    }
}

__global__ void advance_sample_batch_kernel(
    const DFrameParameters* parameters,
    cudaGraphConditionalHandle batch_handle) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const DFrameParameters& frame = *parameters;
    const int next_sample = *frame.batch_sample_index + 1;
    *frame.batch_sample_index = next_sample;
    if (batch_handle != 0) {
        cudaGraphSetConditional(
            batch_handle,
            next_sample < frame.batch_sample_count ? 1U : 0U);
    }
}

__global__ void update_frame_parameters_kernel(
    DFrameParameters* destination,
    DFrameParameters parameters) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *destination = parameters;
    }
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

        if (has_scene_change(changes, SceneChange::Geometry) ||
            has_scene_change(changes, SceneChange::MaterialBindings) ||
            has_scene_change(changes, SceneChange::Materials)) {
            rebuild_emissive_lights(scene);
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
        view_.emissive_lights = emissive_lights_.get();
        view_.emissive_light_count = static_cast<int>(emissive_lights_.size());
        view_.sphere_light_indices = sphere_light_indices_.get();
        view_.triangle_light_indices = triangle_light_indices_.get();
        view_.environment = to_device(scene.environment);
        upload_timer_.end(stream_);
    }

    void update_timing() {
        upload_timer_.update(statistics_.upload_milliseconds);
    }

private:
    void rebuild_emissive_lights(const Scene& scene) {
        emissive_lights_host_.clear();
        const auto material_weight = [](const Material& material) {
            if (!material.emission.allFinite() ||
                (material.emission.array() < 0.0f).any()) {
                return 0.0f;
            }
            const float energy =
                material.emission.x() * 0.2126f +
                material.emission.y() * 0.7152f +
                material.emission.z() * 0.0722f;
            if (!(energy > 0.0f) || !std::isfinite(energy)) {
                return 0.0f;
            }
            if (material.opacity_texture_id < 0 &&
                material.opacity < material.alpha_cutoff) {
                return 0.0f;
            }
            return energy * (material.two_sided ? 2.0f : 1.0f);
        };

        std::vector<float> material_weights(scene.materials.size(), 0.0f);
        bool has_emissive_material = false;
        for (std::size_t index = 0; index < scene.materials.size(); ++index) {
            material_weights[index] = material_weight(scene.materials[index]);
            has_emissive_material =
                has_emissive_material || material_weights[index] > 0.0f;
        }
        if (!has_emissive_material) {
            sphere_light_indices_host_.clear();
            triangle_light_indices_host_.clear();
            statistics_.lighting_upload_bytes += emissive_lights_.upload(
                emissive_lights_host_,
                stream_,
                statistics_);
            statistics_.lighting_upload_bytes += sphere_light_indices_.upload(
                sphere_light_indices_host_,
                stream_,
                statistics_);
            statistics_.lighting_upload_bytes += triangle_light_indices_.upload(
                triangle_light_indices_host_,
                stream_,
                statistics_);
            return;
        }

        sphere_light_indices_host_.assign(scene.spheres.size(), -1);
        triangle_light_indices_host_.assign(scene.triangles.size(), -1);
        std::vector<float> weights;
        weights.reserve(scene.spheres.size() + scene.triangles.size());

        for (std::size_t index = 0; index < scene.spheres.size(); ++index) {
            const Sphere& sphere = scene.spheres[index];
            const int material_id = sphere.material_id();
            if (material_id < 0 ||
                static_cast<std::size_t>(material_id) >= scene.materials.size()) {
                continue;
            }
            const float emission_weight = material_weights[material_id];
            const float area =
                4.0f * kPi * sphere.radius() * sphere.radius();
            const float weight = emission_weight * area;
            if (!(weight > 0.0f) || !std::isfinite(weight)) {
                continue;
            }
            const int light_index =
                static_cast<int>(emissive_lights_host_.size());
            sphere_light_indices_host_[index] = light_index;
            emissive_lights_host_.push_back(DEmissiveLight{
                0,
                static_cast<int>(index),
                material_id,
                area,
                0.0f,
                0.0f});
            weights.push_back(weight);
        }

        for (std::size_t index = 0; index < scene.triangles.size(); ++index) {
            const Triangle& triangle = scene.triangles[index];
            const int material_id = triangle.material_id();
            if (material_id < 0 ||
                static_cast<std::size_t>(material_id) >= scene.materials.size()) {
                continue;
            }
            const float emission_weight = material_weights[material_id];
            const float area = 0.5f *
                (triangle.b() - triangle.a())
                    .cross(triangle.c() - triangle.a())
                    .norm();
            const float weight = emission_weight * area;
            if (!(weight > 0.0f) || !std::isfinite(weight)) {
                continue;
            }
            const int light_index =
                static_cast<int>(emissive_lights_host_.size());
            triangle_light_indices_host_[index] = light_index;
            emissive_lights_host_.push_back(DEmissiveLight{
                1,
                static_cast<int>(index),
                material_id,
                area,
                0.0f,
                0.0f});
            weights.push_back(weight);
        }

        float total_weight = 0.0f;
        for (float weight : weights) {
            total_weight += weight;
        }
        if (total_weight > 0.0f && std::isfinite(total_weight)) {
            float cumulative = 0.0f;
            for (std::size_t index = 0;
                 index < emissive_lights_host_.size();
                 ++index) {
                const float probability = weights[index] / total_weight;
                cumulative += probability;
                emissive_lights_host_[index].selection_pdf = probability;
                emissive_lights_host_[index].cumulative_probability =
                    index + 1 == emissive_lights_host_.size()
                    ? 1.0f
                    : cumulative;
            }
        } else {
            emissive_lights_host_.clear();
            std::fill(
                sphere_light_indices_host_.begin(),
                sphere_light_indices_host_.end(),
                -1);
            std::fill(
                triangle_light_indices_host_.begin(),
                triangle_light_indices_host_.end(),
                -1);
        }

        statistics_.lighting_upload_bytes += emissive_lights_.upload(
            emissive_lights_host_,
            stream_,
            statistics_);
        statistics_.lighting_upload_bytes += sphere_light_indices_.upload(
            sphere_light_indices_host_,
            stream_,
            statistics_);
        statistics_.lighting_upload_bytes += triangle_light_indices_.upload(
            triangle_light_indices_host_,
            stream_,
            statistics_);
    }

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
    DeviceBuffer<DEmissiveLight> emissive_lights_;
    DeviceBuffer<int> sphere_light_indices_;
    DeviceBuffer<int> triangle_light_indices_;
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
    std::vector<DEmissiveLight> emissive_lights_host_;
    std::vector<int> sphere_light_indices_host_;
    std::vector<int> triangle_light_indices_host_;
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
        }
        destroy_graph();
        if (stream_) {
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
        ensure_wavefront_capacity(count);
#if !defined(RENDERER_CUDA_SANITIZER_FALLBACK)
        if (!graph_exec_ ||
            count > static_cast<std::size_t>(graph_capacity_)) {
            rebuild_graph(static_cast<int>(count));
        }
#endif
        const int block_count = static_cast<int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
        initialize_frame_kernel<<<block_count, kThreadsPerBlock, 0, stream_>>>(
            accumulation_.get(),
            random_states_.get(),
            width,
            height,
            seed_offset,
            error_code_);
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
        render_samples(scene, camera, 1, output_surface);
    }

    void render_samples(
        const DScene& scene,
        const Camera& camera,
        int sample_count,
        cudaSurfaceObject_t output_surface = 0) {
        if (sample_count <= 0) {
            throw std::invalid_argument(
                "CUDA wavefront sample batch must be positive");
        }
        const DCamera packed_camera{
            to_device(camera.eye()),
            to_device(camera.forward()),
            to_device(camera.right()),
            to_device(camera.up()),
            camera.viewport_width(),
            camera.viewport_height()};
        const std::size_t count = accumulation_.size();
        check_completed_error();
        const DFrameParameters parameters{
            scene,
            packed_camera,
            accumulation_.get(),
            sample_radiance_,
            random_states_.get(),
            width_,
            height_,
            samples_,
            sample_count,
            batch_sample_index_,
            static_cast<int>(count),
            error_code_,
            output_surface};
        update_frame_parameters_kernel<<<1, 1, 0, stream_>>>(
            frame_parameters_,
            parameters);
        check_cuda(
            cudaGetLastError(),
            "update_frame_parameters_kernel launch");
        trace_timer_.update(statistics_.trace_milliseconds);
        const bool record_timing = !trace_timer_.pending();
        if (record_timing) {
            trace_timer_.begin(stream_);
        }
#if defined(RENDERER_CUDA_SANITIZER_FALLBACK)
        launch_sanitizer_fixed_topology(sample_count);
#else
        check_cuda(
            cudaGraphLaunch(graph_exec_, stream_),
            "cudaGraphLaunch wavefront path");
#endif
        if (record_timing) {
            trace_timer_.end(stream_);
        }
        samples_ += sample_count;
    }

    void synchronize_and_check_errors() {
        check_cuda(cudaStreamSynchronize(stream_), "path rendering stream synchronize");
        throw_if_wavefront_error();
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
        check_completed_error();
    }

private:
#if defined(RENDERER_CUDA_SANITIZER_FALLBACK)
    void launch_sanitizer_fixed_topology(int sample_count) {
        const int capacity = static_cast<int>(accumulation_.size());
        const int block_count =
            (capacity + kThreadsPerBlock - 1) / kThreadsPerBlock;
        prepare_sample_batch_kernel<<<1, 1, 0, stream_>>>(
            batch_sample_index_);
        for (int sample = 0; sample < sample_count; ++sample) {
            prepare_sample_kernel<<<
                block_count,
                kThreadsPerBlock,
                0,
                stream_>>>(
                frame_parameters_,
                path_queues_[0],
                path_counts_,
                path_counts_ + 1,
                bounce_index_,
                0,
                0);
            for (int bounce = 0;
                 bounce < kMaxPathBounces;
                 ++bounce) {
                intersect_wavefront_kernel<<<
                    block_count,
                    kThreadsPerBlock,
                    0,
                    stream_>>>(
                    frame_parameters_,
                    path_queues_[0],
                    path_queues_[1],
                    path_counts_,
                    bounce_index_,
                    hits_);
                shade_wavefront_kernel<<<
                    block_count,
                    kThreadsPerBlock,
                    0,
                    stream_>>>(
                    frame_parameters_,
                    path_queues_[0],
                    path_queues_[1],
                    path_counts_,
                    bounce_index_,
                    hits_,
                    shading_records_);
                sample_emissive_lights_kernel<<<
                    block_count,
                    kThreadsPerBlock,
                    0,
                    stream_>>>(
                    frame_parameters_,
                    shading_records_,
                    shadow_tasks_,
                    path_counts_,
                    bounce_index_);
                direct_visibility_kernel<<<
                    block_count,
                    kThreadsPerBlock,
                    0,
                    stream_>>>(
                    frame_parameters_,
                    shading_records_,
                    shadow_tasks_,
                    path_counts_,
                    bounce_index_);
                advance_wavefront_kernel<<<1, 1, 0, stream_>>>(
                    frame_parameters_,
                    path_counts_,
                    bounce_index_,
                    0);
            }
            finalize_sample_kernel<<<
                block_count,
                kThreadsPerBlock,
                0,
                stream_>>>(frame_parameters_);
            advance_sample_batch_kernel<<<1, 1, 0, stream_>>>(
                frame_parameters_,
                0);
        }
        check_cuda(
            cudaGetLastError(),
            "CUDA sanitizer fixed-topology launch");
    }
#endif

    void ensure_wavefront_capacity(std::size_t count) {
        if (count <= wavefront_capacity_pixels_) {
            return;
        }
        check_cuda(
            cudaStreamSynchronize(stream_),
            "wavefront arena resize synchronize");
        destroy_graph();

        struct ArenaLayout {
            std::size_t sample_radiance = 0;
            std::size_t first_paths = 0;
            std::size_t second_paths = 0;
            std::size_t hits = 0;
            std::size_t shading_records = 0;
            std::size_t shadow_tasks = 0;
            std::size_t path_counts = 0;
            std::size_t bounce_index = 0;
            std::size_t batch_sample_index = 0;
            std::size_t error_code = 0;
            std::size_t frame_parameters = 0;
            std::size_t total_bytes = 0;
        } layout;

        std::size_t offset = 0;
        const auto reserve_region = [&offset](
                                        std::size_t& region_offset,
                                        std::size_t alignment,
                                        std::size_t element_size,
                                        std::size_t element_count) {
            const std::size_t aligned =
                (offset + alignment - 1) & ~(alignment - 1);
            if (element_count > (SIZE_MAX - aligned) / element_size) {
                throw std::overflow_error("CUDA wavefront arena is too large");
            }
            region_offset = aligned;
            offset = aligned + element_size * element_count;
        };
        reserve_region(
            layout.sample_radiance, alignof(DVec3), sizeof(DVec3), count);
        reserve_region(
            layout.first_paths, alignof(DPathState), sizeof(DPathState), count);
        reserve_region(
            layout.second_paths, alignof(DPathState), sizeof(DPathState), count);
        reserve_region(
            layout.hits, alignof(DWavefrontHit), sizeof(DWavefrontHit), count);
        reserve_region(
            layout.shading_records,
            alignof(DShadingRecord),
            sizeof(DShadingRecord),
            count);
        reserve_region(
            layout.shadow_tasks,
            alignof(DShadowTask),
            sizeof(DShadowTask),
            count);
        reserve_region(layout.path_counts, alignof(int), sizeof(int), 2);
        reserve_region(layout.bounce_index, alignof(int), sizeof(int), 1);
        reserve_region(
            layout.batch_sample_index, alignof(int), sizeof(int), 1);
        reserve_region(layout.error_code, alignof(int), sizeof(int), 1);
        reserve_region(
            layout.frame_parameters,
            alignof(DFrameParameters),
            sizeof(DFrameParameters),
            1);
        layout.total_bytes = offset;

        wavefront_arena_.resize(layout.total_bytes, statistics_);
        unsigned char* const base = wavefront_arena_.get();
        sample_radiance_ =
            reinterpret_cast<DVec3*>(base + layout.sample_radiance);
        path_queues_[0] =
            reinterpret_cast<DPathState*>(base + layout.first_paths);
        path_queues_[1] =
            reinterpret_cast<DPathState*>(base + layout.second_paths);
        hits_ = reinterpret_cast<DWavefrontHit*>(base + layout.hits);
        shading_records_ =
            reinterpret_cast<DShadingRecord*>(base + layout.shading_records);
        shadow_tasks_ =
            reinterpret_cast<DShadowTask*>(base + layout.shadow_tasks);
        path_counts_ = reinterpret_cast<int*>(base + layout.path_counts);
        bounce_index_ = reinterpret_cast<int*>(base + layout.bounce_index);
        batch_sample_index_ =
            reinterpret_cast<int*>(base + layout.batch_sample_index);
        error_code_ = reinterpret_cast<int*>(base + layout.error_code);
        frame_parameters_ =
            reinterpret_cast<DFrameParameters*>(base + layout.frame_parameters);
        wavefront_capacity_pixels_ = count;
    }

    void rebuild_graph(int capacity) {
        if (capacity <= 0) {
            throw std::invalid_argument(
                "CUDA wavefront graph capacity must be positive");
        }
        check_cuda(
            cudaStreamSynchronize(stream_),
            "wavefront graph rebuild synchronize");
        destroy_graph();
        const int block_count =
            (capacity + kThreadsPerBlock - 1) / kThreadsPerBlock;
        check_cuda(
            cudaGraphCreate(&graph_, 0),
            "cudaGraphCreate wavefront graph");
        check_cuda(
            cudaGraphConditionalHandleCreate(
                &batch_conditional_handle_,
                graph_,
                1U,
                cudaGraphCondAssignDefault),
            "cudaGraphConditionalHandleCreate sample batch");
        check_cuda(
            cudaGraphConditionalHandleCreate(
                &wavefront_conditional_handle_,
                graph_,
                0U,
                0),
            "cudaGraphConditionalHandleCreate wavefront loop");
        check_cuda(
            cudaGraphConditionalHandleCreate(
                &emissive_conditional_handle_,
                graph_,
                0U,
                0),
            "cudaGraphConditionalHandleCreate emissive sampling");

        DFrameParameters* frame_parameters = frame_parameters_;
        DPathState* first_paths = path_queues_[0];
        DPathState* second_paths = path_queues_[1];
        int* path_counts = path_counts_;
        int* primary_count = path_counts;
        int* secondary_count = path_counts + 1;
        int* bounce_index = bounce_index_;
        int* batch_sample_index = batch_sample_index_;
        DWavefrontHit* hits = hits_;
        DShadingRecord* shading_records = shading_records_;
        DShadowTask* shadow_tasks = shadow_tasks_;

        void* prepare_batch_arguments[] = {&batch_sample_index};
        cudaGraphNode_t prepare_batch_node = add_kernel_node(
            graph_,
            nullptr,
            reinterpret_cast<void*>(prepare_sample_batch_kernel),
            dim3(1),
            dim3(1),
            prepare_batch_arguments);

        cudaGraphNodeParams batch_conditional_parameters{};
        batch_conditional_parameters.type = cudaGraphNodeTypeConditional;
        batch_conditional_parameters.conditional.handle =
            batch_conditional_handle_;
        batch_conditional_parameters.conditional.type =
            cudaGraphCondTypeWhile;
        batch_conditional_parameters.conditional.size = 1;
        cudaGraphNode_t batch_conditional_node = nullptr;
        check_cuda(
            cudaGraphAddNode(
                &batch_conditional_node,
                graph_,
                &prepare_batch_node,
                nullptr,
                1,
                &batch_conditional_parameters),
            "cudaGraphAddNode sample batch while");
        cudaGraph_t sample_body =
            batch_conditional_parameters.conditional.phGraph_out[0];

        cudaGraphConditionalHandle wavefront_handle =
            wavefront_conditional_handle_;
        cudaGraphConditionalHandle emissive_handle =
            emissive_conditional_handle_;
        void* prepare_arguments[] = {
            &frame_parameters,
            &first_paths,
            &primary_count,
            &secondary_count,
            &bounce_index,
            &wavefront_handle,
            &emissive_handle};
        cudaGraphNode_t prepare_node = add_kernel_node(
            sample_body,
            nullptr,
            reinterpret_cast<void*>(prepare_sample_kernel),
            dim3(block_count),
            dim3(kThreadsPerBlock),
            prepare_arguments);

        cudaGraphNodeParams conditional_parameters{};
        conditional_parameters.type = cudaGraphNodeTypeConditional;
        conditional_parameters.conditional.handle =
            wavefront_conditional_handle_;
        conditional_parameters.conditional.type = cudaGraphCondTypeWhile;
        conditional_parameters.conditional.size = 1;
        cudaGraphNode_t conditional_node = nullptr;
        check_cuda(
            cudaGraphAddNode(
                &conditional_node,
                sample_body,
                &prepare_node,
                nullptr,
                1,
                &conditional_parameters),
            "cudaGraphAddNode wavefront while");
        cudaGraph_t body = conditional_parameters.conditional.phGraph_out[0];

        void* intersect_arguments[] = {
            &frame_parameters,
            &first_paths,
            &second_paths,
            &path_counts,
            &bounce_index,
            &hits};
        cudaGraphNode_t body_tail = add_kernel_node(
            body,
            nullptr,
            reinterpret_cast<void*>(intersect_wavefront_kernel),
            dim3(block_count),
            dim3(kThreadsPerBlock),
            intersect_arguments);

        void* shade_arguments[] = {
            &frame_parameters,
            &first_paths,
            &second_paths,
            &path_counts,
            &bounce_index,
            &hits,
            &shading_records};
        body_tail = add_kernel_node(
            body,
            body_tail,
            reinterpret_cast<void*>(shade_wavefront_kernel),
            dim3(block_count),
            dim3(kThreadsPerBlock),
            shade_arguments);

        cudaGraphNodeParams emissive_conditional_parameters{};
        emissive_conditional_parameters.type =
            cudaGraphNodeTypeConditional;
        emissive_conditional_parameters.conditional.handle =
            emissive_conditional_handle_;
        emissive_conditional_parameters.conditional.type =
            cudaGraphCondTypeIf;
        emissive_conditional_parameters.conditional.size = 1;
        cudaGraphNode_t emissive_conditional_node = nullptr;
        check_cuda(
            cudaGraphAddNode(
                &emissive_conditional_node,
                body,
                &body_tail,
                nullptr,
                1,
                &emissive_conditional_parameters),
            "cudaGraphAddNode emissive sampling if");
        cudaGraph_t emissive_body =
            emissive_conditional_parameters.conditional.phGraph_out[0];

        void* emissive_arguments[] = {
            &frame_parameters,
            &shading_records,
            &shadow_tasks,
            &path_counts,
            &bounce_index};
        add_kernel_node(
            emissive_body,
            nullptr,
            reinterpret_cast<void*>(sample_emissive_lights_kernel),
            dim3(block_count),
            dim3(kThreadsPerBlock),
            emissive_arguments);
        body_tail = emissive_conditional_node;

        void* direct_arguments[] = {
            &frame_parameters,
            &shading_records,
            &shadow_tasks,
            &path_counts,
            &bounce_index};
        body_tail = add_kernel_node(
            body,
            body_tail,
            reinterpret_cast<void*>(direct_visibility_kernel),
            dim3(block_count),
            dim3(kThreadsPerBlock),
            direct_arguments);

        cudaGraphConditionalHandle conditional_handle =
            wavefront_conditional_handle_;
        void* advance_arguments[] = {
            &frame_parameters,
            &path_counts,
            &bounce_index,
            &conditional_handle};
        add_kernel_node(
            body,
            body_tail,
            reinterpret_cast<void*>(advance_wavefront_kernel),
            dim3(1),
            dim3(1),
            advance_arguments);

        void* finalize_arguments[] = {&frame_parameters};
        cudaGraphNode_t finalize_node = add_kernel_node(
            sample_body,
            conditional_node,
            reinterpret_cast<void*>(finalize_sample_kernel),
            dim3(block_count),
            dim3(kThreadsPerBlock),
            finalize_arguments);

        cudaGraphConditionalHandle batch_handle =
            batch_conditional_handle_;
        void* advance_batch_arguments[] = {
            &frame_parameters,
            &batch_handle};
        add_kernel_node(
            sample_body,
            finalize_node,
            reinterpret_cast<void*>(advance_sample_batch_kernel),
            dim3(1),
            dim3(1),
            advance_batch_arguments);
        check_cuda(
            cudaGraphInstantiate(
                &graph_exec_,
                graph_,
                nullptr,
                nullptr,
                0),
            "cudaGraphInstantiate wavefront graph");
        graph_capacity_ = capacity;
    }

    cudaGraphNode_t add_kernel_node(
        cudaGraph_t graph,
        cudaGraphNode_t dependency,
        void* function,
        dim3 grid,
        dim3 block,
        void** arguments) {
        cudaKernelNodeParams parameters{};
        parameters.func = function;
        parameters.gridDim = grid;
        parameters.blockDim = block;
        parameters.kernelParams = arguments;
        cudaGraphNode_t node = nullptr;
        const cudaGraphNode_t* dependencies =
            dependency != nullptr ? &dependency : nullptr;
        check_cuda(
            cudaGraphAddKernelNode(
                &node,
                graph,
                dependencies,
                dependency != nullptr ? 1 : 0,
                &parameters),
            "cudaGraphAddKernelNode wavefront graph");
        return node;
    }

    void destroy_graph() noexcept {
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        if (graph_) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
        batch_conditional_handle_ = 0;
        wavefront_conditional_handle_ = 0;
        emissive_conditional_handle_ = 0;
        graph_capacity_ = 0;
    }

    void check_completed_error() {
        if (!stream_) {
            return;
        }
        const cudaError_t query = cudaStreamQuery(stream_);
        if (query == cudaErrorNotReady) {
            return;
        }
        check_cuda(query, "CUDA wavefront stream query");
        throw_if_wavefront_error();
    }

    void throw_if_wavefront_error() {
        int error = 0;
        check_cuda(
            cudaMemcpy(
                &error,
                error_code_,
                sizeof(error),
                cudaMemcpyDeviceToHost),
            "download CUDA wavefront error");
        if (error == 0) {
            return;
        }
        if (error == 1) {
            throw std::runtime_error("CUDA wavefront path queue overflow");
        }
        throw std::runtime_error("CUDA wavefront queue count is invalid");
    }

    CudaPathStatistics& statistics_;
    cudaStream_t stream_ = nullptr;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaGraphConditionalHandle batch_conditional_handle_ = 0;
    cudaGraphConditionalHandle wavefront_conditional_handle_ = 0;
    cudaGraphConditionalHandle emissive_conditional_handle_ = 0;
    int graph_capacity_ = 0;
    CudaEventTimer reset_timer_;
    CudaEventTimer trace_timer_;
    DeviceBuffer<DVec3> accumulation_;
    DeviceBuffer<DVec3> resolved_;
    DeviceBuffer<DPcgState> random_states_;
    DeviceBuffer<unsigned char> wavefront_arena_;
    std::size_t wavefront_capacity_pixels_ = 0;
    DVec3* sample_radiance_ = nullptr;
    DPathState* path_queues_[2]{nullptr, nullptr};
    DWavefrontHit* hits_ = nullptr;
    DShadingRecord* shading_records_ = nullptr;
    DShadowTask* shadow_tasks_ = nullptr;
    int* path_counts_ = nullptr;
    int* bounce_index_ = nullptr;
    int* batch_sample_index_ = nullptr;
    int* error_code_ = nullptr;
    DFrameParameters* frame_parameters_ = nullptr;
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
    frame.render_samples(device_scene.view(), camera, sample_count);
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
