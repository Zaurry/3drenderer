// Shared CUDA scene, traversal, material and lighting implementation.
#pragma once
#define EIGEN_NO_CUDA
#include "render/pathtracer/cuda_pathtracer.h"

#include "acceleration/bvh.h"
#include "core/timer.h"
#include "render/mis_weight.h"
#include "scene/material.h"
#include "scene/texture.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cfloat>
#include <climits>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace renderer {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kMaxPathBounces = 64;
constexpr int kBvhStackCapacity = 64;
constexpr float kPi = 3.14159265358979323846f;
constexpr int kGgxEnergyLutSize = 32;

__device__ __constant__ float
    kGgxDirectionalAlbedoLut[kGgxEnergyLutSize * kGgxEnergyLutSize] = {
#include "render/generated/ggx_directional_albedo_lut.inc"
};
__device__ __constant__ float kGgxAverageAlbedoLut[kGgxEnergyLutSize] = {
#include "render/generated/ggx_average_albedo_lut.inc"
};

#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
constexpr int kDiagnosticPrimaryRays = 0;
constexpr int kDiagnosticContinuationRays = 1;
constexpr int kDiagnosticDirectionalShadowRays = 2;
constexpr int kDiagnosticPointShadowRays = 3;
constexpr int kDiagnosticSpotShadowRays = 4;
constexpr int kDiagnosticEmissiveShadowRays = 5;
constexpr int kDiagnosticEnvironmentShadowRays = 6;
constexpr int kDiagnosticPrimaryHits = 7;
constexpr int kDiagnosticPrimaryMisses = 8;
constexpr int kDiagnosticRaysByBounce = 9;
constexpr int kDiagnosticTerminationByBounce =
    kDiagnosticRaysByBounce + kMaxPathBounces + 1;
constexpr int kDiagnosticCounterCount =
    kDiagnosticTerminationByBounce + kMaxPathBounces + 1;
constexpr int kDiagnosticShadowEmissive = 1;
constexpr int kDiagnosticShadowEnvironment = 2;
#endif

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

struct DVec4 {
    float x;
    float y;
    float z;
    float w;
};

struct DRay {
    DVec3 origin;
    DVec3 direction;
};

struct DMatrix3x4 {
    float values[12];
};

struct DMatrix3x3 {
    float values[9];
};

struct DTraversalTriangle {
    DVec3 v0;
    DVec3 edge1;
    DVec3 edge2;
};

struct DShadingTriangle {
    DVec2 uvs[3];
    DVec2 uv1s[3];
    DVec3 normals[3];
    DVec4 tangents[3];
    DVec3 colors[3];
    float alphas[3];
    unsigned int normal_mask;
    unsigned int uv1_mask;
    unsigned int tangent_mask;
    unsigned int color_mask;
};

struct DSphere {
    DVec3 center;
    float radius;
};

struct DMaterial {
    // NOTE: kernels copy this by value; pack_material() constructs fields by
    // name, so the field order itself is not load-bearing. Do not add a
    // second writer that relies on positional initialization.
    int type;
    DVec3 base_color;
    DVec3 emission;
    float roughness;
    float metallic;
    float ior;
    float opacity;
    float alpha_cutoff;
    float bump_scale;
    float normal_scale;
    float occlusion_strength;
    int alpha_mode;
    int two_sided;
    int diffuse_texture_id;
    int opacity_texture_id;
    int bump_texture_id;
    int base_color_texture_id;
    int metallic_roughness_texture_id;
    int normal_texture_id;
    int occlusion_texture_id;
    int emissive_texture_id;
    DVec4 base_color_texture_transform;
    DVec4 metallic_roughness_texture_transform;
    DVec4 normal_texture_transform;
    DVec4 occlusion_texture_transform;
    DVec4 emissive_texture_transform;
    float base_color_texture_rotation;
    float metallic_roughness_texture_rotation;
    float normal_texture_rotation;
    float occlusion_texture_rotation;
    float emissive_texture_rotation;
    int base_color_texture_texcoord;
    int metallic_roughness_texture_texcoord;
    int normal_texture_texcoord;
    int occlusion_texture_texcoord;
    int emissive_texture_texcoord;
    int pbr_workflow;
    DVec3 specular_color;
    float specular_factor;
    float glossiness;
    int specular_texture_id;
    int specular_color_texture_id;
    int specular_glossiness_texture_id;
    DVec4 specular_texture_transform;
    DVec4 specular_color_texture_transform;
    DVec4 specular_glossiness_texture_transform;
    float specular_texture_rotation;
    float specular_color_texture_rotation;
    float specular_glossiness_texture_rotation;
    int specular_texture_texcoord;
    int specular_color_texture_texcoord;
    int specular_glossiness_texture_texcoord;
};

struct DTexture {
    int width;
    int height;
    int first_pixel;
    int first_alpha;
    int wrap_s;
    int wrap_t;
    int nearest;
    int top_left;
};

struct DBvh4Node {
    DVec3 bounds_min[4];
    DVec3 bounds_max[4];
    int children[4];
    int first[4];
    int counts[4];
    int child_count;
};

struct DPointLight {
    DVec3 position;
    DVec3 intensity;
    float range;
    float source_radius = 0;
    int casts_shadows = 1;
};

struct DDirectionalLight {
    DVec3 direction;
    DVec3 radiance;
    float angular_radius = 0;
    int casts_shadows = 1;
};

struct DSpotLight {
    DVec3 position;
    DVec3 direction;
    DVec3 intensity;
    float range;
    float inner_cosine;
    float outer_cosine;
    float source_radius = 0;
    int casts_shadows = 1;
};

struct DEmissiveLight {
    int primitive_kind;
    int primitive_index;
    int instance_index;
    int material_id;
    float area;
    float selection_pdf;
    float cumulative_probability;
    int casts_shadows = 1;
};

struct DAsset {
    int sphere_first;
    int sphere_count;
    int triangle_first;
    int triangle_count;
    int bvh_root;
    int bvh_node_count;
};

struct DInstance {
    DMatrix3x4 object_to_world;
    DMatrix3x4 world_to_object;
    DMatrix3x3 normal_to_world;
    DVec3 bounds_min;
    DVec3 bounds_max;
    int asset_index;
    int material_offset;
    float orientation_sign;
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
    int stochastic_alpha_test;
    const DTexture* textures;
    int texture_count;
    const DVec3* texels;
    const float* texture_alphas;
    const DSphere* spheres;
    const int* sphere_material_ids;
    int sphere_count;
    const DTraversalTriangle* traversal_triangles;
    const DShadingTriangle* shading_triangles;
    const int* triangle_material_ids;
    int triangle_count;
    const DBvh4Node* bvh_nodes;
    int bvh_node_count;
    const int* primitive_indices;
    const DPointLight* point_lights;
    int point_light_count;
    const DDirectionalLight* directional_lights;
    int directional_light_count;
    const DSpotLight* spot_lights;
    int spot_light_count;
    const DEmissiveLight* emissive_lights;
    int emissive_light_count;
    const int* sphere_light_indices;
    const int* triangle_light_indices;
    const DAsset* assets;
    int asset_count;
    const DInstance* instances;
    int instance_count;
    const DBvh4Node* tlas_nodes;
    int tlas_node_count;
    const int* tlas_primitive_indices;
    int instanced_mode;
    DVec3 environment;
    const DVec3* environment_texels;
    const float* environment_pmf;
    const float* environment_cdf;
    int environment_width;
    int environment_height;
    float environment_intensity;
    float environment_rotation_radians;
    int environment_background_visible;
};

// Device structs are passed to kernels by value and laid out in device
// buffers; they must stay trivially copyable.
static_assert(
    std::is_trivially_copyable_v<DMaterial>,
    "DMaterial must stay trivially copyable for device kernels");
static_assert(
    std::is_trivially_copyable_v<DInstance>,
    "DInstance must stay trivially copyable for device kernels");
static_assert(
    std::is_trivially_copyable_v<DScene>,
    "DScene must stay trivially copyable for device kernels");
static_assert(
    std::is_trivially_copyable_v<DTexture>,
    "DTexture must stay trivially copyable for device kernels");

struct DHit {
    float t;
    DVec3 position;
    DVec2 uv;
    DVec2 uv1;
    DVec3 vertex_color;
    float vertex_alpha;
    DVec4 tangent;
    int has_tangent;
    DVec3 geometric_normal;
    DVec3 shading_normal;
    int material_id;
    int front_face;
    int primitive_kind;
    int primitive_index;
    int instance_index;
    float barycentric_u;
    float barycentric_v;
};

struct DAlphaEvaluation {
    int mode;
    float coverage;
};

struct DCompactHit {
    float t;
    float u;
    float v;
    int primitive_kind;
    int primitive_index;
    int instance_index;
};

struct DSurface {
    DVec3 base_color;
    DVec3 diffuse_color;
    DVec3 specular_f0;
    DVec3 specular_f90;
    DVec3 diffuse_fresnel_f0;
    DVec3 diffuse_fresnel_f90;
    int diffuse_fresnel_uses_max;
    DVec3 emission;
    float opacity;
    float metallic;
    float roughness;
    float occlusion;
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
    DVec3 diffuse_color;
    DVec3 specular_f0;
    DVec3 specular_f90;
    DVec3 diffuse_fresnel_f0;
    DVec3 diffuse_fresnel_f90;
    int diffuse_fresnel_uses_max;
    DVec3 outgoing;
    float metallic;
    float roughness;
    float occlusion;
    DVec3 throughput;
    int pixel_index;
    int valid;
};

struct DShadowTask {
    DRay ray;
    DVec3 contribution;
    float t_max;
    int pixel_index;
    int casts_shadows = 1;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    int diagnostic_kind = 0;
#endif
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
    int pixel_offset;
    int pixel_count;
    int path_capacity;
    int max_bounces;
    int russian_roulette_start_bounce;
    float russian_roulette_min_probability;
    float russian_roulette_max_probability;
    int output_width;
    int output_height;
    int* error_code;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    unsigned long long* diagnostic_counters;
#endif
    cudaSurfaceObject_t output_surface;
};

__host__ DVec3 to_device(const Vec3& value) {
    return DVec3{value.x(), value.y(), value.z()};
}

__host__ DVec2 to_device(const Vec2& value) {
    return DVec2{value.x(), value.y()};
}

__host__ DVec4 to_device(const Vec4& value) {
    return DVec4{value.x(), value.y(), value.z(), value.w()};
}

__host__ DVec4 pack_texture_transform(const TextureTransform& transform) {
    return DVec4{
        transform.offset.x(),
        transform.offset.y(),
        transform.scale.x(),
        transform.scale.y()};
}

__host__ DMatrix3x4 to_device_affine(const Mat4& value) {
    DMatrix3x4 result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 4; ++column) {
            result.values[row * 4 + column] =
                value(row, column);
        }
    }
    return result;
}

__host__ DMatrix3x3 to_device_matrix(const Mat3& value) {
    DMatrix3x3 result{};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            result.values[row * 3 + column] =
                value(row, column);
        }
    }
    return result;
}

struct GpuBvhPrimitiveInfo {
    Bounds3 bounds;
    Vec3 centroid = Vec3::Zero();
};

struct GpuBinaryBvhNode {
    Bounds3 bounds;
    int left = -1;
    int right = -1;
    int first = 0;
    int count = 0;
};

struct GpuBvh4Layout {
    std::vector<DBvh4Node> nodes;
    std::vector<int> primitive_indices;
};

float host_bounds_area(const Bounds3& bounds) {
    if (!bounds.min.allFinite() || !bounds.max.allFinite()) {
        return 0.0f;
    }
    const Vec3 extent = (bounds.max - bounds.min).cwiseMax(0.0f);
    return 2.0f *
        (extent.x() * extent.y() +
         extent.x() * extent.z() +
         extent.y() * extent.z());
}

class GpuBvh4Builder {
public:
    explicit GpuBvh4Builder(
        std::vector<GpuBvhPrimitiveInfo> primitive_info)
        : primitive_info_(std::move(primitive_info)),
          primitive_indices_(primitive_info_.size()) {
        std::iota(primitive_indices_.begin(), primitive_indices_.end(), 0);
    }

    GpuBvh4Layout build() {
        GpuBvh4Layout layout;
        if (primitive_info_.empty()) {
            return layout;
        }
        binary_nodes_.reserve(primitive_info_.size() * 2);
        const int root = build_binary(
            0,
            static_cast<int>(primitive_indices_.size()));
        wide_nodes_.reserve((binary_nodes_.size() + 2) / 3);
        collapse(root);
        layout.nodes = std::move(wide_nodes_);
        layout.primitive_indices = std::move(primitive_indices_);
        return layout;
    }

private:
    static constexpr int kBinCount = 16;
    static constexpr int kMaxLeafTriangles = 8;

    struct Bin {
        Bounds3 bounds;
        int count = 0;
    };

    std::vector<GpuBvhPrimitiveInfo> primitive_info_;
    std::vector<int> primitive_indices_;
    std::vector<GpuBinaryBvhNode> binary_nodes_;
    std::vector<DBvh4Node> wide_nodes_;

    int build_binary(int first, int count) {
        const int node_index = static_cast<int>(binary_nodes_.size());
        binary_nodes_.push_back(GpuBinaryBvhNode{});
        Bounds3 bounds;
        Bounds3 centroid_bounds;
        for (int offset = 0; offset < count; ++offset) {
            const GpuBvhPrimitiveInfo& primitive =
                primitive_info_[primitive_indices_[first + offset]];
            bounds.expand(primitive.bounds);
            centroid_bounds.expand(primitive.centroid);
        }
        binary_nodes_[node_index].bounds = bounds;
        if (count <= kMaxLeafTriangles) {
            binary_nodes_[node_index].first = first;
            binary_nodes_[node_index].count = count;
            return node_index;
        }

        const int axis = centroid_bounds.longest_axis();
        const float minimum = centroid_bounds.min[axis];
        const float maximum = centroid_bounds.max[axis];
        const float extent = maximum - minimum;
        int middle = first;
        if (std::isfinite(extent) && extent > 1.0e-12f) {
            std::array<Bin, kBinCount> bins{};
            for (int offset = 0; offset < count; ++offset) {
                const int primitive_index = primitive_indices_[first + offset];
                const float coordinate =
                    primitive_info_[primitive_index].centroid[axis];
                const int bin_index = std::clamp(
                    static_cast<int>(
                        (coordinate - minimum) *
                        static_cast<float>(kBinCount) / extent),
                    0,
                    kBinCount - 1);
                ++bins[bin_index].count;
                bins[bin_index].bounds.expand(
                    primitive_info_[primitive_index].bounds);
            }

            std::array<int, kBinCount - 1> left_counts{};
            std::array<int, kBinCount - 1> right_counts{};
            std::array<float, kBinCount - 1> left_areas{};
            std::array<float, kBinCount - 1> right_areas{};
            Bounds3 left_bounds;
            int left_count = 0;
            for (int bin = 0; bin < kBinCount - 1; ++bin) {
                if (bins[bin].count > 0) {
                    left_bounds.expand(bins[bin].bounds);
                }
                left_count += bins[bin].count;
                left_counts[bin] = left_count;
                left_areas[bin] = host_bounds_area(left_bounds);
            }
            Bounds3 right_bounds;
            int right_count = 0;
            for (int bin = kBinCount - 1; bin > 0; --bin) {
                if (bins[bin].count > 0) {
                    right_bounds.expand(bins[bin].bounds);
                }
                right_count += bins[bin].count;
                right_counts[bin - 1] = right_count;
                right_areas[bin - 1] = host_bounds_area(right_bounds);
            }

            int best_split = -1;
            double best_cost = DBL_MAX;
            for (int split = 0; split < kBinCount - 1; ++split) {
                if (left_counts[split] == 0 || right_counts[split] == 0) {
                    continue;
                }
                const double cost =
                    static_cast<double>(left_counts[split]) *
                        left_areas[split] +
                    static_cast<double>(right_counts[split]) *
                        right_areas[split];
                if (cost < best_cost) {
                    best_cost = cost;
                    best_split = split;
                }
            }
            if (best_split >= 0) {
                const float split_position =
                    minimum +
                    extent * static_cast<float>(best_split + 1) /
                        static_cast<float>(kBinCount);
                auto begin = primitive_indices_.begin() + first;
                auto end = begin + count;
                middle = static_cast<int>(
                    std::partition(
                        begin,
                        end,
                        [&](int primitive_index) {
                            return primitive_info_[primitive_index]
                                       .centroid[axis] < split_position;
                        }) -
                    primitive_indices_.begin());
            }
        }

        if (middle <= first || middle >= first + count) {
            middle = first + count / 2;
            auto begin = primitive_indices_.begin();
            std::nth_element(
                begin + first,
                begin + middle,
                begin + first + count,
                [&](int lhs, int rhs) {
                    return primitive_info_[lhs].centroid[axis] <
                        primitive_info_[rhs].centroid[axis];
                });
        }

        const int left = build_binary(first, middle - first);
        const int right =
            build_binary(middle, first + count - middle);
        binary_nodes_[node_index].left = left;
        binary_nodes_[node_index].right = right;
        return node_index;
    }

    int collapse(int binary_index) {
        const int wide_index = static_cast<int>(wide_nodes_.size());
        wide_nodes_.push_back(DBvh4Node{});
        std::vector<int> candidates;
        const GpuBinaryBvhNode& root = binary_nodes_[binary_index];
        if (root.count > 0) {
            candidates.push_back(binary_index);
        } else {
            candidates.push_back(root.left);
            candidates.push_back(root.right);
        }
        while (candidates.size() < 4) {
            int expand_index = -1;
            float largest_area = -1.0f;
            for (int index = 0;
                 index < static_cast<int>(candidates.size());
                 ++index) {
                const GpuBinaryBvhNode& candidate =
                    binary_nodes_[candidates[index]];
                if (candidate.count > 0) {
                    continue;
                }
                const float area = host_bounds_area(candidate.bounds);
                if (area > largest_area) {
                    largest_area = area;
                    expand_index = index;
                }
            }
            if (expand_index < 0) {
                break;
            }
            const GpuBinaryBvhNode expanded =
                binary_nodes_[candidates[expand_index]];
            candidates[expand_index] = expanded.left;
            candidates.push_back(expanded.right);
        }

        DBvh4Node node{};
        node.child_count = static_cast<int>(candidates.size());
        for (int slot = 0; slot < 4; ++slot) {
            node.children[slot] = -1;
            node.first[slot] = 0;
            node.counts[slot] = 0;
        }
        for (int slot = 0; slot < node.child_count; ++slot) {
            const GpuBinaryBvhNode& candidate =
                binary_nodes_[candidates[slot]];
            node.bounds_min[slot] = to_device(candidate.bounds.min);
            node.bounds_max[slot] = to_device(candidate.bounds.max);
            if (candidate.count > 0) {
                node.first[slot] = candidate.first;
                node.counts[slot] = candidate.count;
            } else {
                node.children[slot] = collapse(candidates[slot]);
            }
        }
        wide_nodes_[wide_index] = node;
        return wide_index;
    }
};

GpuBvh4Layout build_gpu_bvh4(const std::vector<Triangle>& triangles) {
    std::vector<GpuBvhPrimitiveInfo> primitive_info(
        triangles.size());
    for (std::size_t index = 0; index < triangles.size(); ++index) {
        primitive_info[index].bounds = triangles[index].bounds();
        primitive_info[index].centroid = triangles[index].centroid();
    }
    return GpuBvh4Builder(std::move(primitive_info)).build();
}

GpuBvh4Layout build_gpu_bvh4(
    const std::vector<Bounds3>& primitive_bounds) {
    std::vector<GpuBvhPrimitiveInfo> primitive_info(
        primitive_bounds.size());
    for (std::size_t index = 0;
         index < primitive_bounds.size();
         ++index) {
        primitive_info[index].bounds = primitive_bounds[index];
        primitive_info[index].centroid =
            (primitive_bounds[index].min +
             primitive_bounds[index].max) *
            0.5f;
    }
    return GpuBvh4Builder(std::move(primitive_info)).build();
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

__device__ float max_component(DVec3 value) {
    return fmaxf(value.x, fmaxf(value.y, value.z));
}

__device__ DVec3 transform_point(
    const DMatrix3x4& matrix,
    DVec3 point) {
    return v3(
        matrix.values[0] * point.x +
            matrix.values[1] * point.y +
            matrix.values[2] * point.z +
            matrix.values[3],
        matrix.values[4] * point.x +
            matrix.values[5] * point.y +
            matrix.values[6] * point.z +
            matrix.values[7],
        matrix.values[8] * point.x +
            matrix.values[9] * point.y +
            matrix.values[10] * point.z +
            matrix.values[11]);
}

__device__ DVec3 transform_direction(
    const DMatrix3x4& matrix,
    DVec3 direction) {
    return v3(
        matrix.values[0] * direction.x +
            matrix.values[1] * direction.y +
            matrix.values[2] * direction.z,
        matrix.values[4] * direction.x +
            matrix.values[5] * direction.y +
            matrix.values[6] * direction.z,
        matrix.values[8] * direction.x +
            matrix.values[9] * direction.y +
            matrix.values[10] * direction.z);
}

__device__ DVec3 transform_direction(
    const DMatrix3x3& matrix,
    DVec3 direction) {
    return v3(
        matrix.values[0] * direction.x +
            matrix.values[1] * direction.y +
            matrix.values[2] * direction.z,
        matrix.values[3] * direction.x +
            matrix.values[4] * direction.y +
            matrix.values[5] * direction.z,
        matrix.values[6] * direction.x +
            matrix.values[7] * direction.y +
            matrix.values[8] * direction.z);
}

__device__ float linear_determinant(
    const DMatrix3x4& matrix) {
    return
        matrix.values[0] *
            (matrix.values[5] * matrix.values[10] -
             matrix.values[6] * matrix.values[9]) -
        matrix.values[1] *
            (matrix.values[4] * matrix.values[10] -
             matrix.values[6] * matrix.values[8]) +
        matrix.values[2] *
            (matrix.values[4] * matrix.values[9] -
             matrix.values[5] * matrix.values[8]);
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

template<class Rng>
__device__ DVec3 cosine_weighted_hemisphere(Rng& rng) {
    const float r1 = random_float(rng);
    const float r2 = random_float(rng);
    const float phi = 2.0f * kPi * r1;
    const float radius = sqrtf(r2);
    return normalize(v3(
        radius * cosf(phi),
        radius * sinf(phi),
        sqrtf(fmaxf(0.0f, 1.0f - r2))));
}

__device__ float material_opacity(
    const DScene& scene,
    const DMaterial& material,
    DVec2 uv);
__device__ float material_surface_opacity(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit);
__device__ int effective_alpha_mode(const DMaterial& material);
__device__ DAlphaEvaluation evaluate_alpha(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit);
__device__ DAlphaEvaluation evaluate_alpha(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit,
    int mode);

__device__ unsigned int alpha_hash(unsigned int value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

__device__ float candidate_alpha_sample(
    const DRay& ray,
    int instance_index,
    int primitive_index,
    float u,
    float v) {
    unsigned int hash = alpha_hash(static_cast<unsigned int>(primitive_index + 1));
    hash ^= alpha_hash(static_cast<unsigned int>(instance_index + 2));
    hash ^= alpha_hash(__float_as_uint(ray.direction.x));
    hash ^= alpha_hash(__float_as_uint(ray.direction.y));
    hash ^= alpha_hash(__float_as_uint(ray.direction.z));
    hash ^= alpha_hash(__float_as_uint(u));
    hash ^= alpha_hash(__float_as_uint(v));
    return static_cast<float>(hash >> 8U) * (1.0f / 16777216.0f);
}

__device__ DVec3 inverse_direction(DVec3 direction) {
    constexpr float epsilon = 1.0e-12f;
    const auto inverse_component = [](float value) {
        return fabsf(value) < epsilon
            ? copysignf(FLT_MAX, value == 0.0f ? 1.0f : value)
            : 1.0f / value;
    };
    return v3(
        inverse_component(direction.x),
        inverse_component(direction.y),
        inverse_component(direction.z));
}

__device__ bool intersect_bounds(
    DVec3 bounds_min,
    DVec3 bounds_max,
    const DRay& ray,
    DVec3 inverse,
    float t_min,
    float t_max,
    float& entry_t) {
    const float tx0 =
        (bounds_min.x - ray.origin.x) * inverse.x;
    const float tx1 =
        (bounds_max.x - ray.origin.x) * inverse.x;
    const float ty0 =
        (bounds_min.y - ray.origin.y) * inverse.y;
    const float ty1 =
        (bounds_max.y - ray.origin.y) * inverse.y;
    const float tz0 =
        (bounds_min.z - ray.origin.z) * inverse.z;
    const float tz1 =
        (bounds_max.z - ray.origin.z) * inverse.z;
    const float entry = fmaxf(
        t_min,
        fmaxf(
            fminf(tx0, tx1),
            fmaxf(fminf(ty0, ty1), fminf(tz0, tz1))));
    const float exit = fminf(
        t_max,
        fminf(
            fmaxf(tx0, tx1),
            fminf(fmaxf(ty0, ty1), fmaxf(tz0, tz1))));
    entry_t = entry;
    return exit >= entry;
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
    const DTraversalTriangle& triangle,
    const DRay& ray,
    float t_min,
    float t_max,
    float& hit_t,
    float& hit_u,
    float& hit_v) {
    const DVec3 h = cross(ray.direction, triangle.edge2);
    const float determinant = dot(triangle.edge1, h);
    if (fabsf(determinant) < 1.0e-12f) {
        return false;
    }
    const float inverse = 1.0f / determinant;
    const DVec3 s = sub(ray.origin, triangle.v0);
    const float u = inverse * dot(s, h);
    if (u < 0.0f || u > 1.0f) {
        return false;
    }
    const DVec3 q = cross(s, triangle.edge1);
    const float v = inverse * dot(ray.direction, q);
    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }
    const float t = inverse * dot(triangle.edge2, q);
    if (t < t_min || t > t_max) {
        return false;
    }

    hit_t = t;
    hit_u = u;
    hit_v = v;
    return true;
}

__device__ int triangle_material_id(
    const DScene& scene,
    int instance_index,
    int primitive_index) {
    const int material_slot =
        scene.triangle_material_ids[primitive_index];
    if (material_slot < 0) {
        return instance_index >= 0
            ? 0
            : -1;
    }
    return instance_index >= 0
        ? scene.instances[instance_index].material_offset +
            material_slot
        : material_slot;
}

__device__ int sphere_material_id(
    const DScene& scene,
    int instance_index,
    int primitive_index) {
    const int material_slot =
        scene.sphere_material_ids[primitive_index];
    if (material_slot < 0) {
        return instance_index >= 0
            ? 0
            : -1;
    }
    return instance_index >= 0
        ? scene.instances[instance_index].material_offset +
            material_slot
        : material_slot;
}

__device__ bool triangle_candidate_visible(
    const DScene& scene,
    const DRay& local_ray,
    int instance_index,
    int primitive_index,
    float u,
    float v) {
    const int material_id =
        triangle_material_id(
            scene,
            instance_index,
            primitive_index);
    if (material_id < 0 || material_id >= scene.material_count) {
        return true;
    }
    const DMaterial material = scene.materials[material_id];
    if (!material.two_sided) {
        const DTraversalTriangle& geometry =
            scene.traversal_triangles[primitive_index];
        const DVec3 outward = cross(geometry.edge1, geometry.edge2);
        const float orientation =
            instance_index >= 0
            ? scene.instances[instance_index].orientation_sign
            : 1.0f;
        if (orientation *
                dot(local_ray.direction, outward) >=
            0.0f) {
            return false;
        }
    }
    const int alpha_mode = material.type == static_cast<int>(MaterialType::Pbr)
        ? material.alpha_mode
        : ((material.opacity_texture_id >= 0 || material.opacity < 1.0f)
              ? static_cast<int>(AlphaMode::Mask)
              : static_cast<int>(AlphaMode::Opaque));
    if (alpha_mode == static_cast<int>(AlphaMode::Opaque) ||
        (alpha_mode == static_cast<int>(AlphaMode::Blend) &&
         scene.stochastic_alpha_test == 0)) {
        return true;
    }
    const DShadingTriangle& shading =
        scene.shading_triangles[primitive_index];
    const float w = 1.0f - u - v;
    DHit opacity_hit{};
    opacity_hit.uv = DVec2{
        shading.uvs[0].x * w +
            shading.uvs[1].x * u +
            shading.uvs[2].x * v,
        shading.uvs[0].y * w +
            shading.uvs[1].y * u +
            shading.uvs[2].y * v};
    opacity_hit.uv1 = (shading.uv1_mask & 0x7U) == 0x7U
        ? DVec2{
              shading.uv1s[0].x * w + shading.uv1s[1].x * u + shading.uv1s[2].x * v,
              shading.uv1s[0].y * w + shading.uv1s[1].y * u + shading.uv1s[2].y * v}
        : opacity_hit.uv;
    opacity_hit.vertex_alpha = (shading.color_mask & 0x7U) == 0x7U
        ? shading.alphas[0] * w + shading.alphas[1] * u + shading.alphas[2] * v
        : 1.0f;
    const float opacity = evaluate_alpha(
        scene,
        material,
        opacity_hit,
        alpha_mode).coverage;
    return alpha_mode == static_cast<int>(AlphaMode::Blend)
        ? candidate_alpha_sample(
              local_ray,
              instance_index,
              primitive_index,
              u,
              v) < opacity
        : opacity >= material.alpha_cutoff;
}

__device__ bool sphere_candidate_visible(
    const DScene& scene,
    const DRay& local_ray,
    int instance_index,
    int primitive_index,
    float hit_t) {
    const int material_id =
        sphere_material_id(
            scene,
            instance_index,
            primitive_index);
    if (material_id < 0 || material_id >= scene.material_count) {
        return true;
    }
    const DSphere& sphere = scene.spheres[primitive_index];
    const DVec3 position =
        add(
            local_ray.origin,
            mul(local_ray.direction, hit_t));
    const DVec3 local_outward =
        divv(sub(position, sphere.center), sphere.radius);
    const DVec3 outward =
        instance_index >= 0
        ? transform_direction(
            scene.instances[instance_index].normal_to_world,
            local_outward)
        : local_outward;
    const DVec3 world_direction =
        instance_index >= 0
        ? transform_direction(
            scene.instances[instance_index].object_to_world,
            local_ray.direction)
        : local_ray.direction;
    const bool front_face =
        dot(world_direction, outward) < 0.0f;
    const DMaterial material = scene.materials[material_id];
    if (!material.two_sided && !front_face) {
        return false;
    }
    const int alpha_mode = material.type == static_cast<int>(MaterialType::Pbr)
        ? material.alpha_mode
        : ((material.opacity_texture_id >= 0 || material.opacity < 1.0f)
              ? static_cast<int>(AlphaMode::Mask)
              : static_cast<int>(AlphaMode::Opaque));
    if (alpha_mode == static_cast<int>(AlphaMode::Opaque) ||
        (alpha_mode == static_cast<int>(AlphaMode::Blend) &&
         scene.stochastic_alpha_test == 0)) {
        return true;
    }
    DHit opacity_hit{};
    opacity_hit.uv = DVec2{0.0f, 0.0f};
    opacity_hit.uv1 = opacity_hit.uv;
    opacity_hit.vertex_alpha = 1.0f;
    const float opacity = evaluate_alpha(
        scene,
        material,
        opacity_hit,
        alpha_mode).coverage;
    return alpha_mode == static_cast<int>(AlphaMode::Blend)
        ? candidate_alpha_sample(
              local_ray,
              instance_index,
              primitive_index,
              hit_t,
              0.0f) < opacity
        : opacity >= material.alpha_cutoff;
}

__device__ void sort_child_hits(
    float* near_values,
    int* slots,
    int count) {
    for (int index = 1; index < count; ++index) {
        const float near_value = near_values[index];
        const int slot = slots[index];
        int position = index;
        while (position > 0 &&
               near_values[position - 1] > near_value) {
            near_values[position] = near_values[position - 1];
            slots[position] = slots[position - 1];
            --position;
        }
        near_values[position] = near_value;
        slots[position] = slot;
    }
}

__device__ bool nearest_visible_hit_flat(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max,
    DCompactHit& hit,
    int* error_code) {
    bool found = false;
    float closest = t_max;
    for (int sphere_index = 0; sphere_index < scene.sphere_count; ++sphere_index) {
        float candidate_t = 0.0f;
        if (intersect_sphere_geometry(
                scene.spheres[sphere_index],
                ray,
                t_min,
                closest,
                candidate_t) &&
            sphere_candidate_visible(
                scene,
                ray,
                -1,
                sphere_index,
                candidate_t)) {
            found = true;
            closest = candidate_t;
            hit = DCompactHit{
                candidate_t,
                0.0f,
                0.0f,
                0,
                sphere_index,
                -1};
        }
    }

    if (scene.bvh_node_count <= 0) {
        return found;
    }
    const DVec3 ray_inverse = inverse_direction(ray.direction);
    // Negative stack entries encode ~(parent_node * 4 + child_slot).
    int stack[kBvhStackCapacity];
    int stack_size = 1;
    stack[0] = 0;
    while (stack_size > 0) {
        const int reference = stack[--stack_size];
        if (reference < 0) {
            const int leaf_code = ~reference;
            const int node_index = leaf_code >> 2;
            const int slot = leaf_code & 3;
            const DBvh4Node& leaf_parent = scene.bvh_nodes[node_index];
            for (int offset = 0;
                 offset < leaf_parent.counts[slot];
                 ++offset) {
                const int primitive_index =
                    scene.primitive_indices[
                        leaf_parent.first[slot] + offset];
                float candidate_t = 0.0f;
                float candidate_u = 0.0f;
                float candidate_v = 0.0f;
                if (intersect_triangle_geometry(
                        scene.traversal_triangles[primitive_index],
                        ray,
                        t_min,
                        closest,
                        candidate_t,
                        candidate_u,
                        candidate_v) &&
                    triangle_candidate_visible(
                        scene,
                        ray,
                        -1,
                        primitive_index,
                        candidate_u,
                        candidate_v)) {
                    found = true;
                    closest = candidate_t;
                    hit = DCompactHit{
                        candidate_t,
                        candidate_u,
                        candidate_v,
                        1,
                        primitive_index,
                        -1};
                }
            }
            continue;
        }
        const int node_index = reference;
        const DBvh4Node& node = scene.bvh_nodes[node_index];
        float near_values[4];
        int hit_slots[4];
        int hit_count = 0;
        for (int slot = 0; slot < node.child_count; ++slot) {
            float entry = 0.0f;
            if (intersect_bounds(
                    node.bounds_min[slot],
                    node.bounds_max[slot],
                    ray,
                    ray_inverse,
                    t_min,
                    closest,
                    entry)) {
                near_values[hit_count] = entry;
                hit_slots[hit_count] = slot;
                ++hit_count;
            }
        }
        sort_child_hits(near_values, hit_slots, hit_count);
        for (int hit_index = hit_count - 1; hit_index >= 0; --hit_index) {
            const int slot = hit_slots[hit_index];
            if (near_values[hit_index] > closest) {
                continue;
            }
            if (stack_size >= kBvhStackCapacity) {
                if (error_code) {
                    atomicCAS(error_code, 0, 5);
                }
                return found;
            }
            if (node.counts[slot] > 0) {
                stack[stack_size++] = ~(node_index * 4 + slot);
            } else if (node.children[slot] >= 0) {
                stack[stack_size++] = node.children[slot];
            }
        }
    }
    return found;
}

__device__ bool occluded_scene_flat(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max,
    int* error_code) {
    for (int sphere_index = 0;
         sphere_index < scene.sphere_count;
         ++sphere_index) {
        float candidate_t = 0.0f;
        if (intersect_sphere_geometry(
                scene.spheres[sphere_index],
                ray,
                t_min,
                t_max,
                candidate_t) &&
            sphere_candidate_visible(
                scene,
                ray,
                -1,
                sphere_index,
                candidate_t)) {
            return true;
        }
    }
    if (scene.bvh_node_count <= 0) {
        return false;
    }
    const DVec3 ray_inverse = inverse_direction(ray.direction);
    // Negative stack entries encode ~(parent_node * 4 + child_slot).
    int stack[kBvhStackCapacity];
    int stack_size = 1;
    stack[0] = 0;
    while (stack_size > 0) {
        const int reference = stack[--stack_size];
        if (reference < 0) {
            const int leaf_code = ~reference;
            const int node_index = leaf_code >> 2;
            const int slot = leaf_code & 3;
            const DBvh4Node& leaf_parent = scene.bvh_nodes[node_index];
            for (int offset = 0;
                 offset < leaf_parent.counts[slot];
                 ++offset) {
                const int primitive_index =
                    scene.primitive_indices[
                        leaf_parent.first[slot] + offset];
                float candidate_t = 0.0f;
                float candidate_u = 0.0f;
                float candidate_v = 0.0f;
                if (intersect_triangle_geometry(
                        scene.traversal_triangles[primitive_index],
                        ray,
                        t_min,
                        t_max,
                        candidate_t,
                        candidate_u,
                        candidate_v) &&
                    triangle_candidate_visible(
                        scene,
                        ray,
                        -1,
                        primitive_index,
                        candidate_u,
                        candidate_v)) {
                    return true;
                }
            }
            continue;
        }
        const int node_index = reference;
        const DBvh4Node& node = scene.bvh_nodes[node_index];
        float near_values[4];
        int hit_slots[4];
        int hit_count = 0;
        for (int slot = 0; slot < node.child_count; ++slot) {
            float entry = 0.0f;
            if (intersect_bounds(
                    node.bounds_min[slot],
                    node.bounds_max[slot],
                    ray,
                    ray_inverse,
                    t_min,
                    t_max,
                    entry)) {
                near_values[hit_count] = entry;
                hit_slots[hit_count] = slot;
                ++hit_count;
            }
        }
        sort_child_hits(near_values, hit_slots, hit_count);
        for (int hit_index = hit_count - 1; hit_index >= 0; --hit_index) {
            const int slot = hit_slots[hit_index];
            if (stack_size >= kBvhStackCapacity) {
                if (error_code) {
                    atomicCAS(error_code, 0, 5);
                }
                return false;
            }
            if (node.counts[slot] > 0) {
                stack[stack_size++] = ~(node_index * 4 + slot);
            } else if (node.children[slot] >= 0) {
                stack[stack_size++] = node.children[slot];
            }
        }
    }
    return false;
}

__device__ bool nearest_visible_hit_in_instance(
    const DScene& scene,
    const DRay& world_ray,
    int instance_index,
    float t_min,
    float& closest,
    DCompactHit& hit,
    int* error_code) {
    if (instance_index < 0 ||
        instance_index >= scene.instance_count) {
        return false;
    }
    const DInstance instance = scene.instances[instance_index];
    if (instance.asset_index < 0 ||
        instance.asset_index >= scene.asset_count) {
        if (error_code) {
            atomicCAS(error_code, 0, 7);
        }
        return false;
    }
    const DAsset asset = scene.assets[instance.asset_index];
    const DRay local_ray{
        transform_point(
            instance.world_to_object,
            world_ray.origin),
        transform_direction(
            instance.world_to_object,
            world_ray.direction)};
    bool found = false;
    for (int offset = 0; offset < asset.sphere_count; ++offset) {
        const int primitive_index =
            asset.sphere_first + offset;
        float candidate_t = 0.0f;
        if (intersect_sphere_geometry(
                scene.spheres[primitive_index],
                local_ray,
                t_min,
                closest,
                candidate_t) &&
            sphere_candidate_visible(
                scene,
                local_ray,
                instance_index,
                primitive_index,
                candidate_t)) {
            found = true;
            closest = candidate_t;
            hit = DCompactHit{
                candidate_t,
                0.0f,
                0.0f,
                0,
                primitive_index,
                instance_index};
        }
    }
    if (asset.bvh_node_count <= 0 || asset.bvh_root < 0) {
        return found;
    }

    const DVec3 ray_inverse =
        inverse_direction(local_ray.direction);
    int stack[kBvhStackCapacity];
    int stack_size = 1;
    stack[0] = asset.bvh_root;
    while (stack_size > 0) {
        const int reference = stack[--stack_size];
        if (reference < 0) {
            const int leaf_code = ~reference;
            const int node_index = leaf_code >> 2;
            const int slot = leaf_code & 3;
            const DBvh4Node& leaf_parent =
                scene.bvh_nodes[node_index];
            for (int offset = 0;
                 offset < leaf_parent.counts[slot];
                 ++offset) {
                const int primitive_index =
                    scene.primitive_indices[
                        leaf_parent.first[slot] + offset];
                float candidate_t = 0.0f;
                float candidate_u = 0.0f;
                float candidate_v = 0.0f;
                if (intersect_triangle_geometry(
                        scene.traversal_triangles[
                            primitive_index],
                        local_ray,
                        t_min,
                        closest,
                        candidate_t,
                        candidate_u,
                        candidate_v) &&
                    triangle_candidate_visible(
                        scene,
                        local_ray,
                        instance_index,
                        primitive_index,
                        candidate_u,
                        candidate_v)) {
                    found = true;
                    closest = candidate_t;
                    hit = DCompactHit{
                        candidate_t,
                        candidate_u,
                        candidate_v,
                        1,
                        primitive_index,
                        instance_index};
                }
            }
            continue;
        }
        const DBvh4Node& node = scene.bvh_nodes[reference];
        float near_values[4];
        int hit_slots[4];
        int hit_count = 0;
        for (int slot = 0; slot < node.child_count; ++slot) {
            float entry = 0.0f;
            if (intersect_bounds(
                    node.bounds_min[slot],
                    node.bounds_max[slot],
                    local_ray,
                    ray_inverse,
                    t_min,
                    closest,
                    entry)) {
                near_values[hit_count] = entry;
                hit_slots[hit_count] = slot;
                ++hit_count;
            }
        }
        sort_child_hits(near_values, hit_slots, hit_count);
        for (int index = hit_count - 1; index >= 0; --index) {
            const int slot = hit_slots[index];
            if (near_values[index] > closest) {
                continue;
            }
            if (stack_size >= kBvhStackCapacity) {
                if (error_code) {
                    atomicCAS(error_code, 0, 5);
                }
                return found;
            }
            if (node.counts[slot] > 0) {
                stack[stack_size++] =
                    ~(reference * 4 + slot);
            } else if (node.children[slot] >= 0) {
                stack[stack_size++] = node.children[slot];
            }
        }
    }
    return found;
}

__device__ bool nearest_visible_hit_instanced(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max,
    DCompactHit& hit,
    int* error_code) {
    if (scene.tlas_node_count <= 0 ||
        scene.tlas_nodes == nullptr) {
        bool found = false;
        float closest = t_max;
        for (int instance_index = 0;
             instance_index < scene.instance_count;
             ++instance_index) {
            found =
                nearest_visible_hit_in_instance(
                    scene,
                    ray,
                    instance_index,
                    t_min,
                    closest,
                    hit,
                    error_code) ||
                found;
        }
        return found;
    }

    bool found = false;
    float closest = t_max;
    const DVec3 ray_inverse = inverse_direction(ray.direction);
    int stack[kBvhStackCapacity];
    int stack_size = 1;
    stack[0] = 0;
    while (stack_size > 0) {
        const int reference = stack[--stack_size];
        if (reference < 0) {
            const int leaf_code = ~reference;
            const int node_index = leaf_code >> 2;
            const int slot = leaf_code & 3;
            const DBvh4Node& leaf_parent =
                scene.tlas_nodes[node_index];
            for (int offset = 0;
                 offset < leaf_parent.counts[slot];
                 ++offset) {
                const int instance_index =
                    scene.tlas_primitive_indices[
                        leaf_parent.first[slot] + offset];
                found =
                    nearest_visible_hit_in_instance(
                        scene,
                        ray,
                        instance_index,
                        t_min,
                        closest,
                        hit,
                        error_code) ||
                    found;
            }
            continue;
        }
        const DBvh4Node& node = scene.tlas_nodes[reference];
        float near_values[4];
        int hit_slots[4];
        int hit_count = 0;
        for (int slot = 0; slot < node.child_count; ++slot) {
            float entry = 0.0f;
            if (intersect_bounds(
                    node.bounds_min[slot],
                    node.bounds_max[slot],
                    ray,
                    ray_inverse,
                    t_min,
                    closest,
                    entry)) {
                near_values[hit_count] = entry;
                hit_slots[hit_count] = slot;
                ++hit_count;
            }
        }
        sort_child_hits(near_values, hit_slots, hit_count);
        for (int index = hit_count - 1; index >= 0; --index) {
            const int slot = hit_slots[index];
            if (near_values[index] > closest) {
                continue;
            }
            if (stack_size >= kBvhStackCapacity) {
                if (error_code) {
                    atomicCAS(error_code, 0, 6);
                }
                return found;
            }
            if (node.counts[slot] > 0) {
                stack[stack_size++] =
                    ~(reference * 4 + slot);
            } else if (node.children[slot] >= 0) {
                stack[stack_size++] = node.children[slot];
            }
        }
    }
    return found;
}

__device__ bool occluded_instance(
    const DScene& scene,
    const DRay& world_ray,
    int instance_index,
    float t_min,
    float t_max,
    int* error_code) {
    if (instance_index < 0 ||
        instance_index >= scene.instance_count) {
        return false;
    }
    const DInstance instance = scene.instances[instance_index];
    if (instance.asset_index < 0 ||
        instance.asset_index >= scene.asset_count) {
        if (error_code) {
            atomicCAS(error_code, 0, 7);
        }
        return false;
    }
    const DAsset asset = scene.assets[instance.asset_index];
    const DRay local_ray{
        transform_point(
            instance.world_to_object,
            world_ray.origin),
        transform_direction(
            instance.world_to_object,
            world_ray.direction)};
    for (int offset = 0; offset < asset.sphere_count; ++offset) {
        const int primitive_index =
            asset.sphere_first + offset;
        float candidate_t = 0.0f;
        if (intersect_sphere_geometry(
                scene.spheres[primitive_index],
                local_ray,
                t_min,
                t_max,
                candidate_t) &&
            sphere_candidate_visible(
                scene,
                local_ray,
                instance_index,
                primitive_index,
                candidate_t)) {
            return true;
        }
    }
    if (asset.bvh_node_count <= 0 || asset.bvh_root < 0) {
        return false;
    }
    const DVec3 ray_inverse =
        inverse_direction(local_ray.direction);
    int stack[kBvhStackCapacity];
    int stack_size = 1;
    stack[0] = asset.bvh_root;
    while (stack_size > 0) {
        const int reference = stack[--stack_size];
        if (reference < 0) {
            const int leaf_code = ~reference;
            const int node_index = leaf_code >> 2;
            const int slot = leaf_code & 3;
            const DBvh4Node& leaf_parent =
                scene.bvh_nodes[node_index];
            for (int offset = 0;
                 offset < leaf_parent.counts[slot];
                 ++offset) {
                const int primitive_index =
                    scene.primitive_indices[
                        leaf_parent.first[slot] + offset];
                float candidate_t = 0.0f;
                float candidate_u = 0.0f;
                float candidate_v = 0.0f;
                if (intersect_triangle_geometry(
                        scene.traversal_triangles[
                            primitive_index],
                        local_ray,
                        t_min,
                        t_max,
                        candidate_t,
                        candidate_u,
                        candidate_v) &&
                    triangle_candidate_visible(
                        scene,
                        local_ray,
                        instance_index,
                        primitive_index,
                        candidate_u,
                        candidate_v)) {
                    return true;
                }
            }
            continue;
        }
        const DBvh4Node& node = scene.bvh_nodes[reference];
        float near_values[4];
        int hit_slots[4];
        int hit_count = 0;
        for (int slot = 0; slot < node.child_count; ++slot) {
            float entry = 0.0f;
            if (intersect_bounds(
                    node.bounds_min[slot],
                    node.bounds_max[slot],
                    local_ray,
                    ray_inverse,
                    t_min,
                    t_max,
                    entry)) {
                near_values[hit_count] = entry;
                hit_slots[hit_count] = slot;
                ++hit_count;
            }
        }
        sort_child_hits(near_values, hit_slots, hit_count);
        for (int index = hit_count - 1; index >= 0; --index) {
            const int slot = hit_slots[index];
            if (stack_size >= kBvhStackCapacity) {
                if (error_code) {
                    atomicCAS(error_code, 0, 5);
                }
                return false;
            }
            if (node.counts[slot] > 0) {
                stack[stack_size++] =
                    ~(reference * 4 + slot);
            } else if (node.children[slot] >= 0) {
                stack[stack_size++] = node.children[slot];
            }
        }
    }
    return false;
}

__device__ bool occluded_scene_instanced(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max,
    int* error_code) {
    if (scene.tlas_node_count <= 0 ||
        scene.tlas_nodes == nullptr) {
        for (int instance_index = 0;
             instance_index < scene.instance_count;
             ++instance_index) {
            if (occluded_instance(
                    scene,
                    ray,
                    instance_index,
                    t_min,
                    t_max,
                    error_code)) {
                return true;
            }
        }
        return false;
    }
    const DVec3 ray_inverse = inverse_direction(ray.direction);
    int stack[kBvhStackCapacity];
    int stack_size = 1;
    stack[0] = 0;
    while (stack_size > 0) {
        const int reference = stack[--stack_size];
        if (reference < 0) {
            const int leaf_code = ~reference;
            const int node_index = leaf_code >> 2;
            const int slot = leaf_code & 3;
            const DBvh4Node& leaf_parent =
                scene.tlas_nodes[node_index];
            for (int offset = 0;
                 offset < leaf_parent.counts[slot];
                 ++offset) {
                const int instance_index =
                    scene.tlas_primitive_indices[
                        leaf_parent.first[slot] + offset];
                if (occluded_instance(
                        scene,
                        ray,
                        instance_index,
                        t_min,
                        t_max,
                        error_code)) {
                    return true;
                }
            }
            continue;
        }
        const DBvh4Node& node = scene.tlas_nodes[reference];
        float near_values[4];
        int hit_slots[4];
        int hit_count = 0;
        for (int slot = 0; slot < node.child_count; ++slot) {
            float entry = 0.0f;
            if (intersect_bounds(
                    node.bounds_min[slot],
                    node.bounds_max[slot],
                    ray,
                    ray_inverse,
                    t_min,
                    t_max,
                    entry)) {
                near_values[hit_count] = entry;
                hit_slots[hit_count] = slot;
                ++hit_count;
            }
        }
        sort_child_hits(near_values, hit_slots, hit_count);
        for (int index = hit_count - 1; index >= 0; --index) {
            const int slot = hit_slots[index];
            if (stack_size >= kBvhStackCapacity) {
                if (error_code) {
                    atomicCAS(error_code, 0, 6);
                }
                return false;
            }
            if (node.counts[slot] > 0) {
                stack[stack_size++] =
                    ~(reference * 4 + slot);
            } else if (node.children[slot] >= 0) {
                stack[stack_size++] = node.children[slot];
            }
        }
    }
    return false;
}

__device__ bool nearest_visible_hit(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max,
    DCompactHit& hit,
    int* error_code) {
    return scene.instanced_mode != 0
        ? nearest_visible_hit_instanced(
            scene,
            ray,
            t_min,
            t_max,
            hit,
            error_code)
        : nearest_visible_hit_flat(
            scene,
            ray,
            t_min,
            t_max,
            hit,
            error_code);
}

__device__ bool occluded_scene(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max,
    int* error_code) {
    DScene shadow_scene = scene;
    shadow_scene.stochastic_alpha_test = 1;
    return shadow_scene.instanced_mode != 0
        ? occluded_scene_instanced(
              shadow_scene, ray, t_min, t_max, error_code)
        : occluded_scene_flat(
              shadow_scene, ray, t_min, t_max, error_code);
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
    hit.instance_index = compact.instance_index;
    hit.barycentric_u = compact.u;
    hit.barycentric_v = compact.v;
    if (compact.primitive_kind == 0) {
        const DSphere& sphere = scene.spheres[compact.primitive_index];
        const DVec3 local_position =
            compact.instance_index >= 0
            ? transform_point(
                scene.instances[compact.instance_index]
                    .world_to_object,
                hit.position)
            : hit.position;
        const DVec3 local_outward =
            divv(
                sub(local_position, sphere.center),
                sphere.radius);
        const DVec3 outward =
            compact.instance_index >= 0
            ? transform_direction(
                scene.instances[compact.instance_index]
                    .normal_to_world,
                local_outward)
            : local_outward;
        set_normals(hit, ray, outward, outward);
        hit.uv = DVec2{0.0f, 0.0f};
        hit.uv1 = hit.uv;
        hit.vertex_color = v3(1.0f, 1.0f, 1.0f);
        hit.vertex_alpha = 1.0f;
        hit.tangent = DVec4{0.0f, 0.0f, 0.0f, 0.0f};
        hit.has_tangent = 0;
        hit.material_id = sphere_material_id(
            scene,
            compact.instance_index,
            compact.primitive_index);
        return;
    }

    const DTraversalTriangle& geometry =
        scene.traversal_triangles[compact.primitive_index];
    const DShadingTriangle& triangle =
        scene.shading_triangles[compact.primitive_index];
    const float w = 1.0f - compact.u - compact.v;
    const DVec3 local_geometric =
        normalize(cross(geometry.edge1, geometry.edge2));
    DVec3 local_shading = local_geometric;
    if (reconstruct_shading &&
        (triangle.normal_mask & 0x7U) == 0x7U) {
        local_shading = add(
            add(
                mul(triangle.normals[0], w),
                mul(triangle.normals[1], compact.u)),
            mul(triangle.normals[2], compact.v));
        if (!usable(local_shading)) {
            local_shading = local_geometric;
        } else {
            local_shading = normalize(local_shading);
            if (dot(local_shading, local_geometric) < 0.0f) {
                local_shading = mul(local_shading, -1.0f);
            }
        }
    }
    DVec3 geometric = local_geometric;
    DVec3 shading = local_shading;
    if (compact.instance_index >= 0) {
        const DInstance instance =
            scene.instances[compact.instance_index];
        geometric = mul(
            transform_direction(
                instance.normal_to_world,
                local_geometric),
            instance.orientation_sign);
        shading = transform_direction(
            instance.normal_to_world,
            local_shading);
        if (dot(shading, geometric) < 0.0f) {
            shading = mul(shading, -1.0f);
        }
    }
    hit.uv = DVec2{
        triangle.uvs[0].x * w +
            triangle.uvs[1].x * compact.u +
            triangle.uvs[2].x * compact.v,
        triangle.uvs[0].y * w +
            triangle.uvs[1].y * compact.u +
            triangle.uvs[2].y * compact.v};
    if ((triangle.uv1_mask & 0x7U) == 0x7U) {
        hit.uv1 = DVec2{
            triangle.uv1s[0].x * w +
                triangle.uv1s[1].x * compact.u +
                triangle.uv1s[2].x * compact.v,
            triangle.uv1s[0].y * w +
                triangle.uv1s[1].y * compact.u +
                triangle.uv1s[2].y * compact.v};
    } else {
        hit.uv1 = hit.uv;
    }
    if ((triangle.color_mask & 0x7U) == 0x7U) {
        hit.vertex_color = add(
            add(
                mul(triangle.colors[0], w),
                mul(triangle.colors[1], compact.u)),
            mul(triangle.colors[2], compact.v));
        hit.vertex_alpha = triangle.alphas[0] * w +
            triangle.alphas[1] * compact.u +
            triangle.alphas[2] * compact.v;
    } else {
        hit.vertex_color = v3(1.0f, 1.0f, 1.0f);
        hit.vertex_alpha = 1.0f;
    }
    hit.has_tangent = 0;
    hit.tangent = DVec4{0.0f, 0.0f, 0.0f, 0.0f};
    if ((triangle.tangent_mask & 0x7U) == 0x7U) {
        DVec3 local_tangent = add(
            add(
                mul(v3(triangle.tangents[0].x, triangle.tangents[0].y, triangle.tangents[0].z), w),
                mul(v3(triangle.tangents[1].x, triangle.tangents[1].y, triangle.tangents[1].z), compact.u)),
            mul(v3(triangle.tangents[2].x, triangle.tangents[2].y, triangle.tangents[2].z), compact.v));
        DVec3 world_tangent = local_tangent;
        float handedness = triangle.tangents[0].w * w +
            triangle.tangents[1].w * compact.u +
            triangle.tangents[2].w * compact.v;
        if (compact.instance_index >= 0) {
            const DInstance instance = scene.instances[compact.instance_index];
            world_tangent = transform_direction(instance.object_to_world, local_tangent);
            handedness *= instance.orientation_sign;
        }
        if (usable(world_tangent)) {
            world_tangent = normalize(world_tangent);
            hit.tangent = DVec4{
                world_tangent.x,
                world_tangent.y,
                world_tangent.z,
                handedness < 0.0f ? -1.0f : 1.0f};
            hit.has_tangent = 1;
        }
    }
    set_normals(hit, ray, geometric, shading);
    hit.material_id = triangle_material_id(
        scene,
        compact.instance_index,
        compact.primitive_index);
}

__device__ float addressed_coordinate(float value, int wrap) {
    if (wrap == static_cast<int>(TextureWrap::ClampToEdge)) {
        return fminf(fmaxf(value, 0.0f), 1.0f);
    }
    if (wrap == static_cast<int>(TextureWrap::MirroredRepeat)) {
        const float period = value - floorf(value * 0.5f) * 2.0f;
        return period <= 1.0f ? period : 2.0f - period;
    }
    const float repeated = value - floorf(value);
    return repeated < 0.0f ? repeated + 1.0f : repeated;
}

__device__ int addressed_index(int value, int size, int wrap) {
    if (wrap != static_cast<int>(TextureWrap::Repeat)) {
        return max(0, min(value, size - 1));
    }
    const int wrapped = value % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

__device__ DVec2 transformed_uv(
    DVec2 uv0,
    DVec2 uv1,
    DVec4 transform,
    float rotation,
    int texcoord) {
    DVec2 uv = texcoord == 1 ? uv1 : uv0;
    uv.x *= transform.z;
    uv.y *= transform.w;
    const float cosine = cosf(rotation);
    const float sine = sinf(rotation);
    return DVec2{
        cosine * uv.x - sine * uv.y + transform.x,
        sine * uv.x + cosine * uv.y + transform.y};
}

__device__ DVec3 sample_texture(const DScene& scene, int texture_id, DVec2 uv) {
    if (texture_id < 0 || texture_id >= scene.texture_count) {
        return v3(1.0f, 0.0f, 1.0f);
    }
    const DTexture texture = scene.textures[texture_id];
    if (texture.width <= 0 || texture.height <= 0) {
        return v3(1.0f, 0.0f, 1.0f);
    }
    const float u = addressed_coordinate(uv.x, texture.wrap_s);
    const float source_v = texture.top_left != 0 ? uv.y : 1.0f - uv.y;
    const float v = addressed_coordinate(source_v, texture.wrap_t);
    if (texture.nearest != 0) {
        const int x = addressed_index(
            static_cast<int>(floorf(u * static_cast<float>(texture.width))),
            texture.width,
            texture.wrap_s);
        const int y = addressed_index(
            static_cast<int>(floorf(v * static_cast<float>(texture.height))),
            texture.height,
            texture.wrap_t);
        return scene.texels[texture.first_pixel + y * texture.width + x];
    }
    const float x = u * static_cast<float>(texture.width) - 0.5f;
    const float y = v * static_cast<float>(texture.height) - 0.5f;
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const int ix0 = addressed_index(x0, texture.width, texture.wrap_s);
    const int ix1 = addressed_index(x0 + 1, texture.width, texture.wrap_s);
    const int iy0 = addressed_index(y0, texture.height, texture.wrap_t);
    const int iy1 = addressed_index(y0 + 1, texture.height, texture.wrap_t);
    const DVec3 c00 = scene.texels[texture.first_pixel + iy0 * texture.width + ix0];
    const DVec3 c10 = scene.texels[texture.first_pixel + iy0 * texture.width + ix1];
    const DVec3 c01 = scene.texels[texture.first_pixel + iy1 * texture.width + ix0];
    const DVec3 c11 = scene.texels[texture.first_pixel + iy1 * texture.width + ix1];
    const DVec3 top = add(mul(c00, 1.0f - tx), mul(c10, tx));
    const DVec3 bottom = add(mul(c01, 1.0f - tx), mul(c11, tx));
    return add(mul(top, 1.0f - ty), mul(bottom, ty));
}

__device__ float sample_texture_alpha(const DScene& scene, int texture_id, DVec2 uv) {
    if (texture_id < 0 || texture_id >= scene.texture_count ||
        scene.texture_alphas == nullptr) {
        return 1.0f;
    }
    const DTexture texture = scene.textures[texture_id];
    if (texture.width <= 0 || texture.height <= 0) {
        return 1.0f;
    }
    const float u = addressed_coordinate(uv.x, texture.wrap_s);
    const float source_v = texture.top_left != 0 ? uv.y : 1.0f - uv.y;
    const float v = addressed_coordinate(source_v, texture.wrap_t);
    if (texture.nearest != 0) {
        const int x = addressed_index(
            static_cast<int>(floorf(u * static_cast<float>(texture.width))),
            texture.width,
            texture.wrap_s);
        const int y = addressed_index(
            static_cast<int>(floorf(v * static_cast<float>(texture.height))),
            texture.height,
            texture.wrap_t);
        return scene.texture_alphas[texture.first_alpha + y * texture.width + x];
    }
    const float x = u * static_cast<float>(texture.width) - 0.5f;
    const float y = v * static_cast<float>(texture.height) - 0.5f;
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const int ix0 = addressed_index(x0, texture.width, texture.wrap_s);
    const int ix1 = addressed_index(x0 + 1, texture.width, texture.wrap_s);
    const int iy0 = addressed_index(y0, texture.height, texture.wrap_t);
    const int iy1 = addressed_index(y0 + 1, texture.height, texture.wrap_t);
    const float a00 = scene.texture_alphas[texture.first_alpha + iy0 * texture.width + ix0];
    const float a10 = scene.texture_alphas[texture.first_alpha + iy0 * texture.width + ix1];
    const float a01 = scene.texture_alphas[texture.first_alpha + iy1 * texture.width + ix0];
    const float a11 = scene.texture_alphas[texture.first_alpha + iy1 * texture.width + ix1];
    const float top = a00 * (1.0f - tx) + a10 * tx;
    const float bottom = a01 * (1.0f - tx) + a11 * tx;
    return top * (1.0f - ty) + bottom * ty;
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

__device__ float material_surface_opacity(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit) {
    float opacity = material_opacity(scene, material, hit.uv) * hit.vertex_alpha;
    if (material.base_color_texture_id >= 0 &&
        material.base_color_texture_id < scene.texture_count) {
        const DVec2 uv = transformed_uv(
            hit.uv,
            hit.uv1,
            material.base_color_texture_transform,
            material.base_color_texture_rotation,
            material.base_color_texture_texcoord);
        opacity *= sample_texture_alpha(scene, material.base_color_texture_id, uv);
    }
    return fminf(fmaxf(opacity, 0.0f), 1.0f);
}

void hash_geometry_value(std::uint64_t& hash, std::uint32_t value) {
    hash ^= value;
    hash *= 1099511628211ULL;
}

void hash_geometry_float(std::uint64_t& hash, float value) {
    hash_geometry_value(hash, std::bit_cast<std::uint32_t>(value));
}

template <typename Vector>
void hash_geometry_vector(std::uint64_t& hash, const Vector& value) {
    for (int index = 0; index < value.size(); ++index) {
        hash_geometry_float(hash, value[index]);
    }
}

std::uint64_t scene_geometry_fingerprint(const Scene& scene) {
    std::uint64_t hash = 1469598103934665603ULL;
    hash_geometry_value(hash, static_cast<std::uint32_t>(scene.spheres.size()));
    hash_geometry_value(hash, static_cast<std::uint32_t>(scene.triangles.size()));
    for (const Sphere& sphere : scene.spheres) {
        hash_geometry_vector(hash, sphere.center());
        hash_geometry_float(hash, sphere.radius());
    }
    for (const Triangle& triangle : scene.triangles) {
        for (int vertex_index = 0; vertex_index < 3; ++vertex_index) {
            const TriangleVertex& vertex = triangle.vertex(vertex_index);
            hash_geometry_vector(hash, vertex.position);
            hash_geometry_vector(hash, vertex.uv);
            hash_geometry_vector(hash, vertex.uv1);
            hash_geometry_vector(hash, vertex.normal);
            hash_geometry_vector(hash, vertex.tangent);
            hash_geometry_vector(hash, vertex.color);
            hash_geometry_float(hash, vertex.alpha);
            hash_geometry_value(hash, vertex.has_normal ? 1U : 0U);
            hash_geometry_value(hash, vertex.has_uv1 ? 1U : 0U);
            hash_geometry_value(hash, vertex.has_tangent ? 1U : 0U);
            hash_geometry_value(hash, vertex.has_color ? 1U : 0U);
        }
    }
    return hash;
}

__device__ int effective_alpha_mode(const DMaterial& material) {
    return material.type == static_cast<int>(MaterialType::Pbr)
        ? material.alpha_mode
        : ((material.opacity_texture_id >= 0 || material.opacity < 1.0f)
              ? static_cast<int>(AlphaMode::Mask)
              : static_cast<int>(AlphaMode::Opaque));
}

__device__ DAlphaEvaluation evaluate_alpha(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit) {
    return evaluate_alpha(
        scene,
        material,
        hit,
        effective_alpha_mode(material));
}

__device__ DAlphaEvaluation evaluate_alpha(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit,
    int mode) {
    return DAlphaEvaluation{
        mode,
        mode == static_cast<int>(AlphaMode::Opaque)
            ? 1.0f
            : material_surface_opacity(scene, material, hit)};
}

struct DEnvironmentSample {
    DVec3 direction;
    DVec3 radiance;
    float pdf;
};

__device__ DVec3 rotate_y(DVec3 direction, float radians) {
    const float cosine = cosf(radians);
    const float sine = sinf(radians);
    return v3(
        cosine * direction.x + sine * direction.z,
        direction.y,
        -sine * direction.x + cosine * direction.z);
}

__device__ DVec2 environment_uv(DVec3 direction) {
    direction = normalize(direction);
    float u = atan2f(direction.z, direction.x) / (2.0f * kPi) + 0.5f;
    u -= floorf(u);
    return DVec2{u, acosf(fminf(fmaxf(direction.y, -1.0f), 1.0f)) / kPi};
}

__device__ DVec3 sample_environment_map(const DScene& scene, DVec3 local_direction) {
    if (scene.environment_texels == nullptr || scene.environment_width <= 0 ||
        scene.environment_height <= 0) {
        return scene.environment;
    }
    const DVec2 uv = environment_uv(local_direction);
    const float x = uv.x * static_cast<float>(scene.environment_width) - 0.5f;
    const float y = uv.y * static_cast<float>(scene.environment_height) - 0.5f;
    const int x0 = static_cast<int>(floorf(x));
    const int y0 = max(0, min(static_cast<int>(floorf(y)), scene.environment_height - 1));
    const int y1 = min(y0 + 1, scene.environment_height - 1);
    const int ix0 = addressed_index(x0, scene.environment_width, static_cast<int>(TextureWrap::Repeat));
    const int ix1 = addressed_index(x0 + 1, scene.environment_width, static_cast<int>(TextureWrap::Repeat));
    const float tx = x - floorf(x);
    const float ty = fminf(fmaxf(y - floorf(y), 0.0f), 1.0f);
    const DVec3 a = add(
        mul(scene.environment_texels[y0 * scene.environment_width + ix0], 1.0f - tx),
        mul(scene.environment_texels[y0 * scene.environment_width + ix1], tx));
    const DVec3 b = add(
        mul(scene.environment_texels[y1 * scene.environment_width + ix0], 1.0f - tx),
        mul(scene.environment_texels[y1 * scene.environment_width + ix1], tx));
    return add(mul(a, 1.0f - ty), mul(b, ty));
}

__device__ DVec3 environment_radiance(const DScene& scene, DVec3 world_direction) {
    const DVec3 local = rotate_y(world_direction, -scene.environment_rotation_radians);
    const DVec3 radiance = sample_environment_map(scene, local);
    return mul(
        scene.environment_texels != nullptr
            ? product(scene.environment, radiance)
            : radiance,
        fmaxf(0.0f, scene.environment_intensity));
}

__device__ float environment_pdf(const DScene& scene, DVec3 world_direction) {
    if (scene.environment_pmf == nullptr || scene.environment_width <= 0 ||
        scene.environment_height <= 0) {
        return 1.0f / (4.0f * kPi);
    }
    const DVec2 uv = environment_uv(
        rotate_y(world_direction, -scene.environment_rotation_radians));
    const int x = min(
        static_cast<int>(uv.x * static_cast<float>(scene.environment_width)),
        scene.environment_width - 1);
    const int y = min(
        static_cast<int>(uv.y * static_cast<float>(scene.environment_height)),
        scene.environment_height - 1);
    const float theta0 = kPi * static_cast<float>(y) /
        static_cast<float>(scene.environment_height);
    const float theta1 = kPi * static_cast<float>(y + 1) /
        static_cast<float>(scene.environment_height);
    const float solid_angle = (2.0f * kPi / static_cast<float>(scene.environment_width)) *
        (cosf(theta0) - cosf(theta1));
    return scene.environment_pmf[y * scene.environment_width + x] /
        fmaxf(solid_angle, 1.0e-20f);
}

template<class Rng>
__device__ DEnvironmentSample sample_environment(
    const DScene& scene,
    Rng& rng) {
    if (scene.environment_cdf == nullptr || scene.environment_pmf == nullptr ||
        scene.environment_width <= 0 || scene.environment_height <= 0) {
        const float y = 1.0f - 2.0f * random_float(rng);
        const float radius = sqrtf(fmaxf(0.0f, 1.0f - y * y));
        const float phi = 2.0f * kPi * random_float(rng);
        const DVec3 direction = v3(radius * cosf(phi), y, radius * sinf(phi));
        return DEnvironmentSample{
            direction,
            environment_radiance(scene, direction),
            1.0f / (4.0f * kPi)};
    }
    const int count = scene.environment_width * scene.environment_height;
    const float target = fminf(random_float(rng), 0.99999994f);
    int low = 0;
    int high = count - 1;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (target <= scene.environment_cdf[middle]) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    const int x = low % scene.environment_width;
    const int y = low / scene.environment_width;
    const float theta0 = kPi * static_cast<float>(y) /
        static_cast<float>(scene.environment_height);
    const float theta1 = kPi * static_cast<float>(y + 1) /
        static_cast<float>(scene.environment_height);
    const float cos_theta = cosf(theta0) +
        (cosf(theta1) - cosf(theta0)) * random_float(rng);
    const float u = (static_cast<float>(x) + random_float(rng)) /
        static_cast<float>(scene.environment_width);
    const float phi = (u - 0.5f) * 2.0f * kPi;
    const float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    const DVec3 local_direction = v3(
        cosf(phi) * sin_theta,
        cos_theta,
        sinf(phi) * sin_theta);
    const DVec3 direction = rotate_y(
        local_direction,
        scene.environment_rotation_radians);
    return DEnvironmentSample{
        direction,
        environment_radiance(scene, direction),
        environment_pdf(scene, direction)};
}

__device__ bool intersect_scene_compact(
    const DScene& scene,
    const DRay& ray,
    float t_min,
    float t_max,
    DCompactHit& hit,
    int* error_code = nullptr) {
    return nearest_visible_hit(
        scene,
        ray,
        t_min,
        t_max,
        hit,
        error_code);
}

__device__ DVec3 bumped_normal(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit) {
    const bool normal_map = material.normal_texture_id >= 0 &&
        material.normal_texture_id < scene.texture_count;
    const int texture_id = normal_map
        ? material.normal_texture_id
        : material.bump_texture_id;
    if (texture_id < 0 || texture_id >= scene.texture_count ||
        hit.primitive_kind != 1 || !usable(hit.shading_normal)) {
        return hit.shading_normal;
    }
    const DTexture texture = scene.textures[texture_id];
    if (texture.width <= 0 || texture.height <= 0) {
        return hit.shading_normal;
    }
    const DTraversalTriangle& geometry =
        scene.traversal_triangles[hit.primitive_index];
    const DShadingTriangle& triangle =
        scene.shading_triangles[hit.primitive_index];
    const DVec3 edge1 =
        hit.instance_index >= 0
        ? transform_direction(
            scene.instances[hit.instance_index].object_to_world,
            geometry.edge1)
        : geometry.edge1;
    const DVec3 edge2 =
        hit.instance_index >= 0
        ? transform_direction(
            scene.instances[hit.instance_index].object_to_world,
            geometry.edge2)
        : geometry.edge2;
    DVec2 triangle_uvs[3];
    for (int vertex = 0; vertex < 3; ++vertex) {
        const DVec2 uv1 = (triangle.uv1_mask & 0x7U) == 0x7U
            ? triangle.uv1s[vertex]
            : triangle.uvs[vertex];
        triangle_uvs[vertex] = normal_map
            ? transformed_uv(
                  triangle.uvs[vertex],
                  uv1,
                  material.normal_texture_transform,
                  material.normal_texture_rotation,
                  material.normal_texture_texcoord)
            : triangle.uvs[vertex];
    }
    const float du1 = triangle_uvs[1].x - triangle_uvs[0].x;
    const float dv1 = triangle_uvs[1].y - triangle_uvs[0].y;
    const float du2 = triangle_uvs[2].x - triangle_uvs[0].x;
    const float dv2 = triangle_uvs[2].y - triangle_uvs[0].y;
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
    const DVec2 sample_uv = normal_map
        ? transformed_uv(
              hit.uv,
              hit.uv1,
              material.normal_texture_transform,
              material.normal_texture_rotation,
              material.normal_texture_texcoord)
        : hit.uv;
    if (normal_map) {
        DVec3 mapped = sub(
            mul(sample_texture(scene, texture_id, sample_uv), 2.0f),
            v3(1.0f, 1.0f, 1.0f));
        mapped.x *= material.normal_scale;
        mapped.y *= material.normal_scale;
        const DVec3 candidate = add(
            add(mul(tangent, mapped.x), mul(bitangent, mapped.y)),
            mul(hit.shading_normal, mapped.z));
        if (!usable(candidate)) {
            return hit.shading_normal;
        }
        DVec3 result = normalize(candidate);
        if (dot(result, hit.geometric_normal) < 0.0f) {
            result = mul(result, -1.0f);
        }
        return result;
    }
    const float du = 1.0f / static_cast<float>(texture.width);
    const float dv = 1.0f / static_cast<float>(texture.height);
    const float left = sample_scalar(scene, texture_id, DVec2{sample_uv.x - du, sample_uv.y});
    const float right = sample_scalar(scene, texture_id, DVec2{sample_uv.x + du, sample_uv.y});
    const float down = sample_scalar(scene, texture_id, DVec2{sample_uv.x, sample_uv.y - dv});
    const float up = sample_scalar(scene, texture_id, DVec2{sample_uv.x, sample_uv.y + dv});
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

__device__ float saturate(float value);

__device__ DSurface evaluate_surface(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit) {
    DSurface surface{};
    surface.base_color = product(material.base_color, hit.vertex_color);
    if (material.base_color_texture_id >= 0 &&
        material.base_color_texture_id < scene.texture_count) {
        const DVec2 uv = transformed_uv(
            hit.uv,
            hit.uv1,
            material.base_color_texture_transform,
            material.base_color_texture_rotation,
            material.base_color_texture_texcoord);
        surface.base_color = product(
            surface.base_color,
            sample_texture(scene, material.base_color_texture_id, uv));
    } else if (material.diffuse_texture_id >= 0 &&
               material.diffuse_texture_id < scene.texture_count) {
        surface.base_color = product(
            surface.base_color,
            sample_texture(scene, material.diffuse_texture_id, hit.uv));
    }
    surface.emission = material.emission;
    if (material.emissive_texture_id >= 0 &&
        material.emissive_texture_id < scene.texture_count) {
        const DVec2 uv = transformed_uv(
            hit.uv,
            hit.uv1,
            material.emissive_texture_transform,
            material.emissive_texture_rotation,
            material.emissive_texture_texcoord);
        surface.emission = product(
            surface.emission,
            sample_texture(scene, material.emissive_texture_id, uv));
    }
    surface.metallic = material.type == static_cast<int>(MaterialType::Metal)
        ? 1.0f
        : fminf(fmaxf(material.metallic, 0.0f), 1.0f);
    const bool specular_glossiness =
        material.type == static_cast<int>(MaterialType::Pbr) &&
        material.pbr_workflow == static_cast<int>(PbrWorkflow::SpecularGlossiness);
    if (specular_glossiness) {
        DVec3 specular = v3(
            fmaxf(material.specular_color.x, 0.0f),
            fmaxf(material.specular_color.y, 0.0f),
            fmaxf(material.specular_color.z, 0.0f));
        float glossiness = saturate(material.glossiness);
        if (material.specular_glossiness_texture_id >= 0 &&
            material.specular_glossiness_texture_id < scene.texture_count) {
            const DVec2 uv = transformed_uv(
                hit.uv,
                hit.uv1,
                material.specular_glossiness_texture_transform,
                material.specular_glossiness_texture_rotation,
                material.specular_glossiness_texture_texcoord);
            specular = product(
                specular,
                sample_texture(scene, material.specular_glossiness_texture_id, uv));
            glossiness *= sample_texture_alpha(
                scene,
                material.specular_glossiness_texture_id,
                uv);
        }
        surface.specular_f0 = v3(
            saturate(specular.x),
            saturate(specular.y),
            saturate(specular.z));
        surface.specular_f90 = v3(1.0f, 1.0f, 1.0f);
        surface.diffuse_fresnel_f0 = surface.specular_f0;
        surface.diffuse_fresnel_f90 = surface.specular_f90;
        surface.diffuse_fresnel_uses_max = 0;
        surface.diffuse_color = mul(
            surface.base_color,
            1.0f - saturate(max_component(surface.specular_f0)));
        surface.metallic = 0.0f;
        surface.roughness = fminf(fmaxf(1.0f - glossiness, 0.02f), 1.0f);
    } else {
        surface.roughness = material.type == static_cast<int>(MaterialType::Diffuse)
            ? 1.0f
            : fminf(fmaxf(material.roughness, 0.02f), 1.0f);
        if (material.metallic_roughness_texture_id >= 0 &&
            material.metallic_roughness_texture_id < scene.texture_count) {
            const DVec2 uv = transformed_uv(
                hit.uv,
                hit.uv1,
                material.metallic_roughness_texture_transform,
                material.metallic_roughness_texture_rotation,
                material.metallic_roughness_texture_texcoord);
            const DVec3 packed = sample_texture(
                scene,
                material.metallic_roughness_texture_id,
                uv);
            surface.roughness = fminf(fmaxf(surface.roughness * packed.y, 0.02f), 1.0f);
            surface.metallic = fminf(fmaxf(surface.metallic * packed.z, 0.0f), 1.0f);
        }
        float specular_strength = saturate(material.specular_factor);
        if (material.specular_texture_id >= 0 &&
            material.specular_texture_id < scene.texture_count) {
            const DVec2 uv = transformed_uv(
                hit.uv,
                hit.uv1,
                material.specular_texture_transform,
                material.specular_texture_rotation,
                material.specular_texture_texcoord);
            specular_strength *= sample_texture_alpha(
                scene,
                material.specular_texture_id,
                uv);
        }
        DVec3 specular_color = v3(
            fmaxf(material.specular_color.x, 0.0f),
            fmaxf(material.specular_color.y, 0.0f),
            fmaxf(material.specular_color.z, 0.0f));
        if (material.specular_color_texture_id >= 0 &&
            material.specular_color_texture_id < scene.texture_count) {
            const DVec2 uv = transformed_uv(
                hit.uv,
                hit.uv1,
                material.specular_color_texture_transform,
                material.specular_color_texture_rotation,
                material.specular_color_texture_texcoord);
            specular_color = product(
                specular_color,
                sample_texture(scene, material.specular_color_texture_id, uv));
        }
        const float ior = fmaxf(material.ior, 1.0f);
        const float ratio = (ior - 1.0f) / (ior + 1.0f);
        const float dielectric_base = ratio * ratio;
        const DVec3 dielectric_f0 = mul(
            v3(
                saturate(specular_color.x * dielectric_base),
                saturate(specular_color.y * dielectric_base),
                saturate(specular_color.z * dielectric_base)),
            specular_strength);
        const DVec3 dielectric_f90 = v3(
            specular_strength,
            specular_strength,
            specular_strength);
        surface.specular_f0 = add(
            mul(dielectric_f0, 1.0f - surface.metallic),
            mul(surface.base_color, surface.metallic));
        surface.specular_f90 = add(
            mul(dielectric_f90, 1.0f - surface.metallic),
            mul(v3(1.0f, 1.0f, 1.0f), surface.metallic));
        surface.diffuse_color = mul(surface.base_color, 1.0f - surface.metallic);
        surface.diffuse_fresnel_f0 = dielectric_f0;
        surface.diffuse_fresnel_f90 = dielectric_f90;
        surface.diffuse_fresnel_uses_max = 1;
    }
    surface.occlusion = 1.0f;
    if (material.occlusion_texture_id >= 0 &&
        material.occlusion_texture_id < scene.texture_count) {
        const DVec2 uv = transformed_uv(
            hit.uv,
            hit.uv1,
            material.occlusion_texture_transform,
            material.occlusion_texture_rotation,
            material.occlusion_texture_texcoord);
        const float sampled = sample_texture(
            scene,
            material.occlusion_texture_id,
            uv).x;
        const float strength = fminf(fmaxf(material.occlusion_strength, 0.0f), 1.0f);
        surface.occlusion = 1.0f + (sampled - 1.0f) * strength;
    }
    surface.opacity = material_surface_opacity(scene, material, hit);
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

struct DPbrEvaluation {
    DVec3 brdf;
    float pdf;
    DVec3 diffuse{};
    DVec3 specular{};
};

__device__ float saturate(float value) {
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ float punctual_range_attenuation(float distance, float range) {
    if (!(range > 0.0f)) {
        return 1.0f;
    }
    const float ratio = distance / range;
    const float squared = ratio * ratio;
    const float cutoff = fmaxf(0.0f, 1.0f - squared * squared);
    return cutoff * cutoff;
}

__device__ DVec3 fresnel_schlick(float cosine, DVec3 f0, DVec3 f90) {
    const float factor = powf(1.0f - saturate(cosine), 5.0f);
    return add(f0, mul(sub(f90, f0), factor));
}

__device__ float ggx_distribution(float n_dot_h, float alpha) {
    const float alpha_squared = alpha * alpha;
    const float denominator =
        n_dot_h * n_dot_h * (alpha_squared - 1.0f) + 1.0f;
    return alpha_squared / fmaxf(kPi * denominator * denominator, 1.0e-12f);
}

__device__ float smith_g1(float n_dot_v, float alpha) {
    if (n_dot_v <= 0.0f) {
        return 0.0f;
    }
    const float tangent_squared = fmaxf(
        0.0f,
        (1.0f - n_dot_v * n_dot_v) / fmaxf(n_dot_v * n_dot_v, 1.0e-12f));
    return 2.0f / (1.0f + sqrtf(1.0f + alpha * alpha * tangent_squared));
}

__device__ float smith_lambda(float n_dot_v, float alpha) {
    if (n_dot_v <= 0.0f) {
        return FLT_MAX;
    }
    const float tangent_squared = fmaxf(
        0.0f,
        (1.0f - n_dot_v * n_dot_v) / fmaxf(n_dot_v * n_dot_v, 1.0e-12f));
    return 0.5f * (sqrtf(1.0f + alpha * alpha * tangent_squared) - 1.0f);
}

__device__ float smith_g2(float n_dot_v, float n_dot_l, float alpha) {
    return 1.0f /
        (1.0f + smith_lambda(n_dot_v, alpha) + smith_lambda(n_dot_l, alpha));
}

__device__ float ggx_directional_albedo(float n_dot_v, float roughness) {
    const float x = saturate(n_dot_v) * static_cast<float>(kGgxEnergyLutSize - 1);
    const float y = saturate(roughness) * static_cast<float>(kGgxEnergyLutSize - 1);
    const int x0 = min(static_cast<int>(x), kGgxEnergyLutSize - 2);
    const int y0 = min(static_cast<int>(y), kGgxEnergyLutSize - 2);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float lower_left =
        kGgxDirectionalAlbedoLut[y0 * kGgxEnergyLutSize + x0];
    const float lower_right =
        kGgxDirectionalAlbedoLut[y0 * kGgxEnergyLutSize + x1];
    const float upper_left =
        kGgxDirectionalAlbedoLut[y1 * kGgxEnergyLutSize + x0];
    const float upper_right =
        kGgxDirectionalAlbedoLut[y1 * kGgxEnergyLutSize + x1];
    const float lower = lower_left + (lower_right - lower_left) * tx;
    const float upper = upper_left + (upper_right - upper_left) * tx;
    return saturate(lower + (upper - lower) * ty);
}

__device__ float ggx_average_albedo(float roughness) {
    const float position =
        saturate(roughness) * static_cast<float>(kGgxEnergyLutSize - 1);
    const int lower = min(
        static_cast<int>(position),
        kGgxEnergyLutSize - 2);
    const float blend = position - static_cast<float>(lower);
    return saturate(
        kGgxAverageAlbedoLut[lower] +
        (kGgxAverageAlbedoLut[lower + 1] - kGgxAverageAlbedoLut[lower]) *
            blend);
}

__device__ DVec3 ggx_multiscatter_brdf(
    const DSurface& surface,
    float n_dot_v,
    float n_dot_l,
    float roughness) {
    const float energy_v = ggx_directional_albedo(n_dot_v, roughness);
    const float energy_l = ggx_directional_albedo(n_dot_l, roughness);
    const float average_energy = ggx_average_albedo(roughness);
    const float missing_average = 1.0f - average_energy;
    if (missing_average <= 1.0e-6f) {
        return v3(0.0f, 0.0f, 0.0f);
    }
    const DVec3 average_fresnel = add(
        surface.specular_f0,
        mul(sub(surface.specular_f90, surface.specular_f0), 1.0f / 21.0f));
    const DVec3 denominator = sub(
        v3(1.0f, 1.0f, 1.0f),
        mul(average_fresnel, missing_average));
    const DVec3 numerator = mul(
        product(average_fresnel, average_fresnel),
        average_energy * (1.0f - energy_v) * (1.0f - energy_l) /
            (kPi * missing_average));
    return v3(
        numerator.x / fmaxf(denominator.x, 1.0e-6f),
        numerator.y / fmaxf(denominator.y, 1.0e-6f),
        numerator.z / fmaxf(denominator.z, 1.0e-6f));
}

__device__ float pbr_specular_probability(const DSurface& surface) {
    const DVec3 weights = v3(0.2126f, 0.7152f, 0.0722f);
    const float diffuse_energy = fmaxf(0.0f, dot(surface.diffuse_color, weights));
    const float average_energy = ggx_average_albedo(surface.roughness);
    const float missing_average = 1.0f - average_energy;
    const DVec3 average_fresnel = add(
        surface.specular_f0,
        mul(sub(surface.specular_f90, surface.specular_f0), 1.0f / 21.0f));
    const DVec3 single_scatter = mul(average_fresnel, average_energy);
    const DVec3 multiple_numerator = mul(
        product(average_fresnel, average_fresnel),
        average_energy * missing_average);
    const DVec3 multiple_denominator = sub(
        v3(1.0f, 1.0f, 1.0f),
        mul(average_fresnel, missing_average));
    const DVec3 multiple_scatter = v3(
        multiple_numerator.x / fmaxf(multiple_denominator.x, 1.0e-6f),
        multiple_numerator.y / fmaxf(multiple_denominator.y, 1.0e-6f),
        multiple_numerator.z / fmaxf(multiple_denominator.z, 1.0e-6f));
    const float specular_energy = fmaxf(0.0f, dot(single_scatter, weights));
    const float broad_energy = diffuse_energy +
        fmaxf(0.0f, dot(multiple_scatter, weights));
    const float total = broad_energy + specular_energy;
    if (!(total > 0.0f)) {
        return 0.5f;
    }
    return fminf(fmaxf(specular_energy / total, 0.05f), 0.95f);
}

__device__ DPbrEvaluation evaluate_pbr(
    const DSurface& surface,
    DVec3 normal,
    DVec3 outgoing,
    DVec3 incoming) {
    const float n_dot_v = saturate(dot(normal, outgoing));
    const float n_dot_l = saturate(dot(normal, incoming));
    if (n_dot_v <= 0.0f || n_dot_l <= 0.0f) {
        return DPbrEvaluation{v3(0.0f, 0.0f, 0.0f), 0.0f};
    }
    const DVec3 half_candidate = add(outgoing, incoming);
    if (!usable(half_candidate)) {
        return DPbrEvaluation{v3(0.0f, 0.0f, 0.0f), 0.0f};
    }
    const DVec3 half_vector = normalize(half_candidate);
    const float n_dot_h = saturate(dot(normal, half_vector));
    const float v_dot_h = saturate(dot(outgoing, half_vector));
    const float roughness = fminf(fmaxf(surface.roughness, 0.02f), 1.0f);
    const float alpha = roughness * roughness;
    const float distribution = ggx_distribution(n_dot_h, alpha);
    const float geometry = smith_g2(n_dot_v, n_dot_l, alpha);
    const DVec3 fresnel = fresnel_schlick(
        v_dot_h,
        surface.specular_f0,
        surface.specular_f90);
    const DVec3 diffuse_fresnel = fresnel_schlick(
        v_dot_h,
        surface.diffuse_fresnel_f0,
        surface.diffuse_fresnel_f90);
    const DVec3 specular_single_scatter = mul(
        fresnel,
        distribution * geometry / fmaxf(4.0f * n_dot_v * n_dot_l, 1.0e-12f));
    const DVec3 specular = add(
        specular_single_scatter,
        ggx_multiscatter_brdf(surface, n_dot_v, n_dot_l, roughness));
    const DVec3 diffuse_weight = surface.diffuse_fresnel_uses_max != 0
        ? v3(
              1.0f - max_component(diffuse_fresnel),
              1.0f - max_component(diffuse_fresnel),
              1.0f - max_component(diffuse_fresnel))
        : sub(v3(1.0f, 1.0f, 1.0f), diffuse_fresnel);
    const DVec3 diffuse = mul(
        product(surface.diffuse_color, diffuse_weight),
        1.0f / kPi);
    const float diffuse_pdf = n_dot_l / kPi;
    const float specular_pdf = distribution * smith_g1(n_dot_v, alpha) /
        fmaxf(4.0f * n_dot_v, 1.0e-12f);
    const float probability = pbr_specular_probability(surface);
    return DPbrEvaluation{
        add(diffuse, specular),
        (1.0f - probability) * diffuse_pdf + probability * specular_pdf,
        diffuse, specular};
}

__device__ void tangent_basis(
    DVec3 normal,
    DVec3& tangent,
    DVec3& bitangent) {
    const DVec3 helper = fabsf(normal.z) < 0.999f
        ? v3(0.0f, 0.0f, 1.0f)
        : v3(1.0f, 0.0f, 0.0f);
    tangent = normalize(cross(helper, normal));
    bitangent = cross(normal, tangent);
}

__device__ DVec3 to_local(
    DVec3 direction,
    DVec3 tangent,
    DVec3 bitangent,
    DVec3 normal) {
    return v3(dot(direction, tangent), dot(direction, bitangent), dot(direction, normal));
}

__device__ DVec3 sample_visible_ggx(
    DVec3 view,
    float alpha,
    float u1,
    float u2) {
    DVec3 stretched = normalize(v3(alpha * view.x, alpha * view.y, view.z));
    const float length_xy = stretched.x * stretched.x + stretched.y * stretched.y;
    const DVec3 tangent1 = length_xy > 1.0e-12f
        ? divv(v3(-stretched.y, stretched.x, 0.0f), sqrtf(length_xy))
        : v3(1.0f, 0.0f, 0.0f);
    const DVec3 tangent2 = cross(stretched, tangent1);
    const float radius = sqrtf(saturate(u1));
    const float phi = 2.0f * kPi * saturate(u2);
    const float t1 = radius * cosf(phi);
    float t2 = radius * sinf(phi);
    const float blend = 0.5f * (1.0f + stretched.z);
    t2 = (1.0f - blend) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + blend * t2;
    const float t3 = sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2));
    const DVec3 micro_normal = add(
        add(mul(tangent1, t1), mul(tangent2, t2)),
        mul(stretched, t3));
    return normalize(v3(
        alpha * micro_normal.x,
        alpha * micro_normal.y,
        fmaxf(0.0f, micro_normal.z)));
}

template<class Rng>
__device__ bool scatter(
    const DRay& ray,
    const DHit& hit,
    const DMaterial& material,
    const DSurface& surface,
    Rng& rng,
    DVec3& attenuation,
    DRay& scattered,
    float& bsdf_pdf,
    int& was_delta) {
    if (material.type == static_cast<int>(MaterialType::Diffuse) ||
        material.type == static_cast<int>(MaterialType::Metal) ||
        material.type == static_cast<int>(MaterialType::Pbr)) {
        const DVec3 outgoing = normalize(mul(ray.direction, -1.0f));
        DVec3 tangent;
        DVec3 bitangent;
        tangent_basis(surface.shading_normal, tangent, bitangent);
        DVec3 direction{};
        if (random_float(rng) < pbr_specular_probability(surface)) {
            const DVec3 view_local = to_local(
                outgoing,
                tangent,
                bitangent,
                surface.shading_normal);
            const DVec3 half_local = sample_visible_ggx(
                view_local,
                surface.roughness * surface.roughness,
                random_float(rng),
                random_float(rng));
            const DVec3 half_world = normalize(add(
                add(
                    mul(tangent, half_local.x),
                    mul(bitangent, half_local.y)),
                mul(surface.shading_normal, half_local.z)));
            direction = normalize(add(
                mul(outgoing, -1.0f),
                mul(half_world, 2.0f * dot(outgoing, half_world))));
        } else {
            direction = tangent_to_world(
                cosine_weighted_hemisphere(rng),
                surface.shading_normal);
        }
        const DPbrEvaluation evaluated = evaluate_pbr(
            surface,
            surface.shading_normal,
            outgoing,
            direction);
        const float cosine = fmaxf(0.0f, dot(surface.shading_normal, direction));
        if (!(evaluated.pdf > 0.0f) || cosine <= 0.0f || !finite(evaluated.brdf)) {
            return false;
        }
        attenuation = mul(evaluated.brdf, cosine / evaluated.pdf);
        scattered = DRay{offset_origin(hit.position, hit.geometric_normal, direction), direction};
        bsdf_pdf = evaluated.pdf;
        was_delta = 0;
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
    const DSurface& surface,
    DVec3 outgoing,
    int* error_code
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    , unsigned long long* diagnostic_counters
#endif
    ) {
    DVec3 direct = v3(0.0f, 0.0f, 0.0f);
    const auto evaluated_contribution = [&](DVec3 direction, DVec3 incoming) {
        const DPbrEvaluation evaluated = evaluate_pbr(
            surface,
            surface.shading_normal,
            outgoing,
            direction);
        return mul(
            product(evaluated.brdf, incoming),
            fmaxf(0.0f, dot(surface.shading_normal, direction)));
    };
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
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        atomicAdd(
            diagnostic_counters + kDiagnosticDirectionalShadowRays,
            1ULL);
#endif
        if (!light.casts_shadows || !occluded_scene(
                scene,
                shadow,
                0.0f,
                1.0e30f,
                error_code)) {
            direct = add(direct, evaluated_contribution(light_direction, light.radiance));
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
        const float range_attenuation =
            punctual_range_attenuation(distance, light.range);
        if (range_attenuation <= 0.0f) {
            continue;
        }
        const DVec3 light_direction = divv(to_light, distance);
        const float cosine = fmaxf(0.0f, dot(surface.shading_normal, light_direction));
        if (cosine <= 0.0f) {
            continue;
        }
        const DRay shadow{offset_origin(hit.position, hit.geometric_normal, light_direction), light_direction};
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        atomicAdd(
            diagnostic_counters + kDiagnosticPointShadowRays,
            1ULL);
#endif
        if (!light.casts_shadows || !occluded_scene(
                scene,
                shadow,
                0.0f,
                distance - 1.0e-7f,
                error_code)) {
            direct = add(
                direct,
                evaluated_contribution(
                    light_direction,
                    mul(light.intensity, range_attenuation / distance_squared)));
        }
    }
    for (int index = 0; index < scene.spot_light_count; ++index) {
        const DSpotLight light = scene.spot_lights[index];
        const DVec3 to_light = sub(light.position, hit.position);
        const float distance_squared = length_squared(to_light);
        if (!(distance_squared > 1.0e-12f)) {
            continue;
        }
        const float distance = sqrtf(distance_squared);
        const float range_attenuation =
            punctual_range_attenuation(distance, light.range);
        if (range_attenuation <= 0.0f) {
            continue;
        }
        const DVec3 light_direction = divv(to_light, distance);
        const float cone_cosine = dot(mul(light_direction, -1.0f), normalize(light.direction));
        const float cone = light.inner_cosine <= light.outer_cosine
            ? (cone_cosine >= light.outer_cosine ? 1.0f : 0.0f)
            : saturate(
                  (cone_cosine - light.outer_cosine) /
                  (light.inner_cosine - light.outer_cosine));
        if (cone <= 0.0f || dot(surface.shading_normal, light_direction) <= 0.0f) {
            continue;
        }
        const DRay shadow{
            offset_origin(hit.position, hit.geometric_normal, light_direction),
            light_direction};
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        atomicAdd(
            diagnostic_counters + kDiagnosticSpotShadowRays,
            1ULL);
#endif
        if (!light.casts_shadows || !occluded_scene(
                scene,
                shadow,
                0.0f,
                distance - 1.0e-7f,
                error_code)) {
            direct = add(
                direct,
                evaluated_contribution(
                    light_direction,
                    mul(
                        light.intensity,
                        cone * range_attenuation / distance_squared)));
        }
    }
    return direct;
}

__device__ int emissive_light_index(const DScene& scene, const DHit& hit) {
    if (hit.instance_index >= 0) {
        for (int index = 0;
             index < scene.emissive_light_count;
             ++index) {
            const DEmissiveLight light =
                scene.emissive_lights[index];
            if (light.instance_index == hit.instance_index &&
                light.primitive_kind == hit.primitive_kind &&
                light.primitive_index == hit.primitive_index) {
                return index;
            }
        }
        return -1;
    }
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

__device__ bool environment_direct_active(const DScene& scene) {
    return scene.environment_intensity > 0.0f &&
        (scene.environment_texels != nullptr || max_component(scene.environment) > 0.0f);
}

__device__ float environment_strategy_probability(const DScene& scene) {
    if (!environment_direct_active(scene)) {
        return 0.0f;
    }
    return scene.emissive_light_count > 0 ? 0.5f : 1.0f;
}

__device__ float emissive_strategy_probability(const DScene& scene) {
    if (scene.emissive_light_count <= 0) {
        return 0.0f;
    }
    return environment_direct_active(scene) ? 0.5f : 1.0f;
}

__device__ float sphere_surface_jacobian(
    const DInstance& instance,
    DVec3 local_normal) {
    const float determinant =
        fabsf(linear_determinant(instance.object_to_world));
    const DVec3 transformed_normal =
        transform_direction(
            instance.normal_to_world,
            local_normal);
    return determinant *
        sqrtf(fmaxf(0.0f, length_squared(transformed_normal)));
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
    float sampled_area = light.area;
    if (light.primitive_kind == 0 &&
        light.instance_index >= 0) {
        const DInstance instance =
            scene.instances[light.instance_index];
        const DSphere sphere =
            scene.spheres[light.primitive_index];
        const DVec3 local_position =
            transform_point(
                instance.world_to_object,
                hit.position);
        const DVec3 local_normal =
            divv(
                sub(local_position, sphere.center),
                sphere.radius);
        sampled_area *=
            sphere_surface_jacobian(instance, local_normal);
    }
    return sampled_area > 0.0f
        ? emissive_strategy_probability(scene) * light.selection_pdf * distance_squared /
            (sampled_area * light_cosine)
        : 0.0f;
}

template<class Rng>
__device__ bool sample_emissive_shadow_task(
    const DScene& scene,
    const DHit& hit,
    const DSurface& surface,
    DVec3 outgoing,
    DVec3 throughput,
    int pixel_index,
    Rng& rng,
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
    DVec2 light_uv1{0.0f, 0.0f};
    float light_vertex_alpha = 1.0f;
    float sampled_area = light.area;
    if (light.primitive_kind == 0) {
        if (light.primitive_index < 0 ||
            light.primitive_index >= scene.sphere_count) {
            return false;
        }
        const DSphere sphere = scene.spheres[light.primitive_index];
        const float z = 1.0f - 2.0f * random_float(rng);
        const float radial = sqrtf(fmaxf(0.0f, 1.0f - z * z));
        const float phi = 2.0f * kPi * random_float(rng);
        const DVec3 local_normal =
            v3(radial * cosf(phi), radial * sinf(phi), z);
        const DVec3 local_position =
            add(sphere.center, mul(local_normal, sphere.radius));
        if (light.instance_index >= 0) {
            const DInstance instance =
                scene.instances[light.instance_index];
            light_position = transform_point(
                instance.object_to_world,
                local_position);
            outward_normal = normalize(
                transform_direction(
                    instance.normal_to_world,
                    local_normal));
            sampled_area *= sphere_surface_jacobian(
                instance,
                local_normal);
        } else {
            outward_normal = local_normal;
            light_position = local_position;
        }
    } else if (light.primitive_kind == 1) {
        if (light.primitive_index < 0 ||
            light.primitive_index >= scene.triangle_count) {
            return false;
        }
        const DTraversalTriangle triangle =
            scene.traversal_triangles[light.primitive_index];
        const DShadingTriangle shading =
            scene.shading_triangles[light.primitive_index];
        const float root = sqrtf(random_float(rng));
        const float w0 = 1.0f - root;
        const float w1 = root * (1.0f - random_float(rng));
        const float w2 = 1.0f - w0 - w1;
        const DVec3 local_position = add(
            triangle.v0,
            add(
                mul(triangle.edge1, w1),
                mul(triangle.edge2, w2)));
        const DVec3 local_normal =
            normalize(cross(triangle.edge1, triangle.edge2));
        if (light.instance_index >= 0) {
            const DInstance instance =
                scene.instances[light.instance_index];
            light_position = transform_point(
                instance.object_to_world,
                local_position);
            outward_normal = normalize(mul(
                transform_direction(
                    instance.normal_to_world,
                    local_normal),
                instance.orientation_sign));
        } else {
            light_position = local_position;
            outward_normal = local_normal;
        }
        light_uv = DVec2{
            shading.uvs[0].x * w0 +
                shading.uvs[1].x * w1 +
                shading.uvs[2].x * w2,
            shading.uvs[0].y * w0 +
                shading.uvs[1].y * w1 +
                shading.uvs[2].y * w2};
        light_uv1 = (shading.uv1_mask & 0x7U) == 0x7U
            ? DVec2{
                  shading.uv1s[0].x * w0 +
                      shading.uv1s[1].x * w1 +
                      shading.uv1s[2].x * w2,
                  shading.uv1s[0].y * w0 +
                      shading.uv1s[1].y * w1 +
                      shading.uv1s[2].y * w2}
            : light_uv;
        light_vertex_alpha = (shading.color_mask & 0x7U) == 0x7U
            ? shading.alphas[0] * w0 +
                shading.alphas[1] * w1 +
                shading.alphas[2] * w2
            : 1.0f;
    } else {
        return false;
    }

    const DMaterial light_material = scene.materials[light.material_id];
    DHit light_surface_hit{};
    light_surface_hit.uv = light_uv;
    light_surface_hit.uv1 = light_uv1;
    light_surface_hit.vertex_alpha = light_vertex_alpha;
    const DAlphaEvaluation light_alpha = evaluate_alpha(
        scene,
        light_material,
        light_surface_hit);
    if (light_alpha.mode == static_cast<int>(AlphaMode::Mask) &&
        light_alpha.coverage < light_material.alpha_cutoff) {
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
        sampled_area > 0.0f
        ? emissive_strategy_probability(scene) * light.selection_pdf * distance_squared /
            (sampled_area * light_cosine)
        : 0.0f;
    if (!(light_pdf > 0.0f) || !isfinite(light_pdf)) {
        return false;
    }
    const DPbrEvaluation evaluated = evaluate_pbr(
        surface,
        surface.shading_normal,
        outgoing,
        light_direction);
    const float mis_weight = power_heuristic(light_pdf, evaluated.pdf);
    DVec3 light_emission = light_material.emission;
    if (light_material.emissive_texture_id >= 0 &&
        light_material.emissive_texture_id < scene.texture_count) {
        const DVec2 emissive_uv = transformed_uv(
            light_uv,
            light_uv1,
            light_material.emissive_texture_transform,
            light_material.emissive_texture_rotation,
            light_material.emissive_texture_texcoord);
        light_emission = product(
            light_emission,
            sample_texture(scene, light_material.emissive_texture_id, emissive_uv));
    }
    if (light_alpha.mode == static_cast<int>(AlphaMode::Blend)) {
        light_emission = mul(light_emission, light_alpha.coverage);
    }
    const DVec3 contribution = mul(
        product(
            product(throughput, evaluated.brdf),
            light_emission),
        surface_cosine * mis_weight / light_pdf);
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
    task.casts_shadows = light.casts_shadows;
    return task.t_max > 0.0f;
}

template<class Rng>
__device__ bool sample_environment_shadow_task(
    const DScene& scene,
    const DHit& hit,
    const DSurface& surface,
    DVec3 outgoing,
    DVec3 throughput,
    int pixel_index,
    Rng& rng,
    DShadowTask& task) {
    const float strategy_probability = environment_strategy_probability(scene);
    if (!(strategy_probability > 0.0f)) {
        return false;
    }
    const DEnvironmentSample light = sample_environment(scene, rng);
    const float cosine = dot(surface.shading_normal, light.direction);
    const float light_pdf = strategy_probability * light.pdf;
    if (!(light_pdf > 0.0f) || cosine <= 0.0f ||
        max_component(light.radiance) <= 0.0f) {
        return false;
    }
    const DPbrEvaluation evaluated = evaluate_pbr(
        surface,
        surface.shading_normal,
        outgoing,
        light.direction);
    const float mis_weight = power_heuristic(light_pdf, evaluated.pdf);
    const DVec3 contribution = mul(
        product(product(throughput, evaluated.brdf), light.radiance),
        cosine * mis_weight * surface.occlusion / light_pdf);
    if (!finite(contribution) || max_component(contribution) <= 0.0f) {
        return false;
    }
    task = DShadowTask{
        DRay{
            offset_origin(hit.position, hit.geometric_normal, light.direction),
            light.direction},
        contribution,
        1.0e30f,
        pixel_index};
    return true;
}

#if !defined(RTRT_OPTIX_DEVICE)
__global__ void refit_tlas_kernel(
    DBvh4Node* nodes,
    int node_count,
    const int* primitive_indices,
    const DInstance* instances,
    int instance_count,
    int* error_code) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (int node_index = node_count - 1;
         node_index >= 0;
         --node_index) {
        DBvh4Node& node = nodes[node_index];
        for (int slot = 0; slot < node.child_count; ++slot) {
            DVec3 minimum = v3(FLT_MAX, FLT_MAX, FLT_MAX);
            DVec3 maximum = v3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            if (node.counts[slot] > 0) {
                for (int offset = 0;
                     offset < node.counts[slot];
                     ++offset) {
                    const int instance_index =
                        primitive_indices[
                            node.first[slot] + offset];
                    if (instance_index < 0 ||
                        instance_index >= instance_count) {
                        if (error_code) {
                            atomicCAS(error_code, 0, 7);
                        }
                        continue;
                    }
                    const DInstance instance =
                        instances[instance_index];
                    minimum = v3(
                        fminf(minimum.x, instance.bounds_min.x),
                        fminf(minimum.y, instance.bounds_min.y),
                        fminf(minimum.z, instance.bounds_min.z));
                    maximum = v3(
                        fmaxf(maximum.x, instance.bounds_max.x),
                        fmaxf(maximum.y, instance.bounds_max.y),
                        fmaxf(maximum.z, instance.bounds_max.z));
                }
            } else if (node.children[slot] >= 0 &&
                       node.children[slot] < node_count) {
                const DBvh4Node& child =
                    nodes[node.children[slot]];
                for (int child_slot = 0;
                     child_slot < child.child_count;
                     ++child_slot) {
                    minimum = v3(
                        fminf(
                            minimum.x,
                            child.bounds_min[child_slot].x),
                        fminf(
                            minimum.y,
                            child.bounds_min[child_slot].y),
                        fminf(
                            minimum.z,
                            child.bounds_min[child_slot].z));
                    maximum = v3(
                        fmaxf(
                            maximum.x,
                            child.bounds_max[child_slot].x),
                        fmaxf(
                            maximum.y,
                            child.bounds_max[child_slot].y),
                        fmaxf(
                            maximum.z,
                            child.bounds_max[child_slot].z));
                }
            } else {
                if (error_code) {
                    atomicCAS(error_code, 0, 7);
                }
            }
            node.bounds_min[slot] = minimum;
            node.bounds_max[slot] = maximum;
        }
    }
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

    void resize_exact(
        std::size_t count,
        CudaPathStatistics& statistics) {
        count_ = count;
        if (count == capacity_) {
            return;
        }
        T* replacement = nullptr;
        if (count > 0) {
            check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&replacement),
                    count * sizeof(T)),
                "cudaMalloc exact");
        }
        if (data_) {
            check_cuda(cudaFree(data_), "cudaFree exact");
        }
        data_ = replacement;
        capacity_ = count;
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
        const RenderSceneSnapshot& scene,
        cudaStream_t stream,
        CudaPathStatistics& statistics)
        : stream_(stream), statistics_(statistics) {
        sync(scene, SceneChange::All);
    }

    DScene view() const { return view_; }

    void sync(
        const RenderSceneSnapshot& scene,
        SceneChangeSet changes) {
        upload_timer_.finish(statistics_.upload_milliseconds);
        upload_timer_.begin(stream_);
        const bool rebuild_assets =
            !same_instanced_asset_layout(scene);
        const bool rebuild_topology =
            !rebuild_assets &&
            (scene.instances.size() != instances_host_.size() ||
             has_scene_change(changes, SceneChange::Geometry));
        if (rebuild_assets) {
            rebuild_instanced_scene(scene);
        } else {
            if (has_scene_change(
                    changes,
                    SceneChange::Textures)) {
                update_instanced_textures(scene);
            }
            if (rebuild_topology) {
                rebuild_instanced_topology(scene);
            } else {
                const bool material_bindings_changed =
                    has_scene_change(
                        changes,
                        SceneChange::MaterialBindings);
                if (material_bindings_changed) {
                    update_instanced_material_bindings(scene);
                }
                if (has_scene_change(
                        changes,
                        SceneChange::Materials) ||
                    has_scene_change(
                        changes,
                        SceneChange::Textures)) {
                    update_instanced_materials(scene);
                } else if (material_bindings_changed) {
                    rebuild_instanced_emissive_lights(scene);
                    has_instanced_emissive_candidates_ =
                        !emissive_lights_host_.empty();
                }
                if (has_scene_change(
                        changes,
                        SceneChange::InstanceTransforms)) {
                    update_instanced_transforms(scene);
                }
            }
        }
        if (!rebuild_assets &&
            has_scene_change(changes, SceneChange::Lighting)) {
            update_instanced_lighting(scene);
            bool shadow_policy_changed = false;
            for (auto& light : emissive_lights_host_) {
                if (light.instance_index < 0) continue;
                const int casts = scene.instances[std::size_t(light.instance_index)].emission_casts_shadows ? 1 : 0;
                shadow_policy_changed |= light.casts_shadows != casts;
                light.casts_shadows = casts;
            }
            if (shadow_policy_changed)
                statistics_.lighting_upload_bytes += emissive_lights_.upload(emissive_lights_host_, stream_, statistics_);
        }
        if (has_scene_change(changes, SceneChange::Environment) || rebuild_assets) {
            upload_environment(
                scene.environment,
                scene.environment_map,
                scene.environment_intensity,
                scene.environment_rotation_degrees,
                scene.environment_background_visible);
        }
        update_instanced_view();
        upload_timer_.end(stream_);
    }

    void update_timing() {
        upload_timer_.update(statistics_.upload_milliseconds);
        instance_upload_timer_.update(
            statistics_.instance_upload_milliseconds);
        tlas_refit_timer_.update(
            statistics_.tlas_refit_milliseconds);
    }

private:
    struct CachedAssetBlas {
        std::uint64_t geometry_revision = 0;
        std::uint64_t geometry_fingerprint = 0;
        GpuBvh4Layout layout;
    };

    void upload_environment(
        const Color& fallback,
        const std::shared_ptr<const EnvironmentMap>& map,
        float intensity,
        float rotation_degrees,
        bool background_visible) {
        environment_texels_host_.clear();
        environment_pmf_host_.clear();
        environment_cdf_host_.clear();
        if (map) {
            environment_texels_host_.reserve(map->pixels().size());
            for (const Color& color : map->pixels()) {
                environment_texels_host_.push_back(to_device(color));
            }
            environment_pmf_host_ = map->importance_pmf();
            environment_cdf_host_ = map->importance_cdf();
        }
        statistics_.lighting_upload_bytes += environment_texels_.upload(
            environment_texels_host_, stream_, statistics_);
        statistics_.lighting_upload_bytes += environment_pmf_.upload(
            environment_pmf_host_, stream_, statistics_);
        statistics_.lighting_upload_bytes += environment_cdf_.upload(
            environment_cdf_host_, stream_, statistics_);
        view_.environment = to_device(fallback);
        view_.environment_texels = environment_texels_.get();
        view_.environment_pmf = environment_pmf_.get();
        view_.environment_cdf = environment_cdf_.get();
        view_.environment_width = map ? map->width() : 0;
        view_.environment_height = map ? map->height() : 0;
        view_.environment_intensity = std::max(0.0f, intensity);
        view_.environment_rotation_radians =
            rotation_degrees * kPi / 180.0f;
        view_.environment_background_visible = background_visible ? 1 : 0;
    }

    bool same_instanced_asset_layout(
        const RenderSceneSnapshot& scene) const {
        if (!instanced_assets_initialized_ ||
            scene.assets.size() !=
                asset_ids_host_.size() ||
            scene.assets.size() !=
                asset_geometry_revisions_host_.size() ||
            scene.assets.size() !=
                asset_geometry_fingerprints_host_.size()) {
            return false;
        }
        for (std::size_t index = 0;
             index < scene.assets.size();
             ++index) {
            if (scene.assets[index].asset_id !=
                    asset_ids_host_[index] ||
                scene.assets[index].geometry_revision !=
                    asset_geometry_revisions_host_[index] ||
                scene.assets[index].local_scene.get() !=
                    asset_scenes_host_[index] ||
                !scene.assets[index].local_scene ||
                scene_geometry_fingerprint(
                    *scene.assets[index].local_scene) !=
                    asset_geometry_fingerprints_host_[index]) {
                return false;
            }
        }
        return true;
    }

    static DMaterial pack_material(
        const Material& material,
        int texture_base) {
        const auto texture_id = [texture_base](int id) {
            return id >= 0 ? texture_base + id : -1;
        };
        // Named-field construction: a reordered DMaterial cannot silently
        // swap values the way the previous ~50-field positional aggregate
        // initializer could.
        DMaterial result{};
        result.type = static_cast<int>(material.type);
        result.base_color = to_device(material.base_color);
        result.emission = to_device(material.emission);
        result.roughness = material.roughness;
        result.metallic = material.metallic;
        result.ior = material.ior;
        result.opacity = material.opacity;
        result.alpha_cutoff = material.alpha_cutoff;
        result.bump_scale = material.bump_scale;
        result.normal_scale = material.normal_scale;
        result.occlusion_strength = material.occlusion_strength;
        result.alpha_mode = static_cast<int>(material.alpha_mode);
        result.two_sided = material.two_sided ? 1 : 0;
        result.diffuse_texture_id = texture_id(material.diffuse_texture_id);
        result.opacity_texture_id = texture_id(material.opacity_texture_id);
        result.bump_texture_id = texture_id(material.bump_texture_id);
        result.base_color_texture_id =
            texture_id(material.base_color_texture_id);
        result.metallic_roughness_texture_id =
            texture_id(material.metallic_roughness_texture_id);
        result.normal_texture_id = texture_id(material.normal_texture_id);
        result.occlusion_texture_id =
            texture_id(material.occlusion_texture_id);
        result.emissive_texture_id =
            texture_id(material.emissive_texture_id);
        result.base_color_texture_transform =
            pack_texture_transform(material.base_color_texture_transform);
        result.metallic_roughness_texture_transform =
            pack_texture_transform(material.metallic_roughness_texture_transform);
        result.normal_texture_transform =
            pack_texture_transform(material.normal_texture_transform);
        result.occlusion_texture_transform =
            pack_texture_transform(material.occlusion_texture_transform);
        result.emissive_texture_transform =
            pack_texture_transform(material.emissive_texture_transform);
        result.base_color_texture_rotation =
            material.base_color_texture_transform.rotation;
        result.metallic_roughness_texture_rotation =
            material.metallic_roughness_texture_transform.rotation;
        result.normal_texture_rotation =
            material.normal_texture_transform.rotation;
        result.occlusion_texture_rotation =
            material.occlusion_texture_transform.rotation;
        result.emissive_texture_rotation =
            material.emissive_texture_transform.rotation;
        result.base_color_texture_texcoord =
            material.base_color_texture_transform.texcoord;
        result.metallic_roughness_texture_texcoord =
            material.metallic_roughness_texture_transform.texcoord;
        result.normal_texture_texcoord =
            material.normal_texture_transform.texcoord;
        result.occlusion_texture_texcoord =
            material.occlusion_texture_transform.texcoord;
        result.emissive_texture_texcoord =
            material.emissive_texture_transform.texcoord;
        result.pbr_workflow = static_cast<int>(material.pbr_workflow);
        result.specular_color = to_device(material.specular_color);
        result.specular_factor = material.specular_factor;
        result.glossiness = material.glossiness;
        result.specular_texture_id = texture_id(material.specular_texture_id);
        result.specular_color_texture_id =
            texture_id(material.specular_color_texture_id);
        result.specular_glossiness_texture_id =
            texture_id(material.specular_glossiness_texture_id);
        result.specular_texture_transform =
            pack_texture_transform(material.specular_texture_transform);
        result.specular_color_texture_transform =
            pack_texture_transform(material.specular_color_texture_transform);
        result.specular_glossiness_texture_transform =
            pack_texture_transform(material.specular_glossiness_texture_transform);
        result.specular_texture_rotation =
            material.specular_texture_transform.rotation;
        result.specular_color_texture_rotation =
            material.specular_color_texture_transform.rotation;
        result.specular_glossiness_texture_rotation =
            material.specular_glossiness_texture_transform.rotation;
        result.specular_texture_texcoord =
            material.specular_texture_transform.texcoord;
        result.specular_color_texture_texcoord =
            material.specular_color_texture_transform.texcoord;
        result.specular_glossiness_texture_texcoord =
            material.specular_glossiness_texture_transform.texcoord;
        return result;
    }

    static DInstance pack_instance(
        const RenderSceneInstanceSnapshot& instance,
        int material_offset) {
        if (!instance.object_to_world.allFinite() ||
            !instance.world_to_object.allFinite() ||
            !instance.normal_to_world.allFinite() ||
            !instance.world_bounds.min.allFinite() ||
            !instance.world_bounds.max.allFinite()) {
            throw std::runtime_error(
                "CUDA instance contains a non-finite transform or bounds");
        }
        const float determinant =
            instance.object_to_world
                .topLeftCorner<3, 3>()
                .determinant();
        if (!std::isfinite(determinant) ||
            std::abs(determinant) < 1.0e-12f) {
            throw std::runtime_error(
                "CUDA instance transform is singular");
        }
        DInstance result{};
        result.object_to_world = to_device_affine(instance.object_to_world);
        result.world_to_object = to_device_affine(instance.world_to_object);
        result.normal_to_world = to_device_matrix(instance.normal_to_world);
        result.bounds_min = to_device(instance.world_bounds.min);
        result.bounds_max = to_device(instance.world_bounds.max);
        result.asset_index = instance.asset_index;
        result.material_offset = material_offset;
        result.orientation_sign = determinant < 0.0f ? -1.0f : 1.0f;
        return result;
    }

    static bool instance_surface_metric_changed(
        const DInstance& previous,
        const DInstance& next) {
        for (int first_axis = 0; first_axis < 3; ++first_axis) {
            for (int second_axis = first_axis;
                 second_axis < 3;
                 ++second_axis) {
                float previous_metric = 0.0f;
                float next_metric = 0.0f;
                for (int row = 0; row < 3; ++row) {
                    previous_metric +=
                        previous.object_to_world.values[
                            row * 4 + first_axis] *
                        previous.object_to_world.values[
                            row * 4 + second_axis];
                    next_metric +=
                        next.object_to_world.values[
                            row * 4 + first_axis] *
                        next.object_to_world.values[
                            row * 4 + second_axis];
                }
                const float scale = std::max(
                    1.0f,
                    std::max(
                        std::abs(previous_metric),
                        std::abs(next_metric)));
                if (std::abs(previous_metric - next_metric) >
                    1.0e-5f * scale) {
                    return true;
                }
            }
        }
        return false;
    }

    static float emissive_material_weight(
        const DMaterial& material) {
        const float energy =
            material.emission.x * 0.2126f +
            material.emission.y * 0.7152f +
            material.emission.z * 0.0722f;
        const int alpha_mode =
            material.type == static_cast<int>(MaterialType::Pbr)
            ? material.alpha_mode
            : ((material.opacity_texture_id >= 0 || material.opacity < 1.0f)
                  ? static_cast<int>(AlphaMode::Mask)
                  : static_cast<int>(AlphaMode::Opaque));
        if (!(energy > 0.0f) || !std::isfinite(energy) ||
            (alpha_mode == static_cast<int>(AlphaMode::Mask) &&
             material.opacity_texture_id < 0 &&
             material.base_color_texture_id < 0 &&
             material.opacity < material.alpha_cutoff)) {
            return 0.0f;
        }
        return energy * (material.two_sided ? 2.0f : 1.0f);
    }

    float sphere_area_scale(
        const DInstance& instance) const {
        const float determinant =
            std::abs(
                instance.object_to_world.values[0] *
                    (instance.object_to_world.values[5] *
                         instance.object_to_world.values[10] -
                     instance.object_to_world.values[6] *
                         instance.object_to_world.values[9]) -
                instance.object_to_world.values[1] *
                    (instance.object_to_world.values[4] *
                         instance.object_to_world.values[10] -
                     instance.object_to_world.values[6] *
                         instance.object_to_world.values[8]) +
                instance.object_to_world.values[2] *
                    (instance.object_to_world.values[4] *
                         instance.object_to_world.values[9] -
                     instance.object_to_world.values[5] *
                         instance.object_to_world.values[8]));
        const auto normal_length =
            [&instance](int column) {
                const float x =
                    instance.normal_to_world.values[column];
                const float y =
                    instance.normal_to_world.values[3 + column];
                const float z =
                    instance.normal_to_world.values[6 + column];
                return std::sqrt(x * x + y * y + z * z);
            };
        return determinant *
            (normal_length(0) +
             normal_length(1) +
             normal_length(2)) /
            3.0f;
    }

    float triangle_world_area(
        const DEmissiveLight& light) const {
        const DTraversalTriangle& triangle =
            traversal_triangles_host_[
                static_cast<std::size_t>(
                    light.primitive_index)];
        const DMatrix3x4& transform =
            instances_host_[
                static_cast<std::size_t>(
                    light.instance_index)]
                .object_to_world;
        const auto transform_vector =
            [&transform](DVec3 value) {
                return Vec3(
                    transform.values[0] * value.x +
                        transform.values[1] * value.y +
                        transform.values[2] * value.z,
                    transform.values[4] * value.x +
                        transform.values[5] * value.y +
                        transform.values[6] * value.z,
                    transform.values[8] * value.x +
                        transform.values[9] * value.y +
                        transform.values[10] * value.z);
            };
        return 0.5f *
            transform_vector(triangle.edge1)
                .cross(transform_vector(triangle.edge2))
                .norm();
    }

    void finalize_instanced_emissive_lights() {
        std::vector<float> weights;
        weights.reserve(emissive_lights_host_.size());
        float total_weight = 0.0f;
        for (DEmissiveLight& light : emissive_lights_host_) {
            if (light.primitive_kind == 1) {
                light.area = triangle_world_area(light);
            }
            const float area_for_selection =
                light.primitive_kind == 0
                ? light.area *
                    sphere_area_scale(
                        instances_host_[
                            static_cast<std::size_t>(
                                light.instance_index)])
                : light.area;
            const float material_weight =
                light.material_id >= 0 &&
                    static_cast<std::size_t>(light.material_id) <
                        materials_host_.size()
                ? emissive_material_weight(
                    materials_host_[
                        static_cast<std::size_t>(
                            light.material_id)])
                : 0.0f;
            const float weight =
                area_for_selection * material_weight;
            weights.push_back(
                std::isfinite(weight) && weight > 0.0f
                ? weight
                : 0.0f);
            total_weight += weights.back();
        }
        if (!(total_weight > 0.0f) ||
            !std::isfinite(total_weight)) {
            emissive_lights_host_.clear();
        } else {
            float cumulative = 0.0f;
            for (std::size_t index = 0;
                 index < emissive_lights_host_.size();
                 ++index) {
                const float probability =
                    weights[index] / total_weight;
                cumulative += probability;
                emissive_lights_host_[index].selection_pdf =
                    probability;
                emissive_lights_host_[index]
                    .cumulative_probability =
                    index + 1 ==
                            emissive_lights_host_.size()
                    ? 1.0f
                    : cumulative;
            }
        }
        sphere_light_indices_host_.clear();
        triangle_light_indices_host_.clear();
        statistics_.lighting_upload_bytes +=
            emissive_lights_.upload(
                emissive_lights_host_,
                stream_,
                statistics_);
        statistics_.lighting_upload_bytes +=
            sphere_light_indices_.upload(
                sphere_light_indices_host_,
                stream_,
                statistics_);
        statistics_.lighting_upload_bytes +=
            triangle_light_indices_.upload(
                triangle_light_indices_host_,
                stream_,
                statistics_);
    }

    void rebuild_instanced_emissive_lights(
        const RenderSceneSnapshot& scene) {
        emissive_lights_host_.clear();
        for (std::size_t instance_index = 0;
             instance_index < scene.instances.size();
             ++instance_index) {
            const DInstance& packed_instance =
                instances_host_[instance_index];
            if (packed_instance.asset_index < 0 ||
                static_cast<std::size_t>(
                    packed_instance.asset_index) >=
                    assets_host_.size()) {
                throw std::runtime_error(
                    "CUDA instance has an invalid asset index");
            }
            const DAsset& asset =
                assets_host_[
                    static_cast<std::size_t>(
                        packed_instance.asset_index)];
            for (int offset = 0;
                 offset < asset.sphere_count;
                 ++offset) {
                const int primitive_index =
                    asset.sphere_first + offset;
                const int material_slot =
                    sphere_material_ids_host_[
                        static_cast<std::size_t>(primitive_index)];
                const int material_id = material_slot < 0
                    ? -1
                    : packed_instance.material_offset + material_slot;
                if (material_id < 0 ||
                    static_cast<std::size_t>(material_id) >=
                        materials_host_.size() ||
                    emissive_material_weight(
                        materials_host_[
                            static_cast<std::size_t>(
                                material_id)]) <= 0.0f) {
                    continue;
                }
                const DSphere& sphere =
                    spheres_host_[
                        static_cast<std::size_t>(
                            primitive_index)];
                emissive_lights_host_.push_back(
                    DEmissiveLight{
                        0,
                        primitive_index,
                        static_cast<int>(instance_index),
                        material_id,
                        4.0f * kPi * sphere.radius *
                            sphere.radius,
                        0.0f,
                        0.0f, scene.instances[instance_index].emission_casts_shadows ? 1 : 0});
            }
            for (int offset = 0;
                 offset < asset.triangle_count;
                 ++offset) {
                const int primitive_index =
                    asset.triangle_first + offset;
                const int material_slot =
                    triangle_material_ids_host_[
                        static_cast<std::size_t>(primitive_index)];
                const int material_id = material_slot < 0
                    ? -1
                    : packed_instance.material_offset + material_slot;
                if (material_id < 0 ||
                    static_cast<std::size_t>(material_id) >=
                        materials_host_.size() ||
                    emissive_material_weight(
                        materials_host_[
                            static_cast<std::size_t>(
                                material_id)]) <= 0.0f) {
                    continue;
                }
                emissive_lights_host_.push_back(
                    DEmissiveLight{
                        1,
                        primitive_index,
                        static_cast<int>(instance_index),
                        material_id,
                        0.0f,
                        0.0f,
                        0.0f, scene.instances[instance_index].emission_casts_shadows ? 1 : 0});
            }
        }
        finalize_instanced_emissive_lights();
    }

    void upload_instanced_lights(
        const RenderSceneSnapshot& scene) {
        point_lights_host_.clear();
        point_lights_host_.reserve(scene.point_lights.size());
        for (const PointLight& light : scene.point_lights) {
            point_lights_host_.push_back(
                DPointLight{
                    to_device(light.position),
                    to_device(light.intensity),
                    light.range, light.source_radius, light.casts_shadows ? 1 : 0});
        }
        directional_lights_host_.clear();
        directional_lights_host_.reserve(
            scene.directional_lights.size());
        for (const DirectionalLight& light :
             scene.directional_lights) {
            directional_lights_host_.push_back(
                DDirectionalLight{
                    to_device(light.direction),
                    to_device(light.radiance), light.angular_radius_radians, light.casts_shadows ? 1 : 0});
        }
        spot_lights_host_.clear();
        spot_lights_host_.reserve(scene.spot_lights.size());
        for (const SpotLight& light : scene.spot_lights) {
            spot_lights_host_.push_back(DSpotLight{
                to_device(light.position),
                to_device(light.direction),
                to_device(light.intensity),
                light.range,
                std::cos(light.inner_cone_radians),
                std::cos(light.outer_cone_radians), light.source_radius, light.casts_shadows ? 1 : 0});
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
        statistics_.lighting_upload_bytes +=
            spot_lights_.upload(
                spot_lights_host_,
                stream_,
                statistics_);
    }

    void update_instanced_lighting(
        const RenderSceneSnapshot& scene) {
        upload_instanced_lights(scene);
    }

    void update_instanced_textures(
        const RenderSceneSnapshot& scene) {
        textures_host_.clear();
        texels_host_.clear();
        texture_alphas_host_.clear();
        textures_host_.reserve(scene.textures.size());
        for (const ImageTexture& texture : scene.textures) {
            const int first =
                static_cast<int>(texels_host_.size());
            const int first_alpha =
                static_cast<int>(texture_alphas_host_.size());
            textures_host_.push_back(
                DTexture{
                    texture.width(),
                    texture.height(),
                    first,
                    first_alpha,
                    static_cast<int>(texture.wrap_s()),
                    static_cast<int>(texture.wrap_t()),
                    texture.mag_filter() == TextureFilter::Nearest ? 1 : 0,
                    texture.uv_origin() == TextureUvOrigin::TopLeft ? 1 : 0});
            for (const Color& color : texture.pixels()) {
                texels_host_.push_back(to_device(color));
            }
            texture_alphas_host_.insert(
                texture_alphas_host_.end(),
                texture.alphas().begin(),
                texture.alphas().end());
        }
        statistics_.texture_upload_bytes +=
            textures_.upload(
                textures_host_,
                stream_,
                statistics_);
        statistics_.texture_upload_bytes +=
            texels_.upload(
                texels_host_,
                stream_,
                statistics_);
        statistics_.texture_upload_bytes +=
            texture_alphas_.upload(
                texture_alphas_host_,
                stream_,
                statistics_);
    }

    void update_instanced_materials(
        const RenderSceneSnapshot& scene) {
        if (scene.instances.size() !=
            instances_host_.size()) {
            throw std::runtime_error(
                "CUDA instance material update changed topology");
        }
        materials_host_.clear();
        materials_host_.push_back(
            pack_material(diagnostic_material(), 0));
        for (std::size_t index = 0;
             index < scene.instances.size();
             ++index) {
            const RenderSceneInstanceSnapshot& instance =
                scene.instances[index];
            if (instance.asset_index < 0 ||
                static_cast<std::size_t>(
                    instance.asset_index) >=
                    scene.assets.size()) {
                throw std::runtime_error(
                    "CUDA instance material update references an invalid asset");
            }
            const int material_offset =
                static_cast<int>(materials_host_.size());
            for (const Material& material :
                 instance.materials) {
                materials_host_.push_back(
                    pack_material(material, 0));
            }
            instances_host_[index].material_offset =
                material_offset;
        }
        statistics_.material_upload_bytes +=
            materials_.upload(
                materials_host_,
                stream_,
                statistics_);
        statistics_.instance_upload_bytes +=
            instances_.upload(
                instances_host_,
                stream_,
                statistics_);
        rebuild_instanced_emissive_lights(scene);
        has_instanced_emissive_candidates_ =
            !emissive_lights_host_.empty();
    }

    void update_instanced_material_bindings(
        const RenderSceneSnapshot& scene) {
        if (scene.assets.size() != assets_host_.size()) {
            throw std::runtime_error(
                "CUDA material-binding update changed asset topology");
        }
        std::vector<int> next_sphere_slots =
            sphere_material_ids_host_;
        std::vector<int> next_triangle_slots =
            triangle_material_ids_host_;
        for (std::size_t asset_index = 0;
             asset_index < scene.assets.size();
             ++asset_index) {
            const RenderSceneAssetSnapshot& source =
                scene.assets[asset_index];
            const DAsset& target = assets_host_[asset_index];
            if (source.sphere_material_slots.size() !=
                    static_cast<std::size_t>(target.sphere_count) ||
                source.triangle_material_slots.size() !=
                    static_cast<std::size_t>(target.triangle_count)) {
                throw std::runtime_error(
                    "CUDA material-binding table does not match asset geometry");
            }
            for (int index = 0; index < target.sphere_count; ++index) {
                next_sphere_slots[static_cast<std::size_t>(
                    target.sphere_first + index)] =
                    source.sphere_material_slots[
                        static_cast<std::size_t>(index)].device_value();
            }
            for (int index = 0; index < target.triangle_count; ++index) {
                next_triangle_slots[static_cast<std::size_t>(
                    target.triangle_first + index)] =
                    source.triangle_material_slots[
                        static_cast<std::size_t>(index)].device_value();
            }
        }
        sphere_material_ids_host_ = std::move(next_sphere_slots);
        triangle_material_ids_host_ = std::move(next_triangle_slots);
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

    void rebuild_instanced_topology(
        const RenderSceneSnapshot& scene) {
        materials_host_.clear();
        materials_host_.push_back(
            pack_material(diagnostic_material(), 0));
        instances_host_.clear();
        instances_host_.reserve(scene.instances.size());
        for (const RenderSceneInstanceSnapshot& instance :
             scene.instances) {
            if (instance.asset_index < 0 ||
                static_cast<std::size_t>(
                    instance.asset_index) >=
                    assets_host_.size()) {
                throw std::runtime_error(
                    "CUDA instance topology references an invalid asset");
            }
            const int material_offset =
                static_cast<int>(materials_host_.size());
            for (const Material& material :
                 instance.materials) {
                materials_host_.push_back(
                    pack_material(material, 0));
            }
            instances_host_.push_back(
                pack_instance(instance, material_offset));
        }
        statistics_.material_upload_bytes +=
            materials_.upload(
                materials_host_,
                stream_,
                statistics_);
        statistics_.instance_upload_bytes +=
            instances_.upload(
                instances_host_,
                stream_,
                statistics_);

        std::vector<Bounds3> instance_bounds;
        instance_bounds.reserve(scene.instances.size());
        for (const RenderSceneInstanceSnapshot& instance :
             scene.instances) {
            instance_bounds.push_back(
                instance.world_bounds);
        }
        GpuBvh4Layout tlas =
            build_gpu_bvh4(instance_bounds);
        tlas_nodes_host_ = std::move(tlas.nodes);
        tlas_primitive_indices_host_ =
            std::move(tlas.primitive_indices);
        statistics_.tlas_upload_bytes +=
            tlas_nodes_.upload(
                tlas_nodes_host_,
                stream_,
                statistics_);
        statistics_.tlas_upload_bytes +=
            tlas_primitive_indices_.upload(
                tlas_primitive_indices_host_,
                stream_,
                statistics_);
        ++statistics_.tlas_build_count;
        rebuild_instanced_emissive_lights(scene);
        has_instanced_emissive_candidates_ =
            !emissive_lights_host_.empty();
    }

    void rebuild_instanced_scene(
        const RenderSceneSnapshot& scene) {
        const auto blas_started =
            std::chrono::steady_clock::now();
        materials_host_.clear();
        materials_host_.push_back(
            pack_material(diagnostic_material(), 0));
        textures_host_.clear();
        texels_host_.clear();
        texture_alphas_host_.clear();
        spheres_host_.clear();
        sphere_material_ids_host_.clear();
        traversal_triangles_host_.clear();
        shading_triangles_host_.clear();
        triangle_material_ids_host_.clear();
        bvh_nodes_host_.clear();
        primitive_indices_host_.clear();
        assets_host_.clear();
        asset_ids_host_.clear();
        asset_scenes_host_.clear();
        asset_geometry_revisions_host_.clear();
        asset_geometry_fingerprints_host_.clear();
        asset_ids_host_.reserve(scene.assets.size());
        asset_scenes_host_.reserve(scene.assets.size());
        asset_geometry_revisions_host_.reserve(scene.assets.size());
        asset_geometry_fingerprints_host_.reserve(scene.assets.size());
        assets_host_.reserve(scene.assets.size());
        std::unordered_map<std::uint64_t, CachedAssetBlas>
            next_blas_cache;
        next_blas_cache.reserve(scene.assets.size());

        textures_host_.reserve(scene.textures.size());
        for (const ImageTexture& texture : scene.textures) {
            const int first = static_cast<int>(texels_host_.size());
            const int first_alpha = static_cast<int>(texture_alphas_host_.size());
            textures_host_.push_back(DTexture{
                texture.width(),
                texture.height(),
                first,
                first_alpha,
                static_cast<int>(texture.wrap_s()),
                static_cast<int>(texture.wrap_t()),
                texture.mag_filter() == TextureFilter::Nearest ? 1 : 0,
                texture.uv_origin() == TextureUvOrigin::TopLeft ? 1 : 0});
            for (const Color& color : texture.pixels()) {
                texels_host_.push_back(to_device(color));
            }
            texture_alphas_host_.insert(
                texture_alphas_host_.end(),
                texture.alphas().begin(),
                texture.alphas().end());
        }

        for (const RenderSceneAssetSnapshot& asset_view :
             scene.assets) {
            if (!asset_view.local_scene) {
                throw std::runtime_error(
                    "CUDA instance asset has no local scene");
            }
            const Scene& asset_scene = *asset_view.local_scene;
            if (asset_view.sphere_material_slots.size() !=
                    asset_scene.spheres.size() ||
                asset_view.triangle_material_slots.size() !=
                    asset_scene.triangles.size()) {
                throw std::runtime_error(
                    "CUDA instance asset has an invalid material-slot table");
            }
            asset_ids_host_.push_back(asset_view.asset_id);
            asset_scenes_host_.push_back(
                asset_view.local_scene.get());
            asset_geometry_revisions_host_.push_back(
                asset_view.geometry_revision);
            asset_geometry_fingerprints_host_.push_back(
                scene_geometry_fingerprint(asset_scene));
            DAsset asset{};
            asset.sphere_first =
                static_cast<int>(spheres_host_.size());
            asset.sphere_count =
                static_cast<int>(asset_scene.spheres.size());
            for (std::size_t sphere_index = 0;
                 sphere_index < asset_scene.spheres.size();
                 ++sphere_index) {
                const Sphere& sphere = asset_scene.spheres[sphere_index];
                spheres_host_.push_back(
                    DSphere{
                        to_device(sphere.center()),
                        sphere.radius()});
                sphere_material_ids_host_.push_back(
                    asset_view.sphere_material_slots[sphere_index].device_value());
            }

            asset.triangle_first =
                static_cast<int>(
                    traversal_triangles_host_.size());
            asset.triangle_count =
                static_cast<int>(
                    asset_scene.triangles.size());
            for (std::size_t triangle_index = 0;
                 triangle_index < asset_scene.triangles.size();
                 ++triangle_index) {
                const Triangle& triangle =
                    asset_scene.triangles[triangle_index];
                const TriangleVertex& first =
                    triangle.vertex(0);
                const TriangleVertex& second =
                    triangle.vertex(1);
                const TriangleVertex& third =
                    triangle.vertex(2);
                traversal_triangles_host_.push_back(
                    DTraversalTriangle{
                        to_device(first.position),
                        to_device(Vec3(
                            second.position -
                            first.position)),
                        to_device(Vec3(
                            third.position -
                            first.position))});
                DShadingTriangle shading{};
                for (int vertex = 0; vertex < 3; ++vertex) {
                    const TriangleVertex& source =
                        triangle.vertex(vertex);
                    shading.uvs[vertex] =
                        to_device(source.uv);
                    shading.uv1s[vertex] =
                        to_device(source.uv1);
                    shading.normals[vertex] =
                        to_device(source.normal);
                    shading.tangents[vertex] =
                        to_device(source.tangent);
                    shading.colors[vertex] =
                        to_device(source.color);
                    shading.alphas[vertex] = source.alpha;
                    if (source.has_normal) {
                        shading.normal_mask |=
                            1U << static_cast<unsigned int>(
                                vertex);
                    }
                    if (source.has_uv1) {
                        shading.uv1_mask |=
                            1U << static_cast<unsigned int>(vertex);
                    }
                    if (source.has_tangent) {
                        shading.tangent_mask |=
                            1U << static_cast<unsigned int>(vertex);
                    }
                    if (source.has_color) {
                        shading.color_mask |=
                            1U << static_cast<unsigned int>(vertex);
                    }
                }
                shading_triangles_host_.push_back(shading);
                triangle_material_ids_host_.push_back(
                    asset_view.triangle_material_slots[triangle_index].device_value());
            }

            const std::uint64_t geometry_fingerprint =
                asset_geometry_fingerprints_host_.back();
            GpuBvh4Layout raw_blas;
            const auto cached = asset_blas_cache_.find(
                asset_view.asset_id);
            if (cached != asset_blas_cache_.end() &&
                cached->second.geometry_revision ==
                    asset_view.geometry_revision &&
                cached->second.geometry_fingerprint ==
                    geometry_fingerprint) {
                raw_blas = cached->second.layout;
            } else {
                raw_blas = build_gpu_bvh4(asset_scene.triangles);
                if (!raw_blas.nodes.empty()) {
                    ++statistics_.blas_build_count;
                }
            }
            GpuBvh4Layout blas = raw_blas;
            const int node_offset =
                static_cast<int>(bvh_nodes_host_.size());
            const int primitive_offset =
                static_cast<int>(
                    primitive_indices_host_.size());
            for (DBvh4Node& node : blas.nodes) {
                for (int slot = 0;
                     slot < node.child_count;
                     ++slot) {
                    if (node.children[slot] >= 0) {
                        node.children[slot] += node_offset;
                    }
                    if (node.counts[slot] > 0) {
                        node.first[slot] +=
                            primitive_offset;
                    }
                }
            }
            for (int primitive : blas.primitive_indices) {
                primitive_indices_host_.push_back(
                    asset.triangle_first + primitive);
            }
            asset.bvh_root = blas.nodes.empty()
                ? -1
                : node_offset;
            asset.bvh_node_count =
                static_cast<int>(blas.nodes.size());
            bvh_nodes_host_.insert(
                bvh_nodes_host_.end(),
                blas.nodes.begin(),
                blas.nodes.end());
            next_blas_cache.emplace(
                asset_view.asset_id,
                CachedAssetBlas{
                    asset_view.geometry_revision,
                    geometry_fingerprint,
                    std::move(raw_blas)});
            assets_host_.push_back(asset);
        }
        asset_blas_cache_ = std::move(next_blas_cache);
        const auto blas_finished =
            std::chrono::steady_clock::now();
        statistics_.blas_build_milliseconds =
            std::chrono::duration<float, std::milli>(
                blas_finished - blas_started).count();

        instances_host_.clear();
        instances_host_.reserve(scene.instances.size());
        for (const RenderSceneInstanceSnapshot& instance :
             scene.instances) {
            if (instance.asset_index < 0 ||
                static_cast<std::size_t>(
                    instance.asset_index) >=
                    assets_host_.size()) {
                throw std::runtime_error(
                    "CUDA instance references an invalid asset");
            }
            const int material_offset =
                static_cast<int>(materials_host_.size());
            for (const Material& material :
                 instance.materials) {
                materials_host_.push_back(
                    pack_material(material, 0));
            }
            instances_host_.push_back(
                pack_instance(instance, material_offset));
        }
        statistics_.material_upload_bytes +=
            materials_.upload(
                materials_host_,
                stream_,
                statistics_);
        statistics_.texture_upload_bytes +=
            textures_.upload(
                textures_host_,
                stream_,
                statistics_);
        statistics_.texture_upload_bytes +=
            texels_.upload(
                texels_host_,
                stream_,
                statistics_);
        statistics_.texture_upload_bytes +=
            texture_alphas_.upload(
                texture_alphas_host_,
                stream_,
                statistics_);
        statistics_.geometry_upload_bytes +=
            spheres_.upload(
                spheres_host_,
                stream_,
                statistics_);
        statistics_.material_binding_upload_bytes +=
            sphere_material_ids_.upload(
                sphere_material_ids_host_,
                stream_,
                statistics_);
        statistics_.geometry_upload_bytes +=
            traversal_triangles_.upload(
                traversal_triangles_host_,
                stream_,
                statistics_);
        statistics_.geometry_upload_bytes +=
            shading_triangles_.upload(
                shading_triangles_host_,
                stream_,
                statistics_);
        statistics_.material_binding_upload_bytes +=
            triangle_material_ids_.upload(
                triangle_material_ids_host_,
                stream_,
                statistics_);
        statistics_.bvh_upload_bytes +=
            bvh_nodes_.upload(
                bvh_nodes_host_,
                stream_,
                statistics_);
        statistics_.bvh_upload_bytes +=
            primitive_indices_.upload(
                primitive_indices_host_,
                stream_,
                statistics_);
        statistics_.bvh_upload_bytes +=
            assets_.upload(
                assets_host_,
                stream_,
                statistics_);
        statistics_.instance_upload_bytes +=
            instances_.upload(
                instances_host_,
                stream_,
                statistics_);

        std::vector<Bounds3> instance_bounds;
        instance_bounds.reserve(scene.instances.size());
        for (const RenderSceneInstanceSnapshot& instance :
             scene.instances) {
            instance_bounds.push_back(instance.world_bounds);
        }
        GpuBvh4Layout tlas =
            build_gpu_bvh4(instance_bounds);
        tlas_nodes_host_ = std::move(tlas.nodes);
        tlas_primitive_indices_host_ =
            std::move(tlas.primitive_indices);
        statistics_.tlas_upload_bytes +=
            tlas_nodes_.upload(
                tlas_nodes_host_,
                stream_,
                statistics_);
        statistics_.tlas_upload_bytes +=
            tlas_primitive_indices_.upload(
                tlas_primitive_indices_host_,
                stream_,
                statistics_);
        ++statistics_.tlas_build_count;

        has_instanced_emissive_candidates_ = false;
        rebuild_instanced_emissive_lights(scene);
        has_instanced_emissive_candidates_ =
            !emissive_lights_host_.empty();
        upload_instanced_lights(scene);
        instanced_assets_initialized_ = true;
    }

    void update_instanced_transforms(
        const RenderSceneSnapshot& scene) {
        if (scene.instances.size() !=
            instances_host_.size()) {
            throw std::runtime_error(
                "CUDA instance topology changed without geometry change");
        }
        instance_upload_timer_.finish(
            statistics_.instance_upload_milliseconds);
        instance_upload_timer_.begin(stream_);
        std::vector<unsigned char> emissive_instances(
            instances_host_.size(),
            0);
        for (const DEmissiveLight& light :
             emissive_lights_host_) {
            if (light.instance_index >= 0 &&
                static_cast<std::size_t>(
                    light.instance_index) <
                    emissive_instances.size()) {
                emissive_instances[
                    static_cast<std::size_t>(
                        light.instance_index)] = 1;
            }
        }
        bool emissive_area_changed = false;
        for (std::size_t index = 0;
             index < scene.instances.size();
             ++index) {
            const int material_offset =
                instances_host_[index].material_offset;
            const DInstance next =
                pack_instance(
                    scene.instances[index],
                    material_offset);
            emissive_area_changed =
                emissive_area_changed ||
                (emissive_instances[index] != 0 &&
                 instance_surface_metric_changed(
                     instances_host_[index],
                     next));
            instances_host_[index] = next;
        }
        statistics_.instance_upload_bytes +=
            instances_.upload(
                instances_host_,
                stream_,
                statistics_);
        instance_upload_timer_.end(stream_);

        if (!tlas_nodes_host_.empty()) {
            tlas_refit_timer_.finish(
                statistics_.tlas_refit_milliseconds);
            tlas_refit_timer_.begin(stream_);
            refit_tlas_kernel<<<1, 1, 0, stream_>>>(
                tlas_nodes_.get(),
                static_cast<int>(tlas_nodes_.size()),
                tlas_primitive_indices_.get(),
                instances_.get(),
                static_cast<int>(instances_.size()),
                nullptr);
            check_cuda(
                cudaGetLastError(),
                "refit_tlas_kernel launch");
            tlas_refit_timer_.end(stream_);
            ++statistics_.tlas_refit_count;
        }
        if (has_instanced_emissive_candidates_ &&
            emissive_area_changed) {
            finalize_instanced_emissive_lights();
        }
    }

    void update_instanced_view() {
        view_.materials = materials_.get();
        view_.material_count =
            static_cast<int>(materials_.size());
        view_.stochastic_alpha_test = 0;
        view_.textures = textures_.get();
        view_.texture_count =
            static_cast<int>(textures_.size());
        view_.texels = texels_.get();
        view_.texture_alphas = texture_alphas_.get();
        view_.spheres = spheres_.get();
        view_.sphere_material_ids =
            sphere_material_ids_.get();
        view_.sphere_count =
            static_cast<int>(spheres_.size());
        view_.traversal_triangles =
            traversal_triangles_.get();
        view_.shading_triangles =
            shading_triangles_.get();
        view_.triangle_material_ids =
            triangle_material_ids_.get();
        view_.triangle_count =
            static_cast<int>(
                traversal_triangles_.size());
        view_.bvh_nodes = bvh_nodes_.get();
        view_.bvh_node_count =
            static_cast<int>(bvh_nodes_.size());
        view_.primitive_indices =
            primitive_indices_.get();
        view_.point_lights = point_lights_.get();
        view_.point_light_count =
            static_cast<int>(point_lights_.size());
        view_.directional_lights =
            directional_lights_.get();
        view_.directional_light_count =
            static_cast<int>(
                directional_lights_.size());
        view_.spot_lights = spot_lights_.get();
        view_.spot_light_count = static_cast<int>(spot_lights_.size());
        view_.emissive_lights =
            emissive_lights_.get();
        view_.emissive_light_count =
            static_cast<int>(
                emissive_lights_.size());
        view_.sphere_light_indices = nullptr;
        view_.triangle_light_indices = nullptr;
        view_.assets = assets_.get();
        view_.asset_count =
            static_cast<int>(assets_.size());
        view_.instances = instances_.get();
        view_.instance_count =
            static_cast<int>(instances_.size());
        view_.tlas_nodes = tlas_nodes_.get();
        view_.tlas_node_count =
            static_cast<int>(tlas_nodes_.size());
        view_.tlas_primitive_indices =
            tlas_primitive_indices_.get();
        view_.instanced_mode = 1;
    }

    cudaStream_t stream_ = nullptr;
    CudaPathStatistics& statistics_;
    CudaEventTimer upload_timer_;
    CudaEventTimer instance_upload_timer_;
    CudaEventTimer tlas_refit_timer_;
    DeviceBuffer<DMaterial> materials_;
    DeviceBuffer<DTexture> textures_;
    DeviceBuffer<DVec3> texels_;
    DeviceBuffer<float> texture_alphas_;
    DeviceBuffer<DSphere> spheres_;
    DeviceBuffer<int> sphere_material_ids_;
    DeviceBuffer<DTraversalTriangle> traversal_triangles_;
    DeviceBuffer<DShadingTriangle> shading_triangles_;
    DeviceBuffer<int> triangle_material_ids_;
    DeviceBuffer<DBvh4Node> bvh_nodes_;
    DeviceBuffer<int> primitive_indices_;
    DeviceBuffer<DPointLight> point_lights_;
    DeviceBuffer<DDirectionalLight> directional_lights_;
    DeviceBuffer<DSpotLight> spot_lights_;
    DeviceBuffer<DVec3> environment_texels_;
    DeviceBuffer<float> environment_pmf_;
    DeviceBuffer<float> environment_cdf_;
    DeviceBuffer<DEmissiveLight> emissive_lights_;
    DeviceBuffer<int> sphere_light_indices_;
    DeviceBuffer<int> triangle_light_indices_;
    DeviceBuffer<DAsset> assets_;
    DeviceBuffer<DInstance> instances_;
    DeviceBuffer<DBvh4Node> tlas_nodes_;
    DeviceBuffer<int> tlas_primitive_indices_;
    std::vector<DMaterial> materials_host_;
    std::vector<DTexture> textures_host_;
    std::vector<DVec3> texels_host_;
    std::vector<float> texture_alphas_host_;
    std::vector<DSphere> spheres_host_;
    std::vector<int> sphere_material_ids_host_;
    std::vector<DTraversalTriangle> traversal_triangles_host_;
    std::vector<DShadingTriangle> shading_triangles_host_;
    std::vector<int> triangle_material_ids_host_;
    std::vector<DBvh4Node> bvh_nodes_host_;
    std::vector<int> primitive_indices_host_;
    std::vector<DPointLight> point_lights_host_;
    std::vector<DDirectionalLight> directional_lights_host_;
    std::vector<DSpotLight> spot_lights_host_;
    std::vector<DVec3> environment_texels_host_;
    std::vector<float> environment_pmf_host_;
    std::vector<float> environment_cdf_host_;
    std::vector<DEmissiveLight> emissive_lights_host_;
    std::vector<int> sphere_light_indices_host_;
    std::vector<int> triangle_light_indices_host_;
    std::vector<DAsset> assets_host_;
    std::vector<DInstance> instances_host_;
    std::vector<std::uint64_t> asset_ids_host_;
    std::vector<const Scene*> asset_scenes_host_;
    std::vector<std::uint64_t> asset_geometry_revisions_host_;
    std::vector<std::uint64_t> asset_geometry_fingerprints_host_;
    std::unordered_map<std::uint64_t, CachedAssetBlas>
        asset_blas_cache_;
    std::vector<DBvh4Node> tlas_nodes_host_;
    std::vector<int> tlas_primitive_indices_host_;
    bool has_instanced_emissive_candidates_ = false;
    bool instanced_assets_initialized_ = false;
    DScene view_{};
};


#endif // !RTRT_OPTIX_DEVICE
}  // namespace
}  // namespace renderer
