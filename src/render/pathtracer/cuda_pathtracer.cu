#define EIGEN_NO_CUDA
#include "render/pathtracer/cuda_pathtracer.h"

#include "acceleration/bvh.h"
#include "core/timer.h"
#include "scene/material.h"
#include "scene/texture.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
#include <utility>
#include <vector>

namespace renderer {
namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kMaxPathBounces = 64;
constexpr int kBvhStackCapacity = 64;
constexpr float kPi = 3.14159265358979323846f;

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
};

struct DDirectionalLight {
    DVec3 direction;
    DVec3 radiance;
};

struct DSpotLight {
    DVec3 position;
    DVec3 direction;
    DVec3 intensity;
    float range;
    float inner_cosine;
    float outer_cosine;
};

struct DEmissiveLight {
    int primitive_kind;
    int primitive_index;
    int instance_index;
    int material_id;
    float area;
    float selection_pdf;
    float cumulative_probability;
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

__device__ float material_opacity(
    const DScene& scene,
    const DMaterial& material,
    DVec2 uv);
__device__ float material_surface_opacity(
    const DScene& scene,
    const DMaterial& material,
    const DHit& hit);

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
    const float opacity = material_surface_opacity(scene, material, opacity_hit);
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
    const float opacity = material_surface_opacity(scene, material, opacity_hit);
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

__device__ DVec3 environment_direction(DVec2 uv) {
    const float phi = (uv.x - floorf(uv.x) - 0.5f) * 2.0f * kPi;
    const float theta = fminf(fmaxf(uv.y, 0.0f), 1.0f) * kPi;
    const float sine = sinf(theta);
    return v3(cosf(phi) * sine, cosf(theta), sinf(phi) * sine);
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

__device__ DEnvironmentSample sample_environment(
    const DScene& scene,
    DPcgState& rng) {
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
    const DVec2 uv{
        (static_cast<float>(x) + random_float(rng)) /
            static_cast<float>(scene.environment_width),
        (static_cast<float>(y) + random_float(rng)) /
            static_cast<float>(scene.environment_height)};
    const DVec3 direction = rotate_y(
        environment_direction(uv),
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

__device__ float pbr_specular_probability(const DSurface& surface) {
    const DVec3 weights = v3(0.2126f, 0.7152f, 0.0722f);
    const float diffuse_energy = fmaxf(0.0f, dot(surface.diffuse_color, weights));
    const float specular_energy = fmaxf(0.0f, dot(surface.specular_f0, weights));
    const float total = diffuse_energy + specular_energy;
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
    const float geometry = smith_g1(n_dot_v, alpha) * smith_g1(n_dot_l, alpha);
    const DVec3 fresnel = fresnel_schlick(
        v_dot_h,
        surface.specular_f0,
        surface.specular_f90);
    const DVec3 diffuse_fresnel = fresnel_schlick(
        v_dot_h,
        surface.diffuse_fresnel_f0,
        surface.diffuse_fresnel_f90);
    const DVec3 specular = mul(
        fresnel,
        distribution * geometry / fmaxf(4.0f * n_dot_v * n_dot_l, 1.0e-12f));
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
        (1.0f - probability) * diffuse_pdf + probability * specular_pdf};
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
        if (!occluded_scene(
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
        if (!occluded_scene(
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
        if (!occluded_scene(
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

__device__ bool sample_emissive_shadow_task(
    const DScene& scene,
    const DHit& hit,
    const DSurface& surface,
    DVec3 outgoing,
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
            light_uv,
            light_material.emissive_texture_transform,
            light_material.emissive_texture_rotation,
            light_material.emissive_texture_texcoord);
        light_emission = product(
            light_emission,
            sample_texture(scene, light_material.emissive_texture_id, emissive_uv));
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
    return task.t_max > 0.0f;
}

__device__ bool sample_environment_shadow_task(
    const DScene& scene,
    const DHit& hit,
    const DSurface& surface,
    DVec3 outgoing,
    DVec3 throughput,
    int pixel_index,
    DPcgState& rng,
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
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int sample_index = first;
         sample_index < frame.pixel_count;
         sample_index += stride) {
        const int pixel_index = frame.pixel_offset + sample_index;
        const int x = pixel_index % frame.width;
        const int y = pixel_index / frame.width;
        DPcgState rng = frame.random_states[pixel_index];
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
        primary_paths[sample_index] = DPathState{
            DRay{frame.camera.eye, direction},
            v3(1.0f, 1.0f, 1.0f),
            0.0f,
            1,
            sample_index};
        frame.sample_radiance[sample_index] =
            v3(0.0f, 0.0f, 0.0f);
        frame.random_states[pixel_index] = rng;
    }
    if (first == 0) {
        *primary_count = frame.pixel_count;
        *secondary_count = 0;
        *bounce_index = 0;
        if (wavefront_handle != 0) {
            cudaGraphSetConditional(wavefront_handle, 1U);
        }
        if (emissive_handle != 0) {
            cudaGraphSetConditional(
                emissive_handle,
                (frame.scene.emissive_light_count > 0 ||
                 environment_direct_active(frame.scene))
                    ? 1U
                    : 0U);
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
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int active_index = *bounce_index & 1;
    const DPathState* active_paths =
        active_index == 0 ? first_paths : second_paths;
    int count = path_counts[active_index];
    if (first == 0) {
        path_counts[1 - active_index] = 0;
    }
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = max(0, min(count, frame.path_capacity));
    }
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    if (first == 0) {
        const int bounce = *bounce_index;
        atomicAdd(
            frame.diagnostic_counters +
                (bounce == 0
                    ? kDiagnosticPrimaryRays
                    : kDiagnosticContinuationRays),
            static_cast<unsigned long long>(count));
        atomicAdd(
            frame.diagnostic_counters + kDiagnosticRaysByBounce +
                max(0, min(bounce, kMaxPathBounces)),
            static_cast<unsigned long long>(count));
    }
#endif
    if (first >= count) {
        return;
    }
    DCompactHit hit{};
    const bool found = intersect_scene_compact(
        frame.scene,
        active_paths[first].ray,
        0.0f,
        1.0e30f,
        hit,
        frame.error_code);
    hits[first] = DWavefrontHit{hit, found ? 1 : 0};
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
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    const auto record_termination = [&frame](int surface_bounces) {
        atomicAdd(
            frame.diagnostic_counters + kDiagnosticTerminationByBounce +
                max(0, min(surface_bounces, kMaxPathBounces)),
            1ULL);
    };
    if (bounce == 0) {
        atomicAdd(
            frame.diagnostic_counters +
                (wavefront_hit.found
                    ? kDiagnosticPrimaryHits
                    : kDiagnosticPrimaryMisses),
            1ULL);
    }
#endif
    shading_records[index].valid = 0;
    if (!wavefront_hit.found) {
        if (bounce > 0 || frame.scene.environment_background_visible != 0) {
            const DVec3 incoming = environment_radiance(
                frame.scene,
                path.ray.direction);
            const float weight = bounce == 0 || path.previous_was_delta != 0
                ? 1.0f
                : power_heuristic(
                      path.previous_bsdf_pdf,
                      environment_strategy_probability(frame.scene) *
                          environment_pdf(frame.scene, path.ray.direction));
            frame.sample_radiance[path.pixel_index] = add(
                frame.sample_radiance[path.pixel_index],
                mul(product(path.throughput, incoming), weight));
        }
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce);
#endif
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
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce + 1);
#endif
        return;
    }
    const DMaterial material = frame.scene.materials[hit.material_id];
    const DSurface surface = evaluate_surface(frame.scene, material, hit);
    const int random_index = frame.pixel_offset + path.pixel_index;
    DPcgState rng = frame.random_states[random_index];
    const int alpha_mode = material.type == static_cast<int>(MaterialType::Pbr)
        ? material.alpha_mode
        : ((material.opacity_texture_id >= 0 || material.opacity < 1.0f)
              ? static_cast<int>(AlphaMode::Mask)
              : static_cast<int>(AlphaMode::Opaque));
    const bool transparent_passthrough =
        alpha_mode == static_cast<int>(AlphaMode::Blend) &&
        random_float(rng) >= surface.opacity;

    if (!transparent_passthrough && max_component(surface.emission) > 0.0f) {
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
                product(path.throughput, surface.emission),
                emission_weight));
    }
    if (!transparent_passthrough &&
        material.type == static_cast<int>(MaterialType::Emissive)) {
        frame.random_states[random_index] = rng;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce + 1);
#endif
        return;
    }

    const DVec3 outgoing = normalize(mul(path.ray.direction, -1.0f));
    if (!transparent_passthrough &&
        (material.type == static_cast<int>(MaterialType::Diffuse) ||
         material.type == static_cast<int>(MaterialType::Metal) ||
         material.type == static_cast<int>(MaterialType::Pbr))) {
        shading_records[index] = DShadingRecord{
            hit.position,
            hit.geometric_normal,
            surface.shading_normal,
            surface.base_color,
            surface.diffuse_color,
            surface.specular_f0,
            surface.specular_f90,
            surface.diffuse_fresnel_f0,
            surface.diffuse_fresnel_f90,
            surface.diffuse_fresnel_uses_max,
            outgoing,
            surface.metallic,
            surface.roughness,
            surface.occlusion,
            path.throughput,
            path.pixel_index,
            1};
    }

    DVec3 attenuation{};
    DRay scattered{};
    float bsdf_pdf = 0.0f;
    int was_delta = 1;
    if (transparent_passthrough) {
        attenuation = v3(1.0f, 1.0f, 1.0f);
        scattered = DRay{
            offset_origin(hit.position, hit.geometric_normal, path.ray.direction),
            path.ray.direction};
        bsdf_pdf = 0.0f;
        was_delta = 1;
    } else if (!scatter(
                   path.ray,
                   hit,
                   material,
                   surface,
                   rng,
                   attenuation,
                   scattered,
                   bsdf_pdf,
                   was_delta)) {
        frame.random_states[random_index] = rng;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce + 1);
#endif
        return;
    }

    DVec3 throughput = product(path.throughput, attenuation);
    if (!finite(throughput) || max_component(throughput) <= 0.0f ||
        bounce + 1 >= frame.max_bounces) {
        frame.random_states[random_index] = rng;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce + 1);
#endif
        return;
    }
    if (bounce + 1 >= frame.russian_roulette_start_bounce) {
        const float probability = fminf(
            fmaxf(
                max_component(throughput),
                frame.russian_roulette_min_probability),
            frame.russian_roulette_max_probability);
        if (random_float(rng) >= probability) {
            frame.random_states[random_index] = rng;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
            record_termination(bounce + 1);
#endif
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
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        record_termination(bounce + 1);
#endif
    }
    frame.random_states[random_index] = rng;
}

__global__ void sample_emissive_lights_kernel(
    const DFrameParameters* parameters,
    const DShadingRecord* shading_records,
    DShadowTask* shadow_tasks,
    const int* path_counts,
    const int* bounce_index) {
    const DFrameParameters& frame = *parameters;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    int count = path_counts[*bounce_index & 1];
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = max(0, min(count, frame.path_capacity));
    }
    if (first >= count) {
        return;
    }
    shadow_tasks[first].t_max = 0.0f;
    if (frame.scene.emissive_light_count <= 0 &&
        !environment_direct_active(frame.scene)) {
        return;
    }
    const DShadingRecord record = shading_records[first];
    if (!record.valid) {
        return;
    }
    DHit hit{};
    hit.position = record.position;
    hit.geometric_normal = record.geometric_normal;
    DSurface surface{};
    surface.base_color = record.base_color;
    surface.diffuse_color = record.diffuse_color;
    surface.specular_f0 = record.specular_f0;
    surface.specular_f90 = record.specular_f90;
    surface.diffuse_fresnel_f0 = record.diffuse_fresnel_f0;
    surface.diffuse_fresnel_f90 = record.diffuse_fresnel_f90;
    surface.diffuse_fresnel_uses_max = record.diffuse_fresnel_uses_max;
    surface.metallic = record.metallic;
    surface.roughness = record.roughness;
    surface.occlusion = record.occlusion;
    surface.shading_normal = record.shading_normal;
    const int random_index =
        frame.pixel_offset + record.pixel_index;
    DPcgState rng = frame.random_states[random_index];
    DShadowTask shadow_task{};
    const float environment_probability =
        environment_strategy_probability(frame.scene);
    const bool choose_environment = environment_probability > 0.0f &&
        (frame.scene.emissive_light_count <= 0 ||
         random_float(rng) < environment_probability);
    const bool sampled = choose_environment
        ? sample_environment_shadow_task(
              frame.scene,
              hit,
              surface,
              record.outgoing,
              record.throughput,
              record.pixel_index,
              rng,
              shadow_task)
        : sample_emissive_shadow_task(
              frame.scene,
              hit,
              surface,
              record.outgoing,
              record.throughput,
              record.pixel_index,
              rng,
              shadow_task);
    if (sampled) {
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        shadow_task.diagnostic_kind = choose_environment
            ? kDiagnosticShadowEnvironment
            : kDiagnosticShadowEmissive;
#endif
        shadow_tasks[first] = shadow_task;
    }
    frame.random_states[random_index] = rng;
}

__global__ void direct_visibility_kernel(
    const DFrameParameters* parameters,
    const DShadingRecord* shading_records,
    const int* path_counts,
    const int* bounce_index,
    const DShadowTask* shadow_tasks) {
    const DFrameParameters& frame = *parameters;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    int count = path_counts[*bounce_index & 1];
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = max(0, min(count, frame.path_capacity));
    }
    if (first >= count) {
        return;
    }
    const DShadingRecord record = shading_records[first];
    if (!record.valid) {
        return;
    }
    DHit hit{};
    hit.position = record.position;
    hit.geometric_normal = record.geometric_normal;
    DSurface surface{};
    surface.base_color = record.base_color;
    surface.diffuse_color = record.diffuse_color;
    surface.specular_f0 = record.specular_f0;
    surface.specular_f90 = record.specular_f90;
    surface.diffuse_fresnel_f0 = record.diffuse_fresnel_f0;
    surface.diffuse_fresnel_f90 = record.diffuse_fresnel_f90;
    surface.diffuse_fresnel_uses_max = record.diffuse_fresnel_uses_max;
    surface.metallic = record.metallic;
    surface.roughness = record.roughness;
    surface.occlusion = record.occlusion;
    surface.shading_normal = record.shading_normal;
    DVec3 contribution = product(
        record.throughput,
        direct_lighting(
            frame.scene,
            hit,
            surface,
            record.outgoing,
            frame.error_code
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
            , frame.diagnostic_counters
#endif
            ));
    if (frame.scene.emissive_light_count > 0 ||
        environment_direct_active(frame.scene)) {
        const DShadowTask task = shadow_tasks[first];
        if (task.t_max > 0.0f &&
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
            (atomicAdd(
                frame.diagnostic_counters +
                    (task.diagnostic_kind == kDiagnosticShadowEnvironment
                        ? kDiagnosticEnvironmentShadowRays
                        : kDiagnosticEmissiveShadowRays),
                1ULL), true) &&
#endif
            !occluded_scene(
                frame.scene,
                task.ray,
                0.0f,
                task.t_max,
                frame.error_code)) {
            contribution = add(
                contribution,
                task.contribution);
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
    int count = next_bounce < frame.max_bounces
        ? path_counts[next_bounce & 1]
        : 0;
    if (count < 0 || count > frame.path_capacity) {
        atomicCAS(frame.error_code, 0, 3);
        count = 0;
    }
    if (conditional_handle != 0) {
        cudaGraphSetConditional(
            conditional_handle,
            count > 0 && next_bounce < frame.max_bounces ? 1U : 0U);
    }
}

__global__ void finalize_sample_kernel(
    const DFrameParameters* parameters) {
    const DFrameParameters& frame = *parameters;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int sample_index = first;
         sample_index < frame.pixel_count;
         sample_index += stride) {
        const int pixel_index = frame.pixel_offset + sample_index;
        const DVec3 sum = add(
            frame.accumulation[pixel_index],
            frame.sample_radiance[sample_index]);
        frame.accumulation[pixel_index] = sum;
        if (frame.output_surface != 0) {
            const int completed_samples =
                frame.completed_samples + *frame.batch_sample_index;
            const DVec3 color = divv(
                sum,
                static_cast<float>(completed_samples + 1));
            const int x = pixel_index % frame.width;
            const int y = pixel_index / frame.width;
            const int output_x_begin =
                x * frame.output_width / frame.width;
            const int output_x_end =
                (x + 1) * frame.output_width / frame.width;
            const int output_y_begin =
                y * frame.output_height / frame.height;
            const int output_y_end =
                (y + 1) * frame.output_height / frame.height;
            const float4 output =
                make_float4(color.x, color.y, color.z, 1.0f);
            for (int output_y = output_y_begin;
                 output_y < output_y_end;
                 ++output_y) {
                for (int output_x = output_x_begin;
                     output_x < output_x_end;
                     ++output_x) {
                    surf2Dwrite(
                        output,
                        frame.output_surface,
                        output_x * static_cast<int>(sizeof(float4)),
                        output_y);
                }
            }
        }
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
    int pixel_offset,
    int sample_count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) {
        return;
    }
    resolved[index] = sample_count > 0
        ? divv(
            accumulation[pixel_offset + index],
            static_cast<float>(sample_count))
        : v3(0.0f, 0.0f, 0.0f);
}

__global__ void present_frame_kernel(
    const DVec3* accumulation,
    int width,
    int height,
    int sample_count,
    cudaSurfaceObject_t output_surface) {
    const int pixel_count = width * height;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int index = first; index < pixel_count; index += stride) {
        const DVec3 color = sample_count > 0
            ? divv(
                accumulation[index],
                static_cast<float>(sample_count))
            : v3(0.0f, 0.0f, 0.0f);
        const float4 output =
            make_float4(color.x, color.y, color.z, 1.0f);
        surf2Dwrite(
            output,
            output_surface,
            (index % width) * static_cast<int>(sizeof(float4)),
            index / width);
    }
}

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
        const Scene& scene,
        cudaStream_t stream,
        CudaPathStatistics& statistics)
        : stream_(stream), statistics_(statistics) {
        sync(scene, SceneChange::All);
    }

    CudaSceneStorage(
        const InstancedSceneView& scene,
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
                materials_host_.push_back(pack_material(material, 0));
            }
            statistics_.material_upload_bytes +=
                materials_.upload(materials_host_, stream_, statistics_);
        }

        if (has_scene_change(changes, SceneChange::Textures)) {
            textures_host_.clear();
            texels_host_.clear();
            texture_alphas_host_.clear();
            textures_host_.reserve(scene.textures.size());
            std::size_t texel_count = 0;
            for (const ImageTexture& texture : scene.textures) {
                texel_count += texture.pixels().size();
            }
            texels_host_.reserve(texel_count);
            texture_alphas_host_.reserve(texel_count);
            for (const ImageTexture& texture : scene.textures) {
                const int first = static_cast<int>(texels_host_.size());
                const int first_alpha = static_cast<int>(texture_alphas_host_.size());
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
                textures_.upload(textures_host_, stream_, statistics_);
            statistics_.texture_upload_bytes +=
                texels_.upload(texels_host_, stream_, statistics_);
            statistics_.texture_upload_bytes +=
                texture_alphas_.upload(texture_alphas_host_, stream_, statistics_);
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

            traversal_triangles_host_.clear();
            shading_triangles_host_.clear();
            traversal_triangles_host_.reserve(scene.triangles.size());
            shading_triangles_host_.reserve(scene.triangles.size());
            if (pack_bindings) {
                triangle_material_ids_host_.clear();
                triangle_material_ids_host_.reserve(scene.triangles.size());
            }
            for (const Triangle& triangle : scene.triangles) {
                const TriangleVertex& first = triangle.vertex(0);
                const TriangleVertex& second = triangle.vertex(1);
                const TriangleVertex& third = triangle.vertex(2);
                const Vec3 edge1 = second.position - first.position;
                const Vec3 edge2 = third.position - first.position;
                traversal_triangles_host_.push_back(
                    DTraversalTriangle{
                        to_device(first.position),
                        to_device(edge1),
                        to_device(edge2)});
                DShadingTriangle shading{};
                for (int vertex_index = 0; vertex_index < 3; ++vertex_index) {
                    const TriangleVertex& vertex = triangle.vertex(vertex_index);
                    shading.uvs[vertex_index] = to_device(vertex.uv);
                    shading.uv1s[vertex_index] = to_device(vertex.uv1);
                    shading.normals[vertex_index] =
                        to_device(vertex.normal);
                    shading.tangents[vertex_index] = to_device(vertex.tangent);
                    shading.colors[vertex_index] = to_device(vertex.color);
                    shading.alphas[vertex_index] = vertex.alpha;
                    if (vertex.has_normal) {
                        shading.normal_mask |=
                            1U << static_cast<unsigned int>(vertex_index);
                    }
                    if (vertex.has_uv1) {
                        shading.uv1_mask |=
                            1U << static_cast<unsigned int>(vertex_index);
                    }
                    if (vertex.has_tangent) {
                        shading.tangent_mask |=
                            1U << static_cast<unsigned int>(vertex_index);
                    }
                    if (vertex.has_color) {
                        shading.color_mask |=
                            1U << static_cast<unsigned int>(vertex_index);
                    }
                }
                shading_triangles_host_.push_back(shading);
                if (pack_bindings) {
                    triangle_material_ids_host_.push_back(
                        triangle.material_id());
                }
            }
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

            GpuBvh4Layout bvh = build_gpu_bvh4(scene.triangles);
            bvh_nodes_host_ = std::move(bvh.nodes);
            statistics_.bvh_upload_bytes +=
                bvh_nodes_.upload(bvh_nodes_host_, stream_, statistics_);
            primitive_indices_host_ =
                std::move(bvh.primitive_indices);
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
                        to_device(light.intensity),
                        light.range});
            }
            directional_lights_host_.clear();
            directional_lights_host_.reserve(scene.directional_lights.size());
            for (const DirectionalLight& light : scene.directional_lights) {
                directional_lights_host_.push_back(
                    DDirectionalLight{
                        to_device(light.direction),
                        to_device(light.radiance)});
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
                    std::cos(light.outer_cone_radians)});
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
            statistics_.lighting_upload_bytes += sizeof(DVec3);
        }

        if (has_scene_change(changes, SceneChange::Environment)) {
            upload_environment(
                scene.environment,
                scene.environment_map,
                scene.environment_intensity,
                scene.environment_rotation_degrees,
                scene.environment_background_visible);
        }

        view_.materials = materials_.get();
        view_.material_count = static_cast<int>(materials_.size());
        view_.stochastic_alpha_test = 0;
        view_.textures = textures_.get();
        view_.texture_count = static_cast<int>(textures_.size());
        view_.texels = texels_.get();
        view_.texture_alphas = texture_alphas_.get();
        view_.spheres = spheres_.get();
        view_.sphere_material_ids = sphere_material_ids_.get();
        view_.sphere_count = static_cast<int>(spheres_.size());
        view_.traversal_triangles = traversal_triangles_.get();
        view_.shading_triangles = shading_triangles_.get();
        view_.triangle_material_ids = triangle_material_ids_.get();
        view_.triangle_count =
            static_cast<int>(traversal_triangles_.size());
        view_.bvh_nodes = bvh_nodes_.get();
        view_.bvh_node_count = static_cast<int>(bvh_nodes_.size());
        view_.primitive_indices = primitive_indices_.get();
        view_.point_lights = point_lights_.get();
        view_.point_light_count = static_cast<int>(point_lights_.size());
        view_.directional_lights = directional_lights_.get();
        view_.directional_light_count = static_cast<int>(directional_lights_.size());
        view_.spot_lights = spot_lights_.get();
        view_.spot_light_count = static_cast<int>(spot_lights_.size());
        view_.emissive_lights = emissive_lights_.get();
        view_.emissive_light_count = static_cast<int>(emissive_lights_.size());
        view_.sphere_light_indices = sphere_light_indices_.get();
        view_.triangle_light_indices = triangle_light_indices_.get();
        view_.assets = nullptr;
        view_.asset_count = 0;
        view_.instances = nullptr;
        view_.instance_count = 0;
        view_.tlas_nodes = nullptr;
        view_.tlas_node_count = 0;
        view_.tlas_primitive_indices = nullptr;
        view_.instanced_mode = 0;
        upload_timer_.end(stream_);
    }

    void sync(
        const InstancedSceneView& scene,
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
                if (has_scene_change(
                        changes,
                        SceneChange::MaterialBindings) ||
                    has_scene_change(
                        changes,
                        SceneChange::Materials) ||
                    has_scene_change(
                        changes,
                        SceneChange::Textures)) {
                    update_instanced_materials(scene);
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
        const InstancedSceneView& scene) const {
        if (!instanced_assets_initialized_ ||
            scene.assets.size() !=
                asset_ids_host_.size()) {
            return false;
        }
        for (std::size_t index = 0;
             index < scene.assets.size();
             ++index) {
            if (scene.assets[index].asset_id !=
                    asset_ids_host_[index] ||
                scene.assets[index].local_scene !=
                    asset_scenes_host_[index]) {
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
        return DMaterial{
            static_cast<int>(material.type),
            to_device(material.base_color),
            to_device(material.emission),
            material.roughness,
            material.metallic,
            material.ior,
            material.opacity,
            material.alpha_cutoff,
            material.bump_scale,
            material.normal_scale,
            material.occlusion_strength,
            static_cast<int>(material.alpha_mode),
            material.two_sided ? 1 : 0,
            texture_id(material.diffuse_texture_id),
            texture_id(material.opacity_texture_id),
            texture_id(material.bump_texture_id),
            texture_id(material.base_color_texture_id),
            texture_id(material.metallic_roughness_texture_id),
            texture_id(material.normal_texture_id),
            texture_id(material.occlusion_texture_id),
            texture_id(material.emissive_texture_id),
            pack_texture_transform(material.base_color_texture_transform),
            pack_texture_transform(material.metallic_roughness_texture_transform),
            pack_texture_transform(material.normal_texture_transform),
            pack_texture_transform(material.occlusion_texture_transform),
            pack_texture_transform(material.emissive_texture_transform),
            material.base_color_texture_transform.rotation,
            material.metallic_roughness_texture_transform.rotation,
            material.normal_texture_transform.rotation,
            material.occlusion_texture_transform.rotation,
            material.emissive_texture_transform.rotation,
            material.base_color_texture_transform.texcoord,
            material.metallic_roughness_texture_transform.texcoord,
            material.normal_texture_transform.texcoord,
            material.occlusion_texture_transform.texcoord,
            material.emissive_texture_transform.texcoord,
            static_cast<int>(material.pbr_workflow),
            to_device(material.specular_color),
            material.specular_factor,
            material.glossiness,
            texture_id(material.specular_texture_id),
            texture_id(material.specular_color_texture_id),
            texture_id(material.specular_glossiness_texture_id),
            pack_texture_transform(material.specular_texture_transform),
            pack_texture_transform(material.specular_color_texture_transform),
            pack_texture_transform(material.specular_glossiness_texture_transform),
            material.specular_texture_transform.rotation,
            material.specular_color_texture_transform.rotation,
            material.specular_glossiness_texture_transform.rotation,
            material.specular_texture_transform.texcoord,
            material.specular_color_texture_transform.texcoord,
            material.specular_glossiness_texture_transform.texcoord};
    }

    static DInstance pack_instance(
        const InstancedSceneInstanceView& instance,
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
        return DInstance{
            to_device_affine(instance.object_to_world),
            to_device_affine(instance.world_to_object),
            to_device_matrix(instance.normal_to_world),
            to_device(instance.world_bounds.min),
            to_device(instance.world_bounds.max),
            instance.asset_index,
            material_offset,
            determinant < 0.0f ? -1.0f : 1.0f};
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
        if (!(energy > 0.0f) || !std::isfinite(energy) ||
            (material.opacity_texture_id < 0 &&
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
        const InstancedSceneView& scene) {
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
                const int material_id =
                    packed_instance.material_offset +
                    sphere_material_ids_host_[
                        static_cast<std::size_t>(
                            primitive_index)];
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
                        0.0f});
            }
            for (int offset = 0;
                 offset < asset.triangle_count;
                 ++offset) {
                const int primitive_index =
                    asset.triangle_first + offset;
                const int material_id =
                    packed_instance.material_offset +
                    triangle_material_ids_host_[
                        static_cast<std::size_t>(
                            primitive_index)];
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
                        0.0f});
            }
        }
        finalize_instanced_emissive_lights();
    }

    void upload_instanced_lights(
        const InstancedSceneView& scene) {
        point_lights_host_.clear();
        point_lights_host_.reserve(scene.point_lights.size());
        for (const PointLight& light : scene.point_lights) {
            point_lights_host_.push_back(
                DPointLight{
                    to_device(light.position),
                    to_device(light.intensity),
                    light.range});
        }
        directional_lights_host_.clear();
        directional_lights_host_.reserve(
            scene.directional_lights.size());
        for (const DirectionalLight& light :
             scene.directional_lights) {
            directional_lights_host_.push_back(
                DDirectionalLight{
                    to_device(light.direction),
                    to_device(light.radiance)});
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
                std::cos(light.outer_cone_radians)});
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
        const InstancedSceneView& scene) {
        upload_instanced_lights(scene);
    }

    void update_instanced_textures(
        const InstancedSceneView& scene) {
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
        const InstancedSceneView& scene) {
        if (scene.instances.size() !=
            instances_host_.size()) {
            throw std::runtime_error(
                "CUDA instance material update changed topology");
        }
        materials_host_.clear();
        for (std::size_t index = 0;
             index < scene.instances.size();
             ++index) {
            const InstancedSceneInstanceView& instance =
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

    void rebuild_instanced_topology(
        const InstancedSceneView& scene) {
        materials_host_.clear();
        instances_host_.clear();
        instances_host_.reserve(scene.instances.size());
        for (const InstancedSceneInstanceView& instance :
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
        for (const InstancedSceneInstanceView& instance :
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
        const InstancedSceneView& scene) {
        const auto blas_started =
            std::chrono::steady_clock::now();
        materials_host_.clear();
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
        asset_ids_host_.reserve(scene.assets.size());
        asset_scenes_host_.reserve(scene.assets.size());
        assets_host_.reserve(scene.assets.size());

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

        for (const InstancedSceneAssetView& asset_view :
             scene.assets) {
            if (!asset_view.local_scene) {
                throw std::runtime_error(
                    "CUDA instance asset has no local scene");
            }
            const Scene& asset_scene = *asset_view.local_scene;
            asset_ids_host_.push_back(asset_view.asset_id);
            asset_scenes_host_.push_back(
                asset_view.local_scene);
            DAsset asset{};
            asset.sphere_first =
                static_cast<int>(spheres_host_.size());
            asset.sphere_count =
                static_cast<int>(asset_scene.spheres.size());
            for (const Sphere& sphere : asset_scene.spheres) {
                spheres_host_.push_back(
                    DSphere{
                        to_device(sphere.center()),
                        sphere.radius()});
                sphere_material_ids_host_.push_back(
                    sphere.material_id());
            }

            asset.triangle_first =
                static_cast<int>(
                    traversal_triangles_host_.size());
            asset.triangle_count =
                static_cast<int>(
                    asset_scene.triangles.size());
            for (const Triangle& triangle :
                 asset_scene.triangles) {
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
                    triangle.material_id());
            }

            GpuBvh4Layout blas =
                build_gpu_bvh4(asset_scene.triangles);
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
            if (!blas.nodes.empty()) {
                ++statistics_.blas_build_count;
            }
            assets_host_.push_back(asset);
        }
        const auto blas_finished =
            std::chrono::steady_clock::now();
        statistics_.blas_build_milliseconds =
            std::chrono::duration<float, std::milli>(
                blas_finished - blas_started).count();

        instances_host_.clear();
        instances_host_.reserve(scene.instances.size());
        for (const InstancedSceneInstanceView& instance :
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
        for (const InstancedSceneInstanceView& instance :
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
        const InstancedSceneView& scene) {
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
                -1,
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
                -1,
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
    std::vector<DBvh4Node> tlas_nodes_host_;
    std::vector<int> tlas_primitive_indices_host_;
    bool has_instanced_emissive_candidates_ = false;
    bool instanced_assets_initialized_ = false;
    DScene view_{};
};

class CudaFrameStorage {
public:
    explicit CudaFrameStorage(
        CudaPathStatistics& statistics,
        cudaStream_t shared_stream = nullptr)
        : statistics_(statistics),
          stream_(shared_stream),
          owns_stream_(shared_stream == nullptr) {
        if (owns_stream_) {
            check_cuda(
                cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags");
        }
    }

    ~CudaFrameStorage() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
        }
        destroy_graph();
        if (stream_ && owns_stream_) {
            cudaStreamDestroy(stream_);
        }
    }

    void reset(
        int width,
        int height,
        std::uint64_t seed_offset,
        std::size_t requested_wavefront_capacity = 0) {
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
        const std::size_t wavefront_capacity =
            requested_wavefront_capacity == 0
            ? count
            : std::clamp<std::size_t>(
                requested_wavefront_capacity,
                1,
                count);
        reset_timer_.update(statistics_.reset_milliseconds);
        const bool record_timing = !reset_timer_.pending();
        if (record_timing) {
            reset_timer_.begin(stream_);
        }
        accumulation_.resize(count, statistics_);
        random_states_.resize(count, statistics_);
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        diagnostic_counters_.resize_exact(
            kDiagnosticCounterCount,
            statistics_);
        check_cuda(
            cudaMemsetAsync(
                diagnostic_counters_.get(),
                0,
                kDiagnosticCounterCount * sizeof(unsigned long long),
                stream_),
            "clear CUDA benchmark diagnostic counters");
#endif
        ensure_wavefront_capacity(
            wavefront_capacity,
            requested_wavefront_capacity != 0);
#if !defined(RENDERER_CUDA_SANITIZER_FALLBACK)
        if (!graph_exec_ ||
            wavefront_capacity >
                static_cast<std::size_t>(graph_capacity_)) {
            rebuild_graph(static_cast<int>(wavefront_capacity));
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
        const PathRenderSettings& path_settings,
        cudaSurfaceObject_t output_surface = 0) {
        render_samples(scene, camera, 1, path_settings, output_surface);
    }

    void render_samples(
        const DScene& scene,
        const Camera& camera,
        int sample_count,
        const PathRenderSettings& path_settings,
        cudaSurfaceObject_t output_surface = 0) {
        render_work(
            scene,
            camera,
            sample_count,
            0,
            static_cast<int>(accumulation_.size()),
            kMaxPathBounces,
            width_,
            height_,
            path_settings,
            output_surface,
            true);
    }

    void render_region(
        const DScene& scene,
        const Camera& camera,
        int pixel_offset,
        int pixel_count,
        int max_bounces,
        int output_width,
        int output_height,
        const PathRenderSettings& path_settings,
        cudaSurfaceObject_t output_surface = 0) {
        render_work(
            scene,
            camera,
            1,
            pixel_offset,
            pixel_count,
            max_bounces,
            output_width,
            output_height,
            path_settings,
            output_surface,
            false,
            true);
    }

    void render_region_quantum(
        const DScene& scene,
        const Camera& camera,
        int pixel_offset,
        int pixel_count,
        int max_bounces,
        int output_width,
        int output_height,
        const PathRenderSettings& path_settings) {
        if (pixel_count <= 0) {
            throw std::invalid_argument(
                "CUDA wavefront quantum must contain pixels");
        }
        const int chunk_capacity = static_cast<int>(
            std::min<std::size_t>(
                wavefront_capacity_pixels_,
                static_cast<std::size_t>(INT_MAX)));
        if (chunk_capacity <= 0) {
            throw std::logic_error(
                "CUDA wavefront quantum arena is empty");
        }
        update_trace_timing();
        const bool record_timing = !trace_timer_.pending();
        if (record_timing) {
            trace_timer_.begin(stream_);
        }
        int submitted = 0;
        while (submitted < pixel_count) {
            const int chunk =
                std::min(chunk_capacity, pixel_count - submitted);
            render_work(
                scene,
                camera,
                1,
                pixel_offset + submitted,
                chunk,
                max_bounces,
                output_width,
                output_height,
                path_settings,
                0,
                false,
                false);
            submitted += chunk;
        }
        if (record_timing) {
            trace_timer_.end(stream_);
        }
    }

    void complete_sample() {
        ++samples_;
    }

    void present_surface(cudaSurfaceObject_t output_surface) {
        if (output_surface == 0) {
            throw std::invalid_argument(
                "CUDA frame presentation requires a surface");
        }
        if (samples_ <= 0) {
            throw std::logic_error(
                "CUDA frame presentation requires a completed sample");
        }
        update_presentation_timing();
        const bool record_timing = !presentation_timer_.pending();
        if (record_timing) {
            presentation_timer_.begin(stream_);
        }
        const int pixel_count =
            static_cast<int>(accumulation_.size());
        const int capacity_blocks =
            (pixel_count + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const int block_count = occupancy_grid_size(
            reinterpret_cast<void*>(present_frame_kernel),
            capacity_blocks);
        present_frame_kernel<<<
            block_count,
            kThreadsPerBlock,
            0,
            stream_>>>(
                accumulation_.get(),
                width_,
                height_,
                samples_,
                output_surface);
        check_cuda(cudaGetLastError(), "present_frame_kernel launch");
        if (record_timing) {
            presentation_timer_.end(stream_);
        }
    }

private:
    void render_work(
        const DScene& scene,
        const Camera& camera,
        int sample_count,
        int pixel_offset,
        int pixel_count,
        int max_bounces,
        int output_width,
        int output_height,
        const PathRenderSettings& path_settings,
        cudaSurfaceObject_t output_surface,
        bool completes_sample,
        bool record_trace = true) {
        if (sample_count <= 0) {
            throw std::invalid_argument(
                "CUDA wavefront sample batch must be positive");
        }
        if (pixel_offset < 0 || pixel_count <= 0 ||
            static_cast<std::size_t>(pixel_offset) +
                    static_cast<std::size_t>(pixel_count) >
                accumulation_.size()) {
            throw std::invalid_argument(
                "CUDA wavefront work region is invalid");
        }
        if (static_cast<std::size_t>(pixel_count) >
            wavefront_capacity_pixels_) {
            throw std::invalid_argument(
                "CUDA wavefront work region exceeds arena capacity");
        }
        if (max_bounces <= 0 || max_bounces > kMaxPathBounces) {
            throw std::invalid_argument(
                "CUDA wavefront bounce limit is invalid");
        }
        if (output_width <= 0 || output_height <= 0) {
            throw std::invalid_argument(
                "CUDA output dimensions must be positive");
        }
        const int roulette_start_bounce = std::clamp(
            path_settings.russian_roulette_start_bounce,
            1,
            kMaxPathBounces);
        const float roulette_min_probability = std::clamp(
            path_settings.russian_roulette_min_probability,
            0.01f,
            1.0f);
        const float roulette_max_probability = std::clamp(
            path_settings.russian_roulette_max_probability,
            roulette_min_probability,
            1.0f);
        const DCamera packed_camera{
            to_device(camera.eye()),
            to_device(camera.forward()),
            to_device(camera.right()),
            to_device(camera.up()),
            camera.viewport_width(),
            camera.viewport_height()};
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
            pixel_offset,
            pixel_count,
            static_cast<int>(wavefront_capacity_pixels_),
            max_bounces,
            roulette_start_bounce,
            roulette_min_probability,
            roulette_max_probability,
            output_width,
            output_height,
            error_code_,
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
            diagnostic_counters_.get(),
#endif
            output_surface};
        update_frame_parameters_kernel<<<1, 1, 0, stream_>>>(
            frame_parameters_,
            parameters);
        check_cuda(
            cudaGetLastError(),
            "update_frame_parameters_kernel launch");
        if (record_trace) {
            update_trace_timing();
        }
        const bool record_timing =
            record_trace && !trace_timer_.pending();
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
        if (completes_sample) {
            samples_ += sample_count;
        }
    }

public:
    void synchronize_and_check_errors() {
        check_cuda(cudaStreamSynchronize(stream_), "path rendering stream synchronize");
        throw_if_wavefront_error();
        update_timings();
    }

    std::vector<Color> download_pixels(bool use_pinned_staging = true) {
        return download_region_pixels(
            0,
            static_cast<int>(accumulation_.size()),
            samples_,
            use_pinned_staging);
    }

    CudaPathDiagnosticProfile download_diagnostic_profile() {
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
        std::array<unsigned long long, kDiagnosticCounterCount> counters{};
        diagnostic_counters_.download(counters.data(), stream_);
        synchronize_and_check_errors();
        CudaPathDiagnosticProfile profile;
        profile.primary_rays = counters[kDiagnosticPrimaryRays];
        profile.continuation_rays = counters[kDiagnosticContinuationRays];
        profile.directional_shadow_rays =
            counters[kDiagnosticDirectionalShadowRays];
        profile.point_shadow_rays = counters[kDiagnosticPointShadowRays];
        profile.spot_shadow_rays = counters[kDiagnosticSpotShadowRays];
        profile.emissive_shadow_rays =
            counters[kDiagnosticEmissiveShadowRays];
        profile.environment_shadow_rays =
            counters[kDiagnosticEnvironmentShadowRays];
        profile.primary_hits = counters[kDiagnosticPrimaryHits];
        profile.primary_misses = counters[kDiagnosticPrimaryMisses];
        for (std::size_t bounce = 0;
             bounce < CudaPathDiagnosticProfile::kBounceBins;
             ++bounce) {
            profile.rays_by_bounce[bounce] =
                counters[kDiagnosticRaysByBounce + bounce];
            profile.termination_by_bounce[bounce] =
                counters[kDiagnosticTerminationByBounce + bounce];
        }
        return profile;
#else
        throw std::runtime_error(
            "CUDA path diagnostics require the instrumented benchmark target");
#endif
    }

    std::vector<Color> download_region_pixels(
        int pixel_offset,
        int pixel_count,
        int sample_count,
        bool use_pinned_staging = true) {
        if (pixel_offset < 0 || pixel_count <= 0 ||
            static_cast<std::size_t>(pixel_offset) +
                    static_cast<std::size_t>(pixel_count) >
                accumulation_.size()) {
            throw std::invalid_argument(
                "CUDA download region is invalid");
        }
        const std::size_t count =
            static_cast<std::size_t>(pixel_count);
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
            pixel_offset,
            sample_count);
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
    std::uint64_t trace_generation() const {
        return trace_generation_;
    }
    float last_trace_milliseconds() const {
        return last_trace_milliseconds_;
    }

    void update_timings() {
        reset_timer_.update(statistics_.reset_milliseconds);
        update_trace_timing();
        update_presentation_timing();
        check_completed_error();
    }

private:
    bool update_trace_timing() {
        float milliseconds = last_trace_milliseconds_;
        if (!trace_timer_.update(milliseconds)) {
            return false;
        }
        last_trace_milliseconds_ = milliseconds;
        statistics_.trace_milliseconds = milliseconds;
        ++trace_generation_;
        return true;
    }

    bool update_presentation_timing() {
        float milliseconds =
            statistics_.presentation_milliseconds;
        if (!presentation_timer_.update(milliseconds)) {
            return false;
        }
        statistics_.presentation_milliseconds = milliseconds;
        return true;
    }

#if defined(RENDERER_CUDA_SANITIZER_FALLBACK)
    void launch_sanitizer_fixed_topology(int sample_count) {
        const int capacity =
            static_cast<int>(wavefront_capacity_pixels_);
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
                    path_counts_,
                    bounce_index_,
                    shadow_tasks_);
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

    void ensure_wavefront_capacity(
        std::size_t count,
        bool exact_capacity) {
        if ((!exact_capacity &&
             count <= wavefront_capacity_pixels_) ||
            (exact_capacity &&
             count == wavefront_capacity_pixels_)) {
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
            layout.sample_radiance,
            alignof(DVec3),
            sizeof(DVec3),
            count);
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

        if (exact_capacity) {
            wavefront_arena_.resize_exact(
                layout.total_bytes,
                statistics_);
        } else {
            wavefront_arena_.resize(
                layout.total_bytes,
                statistics_);
        }
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
        const int prepare_block_count = occupancy_grid_size(
            reinterpret_cast<void*>(prepare_sample_kernel),
            block_count);
        const int intersect_block_count = block_count;
        const int emissive_block_count = block_count;
        const int direct_block_count = block_count;
        const int finalize_block_count = occupancy_grid_size(
            reinterpret_cast<void*>(finalize_sample_kernel),
            block_count);
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
            dim3(prepare_block_count),
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
            dim3(intersect_block_count),
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
            dim3(emissive_block_count),
            dim3(kThreadsPerBlock),
            emissive_arguments);
        body_tail = emissive_conditional_node;

        void* direct_arguments[] = {
            &frame_parameters,
            &shading_records,
            &path_counts,
            &bounce_index,
            &shadow_tasks};
        body_tail = add_kernel_node(
            body,
            body_tail,
            reinterpret_cast<void*>(direct_visibility_kernel),
            dim3(direct_block_count),
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
            dim3(finalize_block_count),
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

    int occupancy_grid_size(void* function, int capacity_blocks) {
        int device = 0;
        check_cuda(cudaGetDevice(&device), "cudaGetDevice");
        int multiprocessor_count = 0;
        check_cuda(
            cudaDeviceGetAttribute(
                &multiprocessor_count,
                cudaDevAttrMultiProcessorCount,
                device),
            "cudaDeviceGetAttribute multiprocessor count");
        int resident_blocks = 0;
        check_cuda(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &resident_blocks,
                function,
                kThreadsPerBlock,
                0),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
        return std::max(
            1,
            std::min(
                capacity_blocks,
                2 * resident_blocks * multiprocessor_count));
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
        if (!stream_ || !error_code_) {
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
        if (!error_code_) {
            return;
        }
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
        if (error == 5) {
            throw std::runtime_error(
                "CUDA BVH4 traversal stack overflow");
        }
        if (error == 6) {
            throw std::runtime_error(
                "CUDA TLAS traversal stack overflow");
        }
        if (error == 7) {
            throw std::runtime_error(
                "CUDA instance or TLAS index is invalid");
        }
        throw std::runtime_error("CUDA wavefront queue count is invalid");
    }

    CudaPathStatistics& statistics_;
    cudaStream_t stream_ = nullptr;
    bool owns_stream_ = false;
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graph_exec_ = nullptr;
    cudaGraphConditionalHandle batch_conditional_handle_ = 0;
    cudaGraphConditionalHandle wavefront_conditional_handle_ = 0;
    cudaGraphConditionalHandle emissive_conditional_handle_ = 0;
    int graph_capacity_ = 0;
    CudaEventTimer reset_timer_;
    CudaEventTimer trace_timer_;
    CudaEventTimer presentation_timer_;
    float last_trace_milliseconds_ = 0.0f;
    std::uint64_t trace_generation_ = 0;
    DeviceBuffer<DVec3> accumulation_;
    DeviceBuffer<DVec3> resolved_;
    DeviceBuffer<DPcgState> random_states_;
#if defined(RENDERER_BENCHMARK_DIAGNOSTICS)
    DeviceBuffer<unsigned long long> diagnostic_counters_;
#endif
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
    frame.render_samples(
        device_scene.view(),
        camera,
        sample_count,
        settings.path);
    Image image(settings.width, settings.height);
    image.set_pixels(frame.download_pixels(false));
    return RenderResult{std::move(image), timer.elapsed_seconds(), ExecutionBackend::Cuda};
}

class CudaPathInteractiveRenderer::Impl {
public:
    Impl()
        : frame_(statistics_),
          preview_frame_(statistics_, frame_.stream()) {}

    void reset(
        const Scene& scene,
        const RenderSettings& settings,
        const InstancedSceneView* instanced_scene) {
        scene_is_instanced_ = instanced_scene != nullptr;
        if (instanced_scene) {
            scene_ = std::make_unique<CudaSceneStorage>(
                *instanced_scene,
                frame_.stream(),
                statistics_);
        } else {
            scene_ = std::make_unique<CudaSceneStorage>(
                scene,
                frame_.stream(),
                statistics_);
        }
        frame_.reset(
            settings.width,
            settings.height,
            settings.path.sample_seed_offset);
        reset_interactive_state(
            settings.width,
            settings.height,
            scene_->view().triangle_count);
        preview_has_run_ = false;
    }

    void render_next_frame(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target,
        const InstancedSceneView* instanced_scene) {
        render_next_frame_to_surface(
            scene,
            camera,
            settings,
            frame_state,
            0,
            instanced_scene);
        download_current_frame(target);
    }

    void render_next_frame_to_surface(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        CudaSurfaceHandle surface,
        const InstancedSceneView* instanced_scene) {
        const bool wants_instanced = instanced_scene != nullptr;
        if (!scene_ || wants_instanced != scene_is_instanced_) {
            scene_is_instanced_ = wants_instanced;
            scene_ = instanced_scene
                ? std::make_unique<CudaSceneStorage>(
                    *instanced_scene,
                    frame_.stream(),
                    statistics_)
                : std::make_unique<CudaSceneStorage>(
                    scene,
                    frame_.stream(),
                    statistics_);
        } else if (frame_state.scene_changes != SceneChange::None) {
            if (instanced_scene) {
                scene_->sync(
                    *instanced_scene,
                    frame_state.scene_changes);
            } else {
                scene_->sync(
                    scene,
                    frame_state.scene_changes);
            }
        }
        const bool reset_accumulation =
            frame_.width() != settings.width ||
            frame_.height() != settings.height ||
            frame_state.automatic_interaction_quality !=
                automatic_quality_enabled_ ||
            frame_state.camera_changed ||
            frame_state.scene_changes != SceneChange::None ||
            frame_state.framebuffer_resized ||
            frame_state.reset_requested;
        if (reset_accumulation) {
            frame_.reset(
                settings.width,
                settings.height,
                settings.path.sample_seed_offset,
                frame_state.automatic_interaction_quality
                    ? native_wavefront_capacity(
                        settings.width,
                        settings.height)
                    : static_cast<std::size_t>(settings.width) *
                        static_cast<std::size_t>(settings.height));
            reset_interactive_state(
                settings.width,
                settings.height,
                scene_->view().triangle_count);
            automatic_quality_enabled_ =
                frame_state.automatic_interaction_quality;
        }
        output_width_ = settings.width;
        output_height_ = settings.height;
        native_sweep_published_this_frame_ = false;
        statistics_.presentation_updated = false;

        if (!frame_state.automatic_interaction_quality) {
            active_frame_is_preview_ = false;
            frame_.render_sample(
                scene_->view(),
                camera,
                settings.path,
                static_cast<cudaSurfaceObject_t>(surface));
            statistics_.work_mode = CudaPathWorkMode::FullFrame;
            statistics_.internal_width = settings.width;
            statistics_.internal_height = settings.height;
            statistics_.tile_y = 0;
            statistics_.tile_rows = settings.height;
            statistics_.sweep_progress = 1.0f;
            statistics_.presentation_updated = true;
            scene_->update_timing();
            return;
        }

        const bool interaction_changed =
            frame_state.camera_changed ||
            frame_state.scene_changes != SceneChange::None ||
            frame_state.framebuffer_resized ||
            frame_state.reset_requested;
        if (interaction_changed) {
            idle_frames_ = 0;
            preview_dirty_ = true;
            if (!preview_has_run_ ||
                frame_state.scene_changes != SceneChange::None) {
                preview_scale_tier_ =
                    initial_preview_scale_tier(
                        scene_->view().triangle_count);
                preview_trace_ema_ = 0.0f;
                slow_preview_frames_ = 0;
                fast_preview_frames_ = 0;
            }
        } else if (idle_frames_ < kIdleFramesBeforeNative) {
            ++idle_frames_;
        }

        frame_.update_timings();
        preview_frame_.update_timings();
        update_adaptive_work_size();
        if (idle_frames_ < kIdleFramesBeforeNative) {
            render_interaction_preview(
                camera,
                settings,
                static_cast<cudaSurfaceObject_t>(surface));
        } else {
            render_native_quantum(
                camera,
                settings,
                static_cast<cudaSurfaceObject_t>(surface));
        }
        scene_->update_timing();
    }

    void download_current_frame(Framebuffer& target) {
        const auto download_preview = [&]() {
            const std::vector<Color> preview =
                preview_frame_.download_pixels();
            if (target.width() != output_width_ ||
                target.height() != output_height_) {
                target.resize(output_width_, output_height_);
            }
            for (int y = 0; y < output_height_; ++y) {
                const int source_y = std::min(
                    preview_frame_.height() - 1,
                    y * preview_frame_.height() / output_height_);
                for (int x = 0; x < output_width_; ++x) {
                    const int source_x = std::min(
                        preview_frame_.width() - 1,
                        x * preview_frame_.width() / output_width_);
                    target.set_pixel(
                        x,
                        y,
                        preview[static_cast<std::size_t>(source_y) *
                                    static_cast<std::size_t>(
                                        preview_frame_.width()) +
                                static_cast<std::size_t>(source_x)]);
                }
            }
            host_presentation_initialized_ = true;
        };
        if (active_frame_is_preview_) {
            download_preview();
            return;
        }
        if (last_work_mode_ == CudaPathWorkMode::NativeTile &&
            automatic_quality_enabled_) {
            if (native_sweep_published_this_frame_) {
                if (target.width() != frame_.width() ||
                    target.height() != frame_.height()) {
                    target.resize(frame_.width(), frame_.height());
                }
                target.set_pixels(frame_.download_pixels());
                host_presentation_initialized_ = true;
            } else if (!host_presentation_initialized_ &&
                       preview_has_run_) {
                download_preview();
            }
            return;
        }
        if (target.width() != frame_.width() ||
            target.height() != frame_.height()) {
            target.resize(frame_.width(), frame_.height());
        }
        target.set_pixels(frame_.download_pixels());
        host_presentation_initialized_ = true;
    }

    int accumulated_samples() const {
        return frame_.samples();
    }

    CudaStreamHandle stream_handle() const {
        return reinterpret_cast<CudaStreamHandle>(frame_.stream());
    }

    const CudaPathStatistics& statistics() {
        frame_.update_timings();
        preview_frame_.update_timings();
        if (scene_) {
            scene_->update_timing();
        }
        return statistics_;
    }

    CudaPathDiagnosticProfile download_diagnostic_profile() {
        return frame_.download_diagnostic_profile();
    }

    void set_presentation_state(bool interop_active, bool fallback_active) {
        statistics_.interop_active = interop_active;
        statistics_.fallback_active = fallback_active;
    }

private:
    static constexpr int kIdleFramesBeforeNative = 8;
    static constexpr int kPreviewBounceLimit = 2;
    static constexpr int kNativeArenaRows = 128;
    static constexpr int kMaximumQuantumRows = 512;
    static constexpr float kNativeTargetMilliseconds = 12.0f;
    static constexpr float kNativeHardLimitMilliseconds = 15.0f;

    static std::size_t native_wavefront_capacity(int width, int height) {
        return static_cast<std::size_t>(width) *
            static_cast<std::size_t>(
                std::min(height, kNativeArenaRows));
    }

    static int initial_preview_scale_tier(
        int triangle_count) {
        if (triangle_count >= 1'000'000) {
            return 3;
        }
        if (triangle_count >= 20'000) {
            return 2;
        }
        return 0;
    }

    static int initial_native_quantum_rows(int triangle_count) {
        return triangle_count >= 1'000'000 ? 32 : 64;
    }

    void reset_interactive_state(
        int width,
        int height,
        int triangle_count) {
        output_width_ = width;
        output_height_ = height;
        idle_frames_ = 0;
        tile_y_ = 0;
        quantum_rows_ = std::min(
            initial_native_quantum_rows(triangle_count),
            height);
        native_quantum_row_limit_ =
            triangle_count >= 1'000'000
            ? 80
            : kMaximumQuantumRows;
        last_native_quantum_rows_ = 0;
        last_native_quantum_was_tail_ = false;
        native_ms_per_row_ema_ = 0.0f;
        consumed_native_trace_generation_ =
            frame_.trace_generation();
        consumed_preview_trace_generation_ =
            preview_frame_.trace_generation();
        preview_dirty_ = true;
        active_frame_is_preview_ = false;
        native_sweep_published_this_frame_ = false;
        host_presentation_initialized_ = false;
        last_work_mode_ = CudaPathWorkMode::FullFrame;
        sweep_started_ = std::chrono::steady_clock::now();
        statistics_.internal_width = width;
        statistics_.internal_height = height;
        statistics_.tile_y = 0;
        statistics_.tile_rows = quantum_rows_;
        statistics_.sweep_progress = 0.0f;
        statistics_.complete_sweeps_per_second = 0.0f;
        statistics_.presentation_updated = false;
    }

    void update_adaptive_work_size() {
        if (preview_frame_.trace_generation() !=
            consumed_preview_trace_generation_) {
            consumed_preview_trace_generation_ =
                preview_frame_.trace_generation();
            const float elapsed =
                preview_frame_.last_trace_milliseconds();
            if (elapsed > 0.0f && std::isfinite(elapsed)) {
                preview_trace_ema_ = preview_trace_ema_ > 0.0f
                    ? preview_trace_ema_ * 0.4f + elapsed * 0.6f
                    : elapsed;
                if (preview_trace_ema_ > 12.0f) {
                    ++slow_preview_frames_;
                    fast_preview_frames_ = 0;
                    if (slow_preview_frames_ >= 2 &&
                        preview_scale_tier_ + 1 <
                            static_cast<int>(preview_scales_.size())) {
                        ++preview_scale_tier_;
                        slow_preview_frames_ = 0;
                        preview_dirty_ = true;
                    }
                } else if (preview_trace_ema_ < 8.0f) {
                    ++fast_preview_frames_;
                    slow_preview_frames_ = 0;
                    if (fast_preview_frames_ >= 30 &&
                        preview_scale_tier_ > 0) {
                        --preview_scale_tier_;
                        fast_preview_frames_ = 0;
                        preview_dirty_ = true;
                    }
                } else {
                    slow_preview_frames_ = 0;
                    fast_preview_frames_ = 0;
                }
            }
        }

        if (frame_.trace_generation() ==
            consumed_native_trace_generation_) {
            return;
        }
        consumed_native_trace_generation_ =
            frame_.trace_generation();
        const float elapsed = frame_.last_trace_milliseconds();
        if (last_native_quantum_rows_ <= 0 ||
            last_native_quantum_was_tail_ ||
            !(elapsed > 0.0f) ||
            !std::isfinite(elapsed)) {
            return;
        }
        if (elapsed > kNativeHardLimitMilliseconds) {
            quantum_rows_ =
                std::max(1, last_native_quantum_rows_ / 2);
            return;
        }
        const float milliseconds_per_row =
            elapsed /
            static_cast<float>(last_native_quantum_rows_);
        native_ms_per_row_ema_ =
            native_ms_per_row_ema_ > 0.0f
            ? native_ms_per_row_ema_ * 0.75f +
                milliseconds_per_row * 0.25f
            : milliseconds_per_row;
        int desired_rows = std::clamp(
            static_cast<int>(std::lround(
                kNativeTargetMilliseconds /
                native_ms_per_row_ema_)),
            1,
            native_quantum_row_limit_);
        if (desired_rows > kNativeArenaRows) {
            const int chunk_count = std::clamp(
                static_cast<int>(std::lround(
                    static_cast<float>(desired_rows) /
                    static_cast<float>(kNativeArenaRows))),
                1,
                std::max(
                    1,
                    native_quantum_row_limit_ /
                        kNativeArenaRows));
            desired_rows = chunk_count * kNativeArenaRows;
        }
        const int minimum_rows =
            std::max(1, last_native_quantum_rows_ / 2);
        const int maximum_rows = std::min(
            native_quantum_row_limit_,
            last_native_quantum_rows_ * 2);
        quantum_rows_ = std::clamp(
            desired_rows,
            minimum_rows,
            maximum_rows);
    }

    void render_interaction_preview(
        const Camera& camera,
        const RenderSettings& settings,
        cudaSurfaceObject_t surface) {
        const float scale =
            preview_scales_[static_cast<std::size_t>(preview_scale_tier_)];
        const int preview_width = std::max(
            1,
            static_cast<int>(std::lround(
                static_cast<float>(settings.width) * scale)));
        const int preview_height = std::max(
            1,
            static_cast<int>(std::lround(
                static_cast<float>(settings.height) * scale)));
        if (preview_dirty_ ||
            preview_frame_.width() != preview_width ||
            preview_frame_.height() != preview_height) {
            preview_frame_.reset(
                preview_width,
                preview_height,
                settings.path.sample_seed_offset);
            preview_dirty_ = false;
        }
        preview_frame_.render_region(
            scene_->view(),
            camera,
            0,
            preview_width * preview_height,
            kPreviewBounceLimit,
            settings.width,
            settings.height,
            settings.path,
            surface);
        preview_frame_.complete_sample();
        preview_has_run_ = true;
        active_frame_is_preview_ = true;
        last_work_mode_ = CudaPathWorkMode::InteractionPreview;
        statistics_.work_mode = last_work_mode_;
        statistics_.internal_width = preview_width;
        statistics_.internal_height = preview_height;
        statistics_.tile_y = 0;
        statistics_.tile_rows = preview_height;
        statistics_.sweep_progress = 0.0f;
        statistics_.presentation_updated = true;
    }

    void render_native_quantum(
        const Camera& camera,
        const RenderSettings& settings,
        cudaSurfaceObject_t surface) {
        const int rows = std::min(
            quantum_rows_,
            settings.height - tile_y_);
        last_native_quantum_rows_ = rows;
        last_native_quantum_was_tail_ =
            rows < quantum_rows_;
        frame_.render_region_quantum(
            scene_->view(),
            camera,
            tile_y_ * settings.width,
            rows * settings.width,
            kMaxPathBounces,
            settings.width,
            settings.height,
            settings.path);
        frame_.synchronize_and_check_errors();
        active_frame_is_preview_ = false;
        last_work_mode_ = CudaPathWorkMode::NativeTile;
        tile_y_ += rows;
        if (tile_y_ >= settings.height) {
            frame_.complete_sample();
            if (surface != 0) {
                frame_.present_surface(surface);
            }
            native_sweep_published_this_frame_ = true;
            statistics_.presentation_updated = true;
            tile_y_ = 0;
            const auto completed_at = std::chrono::steady_clock::now();
            const float seconds =
                std::chrono::duration<float>(
                    completed_at - sweep_started_).count();
            if (seconds > 0.0f) {
                statistics_.complete_sweeps_per_second =
                    1.0f / seconds;
            }
            sweep_started_ = completed_at;
        }
        statistics_.work_mode = last_work_mode_;
        statistics_.internal_width = settings.width;
        statistics_.internal_height = rows;
        statistics_.tile_y = tile_y_;
        statistics_.tile_rows = rows;
        statistics_.sweep_progress =
            static_cast<float>(tile_y_) /
            static_cast<float>(settings.height);
    }

    CudaPathStatistics statistics_;
    CudaFrameStorage frame_;
    CudaFrameStorage preview_frame_;
    std::unique_ptr<CudaSceneStorage> scene_;
    const std::array<float, 4> preview_scales_{
        1.0f,
        0.75f,
        0.5f,
        0.25f};
    int preview_scale_tier_ = 0;
    int slow_preview_frames_ = 0;
    int fast_preview_frames_ = 0;
    int idle_frames_ = kIdleFramesBeforeNative;
    int tile_y_ = 0;
    int quantum_rows_ = 64;
    int native_quantum_row_limit_ = kMaximumQuantumRows;
    int last_native_quantum_rows_ = 0;
    int output_width_ = 1;
    int output_height_ = 1;
    float preview_trace_ema_ = 0.0f;
    float native_ms_per_row_ema_ = 0.0f;
    std::uint64_t consumed_native_trace_generation_ = 0;
    std::uint64_t consumed_preview_trace_generation_ = 0;
    bool preview_dirty_ = true;
    bool preview_has_run_ = false;
    bool active_frame_is_preview_ = false;
    bool last_native_quantum_was_tail_ = false;
    bool native_sweep_published_this_frame_ = false;
    bool host_presentation_initialized_ = false;
    bool automatic_quality_enabled_ = false;
    bool scene_is_instanced_ = false;
    CudaPathWorkMode last_work_mode_ = CudaPathWorkMode::FullFrame;
    std::chrono::steady_clock::time_point sweep_started_{};
};

CudaPathInteractiveRenderer::CudaPathInteractiveRenderer()
    : impl_(std::make_unique<Impl>()) {}
CudaPathInteractiveRenderer::~CudaPathInteractiveRenderer() = default;
CudaPathInteractiveRenderer::CudaPathInteractiveRenderer(CudaPathInteractiveRenderer&&) noexcept = default;
CudaPathInteractiveRenderer& CudaPathInteractiveRenderer::operator=(CudaPathInteractiveRenderer&&) noexcept = default;

void CudaPathInteractiveRenderer::reset(
    const Scene& scene,
    const RenderSettings& settings,
    const InstancedSceneView* instanced_scene) {
    impl_->reset(scene, settings, instanced_scene);
}

void CudaPathInteractiveRenderer::render_next_frame(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    Framebuffer& target,
    const InstancedSceneView* instanced_scene) {
    impl_->render_next_frame(
        scene,
        camera,
        settings,
        frame_state,
        target,
        instanced_scene);
}

void CudaPathInteractiveRenderer::render_next_frame_to_surface(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state,
    CudaSurfaceHandle surface,
    const InstancedSceneView* instanced_scene) {
    if (surface == 0) {
        throw std::invalid_argument("CUDA surface output requires a non-zero surface handle");
    }
    impl_->render_next_frame_to_surface(
        scene,
        camera,
        settings,
        frame_state,
        surface,
        instanced_scene);
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

CudaPathDiagnosticProfile
CudaPathInteractiveRenderer::download_diagnostic_profile() {
    return impl_->download_diagnostic_profile();
}

void CudaPathInteractiveRenderer::set_presentation_state(
    bool interop_active,
    bool fallback_active) {
    impl_->set_presentation_state(interop_active, fallback_active);
}

}  // namespace renderer
