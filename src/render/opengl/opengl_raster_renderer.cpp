#include "render/opengl/opengl_raster_renderer.h"

#include "render/opengl/opengl_shadow_math.h"

#include "platform/opengl/gl_shader_program.h"
#include "scene/environment.h"
#include "scene/material.h"

#include <glad/gl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace renderer {

namespace {

struct GpuVertex {
    std::array<float, 3> position{};
    std::array<float, 3> normal{};
    std::array<float, 2> uv{};
    std::array<float, 4> tangent{};
    std::array<float, 2> uv1{};
    std::array<float, 4> color{};
};

struct DrawBatch {
    int material_id = -1;
    GLint first = 0;
    GLsizei count = 0;
};

struct alignas(16) GpuDirectionalLight {
    std::array<float, 4> direction_angular{};
    std::array<float, 4> radiance{};
    std::array<float, 4> shadow{};
};

struct alignas(16) GpuPointLight {
    std::array<float, 4> position_range{};
    std::array<float, 4> intensity{};
    std::array<float, 4> shadow{};
};

struct alignas(16) GpuSpotLight {
    std::array<float, 4> position_range{};
    std::array<float, 4> direction_inner{};
    std::array<float, 4> intensity_outer{};
    std::array<float, 4> shadow{};
};

struct alignas(16) GpuRectAreaLight {
    std::array<float, 4> position_two_sided{};
    std::array<float, 4> axis_u{};
    std::array<float, 4> axis_v{};
    std::array<float, 4> radiance{};
    std::array<float, 4> shadow{};
};

struct ShadowReference {
    int layer = -1;
    int global_slot = -1;
};

enum class ShadowLightType : std::uint8_t {
    Directional,
    Point,
    Spot,
    RectArea,
    DominantEnvironment,
};

struct ShadowSlot {
    ShadowLightType type = ShadowLightType::Directional;
    std::size_t light_index = 0;
    int layer = -1;
    int global_slot = -1;
    bool cube = false;
    Vec3 position = Vec3::Zero();
    Vec3 direction = Vec3(0.0f, 0.0f, -1.0f);
    float near_plane = 0.01f;
    float far_plane = 1.0f;
    std::array<Mat4, 6> matrices{};
};

struct ShadowCandidate {
    ShadowLightType type = ShadowLightType::Directional;
    std::size_t light_index = 0;
    int priority = 0;
    float contribution = 0.0f;
    bool cube = false;
};

struct UniformLocations {
    GLint view_projection = -1;
    GLint camera_position = -1;
    GLint camera_forward = -1;
    GLint camera_right = -1;
    GLint camera_up = -1;
    GLint camera_viewport = -1;
    GLint viewport_size = -1;
    GLint view_to_world = -1;
    GLint environment_color = -1;
    GLint environment_sh = -1;
    GLint environment_intensity = -1;
    GLint environment_rotation_radians = -1;
    GLint environment_mip_count = -1;
    GLint has_environment_map = -1;
    GLint ibl_enabled = -1;
    GLint ltc_area_lights_enabled = -1;
    GLint material_type = -1;
    GLint pbr_workflow = -1;
    GLint base_color = -1;
    GLint emission = -1;
    GLint ior = -1;
    GLint specular_color = -1;
    GLint specular_factor = -1;
    GLint glossiness = -1;
    GLint metallic = -1;
    GLint roughness = -1;
    GLint opacity = -1;
    GLint alpha_cutoff = -1;
    GLint bump_scale = -1;
    GLint normal_scale = -1;
    GLint occlusion_strength = -1;
    GLint alpha_mode = -1;
    GLint two_sided = -1;
    GLint has_base_color_texture = -1;
    GLint has_opacity_texture = -1;
    GLint has_bump_texture = -1;
    GLint has_normal_texture = -1;
    GLint has_metallic_roughness_texture = -1;
    GLint has_occlusion_texture = -1;
    GLint has_emissive_texture = -1;
    GLint has_specular_texture = -1;
    GLint has_specular_color_texture = -1;
    GLint has_specular_glossiness_texture = -1;
    std::array<GLint, 9> texture_offset_scale{};
    std::array<GLint, 9> texture_rotation{};
    std::array<GLint, 9> texture_texcoord{};
    std::array<GLint, 9> texture_top_left{};
    GLint directional_light_count = -1;
    GLint point_light_count = -1;
    GLint spot_light_count = -1;
    GLint rect_area_light_count = -1;
    GLint shadow_matrices = -1;
    GLint shadow_origin_far = -1;
    GLint shadow_direction_near = -1;
    GLint cube_shadow_position_far = -1;
    GLint cube_shadow_near = -1;
    GLint shadow_map_resolution = -1;
    GLint shadow_constant_bias = -1;
    GLint shadow_slope_bias = -1;
    GLint pcss_enabled = -1;
    GLint pcss_blocker_samples = -1;
    GLint pcss_filter_samples = -1;
    GLint pcss_max_penumbra_texels = -1;
    GLint pcss_light_size_scale = -1;
    GLint shadow_debug_view = -1;
    GLint shadow_debug_slot = -1;
    GLint transparent_pass = -1;
    GLint ao_mode = -1;
    GLint ao_bent_normals_enabled = -1;
};

bool valid_material_id(const Scene& scene, int material_id) {
    return material_id >= 0 &&
        static_cast<std::size_t>(material_id) < scene.materials.size();
}

bool valid_texture_id(const Scene& scene, int texture_id) {
    return texture_id >= 0 &&
        static_cast<std::size_t>(texture_id) < scene.textures.size();
}

Vec3 safe_geometric_normal(const Triangle& triangle) {
    const Vec3 normal = (triangle.b() - triangle.a()).cross(triangle.c() - triangle.a());
    return usable_direction(normal) ? normal.normalized() : Vec3::Zero();
}

Vec3 gpu_vertex_normal(const TriangleVertex& vertex, const Vec3& geometric_normal) {
    if (!vertex.has_normal || !usable_direction(vertex.normal)) {
        return geometric_normal;
    }
    return vertex.normal.dot(geometric_normal) < 0.0f ? -vertex.normal : vertex.normal;
}

void triangle_tangent_data(
    const Triangle& triangle,
    const Vec3& normal,
    Vec3& tangent,
    float& handedness) {
    tangent = Vec3::Zero();
    handedness = 0.0f;
    if (!triangle.has_valid_uv_basis()) {
        return;
    }

    const Vec3 edge1 = triangle.b() - triangle.a();
    const Vec3 edge2 = triangle.c() - triangle.a();
    const Vec2 duv1 = triangle.vertex(1).uv - triangle.vertex(0).uv;
    const Vec2 duv2 = triangle.vertex(2).uv - triangle.vertex(0).uv;
    const float determinant = duv1.x() * duv2.y() - duv1.y() * duv2.x();
    if (!std::isfinite(determinant) || std::abs(determinant) <= 1.0e-12f) {
        return;
    }
    const float inverse = 1.0f / determinant;
    const Vec3 raw_tangent = (edge1 * duv2.y() - edge2 * duv1.y()) * inverse;
    const Vec3 raw_bitangent = (edge2 * duv1.x() - edge1 * duv2.x()) * inverse;
    tangent = raw_tangent - normal * raw_tangent.dot(normal);
    if (!usable_direction(tangent)) {
        tangent = Vec3::Zero();
        return;
    }
    tangent.normalize();
    handedness = normal.cross(tangent).dot(raw_bitangent) < 0.0f ? -1.0f : 1.0f;
}

GpuVertex make_gpu_vertex(
    const Triangle& triangle,
    int vertex_index,
    const Vec3& geometric_normal) {
    const TriangleVertex& source = triangle.vertex(vertex_index);
    const Vec3 normal = gpu_vertex_normal(source, geometric_normal);
    Vec3 tangent;
    float handedness = 0.0f;
    if (source.has_tangent && usable_direction(source.tangent.head<3>())) {
        tangent = source.tangent.head<3>().normalized();
        handedness = source.tangent.w() < 0.0f ? -1.0f : 1.0f;
    } else {
        triangle_tangent_data(triangle, normal, tangent, handedness);
    }

    GpuVertex vertex;
    vertex.position = {source.position.x(), source.position.y(), source.position.z()};
    vertex.normal = {normal.x(), normal.y(), normal.z()};
    vertex.uv = {source.uv.x(), source.uv.y()};
    vertex.tangent = {tangent.x(), tangent.y(), tangent.z(), handedness};
    vertex.uv1 = {source.uv1.x(), source.uv1.y()};
    vertex.color = {
        source.color.x(),
        source.color.y(),
        source.color.z(),
        source.alpha};
    return vertex;
}

Mat4 view_projection_matrix(const Camera& camera, float near_plane) {
    Mat4 view = Mat4::Identity();
    view.block<1, 3>(0, 0) = camera.right().transpose();
    view.block<1, 3>(1, 0) = camera.up().transpose();
    view.block<1, 3>(2, 0) = (-camera.forward()).transpose();
    view(0, 3) = -camera.right().dot(camera.eye());
    view(1, 3) = -camera.up().dot(camera.eye());
    view(2, 3) = camera.forward().dot(camera.eye());

    Mat4 projection = Mat4::Zero();
    projection(0, 0) = 2.0f / camera.viewport_width();
    projection(1, 1) = 2.0f / camera.viewport_height();
    projection(2, 2) = -1.0f;
    projection(2, 3) = -2.0f * near_plane;
    projection(3, 2) = -1.0f;
    return projection * view;
}

float color_luminance(const Color& color) {
    return std::max(0.0f, color.dot(Color(0.2126f, 0.7152f, 0.0722f)));
}

Vec3 rotate_y_degrees(const Vec3& direction, float degrees) {
    const float radians = degrees * 0.01745329251994329577f;
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    return Vec3(
        cosine * direction.x() + sine * direction.z(),
        direction.y(),
        -sine * direction.x() + cosine * direction.z());
}

Bounds3 shadow_scene_bounds(const Scene& scene) {
    Bounds3 bounds;
    for (const Triangle& triangle : scene.triangles) {
        bounds.expand(triangle.a());
        bounds.expand(triangle.b());
        bounds.expand(triangle.c());
    }
    if (!bounds.min.allFinite() || !bounds.max.allFinite()) {
        return Bounds3(-Vec3::Ones(), Vec3::Ones());
    }
    return bounds;
}

std::array<Vec3, 8> bounds_corners(const Bounds3& bounds) {
    std::array<Vec3, 8> corners{};
    for (int mask = 0; mask < 8; ++mask) {
        corners[static_cast<std::size_t>(mask)] = Vec3(
            (mask & 1) != 0 ? bounds.max.x() : bounds.min.x(),
            (mask & 2) != 0 ? bounds.max.y() : bounds.min.y(),
            (mask & 4) != 0 ? bounds.max.z() : bounds.min.z());
    }
    return corners;
}

Mat4 look_forward_matrix(
    const Vec3& eye,
    const Vec3& forward_value,
    const Vec3& preferred_up) {
    const Vec3 forward = forward_value.normalized();
    Vec3 right = forward.cross(preferred_up);
    if (!usable_direction(right)) {
        right = forward.cross(
            std::abs(forward.y()) < 0.999f
                ? Vec3(0.0f, 1.0f, 0.0f)
                : Vec3(1.0f, 0.0f, 0.0f));
    }
    right.normalize();
    const Vec3 up = right.cross(forward).normalized();
    Mat4 view = Mat4::Identity();
    view.block<1, 3>(0, 0) = right.transpose();
    view.block<1, 3>(1, 0) = up.transpose();
    view.block<1, 3>(2, 0) = (-forward).transpose();
    view(0, 3) = -right.dot(eye);
    view(1, 3) = -up.dot(eye);
    view(2, 3) = forward.dot(eye);
    return view;
}

Mat4 perspective_matrix(float field_of_view_radians, float near_plane, float far_plane) {
    const float tangent = std::tan(field_of_view_radians * 0.5f);
    Mat4 projection = Mat4::Zero();
    projection(0, 0) = 1.0f / tangent;
    projection(1, 1) = 1.0f / tangent;
    projection(2, 2) = -(far_plane + near_plane) / (far_plane - near_plane);
    projection(2, 3) =
        -(2.0f * far_plane * near_plane) / (far_plane - near_plane);
    projection(3, 2) = -1.0f;
    return projection;
}

Mat4 orthographic_matrix(
    float left,
    float right,
    float bottom,
    float top,
    float near_plane,
    float far_plane) {
    Mat4 projection = Mat4::Identity();
    projection(0, 0) = 2.0f / (right - left);
    projection(1, 1) = 2.0f / (top - bottom);
    projection(2, 2) = -2.0f / (far_plane - near_plane);
    projection(0, 3) = -(right + left) / (right - left);
    projection(1, 3) = -(top + bottom) / (top - bottom);
    projection(2, 3) = -(far_plane + near_plane) / (far_plane - near_plane);
    return projection;
}

void set_uniform(GLint location, int value) {
    if (location >= 0) {
        glUniform1i(location, value);
    }
}

void set_uniform(GLint location, float value) {
    if (location >= 0) {
        glUniform1f(location, value);
    }
}

void set_uniform(GLint location, const Vec3& value) {
    if (location >= 0) {
        glUniform3f(location, value.x(), value.y(), value.z());
    }
}

UniformLocations find_uniforms(GLuint program) {
    UniformLocations uniforms;
    uniforms.view_projection = glGetUniformLocation(program, "u_view_projection");
    uniforms.camera_position = glGetUniformLocation(program, "u_camera_position");
    uniforms.camera_forward = glGetUniformLocation(program, "u_camera_forward");
    uniforms.camera_right = glGetUniformLocation(program, "u_camera_right");
    uniforms.camera_up = glGetUniformLocation(program, "u_camera_up");
    uniforms.camera_viewport = glGetUniformLocation(program, "u_camera_viewport");
    uniforms.viewport_size = glGetUniformLocation(program, "u_viewport_size");
    uniforms.view_to_world = glGetUniformLocation(program, "u_view_to_world");
    uniforms.environment_color = glGetUniformLocation(program, "u_environment_color");
    uniforms.environment_sh = glGetUniformLocation(program, "u_environment_sh[0]");
    uniforms.environment_intensity = glGetUniformLocation(program, "u_environment_intensity");
    uniforms.environment_rotation_radians = glGetUniformLocation(program, "u_environment_rotation_radians");
    uniforms.environment_mip_count = glGetUniformLocation(program, "u_environment_mip_count");
    uniforms.has_environment_map = glGetUniformLocation(program, "u_has_environment_map");
    uniforms.ibl_enabled = glGetUniformLocation(program, "u_ibl_enabled");
    uniforms.ltc_area_lights_enabled = glGetUniformLocation(program, "u_ltc_area_lights_enabled");
    uniforms.material_type = glGetUniformLocation(program, "u_material_type");
    uniforms.pbr_workflow = glGetUniformLocation(program, "u_pbr_workflow");
    uniforms.base_color = glGetUniformLocation(program, "u_base_color");
    uniforms.emission = glGetUniformLocation(program, "u_emission");
    uniforms.ior = glGetUniformLocation(program, "u_ior");
    uniforms.specular_color = glGetUniformLocation(program, "u_specular_color");
    uniforms.specular_factor = glGetUniformLocation(program, "u_specular_factor");
    uniforms.glossiness = glGetUniformLocation(program, "u_glossiness");
    uniforms.metallic = glGetUniformLocation(program, "u_metallic");
    uniforms.roughness = glGetUniformLocation(program, "u_roughness");
    uniforms.opacity = glGetUniformLocation(program, "u_opacity");
    uniforms.alpha_cutoff = glGetUniformLocation(program, "u_alpha_cutoff");
    uniforms.bump_scale = glGetUniformLocation(program, "u_bump_scale");
    uniforms.normal_scale = glGetUniformLocation(program, "u_normal_scale");
    uniforms.occlusion_strength = glGetUniformLocation(program, "u_occlusion_strength");
    uniforms.alpha_mode = glGetUniformLocation(program, "u_alpha_mode");
    uniforms.two_sided = glGetUniformLocation(program, "u_two_sided");
    uniforms.has_base_color_texture = glGetUniformLocation(program, "u_has_base_color_texture");
    uniforms.has_opacity_texture = glGetUniformLocation(program, "u_has_opacity_texture");
    uniforms.has_bump_texture = glGetUniformLocation(program, "u_has_bump_texture");
    uniforms.has_normal_texture = glGetUniformLocation(program, "u_has_normal_texture");
    uniforms.has_metallic_roughness_texture = glGetUniformLocation(program, "u_has_metallic_roughness_texture");
    uniforms.has_occlusion_texture = glGetUniformLocation(program, "u_has_occlusion_texture");
    uniforms.has_emissive_texture = glGetUniformLocation(program, "u_has_emissive_texture");
    uniforms.has_specular_texture = glGetUniformLocation(program, "u_has_specular_texture");
    uniforms.has_specular_color_texture = glGetUniformLocation(program, "u_has_specular_color_texture");
    uniforms.has_specular_glossiness_texture = glGetUniformLocation(program, "u_has_specular_glossiness_texture");
    for (int slot = 0; slot < 9; ++slot) {
        const std::string suffix = "[" + std::to_string(slot) + "]";
        uniforms.texture_offset_scale[static_cast<std::size_t>(slot)] =
            glGetUniformLocation(program, ("u_texture_offset_scale" + suffix).c_str());
        uniforms.texture_rotation[static_cast<std::size_t>(slot)] =
            glGetUniformLocation(program, ("u_texture_rotation" + suffix).c_str());
        uniforms.texture_texcoord[static_cast<std::size_t>(slot)] =
            glGetUniformLocation(program, ("u_texture_texcoord" + suffix).c_str());
        uniforms.texture_top_left[static_cast<std::size_t>(slot)] =
            glGetUniformLocation(program, ("u_texture_top_left" + suffix).c_str());
    }
    uniforms.directional_light_count = glGetUniformLocation(program, "u_directional_light_count");
    uniforms.point_light_count = glGetUniformLocation(program, "u_point_light_count");
    uniforms.spot_light_count = glGetUniformLocation(program, "u_spot_light_count");
    uniforms.rect_area_light_count = glGetUniformLocation(program, "u_rect_area_light_count");
    uniforms.shadow_matrices = glGetUniformLocation(program, "u_shadow_matrices[0]");
    uniforms.shadow_origin_far = glGetUniformLocation(program, "u_shadow_origin_far[0]");
    uniforms.shadow_direction_near = glGetUniformLocation(program, "u_shadow_direction_near[0]");
    uniforms.cube_shadow_position_far = glGetUniformLocation(program, "u_cube_shadow_position_far[0]");
    uniforms.cube_shadow_near = glGetUniformLocation(program, "u_cube_shadow_near[0]");
    uniforms.shadow_map_resolution = glGetUniformLocation(program, "u_shadow_map_resolution");
    uniforms.shadow_constant_bias = glGetUniformLocation(program, "u_shadow_constant_bias");
    uniforms.shadow_slope_bias = glGetUniformLocation(program, "u_shadow_slope_bias");
    uniforms.pcss_enabled = glGetUniformLocation(program, "u_pcss_enabled");
    uniforms.pcss_blocker_samples = glGetUniformLocation(program, "u_pcss_blocker_samples");
    uniforms.pcss_filter_samples = glGetUniformLocation(program, "u_pcss_filter_samples");
    uniforms.pcss_max_penumbra_texels = glGetUniformLocation(program, "u_pcss_max_penumbra_texels");
    uniforms.pcss_light_size_scale = glGetUniformLocation(program, "u_pcss_light_size_scale");
    uniforms.shadow_debug_view = glGetUniformLocation(program, "u_shadow_debug_view");
    uniforms.shadow_debug_slot = glGetUniformLocation(program, "u_shadow_debug_slot");
    uniforms.transparent_pass = glGetUniformLocation(program, "u_transparent_pass");
    uniforms.ao_mode = glGetUniformLocation(program, "u_ao_mode");
    uniforms.ao_bent_normals_enabled =
        glGetUniformLocation(program, "u_ao_bent_normals_enabled");
    return uniforms;
}

std::filesystem::file_time_type file_write_time(const std::filesystem::path& path) {
    std::error_code error;
    const auto write_time = std::filesystem::last_write_time(path, error);
    return error ? std::filesystem::file_time_type::min() : write_time;
}

GLint gl_texture_wrap(TextureWrap wrap) {
    switch (wrap) {
        case TextureWrap::ClampToEdge:
            return GL_CLAMP_TO_EDGE;
        case TextureWrap::MirroredRepeat:
            return GL_MIRRORED_REPEAT;
        case TextureWrap::Repeat:
            return GL_REPEAT;
    }
    return GL_REPEAT;
}

GLint gl_texture_filter(TextureFilter filter) {
    switch (filter) {
        case TextureFilter::Nearest:
            return GL_NEAREST;
        case TextureFilter::Linear:
            return GL_LINEAR;
        case TextureFilter::NearestMipmapNearest:
            return GL_NEAREST_MIPMAP_NEAREST;
        case TextureFilter::LinearMipmapNearest:
            return GL_LINEAR_MIPMAP_NEAREST;
        case TextureFilter::NearestMipmapLinear:
            return GL_NEAREST_MIPMAP_LINEAR;
        case TextureFilter::LinearMipmapLinear:
            return GL_LINEAR_MIPMAP_LINEAR;
    }
    return GL_LINEAR;
}

float radical_inverse(std::uint32_t bits) {
    bits = (bits << 16U) | (bits >> 16U);
    bits = ((bits & 0x55555555U) << 1U) | ((bits & 0xAAAAAAAAU) >> 1U);
    bits = ((bits & 0x33333333U) << 2U) | ((bits & 0xCCCCCCCCU) >> 2U);
    bits = ((bits & 0x0F0F0F0FU) << 4U) | ((bits & 0xF0F0F0F0U) >> 4U);
    bits = ((bits & 0x00FF00FFU) << 8U) | ((bits & 0xFF00FF00U) >> 8U);
    return static_cast<float>(bits) * 2.3283064365386963e-10f;
}

Vec3 cube_direction(int face, float u, float v) {
    const float x = 2.0f * u - 1.0f;
    const float y = 2.0f * v - 1.0f;
    switch (face) {
        case 0:
            return Vec3(1.0f, -y, -x).normalized();
        case 1:
            return Vec3(-1.0f, -y, x).normalized();
        case 2:
            return Vec3(x, 1.0f, y).normalized();
        case 3:
            return Vec3(x, -1.0f, -y).normalized();
        case 4:
            return Vec3(x, -y, 1.0f).normalized();
        default:
            return Vec3(-x, -y, -1.0f).normalized();
    }
}

void orthonormal_basis(const Vec3& normal, Vec3& tangent, Vec3& bitangent) {
    const Vec3 helper = std::abs(normal.z()) < 0.999f
        ? Vec3(0.0f, 0.0f, 1.0f)
        : Vec3(1.0f, 0.0f, 0.0f);
    tangent = helper.cross(normal).normalized();
    bitangent = normal.cross(tangent);
}

Vec3 importance_sample_ggx(float sample_x, float sample_y, float roughness, const Vec3& normal) {
    constexpr float pi = 3.14159265358979323846f;
    const float alpha = roughness * roughness;
    const float alpha_squared = alpha * alpha;
    const float phi = 2.0f * pi * sample_x;
    const float cosine = std::sqrt(
        (1.0f - sample_y) /
        std::max(1.0f + (alpha_squared - 1.0f) * sample_y, 1.0e-8f));
    const float sine = std::sqrt(std::max(0.0f, 1.0f - cosine * cosine));
    Vec3 tangent;
    Vec3 bitangent;
    orthonormal_basis(normal, tangent, bitangent);
    return (tangent * (std::cos(phi) * sine) +
            bitangent * (std::sin(phi) * sine) +
            normal * cosine)
        .normalized();
}

std::vector<float> prefilter_environment_face(
    const EnvironmentMap& map,
    int face,
    int size,
    float roughness) {
    constexpr std::uint32_t sample_count = 64U;
    std::vector<float> pixels(static_cast<std::size_t>(size * size * 4), 0.0f);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const Vec3 normal = cube_direction(
                face,
                (static_cast<float>(x) + 0.5f) / static_cast<float>(size),
                (static_cast<float>(y) + 0.5f) / static_cast<float>(size));
            Color filtered = Color::Zero();
            float total_weight = 0.0f;
            if (roughness <= 1.0e-4f) {
                filtered = map.sample_direction(normal);
                total_weight = 1.0f;
            } else {
                for (std::uint32_t sample_index = 0; sample_index < sample_count; ++sample_index) {
                    const Vec3 half_vector = importance_sample_ggx(
                        (static_cast<float>(sample_index) + 0.5f) / static_cast<float>(sample_count),
                        radical_inverse(sample_index),
                        roughness,
                        normal);
                    const Vec3 incoming =
                        (2.0f * normal.dot(half_vector) * half_vector - normal).normalized();
                    const float weight = std::max(0.0f, normal.dot(incoming));
                    if (weight > 0.0f) {
                        filtered += map.sample_direction(incoming) * weight;
                        total_weight += weight;
                    }
                }
            }
            filtered /= std::max(total_weight, 1.0e-8f);
            const std::size_t base = static_cast<std::size_t>((y * size + x) * 4);
            pixels[base] = filtered.x();
            pixels[base + 1U] = filtered.y();
            pixels[base + 2U] = filtered.z();
            pixels[base + 3U] = 1.0f;
        }
    }
    return pixels;
}

float smith_g1_lut(float cosine, float roughness) {
    const float alpha = roughness * roughness;
    const float tangent_squared =
        std::max(0.0f, (1.0f - cosine * cosine) / std::max(cosine * cosine, 1.0e-8f));
    return 2.0f / (1.0f + std::sqrt(1.0f + alpha * alpha * tangent_squared));
}

std::vector<float> integrate_brdf_lut(int size) {
    constexpr std::uint32_t sample_count = 128U;
    std::vector<float> pixels(static_cast<std::size_t>(size * size * 2), 0.0f);
    for (int y = 0; y < size; ++y) {
        const float roughness = (static_cast<float>(y) + 0.5f) / static_cast<float>(size);
        for (int x = 0; x < size; ++x) {
            const float n_dot_v = std::max(
                (static_cast<float>(x) + 0.5f) / static_cast<float>(size),
                1.0e-4f);
            const Vec3 view(std::sqrt(std::max(0.0f, 1.0f - n_dot_v * n_dot_v)), 0.0f, n_dot_v);
            float scale = 0.0f;
            float bias = 0.0f;
            for (std::uint32_t sample_index = 0; sample_index < sample_count; ++sample_index) {
                const Vec3 half_vector = importance_sample_ggx(
                    (static_cast<float>(sample_index) + 0.5f) / static_cast<float>(sample_count),
                    radical_inverse(sample_index),
                    roughness,
                    Vec3(0.0f, 0.0f, 1.0f));
                const Vec3 light = (2.0f * view.dot(half_vector) * half_vector - view).normalized();
                const float n_dot_l = std::max(light.z(), 0.0f);
                const float n_dot_h = std::max(half_vector.z(), 0.0f);
                const float v_dot_h = std::max(view.dot(half_vector), 0.0f);
                if (n_dot_l <= 0.0f || n_dot_h <= 0.0f) {
                    continue;
                }
                const float geometry =
                    smith_g1_lut(n_dot_v, roughness) * smith_g1_lut(n_dot_l, roughness);
                const float visibility = geometry * v_dot_h /
                    std::max(n_dot_h * n_dot_v, 1.0e-8f);
                const float fresnel = std::pow(1.0f - v_dot_h, 5.0f);
                scale += (1.0f - fresnel) * visibility;
                bias += fresnel * visibility;
            }
            const std::size_t base = static_cast<std::size_t>((y * size + x) * 2);
            pixels[base] = scale / static_cast<float>(sample_count);
            pixels[base + 1U] = bias / static_cast<float>(sample_count);
        }
    }
    return pixels;
}

std::vector<std::uint16_t> load_ltc_dds(
    const std::filesystem::path& path) {
    constexpr int kLutSize = 64;
    constexpr std::size_t kHeaderSize = 128U;
    constexpr std::size_t kValueCount =
        static_cast<std::size_t>(kLutSize * kLutSize * 4);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open LTC LUT: " + path.string());
    }
    std::array<unsigned char, kHeaderSize> header{};
    input.read(
        reinterpret_cast<char*>(header.data()),
        static_cast<std::streamsize>(header.size()));
    const auto read_u32 = [&header](std::size_t offset) {
        return static_cast<std::uint32_t>(header[offset]) |
            (static_cast<std::uint32_t>(header[offset + 1U]) << 8U) |
            (static_cast<std::uint32_t>(header[offset + 2U]) << 16U) |
            (static_cast<std::uint32_t>(header[offset + 3U]) << 24U);
    };
    if (!input || header[0] != 'D' || header[1] != 'D' ||
        header[2] != 'S' || header[3] != ' ' ||
        read_u32(4U) != 124U || read_u32(12U) != kLutSize ||
        read_u32(16U) != kLutSize) {
        throw std::runtime_error("invalid 64x64 LTC DDS LUT: " + path.string());
    }
    std::vector<std::uint16_t> values(kValueCount);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(std::uint16_t)));
    if (!input) {
        throw std::runtime_error("truncated LTC DDS LUT: " + path.string());
    }
    return values;
}

constexpr const char* kSkyVertexShader = R"GLSL(
#version 450 core
out vec2 v_ndc;
void main() {
    vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    vec2 position = positions[gl_VertexID];
    v_ndc = position;
    gl_Position = vec4(position, 0.999999, 1.0);
}
)GLSL";

constexpr const char* kShadowVertexShader = R"GLSL(
#version 450 core
layout(location = 0) in vec3 a_position;
layout(location = 2) in vec2 a_uv;
layout(location = 4) in vec2 a_uv1;
layout(location = 5) in vec4 a_color;
uniform mat4 u_light_view_projection;
out vec3 v_world_position;
out vec2 v_uv;
out vec2 v_uv1;
out float v_vertex_alpha;
void main() {
    v_world_position = a_position;
    v_uv = a_uv;
    v_uv1 = a_uv1;
    v_vertex_alpha = a_color.a;
    gl_Position = u_light_view_projection * vec4(a_position, 1.0);
}
)GLSL";

constexpr const char* kShadowFragmentShader = R"GLSL(
#version 450 core
layout(binding = 0) uniform sampler2D u_base_color_texture;
layout(binding = 1) uniform sampler2D u_opacity_texture;
uniform int u_alpha_mode;
uniform float u_alpha_cutoff;
uniform float u_opacity;
uniform int u_has_base_color_texture;
uniform int u_has_opacity_texture;
uniform vec4 u_texture_offset_scale[2];
uniform float u_texture_rotation[2];
uniform int u_texture_texcoord[2];
uniform int u_texture_top_left[2];
uniform int u_cube_pass;
uniform vec3 u_shadow_origin;
uniform vec3 u_shadow_direction;
uniform float u_shadow_near;
uniform float u_shadow_far;
in vec3 v_world_position;
in vec2 v_uv;
in vec2 v_uv1;
in float v_vertex_alpha;
layout(location = 0) out float out_linear_depth;
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}
vec2 material_uv(int slot) {
    vec2 uv = u_texture_texcoord[slot] == 1 ? v_uv1 : v_uv;
    uv *= u_texture_offset_scale[slot].zw;
    float cosine = cos(u_texture_rotation[slot]);
    float sine = sin(u_texture_rotation[slot]);
    uv = mat2(cosine, sine, -sine, cosine) * uv;
    uv += u_texture_offset_scale[slot].xy;
    if (u_texture_top_left[slot] != 0) {
        uv.y = 1.0 - uv.y;
    }
    return uv;
}
void main() {
    if (u_alpha_mode == 1) {
        float opacity = u_opacity * v_vertex_alpha;
        if (u_has_base_color_texture != 0) {
            opacity *= texture(u_base_color_texture, material_uv(0)).a;
        }
        if (u_has_opacity_texture != 0) {
            opacity *= luminance(texture(u_opacity_texture, material_uv(1)).rgb);
        }
        if (opacity < u_alpha_cutoff) {
            discard;
        }
    }
    float linear_distance = u_cube_pass != 0
        ? length(v_world_position - u_shadow_origin)
        : dot(v_world_position - u_shadow_origin, u_shadow_direction);
    out_linear_depth = clamp(
        (linear_distance - u_shadow_near) /
        max(u_shadow_far - u_shadow_near, 1.0e-6),
        0.0,
        1.0);
}
)GLSL";

constexpr const char* kSkyFragmentShader = R"GLSL(
#version 450 core
layout(binding = 6) uniform samplerCube u_environment_prefilter;
uniform vec3 u_camera_forward;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform float u_viewport_width;
uniform float u_viewport_height;
uniform vec3 u_environment_color;
uniform float u_environment_intensity;
uniform float u_environment_rotation_radians;
in vec2 v_ndc;
layout(location = 0) out vec4 out_linear_color;
vec3 rotate_y(vec3 direction, float radians) {
    float c = cos(radians);
    float s = sin(radians);
    return vec3(c * direction.x + s * direction.z, direction.y, -s * direction.x + c * direction.z);
}
void main() {
    vec3 direction = normalize(
        u_camera_forward +
        v_ndc.x * 0.5 * u_viewport_width * u_camera_right +
        v_ndc.y * 0.5 * u_viewport_height * u_camera_up);
    direction = rotate_y(direction, -u_environment_rotation_radians);
    vec3 radiance = textureLod(u_environment_prefilter, direction, 0.0).rgb;
    out_linear_color = vec4(radiance * u_environment_color * u_environment_intensity, 1.0);
}
)GLSL";

constexpr const char* kCompositeFragmentShader = R"GLSL(
#version 450 core
layout(binding = 0) uniform sampler2D u_opaque;
layout(binding = 1) uniform sampler2D u_accum;
layout(binding = 2) uniform sampler2D u_reveal;
layout(binding = 3) uniform sampler2D u_ao_bent_normal;
layout(binding = 4) uniform sampler2D u_view_normal;
layout(binding = 5) uniform sampler2D u_linear_depth;
uniform int u_ao_debug_view;
uniform float u_scene_radius;
in vec2 v_ndc;
layout(location = 0) out vec4 out_linear_color;
void main() {
    vec2 uv = v_ndc * 0.5 + 0.5;
    if (u_ao_debug_view != 0) {
        float depth = texture(u_linear_depth, uv).r;
        vec3 view_normal = texture(u_view_normal, uv).xyz;
        vec4 ao = texture(u_ao_bent_normal, uv);
        vec3 debug_color = vec3(0.0);
        if (u_ao_debug_view == 1) {
            debug_color = vec3(depth > 0.0 ? ao.a : 1.0);
        } else if (u_ao_debug_view == 2) {
            debug_color = depth > 0.0
                ? normalize(ao.xyz) * 0.5 + 0.5
                : vec3(0.0);
        } else if (u_ao_debug_view == 3) {
            debug_color = depth > 0.0
                ? normalize(view_normal) * 0.5 + 0.5
                : vec3(0.0);
        } else {
            debug_color = vec3(clamp(
                depth / max(u_scene_radius * 4.0, 1.0e-5),
                0.0,
                1.0));
        }
        out_linear_color = vec4(debug_color, 1.0);
        return;
    }
    vec3 opaque = texture(u_opaque, uv).rgb;
    vec4 accum = texture(u_accum, uv);
    float reveal = clamp(texture(u_reveal, uv).r, 0.0, 1.0);
    vec3 transparent = accum.rgb / max(accum.a, 1.0e-5);
    out_linear_color = vec4(transparent * (1.0 - reveal) + opaque * reveal, 1.0);
}
)GLSL";

}  // namespace

class OpenGlRasterRenderer::Impl {
public:
    Impl(
        std::filesystem::path vertex_shader_path,
        std::filesystem::path fragment_shader_path)
        : vertex_shader_path_(std::move(vertex_shader_path)),
          fragment_shader_path_(std::move(fragment_shader_path)) {
        const auto locate_auxiliary_shader = [this](const char* filename) {
            const std::array<std::filesystem::path, 3> candidates{
                fragment_shader_path_.parent_path() / filename,
                vertex_shader_path_.parent_path() / filename,
                std::filesystem::current_path() / "shaders" / "opengl" /
                    filename};
            for (const std::filesystem::path& candidate : candidates) {
                if (std::filesystem::exists(candidate)) {
                    return candidate;
                }
            }
            return candidates.front();
        };
        fullscreen_vertex_path_ = locate_auxiliary_shader("fullscreen.vert");
        ao_gbuffer_fragment_path_ = locate_auxiliary_shader("ao_gbuffer.frag");
        ao_fragment_path_ = locate_auxiliary_shader("ambient_occlusion.frag");
        ao_denoise_fragment_path_ = locate_auxiliary_shader("ao_denoise.frag");
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vertex_buffer_);
        glGenBuffers(1, &directional_light_buffer_);
        glGenBuffers(1, &point_light_buffer_);
        glGenBuffers(1, &spot_light_buffer_);
        glGenBuffers(1, &rect_area_light_buffer_);
        glGenTextures(1, &fallback_texture_);
        glBindTexture(GL_TEXTURE_2D, fallback_texture_);
        const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, white.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        std::string sky_error;
        if (!sky_program_.load_sources(kSkyVertexShader, kSkyFragmentShader, sky_error)) {
            throw std::runtime_error(sky_error);
        }
        if (!composite_program_.load_sources(
                kSkyVertexShader,
                kCompositeFragmentShader,
                sky_error)) {
            throw std::runtime_error(sky_error);
        }
        if (!shadow_program_.load_sources(
                kShadowVertexShader,
                kShadowFragmentShader,
                sky_error)) {
            throw std::runtime_error(sky_error);
        }
        create_brdf_lut();
        create_ltc_luts();
        request_shader_reload();
    }

    ~Impl() {
        shader_program_.reset();
        ao_gbuffer_program_.reset();
        ao_program_.reset();
        ao_denoise_program_.reset();
        release_scene_resources();
        release_output_resources();
        release_shadow_resources();
        if (fallback_texture_ != 0) {
            glDeleteTextures(1, &fallback_texture_);
        }
        if (directional_light_buffer_ != 0) {
            glDeleteBuffers(1, &directional_light_buffer_);
        }
        if (point_light_buffer_ != 0) {
            glDeleteBuffers(1, &point_light_buffer_);
        }
        if (spot_light_buffer_ != 0) {
            glDeleteBuffers(1, &spot_light_buffer_);
        }
        if (rect_area_light_buffer_ != 0) {
            glDeleteBuffers(1, &rect_area_light_buffer_);
        }
        if (environment_cube_ != 0) {
            glDeleteTextures(1, &environment_cube_);
        }
        if (residual_environment_cube_ != 0) {
            glDeleteTextures(1, &residual_environment_cube_);
        }
        if (brdf_lut_ != 0) {
            glDeleteTextures(1, &brdf_lut_);
        }
        if (ltc_matrix_lut_ != 0) {
            glDeleteTextures(1, &ltc_matrix_lut_);
        }
        if (ltc_amplitude_lut_ != 0) {
            glDeleteTextures(1, &ltc_amplitude_lut_);
        }
        if (vertex_buffer_ != 0) {
            glDeleteBuffers(1, &vertex_buffer_);
        }
        if (vao_ != 0) {
            glDeleteVertexArrays(1, &vao_);
        }
    }

    void reset(const Scene& scene) {
        upload_geometry(scene);
        upload_textures(scene);
        upload_environment(scene);
        scene_radius_ = compute_scene_radius(scene);
        shadow_dirty_ = true;
        gpu_lights_dirty_ = true;
    }

    void sync_scene(const Scene& scene, SceneChangeSet changes) {
        if (open_gl_requires_geometry_upload(changes)) {
            upload_geometry(scene);
        }
        if (has_scene_change(changes, SceneChange::Textures)) {
            upload_textures(scene);
        }
        if (has_scene_change(changes, SceneChange::Lighting)) {
            gpu_lights_dirty_ = true;
        }
        if (has_scene_change(changes, SceneChange::Environment)) {
            upload_environment(scene);
            gpu_lights_dirty_ = true;
        }
        if (has_scene_change(changes, SceneChange::Geometry) ||
            has_scene_change(changes, SceneChange::InstanceTransforms)) {
            scene_radius_ = compute_scene_radius(scene);
        }
        if (changes != SceneChange::None) {
            shadow_dirty_ = true;
        }
    }

    void render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state) {
        (void)frame_state;
        reload_shader_if_needed();
        ensure_output(settings.width, settings.height);
        prepare_environment_techniques(scene, settings);
        prepare_shadow_maps(scene, settings);
        if (gpu_lights_dirty_) {
            upload_lights(scene, &settings);
        }

        const Mat4 ao_view_projection = view_projection_matrix(
            camera,
            std::max(1.0e-4f, scene_radius_ * 1.0e-4f));
        const bool ao_active = render_ambient_occlusion(
            scene,
            camera,
            settings,
            ao_view_projection);

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
        glViewport(0, 0, settings.width, settings.height);
        const std::array<GLenum, 3> all_buffers{
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2};
        glDrawBuffers(static_cast<GLsizei>(all_buffers.size()), all_buffers.data());
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(ao_active ? GL_EQUAL : GL_LESS);
        glDepthMask(ao_active ? GL_FALSE : GL_TRUE);
        glDisable(GL_BLEND);
        Color clear_color = Color::Zero();
        if (scene.environment_background_visible && !scene.environment_map) {
            clear_color = scene.environment * std::max(0.0f, scene.environment_intensity);
        }
        const std::array<float, 4> opaque_clear{
            clear_color.x(), clear_color.y(), clear_color.z(), 1.0f};
        const std::array<float, 4> zero{0.0f, 0.0f, 0.0f, 0.0f};
        const std::array<float, 4> one{1.0f, 1.0f, 1.0f, 1.0f};
        glClearBufferfv(GL_COLOR, 0, opaque_clear.data());
        glClearBufferfv(GL_COLOR, 1, zero.data());
        glClearBufferfv(GL_COLOR, 2, one.data());
        if (!ao_active) {
            glClear(GL_DEPTH_BUFFER_BIT);
        }
        if (!shader_program_) {
            glBindFramebuffer(GL_FRAMEBUFFER, composite_framebuffer_);
            glClearColor(0.25f, 0.0f, 0.25f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return;
        }

        if (scene.environment_background_visible && scene.environment_map && environment_cube_ != 0) {
            const GLenum opaque_buffer = GL_COLOR_ATTACHMENT0;
            glDrawBuffers(1, &opaque_buffer);
            glDisable(GL_DEPTH_TEST);
            glUseProgram(sky_program_.id());
            set_uniform(glGetUniformLocation(sky_program_.id(), "u_camera_forward"), camera.forward());
            set_uniform(glGetUniformLocation(sky_program_.id(), "u_camera_right"), camera.right());
            set_uniform(glGetUniformLocation(sky_program_.id(), "u_camera_up"), camera.up());
            set_uniform(glGetUniformLocation(sky_program_.id(), "u_viewport_width"), camera.viewport_width());
            set_uniform(glGetUniformLocation(sky_program_.id(), "u_viewport_height"), camera.viewport_height());
            set_uniform(glGetUniformLocation(sky_program_.id(), "u_environment_color"), scene.environment);
            set_uniform(glGetUniformLocation(sky_program_.id(), "u_environment_intensity"), scene.environment_intensity);
            set_uniform(
                glGetUniformLocation(sky_program_.id(), "u_environment_rotation_radians"),
                scene.environment_rotation_degrees * 0.01745329251994329577f);
            glActiveTexture(GL_TEXTURE6);
            glBindTexture(GL_TEXTURE_CUBE_MAP, environment_cube_);
            glBindVertexArray(vao_);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glEnable(GL_DEPTH_TEST);
        }

        glUseProgram(shader_program_.id());
        const GLenum opaque_buffer = GL_COLOR_ATTACHMENT0;
        glDrawBuffers(1, &opaque_buffer);
        const Mat4 view_projection = view_projection_matrix(
            camera,
            std::max(1.0e-4f, scene_radius_ * 1.0e-4f));
        if (uniforms_.view_projection >= 0) {
            glUniformMatrix4fv(
                uniforms_.view_projection,
                1,
                GL_FALSE,
                view_projection.data());
        }
        set_uniform(uniforms_.camera_position, camera.eye());
        set_uniform(uniforms_.camera_forward, camera.forward());
        set_uniform(uniforms_.camera_right, camera.right());
        set_uniform(uniforms_.camera_up, camera.up());
        if (uniforms_.viewport_size >= 0) {
            glUniform2f(
                uniforms_.viewport_size,
                static_cast<float>(settings.width),
                static_cast<float>(settings.height));
        }
        const int active_ao_mode = ao_active
            ? static_cast<int>(settings.opengl.ambient_occlusion.mode)
            : static_cast<int>(OpenGlAmbientOcclusionMode::Off);
        set_uniform(uniforms_.ao_mode, active_ao_mode);
        set_uniform(
            uniforms_.ao_bent_normals_enabled,
            ao_active &&
                    settings.opengl.ambient_occlusion.mode ==
                        OpenGlAmbientOcclusionMode::Gtao &&
                    settings.opengl.ambient_occlusion.gtao.bent_normals_enabled
                ? 1
                : 0);
        if (uniforms_.view_to_world >= 0) {
            Mat3 view_to_world;
            view_to_world.col(0) = camera.right();
            view_to_world.col(1) = camera.up();
            view_to_world.col(2) = -camera.forward();
            glUniformMatrix3fv(
                uniforms_.view_to_world,
                1,
                GL_FALSE,
                view_to_world.data());
        }
        set_uniform(uniforms_.environment_color, scene.environment);
        set_uniform(uniforms_.environment_intensity, scene.environment_intensity);
        set_uniform(
            uniforms_.environment_rotation_radians,
            scene.environment_rotation_degrees * 0.01745329251994329577f);
        set_uniform(uniforms_.environment_mip_count, static_cast<float>(environment_mip_count_));
        set_uniform(uniforms_.has_environment_map, scene.environment_map ? 1 : 0);
        set_uniform(uniforms_.ibl_enabled, settings.opengl.ibl_enabled ? 1 : 0);
        set_uniform(
            uniforms_.ltc_area_lights_enabled,
            settings.opengl.ltc_area_lights_enabled ? 1 : 0);
        if (uniforms_.environment_sh >= 0) {
            std::array<float, 27> sh{};
            if (active_ibl_environment_) {
                const auto& coefficients = active_ibl_environment_->radiance_sh();
                for (std::size_t index = 0; index < coefficients.size(); ++index) {
                    sh[index * 3U] = coefficients[index].x();
                    sh[index * 3U + 1U] = coefficients[index].y();
                    sh[index * 3U + 2U] = coefficients[index].z();
                }
            }
            glUniform3fv(uniforms_.environment_sh, 9, sh.data());
        }
        set_uniform(
            uniforms_.directional_light_count,
            gpu_directional_light_count_);
        set_uniform(
            uniforms_.point_light_count,
            static_cast<int>(scene.point_lights.size()));
        set_uniform(
            uniforms_.spot_light_count,
            static_cast<int>(scene.spot_lights.size()));
        set_uniform(
            uniforms_.rect_area_light_count,
            static_cast<int>(scene.rect_area_lights.size()));

        std::array<Mat4, 32> shadow_matrices;
        std::array<float, 32 * 4> shadow_origin_far{};
        std::array<float, 32 * 4> shadow_direction_near{};
        std::array<float, 32 * 4> cube_position_far{};
        std::array<float, 32> cube_near{};
        for (Mat4& matrix : shadow_matrices) {
            matrix = Mat4::Identity();
        }
        for (const ShadowSlot& slot : shadow_slots_) {
            if (slot.layer < 0 || slot.layer >= 32) {
                continue;
            }
            const std::size_t layer = static_cast<std::size_t>(slot.layer);
            if (slot.cube) {
                cube_position_far[layer * 4U] = slot.position.x();
                cube_position_far[layer * 4U + 1U] = slot.position.y();
                cube_position_far[layer * 4U + 2U] = slot.position.z();
                cube_position_far[layer * 4U + 3U] = slot.far_plane;
                cube_near[layer] = slot.near_plane;
            } else {
                shadow_matrices[layer] = slot.matrices[0];
                shadow_origin_far[layer * 4U] = slot.position.x();
                shadow_origin_far[layer * 4U + 1U] = slot.position.y();
                shadow_origin_far[layer * 4U + 2U] = slot.position.z();
                shadow_origin_far[layer * 4U + 3U] = slot.far_plane;
                shadow_direction_near[layer * 4U] = slot.direction.x();
                shadow_direction_near[layer * 4U + 1U] = slot.direction.y();
                shadow_direction_near[layer * 4U + 2U] = slot.direction.z();
                shadow_direction_near[layer * 4U + 3U] = slot.near_plane;
            }
        }
        if (uniforms_.shadow_matrices >= 0) {
            glUniformMatrix4fv(
                uniforms_.shadow_matrices,
                32,
                GL_FALSE,
                shadow_matrices[0].data());
        }
        if (uniforms_.shadow_origin_far >= 0) {
            glUniform4fv(uniforms_.shadow_origin_far, 32, shadow_origin_far.data());
        }
        if (uniforms_.shadow_direction_near >= 0) {
            glUniform4fv(
                uniforms_.shadow_direction_near,
                32,
                shadow_direction_near.data());
        }
        if (uniforms_.cube_shadow_position_far >= 0) {
            glUniform4fv(
                uniforms_.cube_shadow_position_far,
                32,
                cube_position_far.data());
        }
        if (uniforms_.cube_shadow_near >= 0) {
            glUniform1fv(uniforms_.cube_shadow_near, 32, cube_near.data());
        }
        const auto& shadow_settings = settings.opengl.shadow_map;
        const auto& pcss_settings = settings.opengl.pcss;
        set_uniform(uniforms_.shadow_map_resolution,
            static_cast<float>(std::max(shadow_resolution_, 1)));
        set_uniform(uniforms_.shadow_constant_bias,
            std::max(0.0f, shadow_settings.constant_bias));
        set_uniform(uniforms_.shadow_slope_bias,
            std::max(0.0f, shadow_settings.slope_bias));
        set_uniform(uniforms_.pcss_enabled, pcss_settings.enabled ? 1 : 0);
        set_uniform(uniforms_.pcss_blocker_samples,
            std::clamp(pcss_settings.blocker_samples, 1, 64));
        set_uniform(uniforms_.pcss_filter_samples,
            std::clamp(pcss_settings.filter_samples, 1, 64));
        set_uniform(uniforms_.pcss_max_penumbra_texels,
            std::clamp(pcss_settings.max_penumbra_texels, 0.0f, 256.0f));
        set_uniform(uniforms_.pcss_light_size_scale,
            std::max(0.0f, pcss_settings.light_size_scale));
        set_uniform(uniforms_.shadow_debug_view,
            static_cast<int>(shadow_settings.debug_view));
        set_uniform(uniforms_.shadow_debug_slot,
            std::max(0, shadow_settings.debug_shadow_slot));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, directional_light_buffer_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, point_light_buffer_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, spot_light_buffer_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, rect_area_light_buffer_);
        glActiveTexture(GL_TEXTURE9);
        glBindTexture(
            GL_TEXTURE_CUBE_MAP,
            residual_environment_cube_ != 0
                ? residual_environment_cube_
                : environment_cube_);
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D, brdf_lut_);
        glActiveTexture(GL_TEXTURE11);
        glBindTexture(GL_TEXTURE_2D, ltc_matrix_lut_);
        glActiveTexture(GL_TEXTURE12);
        glBindTexture(GL_TEXTURE_2D, ltc_amplitude_lut_);
        glActiveTexture(GL_TEXTURE13);
        glBindTexture(GL_TEXTURE_2D_ARRAY, shadow_2d_array_);
        glActiveTexture(GL_TEXTURE14);
        glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, shadow_cube_array_);
        glActiveTexture(GL_TEXTURE15);
        glBindTexture(
            GL_TEXTURE_2D,
            ao_active ? ao_resolved_texture_ : fallback_texture_);
        glBindVertexArray(vao_);
        set_uniform(uniforms_.transparent_pass, 0);
        const auto draw_batches = [&](bool blended) {
        for (const DrawBatch& batch : batches_) {
            const Material fallback;
            const Material& material = valid_material_id(scene, batch.material_id)
                ? scene.materials[static_cast<std::size_t>(batch.material_id)]
                : fallback;
            const bool material_blended = material.alpha_mode == AlphaMode::Blend;
            if (material_blended != blended) {
                continue;
            }
            bind_material(
                scene,
                material,
                valid_material_id(scene, batch.material_id),
                uniforms_);
            if (material.two_sided || !valid_material_id(scene, batch.material_id)) {
                glDisable(GL_CULL_FACE);
            } else {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
                glFrontFace(GL_CCW);
            }
            glDrawArrays(GL_TRIANGLES, batch.first, batch.count);
        }
        };
        draw_batches(false);
        const std::array<GLenum, 3> transparency_buffers{
            GL_NONE,
            GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2};
        glDrawBuffers(
            static_cast<GLsizei>(transparency_buffers.size()),
            transparency_buffers.data());
        glEnablei(GL_BLEND, 1);
        glEnablei(GL_BLEND, 2);
        glBlendEquationi(1, GL_FUNC_ADD);
        glBlendEquationi(2, GL_FUNC_ADD);
        glBlendFunci(1, GL_ONE, GL_ONE);
        glBlendFunci(2, GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
        set_uniform(uniforms_.transparent_pass, 1);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_FALSE);
        draw_batches(true);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
        glDisablei(GL_BLEND, 1);
        glDisablei(GL_BLEND, 2);
        glDisable(GL_CULL_FACE);

        glBindFramebuffer(GL_FRAMEBUFFER, composite_framebuffer_);
        glViewport(0, 0, settings.width, settings.height);
        glDisable(GL_DEPTH_TEST);
        glUseProgram(composite_program_.id());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, opaque_texture_);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, transparency_accum_texture_);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, transparency_reveal_texture_);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(
            GL_TEXTURE_2D,
            ao_active ? ao_resolved_texture_ : fallback_texture_);
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, ao_view_normal_texture_);
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, ao_linear_depth_texture_);
        set_uniform(
            glGetUniformLocation(composite_program_.id(), "u_ao_debug_view"),
            ao_active
                ? static_cast<int>(
                      settings.opengl.ambient_occlusion.debug_view)
                : 0);
        set_uniform(
            glGetUniformLocation(composite_program_.id(), "u_scene_radius"),
            scene_radius_);
        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void set_auto_reload(bool enabled) {
        auto_reload_ = enabled;
    }

    void request_shader_reload() {
        reload_requested_ = true;
        next_shader_check_ = std::chrono::steady_clock::time_point::min();
    }

    void reload_shader_if_needed() {
        const auto now = std::chrono::steady_clock::now();
        if (!reload_requested_ && (!auto_reload_ || now < next_shader_check_)) {
            return;
        }
        next_shader_check_ = now + std::chrono::milliseconds(250);

        const auto vertex_write_time = file_write_time(vertex_shader_path_);
        const auto fragment_write_time = file_write_time(fragment_shader_path_);
        const std::array<std::filesystem::file_time_type, 4> ao_write_times{
            file_write_time(fullscreen_vertex_path_),
            file_write_time(ao_gbuffer_fragment_path_),
            file_write_time(ao_fragment_path_),
            file_write_time(ao_denoise_fragment_path_)};
        const bool main_changed =
            vertex_write_time != vertex_write_time_ ||
            fragment_write_time != fragment_write_time_;
        const bool ao_changed = main_changed || ao_write_times != ao_write_times_;
        if (!reload_requested_ &&
            !main_changed && !ao_changed) {
            return;
        }
        const bool reload_all = reload_requested_;
        reload_requested_ = false;
        vertex_write_time_ = vertex_write_time;
        fragment_write_time_ = fragment_write_time;
        ao_write_times_ = ao_write_times;

        std::string errors;
        if (reload_all || main_changed) {
            GlShaderProgram replacement;
            std::string error;
            if (replacement.load(
                    vertex_shader_path_, fragment_shader_path_, error)) {
                shader_program_ = std::move(replacement);
                uniforms_ = find_uniforms(shader_program_.id());
            } else {
                errors = "Raster shader: " + error;
            }
        }
        if (reload_all || ao_changed) {
            GlShaderProgram gbuffer_replacement;
            GlShaderProgram ao_replacement;
            GlShaderProgram denoise_replacement;
            std::string error;
            bool valid = gbuffer_replacement.load(
                vertex_shader_path_, ao_gbuffer_fragment_path_, error);
            if (!valid) {
                error = "AO G-buffer shader: " + error;
            } else if (!ao_replacement.load(
                           fullscreen_vertex_path_, ao_fragment_path_, error)) {
                valid = false;
                error = "AO evaluation shader: " + error;
            } else if (!denoise_replacement.load(
                           fullscreen_vertex_path_,
                           ao_denoise_fragment_path_,
                           error)) {
                valid = false;
                error = "AO denoise shader: " + error;
            }
            if (valid) {
                ao_gbuffer_program_ = std::move(gbuffer_replacement);
                ao_program_ = std::move(ao_replacement);
                ao_denoise_program_ = std::move(denoise_replacement);
                ao_gbuffer_uniforms_ = find_uniforms(ao_gbuffer_program_.id());
                ao_shaders_available_ = true;
            } else {
                ao_shaders_available_ = static_cast<bool>(ao_gbuffer_program_) &&
                    static_cast<bool>(ao_program_) &&
                    static_cast<bool>(ao_denoise_program_);
                if (!errors.empty()) {
                    errors += '\n';
                }
                errors += error;
            }
        }
        shader_error_ = std::move(errors);
    }

    unsigned int output_texture() const {
        return color_texture_;
    }

    int output_width() const {
        return output_width_;
    }

    int output_height() const {
        return output_height_;
    }

    bool auto_reload() const {
        return auto_reload_;
    }

    bool has_valid_shader() const {
        return static_cast<bool>(shader_program_);
    }

    const std::string& shader_error() const {
        return shader_error_;
    }

    const std::filesystem::path& vertex_shader_path() const {
        return vertex_shader_path_;
    }

    const std::filesystem::path& fragment_shader_path() const {
        return fragment_shader_path_;
    }

    OpenGlTechniqueDiagnostics technique_diagnostics() const {
        OpenGlTechniqueDiagnostics result;
        result.active_shadow_slots = static_cast<int>(shadow_slots_.size());
        result.budget_excluded_lights = shadow_budget_excluded_count_;
        result.hardware_excluded_lights = shadow_hardware_excluded_count_;
        result.dominant_light_valid = dominant_environment_.valid;
        result.dominant_light_direction = dominant_environment_.direction;
        result.dominant_light_integrated_radiance =
            dominant_environment_.integrated_radiance;
        result.dominant_light_energy_fraction =
            dominant_environment_.energy_fraction;
        result.dominant_light_angular_radius_radians =
            dominant_environment_.angular_radius_radians;
        return result;
    }

private:
    std::filesystem::path vertex_shader_path_;
    std::filesystem::path fragment_shader_path_;
    std::filesystem::path fullscreen_vertex_path_;
    std::filesystem::path ao_gbuffer_fragment_path_;
    std::filesystem::path ao_fragment_path_;
    std::filesystem::path ao_denoise_fragment_path_;
    std::filesystem::file_time_type vertex_write_time_ =
        std::filesystem::file_time_type::min();
    std::filesystem::file_time_type fragment_write_time_ =
        std::filesystem::file_time_type::min();
    std::array<std::filesystem::file_time_type, 4> ao_write_times_{};
    std::chrono::steady_clock::time_point next_shader_check_ =
        std::chrono::steady_clock::time_point::min();
    bool auto_reload_ = true;
    bool reload_requested_ = true;
    std::string shader_error_;
    GlShaderProgram shader_program_;
    UniformLocations uniforms_;
    GlShaderProgram ao_gbuffer_program_;
    GlShaderProgram ao_program_;
    GlShaderProgram ao_denoise_program_;
    UniformLocations ao_gbuffer_uniforms_;
    bool ao_shaders_available_ = false;

    GLuint vao_ = 0;
    GLuint vertex_buffer_ = 0;
    GLuint directional_light_buffer_ = 0;
    GLuint point_light_buffer_ = 0;
    GLuint spot_light_buffer_ = 0;
    GLuint rect_area_light_buffer_ = 0;
    GLuint fallback_texture_ = 0;
    GLuint environment_cube_ = 0;
    GLuint residual_environment_cube_ = 0;
    GLuint brdf_lut_ = 0;
    GLuint ltc_matrix_lut_ = 0;
    GLuint ltc_amplitude_lut_ = 0;
    int environment_mip_count_ = 1;
    int source_environment_mip_count_ = 1;
    const EnvironmentMap* uploaded_environment_ = nullptr;
    std::shared_ptr<const EnvironmentMap> active_ibl_environment_;
    DominantEnvironmentLight dominant_environment_;
    bool dominant_environment_dirty_ = true;
    bool dominant_environment_enabled_ = false;
    float dominant_environment_threshold_ev_ = 3.0f;
    float dominant_environment_minimum_energy_ = 0.01f;
    int gpu_directional_light_count_ = 0;
    bool gpu_lights_dirty_ = true;
    GlShaderProgram sky_program_;
    GlShaderProgram composite_program_;
    GlShaderProgram shadow_program_;
    std::vector<GLuint> scene_textures_;
    std::vector<DrawBatch> batches_;
    float scene_radius_ = 1.0f;

    GLuint shadow_framebuffer_ = 0;
    GLuint shadow_depth_renderbuffer_ = 0;
    GLuint shadow_2d_array_ = 0;
    GLuint shadow_cube_array_ = 0;
    int shadow_resolution_ = 0;
    int shadow_2d_layers_ = 0;
    int shadow_cube_layers_ = 0;
    bool shadow_dirty_ = true;
    ShadowMapRenderSettings cached_shadow_settings_{};
    bool cached_shadow_enabled_ = false;
    bool cached_ltc_area_lights_enabled_ = true;
    float cached_dominant_intensity_scale_ = 1.0f;
    std::vector<ShadowSlot> shadow_slots_;
    std::vector<ShadowReference> directional_shadow_references_;
    std::vector<ShadowReference> point_shadow_references_;
    std::vector<ShadowReference> spot_shadow_references_;
    std::vector<ShadowReference> rect_shadow_references_;
    ShadowReference dominant_shadow_reference_;
    int shadow_budget_excluded_count_ = 0;
    int shadow_hardware_excluded_count_ = 0;

    GLuint framebuffer_ = 0;
    GLuint color_texture_ = 0;
    GLuint opaque_texture_ = 0;
    GLuint transparency_accum_texture_ = 0;
    GLuint transparency_reveal_texture_ = 0;
    GLuint composite_framebuffer_ = 0;
    GLuint depth_texture_ = 0;
    GLuint ao_gbuffer_framebuffer_ = 0;
    GLuint ao_framebuffer_ = 0;
    GLuint ao_view_normal_texture_ = 0;
    GLuint ao_linear_depth_texture_ = 0;
    GLuint ao_raw_texture_ = 0;
    GLuint ao_temporary_texture_ = 0;
    GLuint ao_filtered_texture_ = 0;
    GLuint ao_resolved_texture_ = 0;
    int output_width_ = 0;
    int output_height_ = 0;

    void create_brdf_lut() {
        constexpr int size = 256;
        const std::vector<float> pixels = integrate_brdf_lut(size);
        glGenTextures(1, &brdf_lut_);
        glBindTexture(GL_TEXTURE_2D, brdf_lut_);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RG16F,
            size,
            size,
            0,
            GL_RG,
            GL_FLOAT,
            pixels.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }

    void create_ltc_luts() {
        constexpr int size = 64;
        const auto locate_lut = [this](const char* filename) {
            const std::array<std::filesystem::path, 4> candidates{
                fragment_shader_path_.parent_path() / filename,
                vertex_shader_path_.parent_path() / filename,
                std::filesystem::current_path() / "shaders" / "opengl" / filename,
                std::filesystem::current_path() / filename};
            for (const std::filesystem::path& candidate : candidates) {
                if (std::filesystem::exists(candidate)) {
                    return candidate;
                }
            }
            throw std::runtime_error(
                "failed to locate built-in LTC LUT: " +
                std::string(filename));
        };
        const auto create_lut = [](
            const std::filesystem::path& path,
            GLuint& texture) {
            const std::vector<std::uint16_t> pixels = load_ltc_dds(path);
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA16F,
                size,
                size,
                0,
                GL_RGBA,
                GL_HALF_FLOAT,
                pixels.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        };
        create_lut(locate_lut("ltc_1.dds"), ltc_matrix_lut_);
        create_lut(locate_lut("ltc_2.dds"), ltc_amplitude_lut_);
    }

    void upload_environment_cube(
        const EnvironmentMap* environment,
        GLuint& texture,
        int& mip_count) {
        if (texture != 0) {
            glDeleteTextures(1, &texture);
            texture = 0;
        }
        mip_count = 1;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_CUBE_MAP, texture);
        if (!environment) {
            const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
            for (int face = 0; face < 6; ++face) {
                glTexImage2D(
                    GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                    0,
                    GL_RGBA16F,
                    1,
                    1,
                    0,
                    GL_RGBA,
                    GL_FLOAT,
                    white.data());
            }
        } else {
            constexpr int base_size = 256;
            mip_count = 1 + static_cast<int>(std::floor(std::log2(base_size)));
            for (int mip = 0; mip < mip_count; ++mip) {
                const int size = std::max(1, base_size >> mip);
                const float roughness = mip_count > 1
                    ? static_cast<float>(mip) / static_cast<float>(mip_count - 1)
                    : 0.0f;
                for (int face = 0; face < 6; ++face) {
                    const std::vector<float> pixels = prefilter_environment_face(
                        *environment,
                        face,
                        size,
                        roughness);
                    glTexImage2D(
                        GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                        mip,
                        GL_RGBA16F,
                        size,
                        size,
                        0,
                        GL_RGBA,
                        GL_FLOAT,
                        pixels.data());
                }
            }
        }
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    }

    void upload_environment(const Scene& scene) {
        const EnvironmentMap* environment = scene.environment_map.get();
        if (environment == uploaded_environment_) {
            return;
        }
        uploaded_environment_ = environment;
        upload_environment_cube(
            environment,
            environment_cube_,
            source_environment_mip_count_);
        if (residual_environment_cube_ != 0) {
            glDeleteTextures(1, &residual_environment_cube_);
            residual_environment_cube_ = 0;
        }
        environment_mip_count_ = source_environment_mip_count_;
        active_ibl_environment_ = scene.environment_map;
        dominant_environment_ = DominantEnvironmentLight{};
        dominant_environment_dirty_ = true;
    }

    void prepare_environment_techniques(
        const Scene& scene,
        const RenderSettings& settings) {
        const auto& configured = settings.opengl.dominant_light;
        const float threshold_ev = std::clamp(configured.peak_threshold_ev, 0.0f, 20.0f);
        const float minimum_energy = std::clamp(
            configured.minimum_energy_fraction,
            0.0f,
            1.0f);
        if (!dominant_environment_dirty_ &&
            dominant_environment_enabled_ == configured.enabled &&
            dominant_environment_threshold_ev_ == threshold_ev &&
            dominant_environment_minimum_energy_ == minimum_energy) {
            return;
        }
        dominant_environment_dirty_ = false;
        dominant_environment_enabled_ = configured.enabled;
        dominant_environment_threshold_ev_ = threshold_ev;
        dominant_environment_minimum_energy_ = minimum_energy;
        shadow_dirty_ = true;
        gpu_lights_dirty_ = true;
        dominant_environment_ = configured.enabled
            ? extract_dominant_environment_light(
                  scene.environment_map,
                  threshold_ev,
                  minimum_energy)
            : DominantEnvironmentLight{};
        active_ibl_environment_ = dominant_environment_.valid
            ? dominant_environment_.residual_map
            : scene.environment_map;
        if (residual_environment_cube_ != 0) {
            glDeleteTextures(1, &residual_environment_cube_);
            residual_environment_cube_ = 0;
        }
        environment_mip_count_ = source_environment_mip_count_;
        if (dominant_environment_.valid && active_ibl_environment_) {
            upload_environment_cube(
                active_ibl_environment_.get(),
                residual_environment_cube_,
                environment_mip_count_);
        }
    }

    float far_distance_for_position(
        const Vec3& position,
        const Bounds3& bounds,
        float range) const {
        if (range > 0.0f) {
            return range;
        }
        float far_distance = 0.0f;
        for (const Vec3& corner : bounds_corners(bounds)) {
            far_distance = std::max(far_distance, (corner - position).norm());
        }
        return std::max(far_distance, scene_radius_ * 2.0f);
    }

    void configure_directional_shadow(
        ShadowSlot& slot,
        const Bounds3& bounds,
        float padding_fraction,
        int resolution) const {
        const OpenGlDirectionalShadowFit fit =
            open_gl_fit_directional_shadow(
                bounds,
                slot.direction,
                padding_fraction,
                resolution);
        slot.position = fit.position;
        slot.direction = fit.direction;
        slot.near_plane = fit.near_plane;
        slot.far_plane = fit.far_plane;
        slot.matrices[0] = orthographic_matrix(
            -fit.half_width,
            fit.half_width,
            -fit.half_height,
            fit.half_height,
            slot.near_plane,
            slot.far_plane) *
            look_forward_matrix(slot.position, fit.direction, fit.up);
    }

    void configure_cube_shadow(
        ShadowSlot& slot,
        const Bounds3& bounds,
        float range) const {
        slot.near_plane = std::max(scene_radius_ * 1.0e-4f, 1.0e-4f);
        slot.far_plane = std::max(
            far_distance_for_position(slot.position, bounds, range),
            slot.near_plane * 2.0f);
        const Mat4 projection = perspective_matrix(
            1.57079632679f,
            slot.near_plane,
            slot.far_plane);
        const std::array<Vec3, 6> directions{
            Vec3(1.0f, 0.0f, 0.0f),
            Vec3(-1.0f, 0.0f, 0.0f),
            Vec3(0.0f, 1.0f, 0.0f),
            Vec3(0.0f, -1.0f, 0.0f),
            Vec3(0.0f, 0.0f, 1.0f),
            Vec3(0.0f, 0.0f, -1.0f)};
        const std::array<Vec3, 6> up{
            Vec3(0.0f, -1.0f, 0.0f),
            Vec3(0.0f, -1.0f, 0.0f),
            Vec3(0.0f, 0.0f, 1.0f),
            Vec3(0.0f, 0.0f, -1.0f),
            Vec3(0.0f, -1.0f, 0.0f),
            Vec3(0.0f, -1.0f, 0.0f)};
        for (std::size_t face = 0; face < directions.size(); ++face) {
            slot.matrices[face] = projection * look_forward_matrix(
                slot.position,
                directions[face],
                up[face]);
        }
    }

    void configure_spot_shadow(
        ShadowSlot& slot,
        const SpotLight& light,
        const Bounds3& bounds) const {
        slot.position = light.position;
        slot.direction = usable_direction(light.direction)
            ? light.direction.normalized()
            : Vec3(0.0f, 0.0f, -1.0f);
        slot.near_plane = std::max(scene_radius_ * 1.0e-4f, 1.0e-4f);
        slot.far_plane = std::max(
            far_distance_for_position(light.position, bounds, light.range),
            slot.near_plane * 2.0f);
        const float field_of_view = std::clamp(
            2.0f * light.outer_cone_radians,
            0.01f,
            3.13159265f);
        slot.matrices[0] = perspective_matrix(
            field_of_view,
            slot.near_plane,
            slot.far_plane) *
            look_forward_matrix(
                slot.position,
                slot.direction,
                Vec3(0.0f, 1.0f, 0.0f));
    }

    void ensure_shadow_resources(int resolution, int layers_2d, int cube_layers) {
        if (shadow_framebuffer_ != 0 && resolution == shadow_resolution_ &&
            layers_2d == shadow_2d_layers_ && cube_layers == shadow_cube_layers_) {
            return;
        }
        release_shadow_resources();
        shadow_resolution_ = resolution;
        shadow_2d_layers_ = layers_2d;
        shadow_cube_layers_ = cube_layers;
        if (layers_2d == 0 && cube_layers == 0) {
            return;
        }
        glGenFramebuffers(1, &shadow_framebuffer_);
        glGenRenderbuffers(1, &shadow_depth_renderbuffer_);
        glBindRenderbuffer(GL_RENDERBUFFER, shadow_depth_renderbuffer_);
        glRenderbufferStorage(
            GL_RENDERBUFFER,
            GL_DEPTH_COMPONENT24,
            resolution,
            resolution);
        glBindFramebuffer(GL_FRAMEBUFFER, shadow_framebuffer_);
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER,
            GL_DEPTH_ATTACHMENT,
            GL_RENDERBUFFER,
            shadow_depth_renderbuffer_);
        const std::array<float, 4> border{1.0f, 1.0f, 1.0f, 1.0f};
        if (layers_2d > 0) {
            glGenTextures(1, &shadow_2d_array_);
            glBindTexture(GL_TEXTURE_2D_ARRAY, shadow_2d_array_);
            glTexImage3D(
                GL_TEXTURE_2D_ARRAY,
                0,
                GL_R32F,
                resolution,
                resolution,
                layers_2d,
                0,
                GL_RED,
                GL_FLOAT,
                nullptr);
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
            glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
            glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, border.data());
        }
        if (cube_layers > 0) {
            glGenTextures(1, &shadow_cube_array_);
            glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, shadow_cube_array_);
            glTexImage3D(
                GL_TEXTURE_CUBE_MAP_ARRAY,
                0,
                GL_R32F,
                resolution,
                resolution,
                cube_layers * 6,
                0,
                GL_RED,
                GL_FLOAT,
                nullptr);
            glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void bind_shadow_material(const Scene& scene, const Material& material) {
        const AlphaMode alpha_mode = material.type == MaterialType::Pbr
            ? material.alpha_mode
            : (material.opacity_texture_id >= 0 || material.opacity < 1.0f
                  ? AlphaMode::Mask
                  : AlphaMode::Opaque);
        set_uniform(glGetUniformLocation(shadow_program_.id(), "u_alpha_mode"),
            static_cast<int>(alpha_mode));
        set_uniform(glGetUniformLocation(shadow_program_.id(), "u_alpha_cutoff"),
            material.alpha_cutoff);
        set_uniform(glGetUniformLocation(shadow_program_.id(), "u_opacity"), material.opacity);
        const int base_texture = material.base_color_texture_id >= 0
            ? material.base_color_texture_id
            : material.diffuse_texture_id;
        const std::array<int, 2> texture_ids{base_texture, material.opacity_texture_id};
        const std::array<TextureTransform, 2> transforms{
            material.base_color_texture_transform,
            TextureTransform{}};
        for (int slot = 0; slot < 2; ++slot) {
            const bool valid = valid_texture_id(scene, texture_ids[slot]);
            const std::string suffix = "[" + std::to_string(slot) + "]";
            const TextureTransform& transform = transforms[static_cast<std::size_t>(slot)];
            const GLint offset = glGetUniformLocation(
                shadow_program_.id(),
                ("u_texture_offset_scale" + suffix).c_str());
            if (offset >= 0) {
                glUniform4f(
                    offset,
                    transform.offset.x(),
                    transform.offset.y(),
                    transform.scale.x(),
                    transform.scale.y());
            }
            set_uniform(glGetUniformLocation(
                    shadow_program_.id(),
                    ("u_texture_rotation" + suffix).c_str()),
                transform.rotation);
            set_uniform(glGetUniformLocation(
                    shadow_program_.id(),
                    ("u_texture_texcoord" + suffix).c_str()),
                transform.texcoord);
            set_uniform(glGetUniformLocation(
                    shadow_program_.id(),
                    ("u_texture_top_left" + suffix).c_str()),
                valid && scene.textures[static_cast<std::size_t>(texture_ids[slot])].uv_origin() ==
                        TextureUvOrigin::TopLeft
                    ? 1
                    : 0);
            set_uniform(glGetUniformLocation(
                    shadow_program_.id(),
                    slot == 0 ? "u_has_base_color_texture" : "u_has_opacity_texture"),
                valid ? 1 : 0);
            glActiveTexture(GL_TEXTURE0 + slot);
            glBindTexture(
                GL_TEXTURE_2D,
                valid ? scene_textures_[static_cast<std::size_t>(texture_ids[slot])]
                      : fallback_texture_);
        }
    }

    void render_shadow_maps(const Scene& scene) {
        if (shadow_slots_.empty() || shadow_framebuffer_ == 0) {
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, shadow_framebuffer_);
        glViewport(0, 0, shadow_resolution_, shadow_resolution_);
        const GLenum color_attachment = GL_COLOR_ATTACHMENT0;
        glDrawBuffers(1, &color_attachment);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);
        glDisable(GL_BLEND);
        glDisable(GL_CULL_FACE);
        glUseProgram(shadow_program_.id());
        glBindVertexArray(vao_);
        const GLint matrix_location = glGetUniformLocation(
            shadow_program_.id(),
            "u_light_view_projection");
        for (const ShadowSlot& slot : shadow_slots_) {
            const int pass_count = slot.cube ? 6 : 1;
            for (int pass = 0; pass < pass_count; ++pass) {
                glFramebufferTextureLayer(
                    GL_FRAMEBUFFER,
                    GL_COLOR_ATTACHMENT0,
                    slot.cube ? shadow_cube_array_ : shadow_2d_array_,
                    0,
                    slot.cube ? slot.layer * 6 + pass : slot.layer);
                const std::array<float, 4> clear_depth{1.0f, 1.0f, 1.0f, 1.0f};
                glClearBufferfv(GL_COLOR, 0, clear_depth.data());
                glClear(GL_DEPTH_BUFFER_BIT);
                if (matrix_location >= 0) {
                    glUniformMatrix4fv(
                        matrix_location,
                        1,
                        GL_FALSE,
                        slot.matrices[static_cast<std::size_t>(pass)].data());
                }
                set_uniform(glGetUniformLocation(shadow_program_.id(), "u_cube_pass"),
                    slot.cube ? 1 : 0);
                set_uniform(glGetUniformLocation(shadow_program_.id(), "u_shadow_origin"),
                    slot.position);
                set_uniform(glGetUniformLocation(shadow_program_.id(), "u_shadow_direction"),
                    slot.direction);
                set_uniform(glGetUniformLocation(shadow_program_.id(), "u_shadow_near"),
                    slot.near_plane);
                set_uniform(glGetUniformLocation(shadow_program_.id(), "u_shadow_far"),
                    slot.far_plane);
                for (const DrawBatch& batch : batches_) {
                    const Material fallback;
                    const bool valid = valid_material_id(scene, batch.material_id);
                    const Material& material = valid
                        ? scene.materials[static_cast<std::size_t>(batch.material_id)]
                        : fallback;
                    if (material.alpha_mode == AlphaMode::Blend) {
                        continue;
                    }
                    bind_shadow_material(scene, material);
                    glDrawArrays(GL_TRIANGLES, batch.first, batch.count);
                }
            }
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    void prepare_shadow_maps(
        const Scene& scene,
        const RenderSettings& settings) {
        const ShadowMapRenderSettings configured = settings.opengl.shadow_map;
        if (!(configured == cached_shadow_settings_) ||
            cached_shadow_enabled_ != configured.enabled ||
            cached_ltc_area_lights_enabled_ !=
                settings.opengl.ltc_area_lights_enabled ||
            cached_dominant_intensity_scale_ !=
                settings.opengl.dominant_light.intensity_scale) {
            shadow_dirty_ = true;
        }
        if (!shadow_dirty_) {
            return;
        }
        shadow_dirty_ = false;
        cached_shadow_settings_ = configured;
        cached_shadow_enabled_ = configured.enabled;
        cached_ltc_area_lights_enabled_ =
            settings.opengl.ltc_area_lights_enabled;
        cached_dominant_intensity_scale_ =
            settings.opengl.dominant_light.intensity_scale;
        shadow_slots_.clear();
        directional_shadow_references_.assign(
            scene.directional_lights.size(), ShadowReference{});
        point_shadow_references_.assign(scene.point_lights.size(), ShadowReference{});
        spot_shadow_references_.assign(scene.spot_lights.size(), ShadowReference{});
        rect_shadow_references_.assign(scene.rect_area_lights.size(), ShadowReference{});
        dominant_shadow_reference_ = ShadowReference{};
        shadow_budget_excluded_count_ = 0;
        shadow_hardware_excluded_count_ = 0;
        gpu_lights_dirty_ = true;
        if (!configured.enabled) {
            release_shadow_resources();
            return;
        }

        const Bounds3 bounds = shadow_scene_bounds(scene);
        const Vec3 center = (bounds.min + bounds.max) * 0.5f;
        std::vector<ShadowCandidate> candidates;
        for (std::size_t index = 0; index < scene.directional_lights.size(); ++index) {
            const DirectionalLight& light = scene.directional_lights[index];
            if (light.casts_shadows) {
                candidates.push_back({
                    ShadowLightType::Directional,
                    index,
                    light.shadow_priority,
                    color_luminance(light.radiance),
                    false});
            }
        }
        for (std::size_t index = 0; index < scene.point_lights.size(); ++index) {
            const PointLight& light = scene.point_lights[index];
            if (light.casts_shadows) {
                candidates.push_back({
                    ShadowLightType::Point,
                    index,
                    light.shadow_priority,
                    color_luminance(light.intensity) /
                        std::max((light.position - center).squaredNorm(), 1.0e-4f),
                    true});
            }
        }
        for (std::size_t index = 0; index < scene.spot_lights.size(); ++index) {
            const SpotLight& light = scene.spot_lights[index];
            if (!light.casts_shadows) {
                continue;
            }
            const Vec3 to_center = center - light.position;
            const float cone = usable_direction(to_center) && usable_direction(light.direction)
                ? std::max(0.0f, to_center.normalized().dot(light.direction.normalized()))
                : 0.0f;
            candidates.push_back({
                ShadowLightType::Spot,
                index,
                light.shadow_priority,
                color_luminance(light.intensity) * cone /
                    std::max(to_center.squaredNorm(), 1.0e-4f),
                false});
        }
        for (std::size_t index = 0;
             settings.opengl.ltc_area_lights_enabled &&
             index < scene.rect_area_lights.size();
             ++index) {
            const RectAreaLight& light = scene.rect_area_lights[index];
            if (!light.casts_shadows) {
                continue;
            }
            const float area = 4.0f * light.axis_u.cross(light.axis_v).norm();
            const float sided_contribution =
                rect_area_light_emits_toward(light, center) ? 1.0f : 0.0f;
            candidates.push_back({
                ShadowLightType::RectArea,
                index,
                light.shadow_priority,
                sided_contribution * color_luminance(light.radiance) * area /
                    std::max((light.position - center).squaredNorm(), 1.0e-4f),
                true});
        }
        if (dominant_environment_.valid &&
            settings.opengl.dominant_light.enabled) {
            const Color radiance = dominant_environment_.integrated_radiance
                .cwiseProduct(scene.environment) *
                (std::max(0.0f, scene.environment_intensity) *
                 std::max(0.0f, settings.opengl.dominant_light.intensity_scale));
            candidates.push_back({
                ShadowLightType::DominantEnvironment,
                0U,
                0,
                color_luminance(radiance),
                false});
        }
        std::stable_sort(
            candidates.begin(),
            candidates.end(),
            [](const ShadowCandidate& left, const ShadowCandidate& right) {
                return open_gl_shadow_budget_precedes(
                    left.priority,
                    left.contribution,
                    right.priority,
                    right.contribution);
            });
        const int budget = std::clamp(configured.max_shadow_lights, 1, 32);
        if (static_cast<int>(candidates.size()) > budget) {
            shadow_budget_excluded_count_ =
                static_cast<int>(candidates.size()) - budget;
            candidates.resize(static_cast<std::size_t>(budget));
        }
        GLint hardware_layers = 0;
        glGetIntegerv(GL_MAX_ARRAY_TEXTURE_LAYERS, &hardware_layers);
        const int maximum_2d_layers = std::max(0, hardware_layers);
        const int maximum_cube_layers = std::max(0, hardware_layers / 6);
        int layers_2d = 0;
        int cube_layers = 0;
        const int resolution = std::clamp(configured.resolution, 128, 4096);
        for (const ShadowCandidate& candidate : candidates) {
            int layer = -1;
            if (candidate.cube) {
                if (cube_layers < maximum_cube_layers) {
                    layer = cube_layers++;
                }
            } else if (layers_2d < maximum_2d_layers) {
                layer = layers_2d++;
            }
            if (layer < 0) {
                ++shadow_hardware_excluded_count_;
                continue;
            }
            ShadowSlot slot;
            slot.type = candidate.type;
            slot.light_index = candidate.light_index;
            slot.layer = layer;
            slot.global_slot = static_cast<int>(shadow_slots_.size());
            slot.cube = candidate.cube;
            const ShadowReference reference{slot.layer, slot.global_slot};
            switch (candidate.type) {
                case ShadowLightType::Directional: {
                    const DirectionalLight& light =
                        scene.directional_lights[candidate.light_index];
                    slot.direction = light.direction;
                    configure_directional_shadow(
                        slot, bounds, configured.projection_padding, resolution);
                    directional_shadow_references_[candidate.light_index] = reference;
                    break;
                }
                case ShadowLightType::Point: {
                    const PointLight& light = scene.point_lights[candidate.light_index];
                    slot.position = light.position;
                    configure_cube_shadow(slot, bounds, light.range);
                    point_shadow_references_[candidate.light_index] = reference;
                    break;
                }
                case ShadowLightType::Spot: {
                    const SpotLight& light = scene.spot_lights[candidate.light_index];
                    configure_spot_shadow(slot, light, bounds);
                    spot_shadow_references_[candidate.light_index] = reference;
                    break;
                }
                case ShadowLightType::RectArea: {
                    const RectAreaLight& light =
                        scene.rect_area_lights[candidate.light_index];
                    slot.position = light.position;
                    configure_cube_shadow(slot, bounds, 0.0f);
                    rect_shadow_references_[candidate.light_index] = reference;
                    break;
                }
                case ShadowLightType::DominantEnvironment: {
                    slot.direction = -rotate_y_degrees(
                        dominant_environment_.direction,
                        scene.environment_rotation_degrees);
                    configure_directional_shadow(
                        slot, bounds, configured.projection_padding, resolution);
                    dominant_shadow_reference_ = reference;
                    break;
                }
            }
            shadow_slots_.push_back(std::move(slot));
        }
        ensure_shadow_resources(resolution, layers_2d, cube_layers);
        render_shadow_maps(scene);
    }

    void upload_geometry(const Scene& scene) {
        const std::size_t bucket_count = scene.materials.size() + 1U;
        std::vector<std::size_t> vertex_counts(bucket_count, 0U);
        for (const Triangle& triangle : scene.triangles) {
            if (!usable_direction(safe_geometric_normal(triangle))) {
                continue;
            }
            const std::size_t bucket = valid_material_id(scene, triangle.material_id())
                ? static_cast<std::size_t>(triangle.material_id()) + 1U
                : 0U;
            vertex_counts[bucket] += 3U;
        }

        std::vector<std::size_t> offsets(bucket_count, 0U);
        std::size_t total_vertices = 0;
        batches_.clear();
        for (std::size_t bucket = 0; bucket < bucket_count; ++bucket) {
            const std::size_t gl_vertex_limit =
                static_cast<std::size_t>(std::numeric_limits<GLsizei>::max());
            if (vertex_counts[bucket] > gl_vertex_limit - total_vertices) {
                throw std::runtime_error("OpenGL scene has too many expanded vertices");
            }
            offsets[bucket] = total_vertices;
            if (vertex_counts[bucket] > 0U) {
                batches_.push_back(DrawBatch{
                    bucket == 0U ? -1 : static_cast<int>(bucket - 1U),
                    static_cast<GLint>(total_vertices),
                    static_cast<GLsizei>(vertex_counts[bucket])});
            }
            total_vertices += vertex_counts[bucket];
        }

        std::vector<GpuVertex> vertices(total_vertices);
        std::vector<std::size_t> cursors = offsets;
        for (const Triangle& triangle : scene.triangles) {
            const Vec3 geometric_normal = safe_geometric_normal(triangle);
            if (!usable_direction(geometric_normal)) {
                continue;
            }
            const std::size_t bucket = valid_material_id(scene, triangle.material_id())
                ? static_cast<std::size_t>(triangle.material_id()) + 1U
                : 0U;
            for (int vertex_index = 0; vertex_index < 3; ++vertex_index) {
                vertices[cursors[bucket]++] =
                    make_gpu_vertex(triangle, vertex_index, geometric_normal);
            }
        }

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
        glBufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(vertices.size() * sizeof(GpuVertex)),
            vertices.empty() ? nullptr : vertices.data(),
            GL_STATIC_DRAW);
        constexpr GLsizei stride = static_cast<GLsizei>(sizeof(GpuVertex));
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(0));
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(
            1, 3, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(GpuVertex, normal)));
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(
            2, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(GpuVertex, uv)));
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(
            3, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(GpuVertex, tangent)));
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(
            4, 2, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(GpuVertex, uv1)));
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(
            5, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<void*>(offsetof(GpuVertex, color)));
    }

    void upload_textures(const Scene& scene) {
        release_scene_textures();
        scene_textures_.resize(scene.textures.size(), 0);
        if (!scene_textures_.empty()) {
            glGenTextures(
                static_cast<GLsizei>(scene_textures_.size()),
                scene_textures_.data());
        }
        for (std::size_t texture_index = 0; texture_index < scene.textures.size(); ++texture_index) {
            const ImageTexture& texture = scene.textures[texture_index];
            glBindTexture(GL_TEXTURE_2D, scene_textures_[texture_index]);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA16F,
                texture.width(),
                texture.height(),
                0,
                GL_RGBA,
                GL_FLOAT,
                nullptr);
            constexpr std::size_t kUploadChunkBytes = 4U * 1024U * 1024U;
            const std::size_t row_values =
                static_cast<std::size_t>(texture.width()) * 4U;
            const int chunk_rows = std::max(
                1,
                std::min(
                    texture.height(),
                    static_cast<int>(kUploadChunkBytes /
                        std::max<std::size_t>(row_values * sizeof(float), 1U))));
            std::vector<float> pixels(
                static_cast<std::size_t>(chunk_rows) * row_values);
            for (int destination_y = 0;
                 destination_y < texture.height();
                 destination_y += chunk_rows) {
                const int rows = std::min(chunk_rows, texture.height() - destination_y);
                for (int row = 0; row < rows; ++row) {
                    const int source_y = texture.height() - 1 - (destination_y + row);
                    for (int x = 0; x < texture.width(); ++x) {
                        const std::size_t source_index =
                            static_cast<std::size_t>(source_y * texture.width() + x);
                        const std::size_t destination_index =
                            (static_cast<std::size_t>(row) *
                             static_cast<std::size_t>(texture.width()) +
                             static_cast<std::size_t>(x)) * 4U;
                        const Color& color = texture.pixels()[source_index];
                        pixels[destination_index] = color.x();
                        pixels[destination_index + 1U] = color.y();
                        pixels[destination_index + 2U] = color.z();
                        pixels[destination_index + 3U] = texture.alphas()[source_index];
                    }
                }
                glTexSubImage2D(
                    GL_TEXTURE_2D,
                    0,
                    0,
                    destination_y,
                    texture.width(),
                    rows,
                    GL_RGBA,
                    GL_FLOAT,
                    pixels.data());
            }
            glGenerateMipmap(GL_TEXTURE_2D);
            glTexParameteri(
                GL_TEXTURE_2D,
                GL_TEXTURE_MIN_FILTER,
                gl_texture_filter(texture.min_filter()));
            glTexParameteri(
                GL_TEXTURE_2D,
                GL_TEXTURE_MAG_FILTER,
                texture.mag_filter() == TextureFilter::Nearest ? GL_NEAREST : GL_LINEAR);
            glTexParameteri(
                GL_TEXTURE_2D,
                GL_TEXTURE_WRAP_S,
                gl_texture_wrap(texture.wrap_s()));
            glTexParameteri(
                GL_TEXTURE_2D,
                GL_TEXTURE_WRAP_T,
                gl_texture_wrap(texture.wrap_t()));
        }
    }

    void upload_lights(
        const Scene& scene,
        const RenderSettings* settings = nullptr) {
        std::vector<GpuDirectionalLight> directional;
        directional.reserve(scene.directional_lights.size() + 1U);
        for (std::size_t index = 0; index < scene.directional_lights.size(); ++index) {
            const DirectionalLight& light = scene.directional_lights[index];
            const ShadowReference shadow = index < directional_shadow_references_.size()
                ? directional_shadow_references_[index]
                : ShadowReference{};
            directional.push_back(GpuDirectionalLight{
                {
                    light.direction.x(),
                    light.direction.y(),
                    light.direction.z(),
                    light.angular_radius_radians},
                {light.radiance.x(), light.radiance.y(), light.radiance.z(), 0.0f},
                {
                    light.angular_radius_radians,
                    static_cast<float>(shadow.layer),
                    static_cast<float>(shadow.global_slot),
                    0.0f}});
        }
        std::vector<GpuPointLight> point;
        point.reserve(scene.point_lights.size());
        for (std::size_t index = 0; index < scene.point_lights.size(); ++index) {
            const PointLight& light = scene.point_lights[index];
            const ShadowReference shadow = index < point_shadow_references_.size()
                ? point_shadow_references_[index]
                : ShadowReference{};
            point.push_back(GpuPointLight{
                {light.position.x(), light.position.y(), light.position.z(), light.range},
                {light.intensity.x(), light.intensity.y(), light.intensity.z(), 0.0f},
                {
                    light.source_radius,
                    static_cast<float>(shadow.layer),
                    static_cast<float>(shadow.global_slot),
                    0.0f}});
        }
        std::vector<GpuSpotLight> spots;
        spots.reserve(scene.spot_lights.size());
        for (std::size_t index = 0; index < scene.spot_lights.size(); ++index) {
            const SpotLight& light = scene.spot_lights[index];
            const ShadowReference shadow = index < spot_shadow_references_.size()
                ? spot_shadow_references_[index]
                : ShadowReference{};
            spots.push_back(GpuSpotLight{
                {light.position.x(), light.position.y(), light.position.z(), light.range},
                {
                    light.direction.x(),
                    light.direction.y(),
                    light.direction.z(),
                    std::cos(light.inner_cone_radians)},
                {
                    light.intensity.x(),
                    light.intensity.y(),
                    light.intensity.z(),
                    std::cos(light.outer_cone_radians)},
                {
                    light.source_radius,
                    static_cast<float>(shadow.layer),
                    static_cast<float>(shadow.global_slot),
                    0.0f}});
        }
        std::vector<GpuRectAreaLight> rectangles;
        rectangles.reserve(scene.rect_area_lights.size());
        for (std::size_t index = 0; index < scene.rect_area_lights.size(); ++index) {
            const RectAreaLight& light = scene.rect_area_lights[index];
            const ShadowReference shadow = index < rect_shadow_references_.size()
                ? rect_shadow_references_[index]
                : ShadowReference{};
            const float equivalent_radius = std::sqrt(std::max(
                4.0f * light.axis_u.cross(light.axis_v).norm() /
                    3.14159265358979323846f,
                0.0f));
            rectangles.push_back(GpuRectAreaLight{
                {
                    light.position.x(),
                    light.position.y(),
                    light.position.z(),
                    light.two_sided ? 1.0f : 0.0f},
                {light.axis_u.x(), light.axis_u.y(), light.axis_u.z(), 0.0f},
                {light.axis_v.x(), light.axis_v.y(), light.axis_v.z(), 0.0f},
                {light.radiance.x(), light.radiance.y(), light.radiance.z(), 0.0f},
                {
                    equivalent_radius,
                    static_cast<float>(shadow.layer),
                    static_cast<float>(shadow.global_slot),
                    light.casts_shadows ? 1.0f : 0.0f}});
        }
        if (settings && dominant_environment_.valid &&
            settings->opengl.dominant_light.enabled) {
            const Vec3 toward_light = rotate_y_degrees(
                dominant_environment_.direction,
                scene.environment_rotation_degrees);
            const Color radiance = dominant_environment_.integrated_radiance
                .cwiseProduct(scene.environment) *
                (std::max(0.0f, scene.environment_intensity) *
                 std::max(0.0f, settings->opengl.dominant_light.intensity_scale));
            directional.push_back(GpuDirectionalLight{
                {
                    -toward_light.x(),
                    -toward_light.y(),
                    -toward_light.z(),
                    dominant_environment_.angular_radius_radians},
                {radiance.x(), radiance.y(), radiance.z(), 0.0f},
                {
                    dominant_environment_.angular_radius_radians,
                    static_cast<float>(dominant_shadow_reference_.layer),
                    static_cast<float>(dominant_shadow_reference_.global_slot),
                    0.0f}});
        }
        gpu_directional_light_count_ = static_cast<int>(directional.size());
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directional_light_buffer_);
        const GpuDirectionalLight empty_directional;
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<GLsizeiptr>(
                std::max<std::size_t>(directional.size(), 1U) *
                sizeof(GpuDirectionalLight)),
            directional.empty() ? &empty_directional : directional.data(),
            GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, point_light_buffer_);
        const GpuPointLight empty_point;
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<GLsizeiptr>(
                std::max<std::size_t>(point.size(), 1U) * sizeof(GpuPointLight)),
            point.empty() ? &empty_point : point.data(),
            GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, spot_light_buffer_);
        const GpuSpotLight empty_spot;
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<GLsizeiptr>(
                std::max<std::size_t>(spots.size(), 1U) * sizeof(GpuSpotLight)),
            spots.empty() ? &empty_spot : spots.data(),
            GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, rect_area_light_buffer_);
        const GpuRectAreaLight empty_rectangle;
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<GLsizeiptr>(
                std::max<std::size_t>(rectangles.size(), 1U) *
                sizeof(GpuRectAreaLight)),
            rectangles.empty() ? &empty_rectangle : rectangles.data(),
            GL_DYNAMIC_DRAW);
        gpu_lights_dirty_ = false;
    }

    void bind_material(
        const Scene& scene,
        const Material& material,
        bool material_valid,
        const UniformLocations& locations) {
        set_uniform(
            locations.material_type,
            material_valid ? static_cast<int>(material.type) : -1);
        set_uniform(locations.pbr_workflow, static_cast<int>(material.pbr_workflow));
        set_uniform(
            locations.base_color,
            material_valid ? material.base_color : Color(1.0f, 0.0f, 1.0f));
        set_uniform(locations.emission, material.emission);
        set_uniform(locations.ior, material.ior);
        set_uniform(locations.specular_color, material.specular_color);
        set_uniform(locations.specular_factor, material.specular_factor);
        set_uniform(locations.glossiness, material.glossiness);
        set_uniform(locations.metallic, material.metallic);
        set_uniform(locations.roughness, material.roughness);
        set_uniform(locations.opacity, material.opacity);
        set_uniform(locations.alpha_cutoff, material.alpha_cutoff);
        set_uniform(locations.bump_scale, material.bump_scale);
        set_uniform(locations.normal_scale, material.normal_scale);
        set_uniform(locations.occlusion_strength, material.occlusion_strength);
        const AlphaMode alpha_mode = material.type == MaterialType::Pbr
            ? material.alpha_mode
            : (material.opacity_texture_id >= 0 || material.opacity < 1.0f
                  ? AlphaMode::Mask
                  : AlphaMode::Opaque);
        set_uniform(locations.alpha_mode, static_cast<int>(alpha_mode));
        set_uniform(locations.two_sided, material.two_sided ? 1 : 0);

        const int base_color_texture = material.base_color_texture_id >= 0
            ? material.base_color_texture_id
            : material.diffuse_texture_id;
        const int normal_texture = material.normal_texture_id >= 0
            ? material.normal_texture_id
            : material.bump_texture_id;
        const std::array<int, 9> texture_ids{
            base_color_texture,
            material.opacity_texture_id,
            normal_texture,
            material.metallic_roughness_texture_id,
            material.occlusion_texture_id,
            material.emissive_texture_id,
            material.specular_texture_id,
            material.specular_color_texture_id,
            material.specular_glossiness_texture_id};
        const TextureTransform identity_transform;
        const std::array<TextureTransform, 9> texture_transforms{
            material.base_color_texture_transform,
            identity_transform,
            material.normal_texture_id >= 0
                ? material.normal_texture_transform
                : identity_transform,
            material.metallic_roughness_texture_transform,
            material.occlusion_texture_transform,
            material.emissive_texture_transform,
            material.specular_texture_transform,
            material.specular_color_texture_transform,
            material.specular_glossiness_texture_transform};
        const std::array<GLint, 9> presence_uniforms{
            locations.has_base_color_texture,
            locations.has_opacity_texture,
            material.normal_texture_id >= 0
                ? locations.has_normal_texture
                : locations.has_bump_texture,
            locations.has_metallic_roughness_texture,
            locations.has_occlusion_texture,
            locations.has_emissive_texture,
            locations.has_specular_texture,
            locations.has_specular_color_texture,
            locations.has_specular_glossiness_texture};
        set_uniform(locations.has_normal_texture, 0);
        set_uniform(locations.has_bump_texture, 0);
        for (int slot = 0; slot < 9; ++slot) {
            const TextureTransform& transform =
                texture_transforms[static_cast<std::size_t>(slot)];
            const GLint offset_scale_location =
                locations.texture_offset_scale[static_cast<std::size_t>(slot)];
            if (offset_scale_location >= 0) {
                glUniform4f(
                    offset_scale_location,
                    transform.offset.x(),
                    transform.offset.y(),
                    transform.scale.x(),
                    transform.scale.y());
            }
            set_uniform(
                locations.texture_rotation[static_cast<std::size_t>(slot)],
                transform.rotation);
            set_uniform(
                locations.texture_texcoord[static_cast<std::size_t>(slot)],
                transform.texcoord);
            const bool valid = material_valid && valid_texture_id(scene, texture_ids[slot]);
            set_uniform(
                locations.texture_top_left[static_cast<std::size_t>(slot)],
                valid && scene.textures[static_cast<std::size_t>(texture_ids[slot])].uv_origin() ==
                    TextureUvOrigin::TopLeft
                    ? 1
                    : 0);
            set_uniform(presence_uniforms[slot], valid ? 1 : 0);
            glActiveTexture(GL_TEXTURE0 + slot);
            glBindTexture(
                GL_TEXTURE_2D,
                valid ? scene_textures_[static_cast<std::size_t>(texture_ids[slot])]
                      : fallback_texture_);
        }
    }

    bool render_ambient_occlusion(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const Mat4& view_projection) {
        const auto& ao_settings = settings.opengl.ambient_occlusion;
        if (!ao_shaders_available_ ||
            ao_settings.mode == OpenGlAmbientOcclusionMode::Off ||
            ao_gbuffer_framebuffer_ == 0 || ao_framebuffer_ == 0) {
            ao_resolved_texture_ = ao_raw_texture_;
            return false;
        }
        if (!settings.opengl.ibl_enabled &&
            ao_settings.debug_view == OpenGlAmbientOcclusionDebugView::Final) {
            ao_resolved_texture_ = ao_raw_texture_;
            return false;
        }

        glBindFramebuffer(GL_FRAMEBUFFER, ao_gbuffer_framebuffer_);
        glViewport(0, 0, settings.width, settings.height);
        const std::array<GLenum, 2> gbuffer_outputs{
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1};
        glDrawBuffers(
            static_cast<GLsizei>(gbuffer_outputs.size()),
            gbuffer_outputs.data());
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
        const std::array<float, 4> zero{0.0f, 0.0f, 0.0f, 0.0f};
        glClearBufferfv(GL_COLOR, 0, zero.data());
        glClearBufferfv(GL_COLOR, 1, zero.data());
        glClear(GL_DEPTH_BUFFER_BIT);

        glUseProgram(ao_gbuffer_program_.id());
        if (ao_gbuffer_uniforms_.view_projection >= 0) {
            glUniformMatrix4fv(
                ao_gbuffer_uniforms_.view_projection,
                1,
                GL_FALSE,
                view_projection.data());
        }
        set_uniform(ao_gbuffer_uniforms_.camera_position, camera.eye());
        set_uniform(ao_gbuffer_uniforms_.camera_forward, camera.forward());
        set_uniform(ao_gbuffer_uniforms_.camera_right, camera.right());
        set_uniform(ao_gbuffer_uniforms_.camera_up, camera.up());
        glBindVertexArray(vao_);
        for (const DrawBatch& batch : batches_) {
            const Material fallback;
            const bool valid = valid_material_id(scene, batch.material_id);
            const Material& material = valid
                ? scene.materials[static_cast<std::size_t>(batch.material_id)]
                : fallback;
            if (material.alpha_mode == AlphaMode::Blend) {
                continue;
            }
            bind_material(scene, material, valid, ao_gbuffer_uniforms_);
            if (material.two_sided || !valid) {
                glDisable(GL_CULL_FACE);
            } else {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
                glFrontFace(GL_CCW);
            }
            glDrawArrays(GL_TRIANGLES, batch.first, batch.count);
        }
        glDisable(GL_CULL_FACE);

        const float radius_scale =
            ao_settings.mode == OpenGlAmbientOcclusionMode::Ssao
            ? ao_settings.ssao.radius_scale
            : ao_settings.gtao.radius_scale;
        const float radius_world = std::max(
            1.0e-5f,
            scene_radius_ * std::clamp(radius_scale, 0.005f, 0.5f));

        glBindFramebuffer(GL_FRAMEBUFFER, ao_framebuffer_);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            ao_raw_texture_,
            0);
        const GLenum ao_output = GL_COLOR_ATTACHMENT0;
        glDrawBuffers(1, &ao_output);
        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glUseProgram(ao_program_.id());
        glUniform2f(
            glGetUniformLocation(ao_program_.id(), "u_resolution"),
            static_cast<float>(settings.width),
            static_cast<float>(settings.height));
        glUniform2f(
            glGetUniformLocation(ao_program_.id(), "u_camera_viewport"),
            camera.viewport_width(),
            camera.viewport_height());
        set_uniform(
            glGetUniformLocation(ao_program_.id(), "u_ao_mode"),
            static_cast<int>(ao_settings.mode));
        set_uniform(
            glGetUniformLocation(ao_program_.id(), "u_radius_world"),
            radius_world);
        set_uniform(
            glGetUniformLocation(ao_program_.id(), "u_ssao_sample_count"),
            std::clamp(ao_settings.ssao.sample_count, 8, 64));
        set_uniform(
            glGetUniformLocation(ao_program_.id(), "u_ssao_bias_fraction"),
            std::clamp(ao_settings.ssao.depth_bias_fraction, 0.0f, 0.2f));
        set_uniform(
            glGetUniformLocation(ao_program_.id(), "u_ssao_intensity"),
            std::clamp(ao_settings.ssao.intensity, 0.0f, 4.0f));
        set_uniform(
            glGetUniformLocation(ao_program_.id(), "u_gtao_slice_count"),
            std::clamp(ao_settings.gtao.slice_count, 1, 8));
        set_uniform(
            glGetUniformLocation(ao_program_.id(), "u_gtao_samples_per_side"),
            std::clamp(ao_settings.gtao.samples_per_side, 1, 8));
        set_uniform(
            glGetUniformLocation(ao_program_.id(), "u_gtao_falloff_fraction"),
            std::clamp(ao_settings.gtao.falloff_fraction, 0.05f, 1.0f));
        set_uniform(
            glGetUniformLocation(ao_program_.id(), "u_gtao_thickness_fraction"),
            std::clamp(ao_settings.gtao.thickness_fraction, 0.0f, 1.0f));
        set_uniform(
            glGetUniformLocation(ao_program_.id(), "u_gtao_intensity"),
            std::clamp(ao_settings.gtao.intensity, 0.0f, 4.0f));
        set_uniform(
            glGetUniformLocation(
                ao_program_.id(),
                "u_gtao_bent_normals_enabled"),
            ao_settings.gtao.bent_normals_enabled ? 1 : 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, ao_linear_depth_texture_);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, ao_view_normal_texture_);
        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        ao_resolved_texture_ = ao_raw_texture_;

        if (ao_settings.denoise.enabled) {
            const auto denoise_pass = [&](GLuint source, GLuint target, float x, float y) {
                glFramebufferTexture2D(
                    GL_FRAMEBUFFER,
                    GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_2D,
                    target,
                    0);
                glUseProgram(ao_denoise_program_.id());
                glUniform2f(
                    glGetUniformLocation(
                        ao_denoise_program_.id(), "u_resolution"),
                    static_cast<float>(settings.width),
                    static_cast<float>(settings.height));
                glUniform2f(
                    glGetUniformLocation(
                        ao_denoise_program_.id(), "u_filter_axis"),
                    x,
                    y);
                set_uniform(
                    glGetUniformLocation(
                        ao_denoise_program_.id(), "u_kernel_radius"),
                    std::clamp(ao_settings.denoise.kernel_radius, 1, 4));
                set_uniform(
                    glGetUniformLocation(
                        ao_denoise_program_.id(), "u_radius_world"),
                    radius_world);
                set_uniform(
                    glGetUniformLocation(
                        ao_denoise_program_.id(), "u_depth_sigma_fraction"),
                    std::clamp(
                        ao_settings.denoise.depth_sigma_fraction,
                        0.01f,
                        1.0f));
                set_uniform(
                    glGetUniformLocation(
                        ao_denoise_program_.id(), "u_normal_power"),
                    std::clamp(ao_settings.denoise.normal_power, 1.0f, 64.0f));
                glActiveTexture(GL_TEXTURE0);
                glBindTexture(GL_TEXTURE_2D, source);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, ao_linear_depth_texture_);
                glActiveTexture(GL_TEXTURE2);
                glBindTexture(GL_TEXTURE_2D, ao_view_normal_texture_);
                glDrawArrays(GL_TRIANGLES, 0, 3);
            };
            denoise_pass(ao_raw_texture_, ao_temporary_texture_, 1.0f, 0.0f);
            denoise_pass(ao_temporary_texture_, ao_filtered_texture_, 0.0f, 1.0f);
            ao_resolved_texture_ = ao_filtered_texture_;
        }
        glDepthMask(GL_TRUE);
        return true;
    }

    void ensure_output(int width, int height) {
        if (framebuffer_ != 0 && width == output_width_ && height == output_height_) {
            return;
        }
        release_output_resources();
        glGenFramebuffers(1, &framebuffer_);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);

        glGenTextures(1, &opaque_texture_);
        glBindTexture(GL_TEXTURE_2D, opaque_texture_);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, opaque_texture_, 0);

        glGenTextures(1, &transparency_accum_texture_);
        glBindTexture(GL_TEXTURE_2D, transparency_accum_texture_);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT1,
            GL_TEXTURE_2D,
            transparency_accum_texture_,
            0);

        glGenTextures(1, &transparency_reveal_texture_);
        glBindTexture(GL_TEXTURE_2D, transparency_reveal_texture_);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R16F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT2,
            GL_TEXTURE_2D,
            transparency_reveal_texture_,
            0);

        glGenTextures(1, &depth_texture_);
        glBindTexture(GL_TEXTURE_2D, depth_texture_);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_DEPTH_COMPONENT32F,
            width,
            height,
            0,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
            nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_DEPTH_ATTACHMENT,
            GL_TEXTURE_2D,
            depth_texture_,
            0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            throw std::runtime_error("OpenGL raster framebuffer is incomplete");
        }

        glGenFramebuffers(1, &ao_gbuffer_framebuffer_);
        glBindFramebuffer(GL_FRAMEBUFFER, ao_gbuffer_framebuffer_);
        glGenTextures(1, &ao_view_normal_texture_);
        glBindTexture(GL_TEXTURE_2D, ao_view_normal_texture_);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB16F, width, height, 0, GL_RGB, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            ao_view_normal_texture_,
            0);

        glGenTextures(1, &ao_linear_depth_texture_);
        glBindTexture(GL_TEXTURE_2D, ao_linear_depth_texture_);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT1,
            GL_TEXTURE_2D,
            ao_linear_depth_texture_,
            0);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_DEPTH_ATTACHMENT,
            GL_TEXTURE_2D,
            depth_texture_,
            0);
        const std::array<GLenum, 2> ao_gbuffer_outputs{
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1};
        glDrawBuffers(
            static_cast<GLsizei>(ao_gbuffer_outputs.size()),
            ao_gbuffer_outputs.data());
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            throw std::runtime_error("OpenGL AO G-buffer is incomplete");
        }

        const auto create_ao_texture = [width, height](GLuint& texture) {
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA16F,
                width,
                height,
                0,
                GL_RGBA,
                GL_FLOAT,
                nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        };
        create_ao_texture(ao_raw_texture_);
        create_ao_texture(ao_temporary_texture_);
        create_ao_texture(ao_filtered_texture_);
        ao_resolved_texture_ = ao_raw_texture_;
        glGenFramebuffers(1, &ao_framebuffer_);
        glBindFramebuffer(GL_FRAMEBUFFER, ao_framebuffer_);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            ao_raw_texture_,
            0);
        const GLenum ao_output = GL_COLOR_ATTACHMENT0;
        glDrawBuffers(1, &ao_output);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            throw std::runtime_error("OpenGL AO framebuffer is incomplete");
        }

        glGenFramebuffers(1, &composite_framebuffer_);
        glBindFramebuffer(GL_FRAMEBUFFER, composite_framebuffer_);
        glGenTextures(1, &color_texture_);
        glBindTexture(GL_TEXTURE_2D, color_texture_);
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_texture_, 0);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            throw std::runtime_error("OpenGL transparency composite framebuffer is incomplete");
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        output_width_ = width;
        output_height_ = height;
    }

    float compute_scene_radius(const Scene& scene) const {
        Vec3 minimum = Vec3::Constant(std::numeric_limits<float>::infinity());
        Vec3 maximum = Vec3::Constant(-std::numeric_limits<float>::infinity());
        bool has_position = false;
        for (const Triangle& triangle : scene.triangles) {
            for (int index = 0; index < 3; ++index) {
                minimum = minimum.cwiseMin(triangle.vertex(index).position);
                maximum = maximum.cwiseMax(triangle.vertex(index).position);
                has_position = true;
            }
        }
        if (!has_position) {
            return 1.0f;
        }
        return std::max(0.5f, (maximum - minimum).norm() * 0.5f);
    }

    void release_scene_textures() {
        if (!scene_textures_.empty()) {
            glDeleteTextures(
                static_cast<GLsizei>(scene_textures_.size()),
                scene_textures_.data());
            scene_textures_.clear();
        }
    }

    void release_scene_resources() {
        release_scene_textures();
        batches_.clear();
    }

    void release_shadow_resources() {
        if (shadow_2d_array_ != 0) {
            glDeleteTextures(1, &shadow_2d_array_);
            shadow_2d_array_ = 0;
        }
        if (shadow_cube_array_ != 0) {
            glDeleteTextures(1, &shadow_cube_array_);
            shadow_cube_array_ = 0;
        }
        if (shadow_depth_renderbuffer_ != 0) {
            glDeleteRenderbuffers(1, &shadow_depth_renderbuffer_);
            shadow_depth_renderbuffer_ = 0;
        }
        if (shadow_framebuffer_ != 0) {
            glDeleteFramebuffers(1, &shadow_framebuffer_);
            shadow_framebuffer_ = 0;
        }
        shadow_resolution_ = 0;
        shadow_2d_layers_ = 0;
        shadow_cube_layers_ = 0;
    }

    void release_output_resources() {
        if (depth_texture_ != 0) {
            glDeleteTextures(1, &depth_texture_);
            depth_texture_ = 0;
        }
        for (GLuint* texture : {
                 &ao_view_normal_texture_,
                 &ao_linear_depth_texture_,
                 &ao_raw_texture_,
                 &ao_temporary_texture_,
                 &ao_filtered_texture_}) {
            if (*texture != 0) {
                glDeleteTextures(1, texture);
                *texture = 0;
            }
        }
        ao_resolved_texture_ = 0;
        if (color_texture_ != 0) {
            glDeleteTextures(1, &color_texture_);
            color_texture_ = 0;
        }
        if (opaque_texture_ != 0) {
            glDeleteTextures(1, &opaque_texture_);
            opaque_texture_ = 0;
        }
        if (transparency_accum_texture_ != 0) {
            glDeleteTextures(1, &transparency_accum_texture_);
            transparency_accum_texture_ = 0;
        }
        if (transparency_reveal_texture_ != 0) {
            glDeleteTextures(1, &transparency_reveal_texture_);
            transparency_reveal_texture_ = 0;
        }
        if (composite_framebuffer_ != 0) {
            glDeleteFramebuffers(1, &composite_framebuffer_);
            composite_framebuffer_ = 0;
        }
        if (ao_framebuffer_ != 0) {
            glDeleteFramebuffers(1, &ao_framebuffer_);
            ao_framebuffer_ = 0;
        }
        if (ao_gbuffer_framebuffer_ != 0) {
            glDeleteFramebuffers(1, &ao_gbuffer_framebuffer_);
            ao_gbuffer_framebuffer_ = 0;
        }
        if (framebuffer_ != 0) {
            glDeleteFramebuffers(1, &framebuffer_);
            framebuffer_ = 0;
        }
        output_width_ = 0;
        output_height_ = 0;
    }
};

OpenGlRasterRenderer::OpenGlRasterRenderer(
    std::filesystem::path vertex_shader_path,
    std::filesystem::path fragment_shader_path)
    : impl_(std::make_unique<Impl>(
          std::move(vertex_shader_path),
          std::move(fragment_shader_path))) {}

OpenGlRasterRenderer::~OpenGlRasterRenderer() = default;

void OpenGlRasterRenderer::reset(const Scene& scene) {
    impl_->reset(scene);
}

void OpenGlRasterRenderer::sync_scene(
    const Scene& scene,
    SceneChangeSet changes) {
    impl_->sync_scene(scene, changes);
}

void OpenGlRasterRenderer::render(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const InteractiveFrameState& frame_state) {
    impl_->render(scene, camera, settings, frame_state);
}

unsigned int OpenGlRasterRenderer::output_texture() const {
    return impl_->output_texture();
}

int OpenGlRasterRenderer::output_width() const {
    return impl_->output_width();
}

int OpenGlRasterRenderer::output_height() const {
    return impl_->output_height();
}

void OpenGlRasterRenderer::set_auto_reload(bool enabled) {
    impl_->set_auto_reload(enabled);
}

bool OpenGlRasterRenderer::auto_reload() const {
    return impl_->auto_reload();
}

void OpenGlRasterRenderer::request_shader_reload() {
    impl_->request_shader_reload();
}

bool OpenGlRasterRenderer::has_valid_shader() const {
    return impl_->has_valid_shader();
}

const std::string& OpenGlRasterRenderer::shader_error() const {
    return impl_->shader_error();
}

const std::filesystem::path& OpenGlRasterRenderer::vertex_shader_path() const {
    return impl_->vertex_shader_path();
}

const std::filesystem::path& OpenGlRasterRenderer::fragment_shader_path() const {
    return impl_->fragment_shader_path();
}

OpenGlTechniqueDiagnostics OpenGlRasterRenderer::technique_diagnostics() const {
    return impl_->technique_diagnostics();
}

}  // namespace renderer
