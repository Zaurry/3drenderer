#include "render/opengl/opengl_raster_renderer.h"

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

struct alignas(16) GpuLight {
    std::array<float, 4> vector{};
    std::array<float, 4> color{};
};

struct alignas(16) GpuSpotLight {
    std::array<float, 4> position_range{};
    std::array<float, 4> direction_inner{};
    std::array<float, 4> intensity_outer{};
};

struct UniformLocations {
    GLint view_projection = -1;
    GLint camera_position = -1;
    GLint environment_color = -1;
    GLint environment_sh = -1;
    GLint environment_intensity = -1;
    GLint environment_rotation_radians = -1;
    GLint environment_mip_count = -1;
    GLint has_environment_map = -1;
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
    GLint transparent_pass = -1;
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
    uniforms.environment_color = glGetUniformLocation(program, "u_environment_color");
    uniforms.environment_sh = glGetUniformLocation(program, "u_environment_sh[0]");
    uniforms.environment_intensity = glGetUniformLocation(program, "u_environment_intensity");
    uniforms.environment_rotation_radians = glGetUniformLocation(program, "u_environment_rotation_radians");
    uniforms.environment_mip_count = glGetUniformLocation(program, "u_environment_mip_count");
    uniforms.has_environment_map = glGetUniformLocation(program, "u_has_environment_map");
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
    uniforms.transparent_pass = glGetUniformLocation(program, "u_transparent_pass");
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
in vec2 v_ndc;
layout(location = 0) out vec4 out_linear_color;
void main() {
    vec2 uv = v_ndc * 0.5 + 0.5;
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
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vertex_buffer_);
        glGenBuffers(1, &directional_light_buffer_);
        glGenBuffers(1, &point_light_buffer_);
        glGenBuffers(1, &spot_light_buffer_);
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
        create_brdf_lut();
        request_shader_reload();
    }

    ~Impl() {
        shader_program_.reset();
        release_scene_resources();
        release_output_resources();
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
        if (environment_cube_ != 0) {
            glDeleteTextures(1, &environment_cube_);
        }
        if (brdf_lut_ != 0) {
            glDeleteTextures(1, &brdf_lut_);
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
        upload_lights(scene);
        upload_environment(scene);
        scene_radius_ = compute_scene_radius(scene);
    }

    void sync_scene(const Scene& scene, SceneChangeSet changes) {
        if (open_gl_requires_geometry_upload(changes)) {
            upload_geometry(scene);
        }
        if (has_scene_change(changes, SceneChange::Textures)) {
            upload_textures(scene);
        }
        if (has_scene_change(changes, SceneChange::Lighting)) {
            upload_lights(scene);
        }
        if (has_scene_change(changes, SceneChange::Environment)) {
            upload_environment(scene);
        }
        if (has_scene_change(changes, SceneChange::Geometry) ||
            has_scene_change(changes, SceneChange::InstanceTransforms)) {
            scene_radius_ = compute_scene_radius(scene);
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

        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
        glViewport(0, 0, settings.width, settings.height);
        const std::array<GLenum, 3> all_buffers{
            GL_COLOR_ATTACHMENT0,
            GL_COLOR_ATTACHMENT1,
            GL_COLOR_ATTACHMENT2};
        glDrawBuffers(static_cast<GLsizei>(all_buffers.size()), all_buffers.data());
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
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
        glClear(GL_DEPTH_BUFFER_BIT);
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
        set_uniform(uniforms_.environment_color, scene.environment);
        set_uniform(uniforms_.environment_intensity, scene.environment_intensity);
        set_uniform(
            uniforms_.environment_rotation_radians,
            scene.environment_rotation_degrees * 0.01745329251994329577f);
        set_uniform(uniforms_.environment_mip_count, static_cast<float>(environment_mip_count_));
        set_uniform(uniforms_.has_environment_map, scene.environment_map ? 1 : 0);
        if (uniforms_.environment_sh >= 0) {
            std::array<float, 27> sh{};
            if (scene.environment_map) {
                const auto& coefficients = scene.environment_map->radiance_sh();
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
            static_cast<int>(scene.directional_lights.size()));
        set_uniform(
            uniforms_.point_light_count,
            static_cast<int>(scene.point_lights.size()));
        set_uniform(
            uniforms_.spot_light_count,
            static_cast<int>(scene.spot_lights.size()));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, directional_light_buffer_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, point_light_buffer_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, spot_light_buffer_);
        glActiveTexture(GL_TEXTURE9);
        glBindTexture(GL_TEXTURE_CUBE_MAP, environment_cube_);
        glActiveTexture(GL_TEXTURE10);
        glBindTexture(GL_TEXTURE_2D, brdf_lut_);
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
            bind_material(scene, material, valid_material_id(scene, batch.material_id));
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
        glDepthMask(GL_FALSE);
        draw_batches(true);
        glDepthMask(GL_TRUE);
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
        if (!reload_requested_ &&
            vertex_write_time == vertex_write_time_ &&
            fragment_write_time == fragment_write_time_) {
            return;
        }
        reload_requested_ = false;
        vertex_write_time_ = vertex_write_time;
        fragment_write_time_ = fragment_write_time;

        GlShaderProgram replacement;
        std::string error;
        if (!replacement.load(vertex_shader_path_, fragment_shader_path_, error)) {
            shader_error_ = std::move(error);
            return;
        }
        shader_program_ = std::move(replacement);
        uniforms_ = find_uniforms(shader_program_.id());
        shader_error_.clear();
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

private:
    std::filesystem::path vertex_shader_path_;
    std::filesystem::path fragment_shader_path_;
    std::filesystem::file_time_type vertex_write_time_ =
        std::filesystem::file_time_type::min();
    std::filesystem::file_time_type fragment_write_time_ =
        std::filesystem::file_time_type::min();
    std::chrono::steady_clock::time_point next_shader_check_ =
        std::chrono::steady_clock::time_point::min();
    bool auto_reload_ = true;
    bool reload_requested_ = true;
    std::string shader_error_;
    GlShaderProgram shader_program_;
    UniformLocations uniforms_;

    GLuint vao_ = 0;
    GLuint vertex_buffer_ = 0;
    GLuint directional_light_buffer_ = 0;
    GLuint point_light_buffer_ = 0;
    GLuint spot_light_buffer_ = 0;
    GLuint fallback_texture_ = 0;
    GLuint environment_cube_ = 0;
    GLuint brdf_lut_ = 0;
    int environment_mip_count_ = 1;
    const EnvironmentMap* uploaded_environment_ = nullptr;
    GlShaderProgram sky_program_;
    GlShaderProgram composite_program_;
    std::vector<GLuint> scene_textures_;
    std::vector<DrawBatch> batches_;
    float scene_radius_ = 1.0f;

    GLuint framebuffer_ = 0;
    GLuint color_texture_ = 0;
    GLuint opaque_texture_ = 0;
    GLuint transparency_accum_texture_ = 0;
    GLuint transparency_reveal_texture_ = 0;
    GLuint composite_framebuffer_ = 0;
    GLuint depth_renderbuffer_ = 0;
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

    void upload_environment(const Scene& scene) {
        const EnvironmentMap* environment = scene.environment_map.get();
        if (environment == uploaded_environment_) {
            return;
        }
        if (environment_cube_ != 0) {
            glDeleteTextures(1, &environment_cube_);
            environment_cube_ = 0;
        }
        uploaded_environment_ = environment;
        environment_mip_count_ = 1;
        glGenTextures(1, &environment_cube_);
        glBindTexture(GL_TEXTURE_CUBE_MAP, environment_cube_);
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
            environment_mip_count_ = 1 + static_cast<int>(std::floor(std::log2(base_size)));
            for (int mip = 0; mip < environment_mip_count_; ++mip) {
                const int size = std::max(1, base_size >> mip);
                const float roughness = environment_mip_count_ > 1
                    ? static_cast<float>(mip) / static_cast<float>(environment_mip_count_ - 1)
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

    void upload_lights(const Scene& scene) {
        std::vector<GpuLight> directional;
        directional.reserve(scene.directional_lights.size());
        for (const DirectionalLight& light : scene.directional_lights) {
            directional.push_back(GpuLight{
                {light.direction.x(), light.direction.y(), light.direction.z(), 0.0f},
                {light.radiance.x(), light.radiance.y(), light.radiance.z(), 0.0f}});
        }
        std::vector<GpuLight> point;
        point.reserve(scene.point_lights.size());
        for (const PointLight& light : scene.point_lights) {
            point.push_back(GpuLight{
                {light.position.x(), light.position.y(), light.position.z(), light.range},
                {light.intensity.x(), light.intensity.y(), light.intensity.z(), 0.0f}});
        }
        std::vector<GpuSpotLight> spots;
        spots.reserve(scene.spot_lights.size());
        for (const SpotLight& light : scene.spot_lights) {
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
                    std::cos(light.outer_cone_radians)}});
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directional_light_buffer_);
        const GpuLight empty_light;
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<GLsizeiptr>(std::max<std::size_t>(directional.size(), 1U) * sizeof(GpuLight)),
            directional.empty() ? &empty_light : directional.data(),
            GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, point_light_buffer_);
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<GLsizeiptr>(std::max<std::size_t>(point.size(), 1U) * sizeof(GpuLight)),
            point.empty() ? &empty_light : point.data(),
            GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, spot_light_buffer_);
        const GpuSpotLight empty_spot;
        glBufferData(
            GL_SHADER_STORAGE_BUFFER,
            static_cast<GLsizeiptr>(
                std::max<std::size_t>(spots.size(), 1U) * sizeof(GpuSpotLight)),
            spots.empty() ? &empty_spot : spots.data(),
            GL_DYNAMIC_DRAW);
    }

    void bind_material(const Scene& scene, const Material& material, bool material_valid) {
        set_uniform(
            uniforms_.material_type,
            material_valid ? static_cast<int>(material.type) : -1);
        set_uniform(uniforms_.pbr_workflow, static_cast<int>(material.pbr_workflow));
        set_uniform(
            uniforms_.base_color,
            material_valid ? material.base_color : Color(1.0f, 0.0f, 1.0f));
        set_uniform(uniforms_.emission, material.emission);
        set_uniform(uniforms_.ior, material.ior);
        set_uniform(uniforms_.specular_color, material.specular_color);
        set_uniform(uniforms_.specular_factor, material.specular_factor);
        set_uniform(uniforms_.glossiness, material.glossiness);
        set_uniform(uniforms_.metallic, material.metallic);
        set_uniform(uniforms_.roughness, material.roughness);
        set_uniform(uniforms_.opacity, material.opacity);
        set_uniform(uniforms_.alpha_cutoff, material.alpha_cutoff);
        set_uniform(uniforms_.bump_scale, material.bump_scale);
        set_uniform(uniforms_.normal_scale, material.normal_scale);
        set_uniform(uniforms_.occlusion_strength, material.occlusion_strength);
        const AlphaMode alpha_mode = material.type == MaterialType::Pbr
            ? material.alpha_mode
            : (material.opacity_texture_id >= 0 || material.opacity < 1.0f
                  ? AlphaMode::Mask
                  : AlphaMode::Opaque);
        set_uniform(uniforms_.alpha_mode, static_cast<int>(alpha_mode));
        set_uniform(uniforms_.two_sided, material.two_sided ? 1 : 0);

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
            uniforms_.has_base_color_texture,
            uniforms_.has_opacity_texture,
            material.normal_texture_id >= 0
                ? uniforms_.has_normal_texture
                : uniforms_.has_bump_texture,
            uniforms_.has_metallic_roughness_texture,
            uniforms_.has_occlusion_texture,
            uniforms_.has_emissive_texture,
            uniforms_.has_specular_texture,
            uniforms_.has_specular_color_texture,
            uniforms_.has_specular_glossiness_texture};
        set_uniform(uniforms_.has_normal_texture, 0);
        set_uniform(uniforms_.has_bump_texture, 0);
        for (int slot = 0; slot < 9; ++slot) {
            const TextureTransform& transform =
                texture_transforms[static_cast<std::size_t>(slot)];
            const GLint offset_scale_location =
                uniforms_.texture_offset_scale[static_cast<std::size_t>(slot)];
            if (offset_scale_location >= 0) {
                glUniform4f(
                    offset_scale_location,
                    transform.offset.x(),
                    transform.offset.y(),
                    transform.scale.x(),
                    transform.scale.y());
            }
            set_uniform(
                uniforms_.texture_rotation[static_cast<std::size_t>(slot)],
                transform.rotation);
            set_uniform(
                uniforms_.texture_texcoord[static_cast<std::size_t>(slot)],
                transform.texcoord);
            const bool valid = material_valid && valid_texture_id(scene, texture_ids[slot]);
            set_uniform(
                uniforms_.texture_top_left[static_cast<std::size_t>(slot)],
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

        glGenRenderbuffers(1, &depth_renderbuffer_);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_renderbuffer_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_renderbuffer_);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            throw std::runtime_error("OpenGL raster framebuffer is incomplete");
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

    void release_output_resources() {
        if (depth_renderbuffer_ != 0) {
            glDeleteRenderbuffers(1, &depth_renderbuffer_);
            depth_renderbuffer_ = 0;
        }
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

}  // namespace renderer
