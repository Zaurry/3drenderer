#include "render/opengl/opengl_raster_renderer.h"

#include "platform/opengl/gl_shader_program.h"
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

struct UniformLocations {
    GLint view_projection = -1;
    GLint camera_position = -1;
    GLint environment = -1;
    GLint material_type = -1;
    GLint base_color = -1;
    GLint emission = -1;
    GLint opacity = -1;
    GLint alpha_cutoff = -1;
    GLint bump_scale = -1;
    GLint two_sided = -1;
    GLint has_diffuse_texture = -1;
    GLint has_opacity_texture = -1;
    GLint has_bump_texture = -1;
    GLint directional_light_count = -1;
    GLint point_light_count = -1;
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
    triangle_tangent_data(triangle, normal, tangent, handedness);

    GpuVertex vertex;
    vertex.position = {source.position.x(), source.position.y(), source.position.z()};
    vertex.normal = {normal.x(), normal.y(), normal.z()};
    vertex.uv = {source.uv.x(), source.uv.y()};
    vertex.tangent = {tangent.x(), tangent.y(), tangent.z(), handedness};
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
    uniforms.environment = glGetUniformLocation(program, "u_environment");
    uniforms.material_type = glGetUniformLocation(program, "u_material_type");
    uniforms.base_color = glGetUniformLocation(program, "u_base_color");
    uniforms.emission = glGetUniformLocation(program, "u_emission");
    uniforms.opacity = glGetUniformLocation(program, "u_opacity");
    uniforms.alpha_cutoff = glGetUniformLocation(program, "u_alpha_cutoff");
    uniforms.bump_scale = glGetUniformLocation(program, "u_bump_scale");
    uniforms.two_sided = glGetUniformLocation(program, "u_two_sided");
    uniforms.has_diffuse_texture = glGetUniformLocation(program, "u_has_diffuse_texture");
    uniforms.has_opacity_texture = glGetUniformLocation(program, "u_has_opacity_texture");
    uniforms.has_bump_texture = glGetUniformLocation(program, "u_has_bump_texture");
    uniforms.directional_light_count = glGetUniformLocation(program, "u_directional_light_count");
    uniforms.point_light_count = glGetUniformLocation(program, "u_point_light_count");
    return uniforms;
}

std::filesystem::file_time_type file_write_time(const std::filesystem::path& path) {
    std::error_code error;
    const auto write_time = std::filesystem::last_write_time(path, error);
    return error ? std::filesystem::file_time_type::min() : write_time;
}

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
        glGenTextures(1, &fallback_texture_);
        glBindTexture(GL_TEXTURE_2D, fallback_texture_);
        const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA32F, 1, 1, 0, GL_RGBA, GL_FLOAT, white.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
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
        scene_radius_ = compute_scene_radius(scene);
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
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDisable(GL_BLEND);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (!shader_program_) {
            glClearColor(0.25f, 0.0f, 0.25f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            return;
        }

        upload_lights(scene);
        glUseProgram(shader_program_.id());
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
        set_uniform(uniforms_.environment, scene.environment);
        set_uniform(
            uniforms_.directional_light_count,
            static_cast<int>(scene.directional_lights.size()));
        set_uniform(
            uniforms_.point_light_count,
            static_cast<int>(scene.point_lights.size()));

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, directional_light_buffer_);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, point_light_buffer_);
        glBindVertexArray(vao_);
        for (const DrawBatch& batch : batches_) {
            const Material fallback;
            const Material& material = valid_material_id(scene, batch.material_id)
                ? scene.materials[static_cast<std::size_t>(batch.material_id)]
                : fallback;
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
        glDisable(GL_CULL_FACE);
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
    GLuint fallback_texture_ = 0;
    std::vector<GLuint> scene_textures_;
    std::vector<DrawBatch> batches_;
    float scene_radius_ = 1.0f;

    GLuint framebuffer_ = 0;
    GLuint color_texture_ = 0;
    GLuint depth_renderbuffer_ = 0;
    int output_width_ = 0;
    int output_height_ = 0;

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
            std::vector<float> pixels;
            pixels.reserve(
                static_cast<std::size_t>(texture.width()) *
                static_cast<std::size_t>(texture.height()) * 4U);
            for (int y = texture.height() - 1; y >= 0; --y) {
                for (int x = 0; x < texture.width(); ++x) {
                    const Color& color = texture.pixels()[
                        static_cast<std::size_t>(y * texture.width() + x)];
                    pixels.push_back(color.x());
                    pixels.push_back(color.y());
                    pixels.push_back(color.z());
                    pixels.push_back(1.0f);
                }
            }
            glBindTexture(GL_TEXTURE_2D, scene_textures_[texture_index]);
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA32F,
                texture.width(),
                texture.height(),
                0,
                GL_RGBA,
                GL_FLOAT,
                pixels.data());
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
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
                {light.position.x(), light.position.y(), light.position.z(), 0.0f},
                {light.intensity.x(), light.intensity.y(), light.intensity.z(), 0.0f}});
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
    }

    void bind_material(const Scene& scene, const Material& material, bool material_valid) {
        set_uniform(
            uniforms_.material_type,
            material_valid ? static_cast<int>(material.type) : -1);
        set_uniform(
            uniforms_.base_color,
            material_valid ? material.base_color : Color(1.0f, 0.0f, 1.0f));
        set_uniform(uniforms_.emission, material.emission);
        set_uniform(uniforms_.opacity, material.opacity);
        set_uniform(uniforms_.alpha_cutoff, material.alpha_cutoff);
        set_uniform(uniforms_.bump_scale, material.bump_scale);
        set_uniform(uniforms_.two_sided, material.two_sided ? 1 : 0);

        const std::array<int, 3> texture_ids{
            material.diffuse_texture_id,
            material.opacity_texture_id,
            material.bump_texture_id};
        const std::array<GLint, 3> presence_uniforms{
            uniforms_.has_diffuse_texture,
            uniforms_.has_opacity_texture,
            uniforms_.has_bump_texture};
        for (int slot = 0; slot < 3; ++slot) {
            const bool valid = material_valid && valid_texture_id(scene, texture_ids[slot]);
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

        glGenRenderbuffers(1, &depth_renderbuffer_);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_renderbuffer_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_renderbuffer_);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            throw std::runtime_error("OpenGL raster framebuffer is incomplete");
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
