#pragma once

#include "render/interactive/interactive_render_session.h"
#include "render/interactive/render_frame_output.h"

#include <filesystem>
#include <memory>
#include <string>

namespace renderer {

struct OpenGlTechniqueDiagnostics {
    int active_shadow_slots = 0;
    int budget_excluded_lights = 0;
    int hardware_excluded_lights = 0;
    bool dominant_light_valid = false;
    Vec3 dominant_light_direction = Vec3(0.0f, 1.0f, 0.0f);
    Color dominant_light_integrated_radiance = Color::Zero();
    float dominant_light_energy_fraction = 0.0f;
    float dominant_light_angular_radius_radians = 0.0f;
};

constexpr bool open_gl_requires_geometry_upload(SceneChangeSet changes) {
    return has_scene_change(changes, SceneChange::Geometry) ||
        has_scene_change(changes, SceneChange::MaterialBindings) ||
        has_scene_change(changes, SceneChange::InstanceTransforms);
}

class OpenGlRasterRenderer : public TextureLifetimeOwner {
public:
    OpenGlRasterRenderer(
        std::filesystem::path vertex_shader_path,
        std::filesystem::path fragment_shader_path);
    ~OpenGlRasterRenderer();

    OpenGlRasterRenderer(const OpenGlRasterRenderer&) = delete;
    OpenGlRasterRenderer& operator=(const OpenGlRasterRenderer&) = delete;

    void reset(const Scene& scene);
    void sync_scene(const Scene& scene, SceneChangeSet changes);
    void render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state);

    unsigned int output_texture() const;
    int output_width() const;
    int output_height() const;

    void set_auto_reload(bool enabled);
    bool auto_reload() const;
    void request_shader_reload();
    bool has_valid_shader() const;
    const std::string& shader_error() const;
    const std::filesystem::path& vertex_shader_path() const;
    const std::filesystem::path& fragment_shader_path() const;
    OpenGlTechniqueDiagnostics technique_diagnostics() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace renderer
