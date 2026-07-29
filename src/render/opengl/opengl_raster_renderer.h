#pragma once

#include "render/interactive/interactive_render_session.h"

#include <filesystem>
#include <memory>
#include <string>

namespace renderer {

class OpenGlRasterRenderer {
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

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace renderer
