#include "render/opengl/opengl_ddgi_pass.h"
#include <chrono>
#include <glad/gl.h>
#include <stdexcept>

namespace renderer {
OpenGlDdgiPass::OpenGlDdgiPass(bool disable_interop)
    : force_host_(disable_interop), host_(disable_interop) {}
OpenGlDdgiPass::~OpenGlDdgiPass() {
    glDeleteTextures(3, host_textures_.data());
}

const DdgiFrameResources &OpenGlDdgiPass::update(const RenderSceneSnapshot &snapshot,
                                                 const RenderSettings &settings,
                                                 const InteractiveFrameState &state) {
    frame_.active = false;
    statistics_.active = false;
    statistics_.updated_probes = 0;
    const bool enabled =
        settings.opengl.ddgi.enabled && settings.opengl.npr.style == OpenGlRenderStyle::Realistic;
    if (!enabled) {
        was_enabled_ = false;
        statistics_.status = "disabled";
        return frame_;
    }
    if (!was_enabled_ || settings.opengl.ddgi.reset_generation != reset_generation_ ||
        settings.path.cuda_device != device_id_)
        failed_ = false;
    was_enabled_ = true;
    reset_generation_ = settings.opengl.ddgi.reset_generation;
    if (failed_)
        return frame_;
    try {
        if (!volume_ || settings.path.cuda_device != device_id_) {
            for (auto &texture : interop_)
                texture.release_texture();
            volume_.reset();
            host_ = force_host_;
            statistics_.detail.clear();
            device_id_ = settings.path.cuda_device;
            auto device = force_host_ ? std::optional<CudaDeviceContext>{}
                                      : select_cuda_device_for_current_opengl_context(
                                            settings.path.cuda_device, &statistics_.detail);
            if (!device) {
                host_ = true;
                device = CudaDeviceContext::try_create(std::max(0, settings.path.cuda_device),
                                                       &statistics_.detail);
            }
            if (!device) {
                statistics_.status = "unavailable";
                failed_ = true;
                return frame_;
            }
            device->activate();
            volume_ = std::make_unique<CudaDdgiVolume>(*device);
            device_id_ = settings.path.cuda_device;
            if (!host_)
                for (auto &texture : interop_)
                    if (!texture.initialize(*device)) {
                        host_ = true;
                        statistics_.detail = texture.reason();
                        break;
                    }
        }
        volume_->update(snapshot, settings.opengl, state);
        const auto export_start = std::chrono::steady_clock::now();
        const auto layout = volume_->statistics().layout;
        const std::array<int, 3> widths{layout.width(8), layout.width(16), layout.count()};
        const std::array<int, 3> heights{layout.height(8), layout.height(16), 2};
        GLint max_size = 0;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &max_size);
        for (int width : widths)
            if (width > max_size)
                throw std::runtime_error("DDGI atlas exceeds GL_MAX_TEXTURE_SIZE");
        for (int height : heights)
            if (height > max_size)
                throw std::runtime_error("DDGI atlas exceeds GL_MAX_TEXTURE_SIZE");
        if (!host_) {
            std::array<CudaSurfaceHandle, 3> surfaces{};
            bool mapped = true;
            for (int i = 0; i < 3; ++i)
                if (!interop_[i].begin_frame(widths[i], heights[i], volume_->stream_handle(),
                                             surfaces[i])) {
                    statistics_.detail = interop_[i].reason();
                    mapped = false;
                    break;
                }
            if (mapped) {
                volume_->export_atlases(surfaces[0], surfaces[1], surfaces[2]);
                for (auto &texture : interop_)
                    if (!texture.end_frame(volume_->stream_handle())) {
                        statistics_.detail = texture.reason();
                        mapped = false;
                    }
            }
            if (!mapped) {
                for (auto &texture : interop_)
                    texture.cancel_frame();
                host_ = true;
            }
        }
        if (host_) {
            const auto data = volume_->download_atlases();
            const std::array<const void *, 3> pixels{data.irradiance.data(), data.distance.data(),
                                                     data.metadata.data()};
            const bool allocate = host_textures_[0] == 0 || host_layout_ != layout;
            if (host_textures_[0] == 0)
                glGenTextures(3, host_textures_.data());
            for (int i = 0; i < 3; ++i) {
                glBindTexture(GL_TEXTURE_2D, host_textures_[i]);
                if (allocate) {
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, widths[i], heights[i], 0, GL_RGBA,
                                 GL_FLOAT, pixels[i]);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                } else
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, widths[i], heights[i], GL_RGBA,
                                    GL_FLOAT, pixels[i]);
            }
            host_layout_ = layout;
        }
        const std::string detail = statistics_.detail;
        statistics_ = volume_->statistics();
        statistics_.detail = detail;
        statistics_.export_ms = std::chrono::duration<float, std::milli>(
                                    std::chrono::steady_clock::now() - export_start)
                                    .count();
        statistics_.status = host_ ? "fallback" : "active";
        statistics_.memory_bytes += std::uint64_t(widths[0] * heights[0] + widths[1] * heights[1] +
                                                  widths[2] * heights[2]) *
                                    16;
        frame_ = {true,
                  host_ ? host_textures_[0] : interop_[0].texture(),
                  host_ ? host_textures_[1] : interop_[1].texture(),
                  host_ ? host_textures_[2] : interop_[2].texture(),
                  layout,
                  statistics_.frame_index};
    } catch (const std::exception &error) {
        for (auto &texture : interop_)
            texture.cancel_frame();
        statistics_.status = "error";
        statistics_.detail = error.what();
        statistics_.active = false;
        failed_ = true;
    }
    return frame_;
}
} // namespace renderer
