#include "render/interactive/viewer_render_backend.h"

#include "platform/opengl/cuda_opengl_interop.h"
#include "render/interactive/path_interactive_session.h"
#include "render/opengl/opengl_raster_renderer.h"

#include <stdexcept>
#include <utility>

namespace renderer {
namespace {

std::string interop_state_name(CudaOpenGlInteropState state) {
    switch (state) {
    case CudaOpenGlInteropState::Ready:
        return "ready";
    case CudaOpenGlInteropState::Active:
        return "active";
    case CudaOpenGlInteropState::Fallback:
        return "fallback";
    case CudaOpenGlInteropState::Unavailable:
    default:
        return "unavailable";
    }
}

class OpenGlViewerRenderBackend final : public ViewerRenderBackend {
public:
    OpenGlViewerRenderBackend(
        std::filesystem::path vertex_shader_path,
        std::filesystem::path fragment_shader_path)
        : renderer_(
              std::move(vertex_shader_path),
              std::move(fragment_shader_path)) {}

    InteractiveRenderMode mode() const override {
        return InteractiveRenderMode::OpenGl;
    }

    RenderModeCapability capabilities() const override {
        return render_mode_descriptor(mode()).capabilities;
    }

    void reset(const Scene& scene, const RenderSettings&) override {
        renderer_.reset(scene);
        update_output();
    }

    const RenderFrameOutput& render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state) override {
        if (frame_state.scene_changes != SceneChange::None) {
            renderer_.sync_scene(scene, frame_state.scene_changes);
        }
        renderer_.render(scene, camera, settings, frame_state);
        update_output();
        return output_;
    }

    const RenderFrameOutput& output() const override {
        return output_;
    }

    ViewerRenderBackendStatistics statistics() const override {
        ViewerRenderBackendStatistics result;
        result.shader_valid = renderer_.has_valid_shader();
        result.shader_auto_reload = renderer_.auto_reload();
        result.shader_error = renderer_.shader_error();
        result.shader_vertex_path = renderer_.vertex_shader_path().string();
        result.shader_fragment_path = renderer_.fragment_shader_path().string();
        return result;
    }

    void set_shader_auto_reload(bool enabled) override {
        renderer_.set_auto_reload(enabled);
    }

    void request_shader_reload() override {
        renderer_.request_shader_reload();
    }

private:
    OpenGlRasterRenderer renderer_;
    RenderFrameOutput output_;

    void update_output() {
        if (renderer_.output_texture() == 0) {
            return;
        }
        output_ = RenderFrameOutput::texture(RenderTextureView{
            renderer_.output_texture(),
            renderer_.output_width(),
            renderer_.output_height(),
            false});
    }
};

class PathViewerRenderBackend final : public ViewerRenderBackend {
public:
    InteractiveRenderMode mode() const override {
        return InteractiveRenderMode::Path;
    }

    RenderModeCapability capabilities() const override {
        return render_mode_descriptor(mode()).capabilities;
    }

    void reset(const Scene& scene, const RenderSettings& settings) override {
        session_.reset(scene, settings);
        framebuffer_.resize(settings.width, settings.height);
        output_ = RenderFrameOutput::host(framebuffer_);
        if (session_.active_backend() == ExecutionBackend::Cuda) {
            interop_.initialize();
        } else {
            interop_.release_texture();
        }
    }

    const RenderFrameOutput& render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state) override {
        if (session_.active_backend() == ExecutionBackend::Cuda &&
            interop_.state() != CudaOpenGlInteropState::Fallback &&
            interop_.state() != CudaOpenGlInteropState::Unavailable) {
            const CudaStreamHandle stream = session_.cuda_stream_handle();
            CudaSurfaceHandle surface = 0;
            if (interop_.begin_frame(
                    settings.width,
                    settings.height,
                    stream,
                    surface)) {
                try {
                    session_.render_next_frame_to_cuda_surface(
                        scene,
                        camera,
                        settings,
                        frame_state,
                        surface);
                } catch (...) {
                    interop_.cancel_frame();
                    throw;
                }
                if (interop_.end_frame(stream)) {
                    session_.set_cuda_presentation_state(true, false);
                    output_ = RenderFrameOutput::texture(RenderTextureView{
                        interop_.texture(),
                        interop_.width(),
                        interop_.height(),
                        true});
                    return output_;
                }
                session_.download_current_cuda_frame(framebuffer_);
                session_.set_cuda_presentation_state(false, true);
                output_ = RenderFrameOutput::host(framebuffer_);
                return output_;
            }
        }

        session_.render_next_frame(
            scene,
            camera,
            settings,
            frame_state,
            framebuffer_);
        session_.set_cuda_presentation_state(false, true);
        output_ = RenderFrameOutput::host(framebuffer_);
        return output_;
    }

    const RenderFrameOutput& output() const override {
        return output_;
    }

    ViewerRenderBackendStatistics statistics() const override {
        ViewerRenderBackendStatistics result;
        result.accumulated_samples = session_.accumulated_samples();
        result.path_backend = session_.active_backend();
        result.interop_status = interop_state_name(interop_.state());
        result.interop_detail = interop_.reason();
        if (const CudaPathStatistics* cuda = session_.cuda_statistics()) {
            result.cuda = *cuda;
        }
        return result;
    }

private:
    PathInteractiveSession session_;
    CudaOpenGlInteropTexture interop_;
    Framebuffer framebuffer_{1, 1};
    RenderFrameOutput output_;
};

}  // namespace

std::unique_ptr<ViewerRenderBackend> make_viewer_render_backend(
    InteractiveRenderMode mode,
    const std::filesystem::path& vertex_shader_path,
    const std::filesystem::path& fragment_shader_path) {
    if (mode == InteractiveRenderMode::OpenGl) {
        return std::make_unique<OpenGlViewerRenderBackend>(
            vertex_shader_path,
            fragment_shader_path);
    }
    if (mode == InteractiveRenderMode::Path) {
        return std::make_unique<PathViewerRenderBackend>();
    }
    throw std::invalid_argument("unsupported interactive render mode");
}

}  // namespace renderer
