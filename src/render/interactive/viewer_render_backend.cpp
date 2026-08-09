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

class OpenGlViewerRenderBackend final
    : public ViewerRenderBackend,
      public OpenGlShaderControl {
public:
    OpenGlViewerRenderBackend(
        std::filesystem::path vertex_shader_path,
        std::filesystem::path fragment_shader_path)
        : renderer_(std::make_shared<OpenGlRasterRenderer>(
              std::move(vertex_shader_path),
              std::move(fragment_shader_path))) {}

    InteractiveRenderMode mode() const override {
        return InteractiveRenderMode::OpenGl;
    }

    RenderModeCapability capabilities() const override {
        return render_mode_descriptor(mode()).capabilities;
    }

    void reset(
        const RenderSceneSnapshot& snapshot,
        const RenderSettings&) override {
        render_scene_ = flatten_render_scene_snapshot(snapshot);
        renderer_->reset(render_scene_);
        revisions_ = snapshot.revisions;
        has_revisions_ = !revisions_.empty();
        update_output();
    }

    const RenderFrameOutput& render(
        const RenderSceneSnapshot& snapshot,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state) override {
        const SceneRevisions current_revisions = snapshot.revisions;
        SceneChangeSet changes = frame_state.scene_changes;
        if (!current_revisions.empty()) {
            changes = has_revisions_
                ? scene_changes_between(revisions_, current_revisions)
                : SceneChange::All;
            revisions_ = current_revisions;
            has_revisions_ = true;
        }
        if (changes != SceneChange::None) {
            render_scene_ = flatten_render_scene_snapshot(snapshot);
            renderer_->sync_scene(render_scene_, changes);
        }
        InteractiveFrameState effective_frame_state = frame_state;
        effective_frame_state.scene_changes = changes;
        renderer_->render(
            render_scene_,
            camera,
            settings,
            effective_frame_state);
        update_output();
        return output_;
    }

    const RenderFrameOutput& output() const override {
        return output_;
    }

    ViewerRenderBackendStatistics statistics() const override {
        OpenGlViewerStatistics result;
        result.shader_valid = renderer_->has_valid_shader();
        result.shader_auto_reload = renderer_->auto_reload();
        result.shader_error = renderer_->shader_error();
        result.shader_vertex_path = renderer_->vertex_shader_path().string();
        result.shader_fragment_path = renderer_->fragment_shader_path().string();
        return result;
    }

    void set_shader_auto_reload(bool enabled) override {
        renderer_->set_auto_reload(enabled);
    }

    void request_shader_reload() override {
        renderer_->request_shader_reload();
    }

private:
    std::shared_ptr<OpenGlRasterRenderer> renderer_;
    Scene render_scene_;
    RenderFrameOutput output_;
    SceneRevisions revisions_;
    bool has_revisions_ = false;

    void update_output() {
        if (renderer_->output_texture() == 0) {
            return;
        }
        output_ = OpenGlTextureHandle{
            renderer_->output_texture(),
            renderer_->output_width(),
            renderer_->output_height(),
            false,
            renderer_};
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

    void reset(
        const RenderSceneSnapshot& snapshot,
        const RenderSettings& settings) override {
        std::string selection_reason;
        auto device_context =
            select_cuda_device_for_current_opengl_context(
                settings.path.cuda_device,
                &selection_reason);
        if (!device_context) {
            throw std::runtime_error(
                "CUDA Path cannot use the current OpenGL context: " +
                selection_reason);
        }
        session_.reset(snapshot, settings);
        revisions_ = snapshot.revisions;
        has_revisions_ = !revisions_.empty();
        framebuffer_->resize(settings.width, settings.height);
        output_ = HostFrameHandle{framebuffer_};
        interop_->initialize(*device_context);
    }

    const RenderFrameOutput& render(
        const RenderSceneSnapshot& snapshot,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state) override {
        const SceneRevisions current_revisions = snapshot.revisions;
        InteractiveFrameState effective_frame_state = frame_state;
        if (!current_revisions.empty()) {
            effective_frame_state.scene_changes = has_revisions_
                ? scene_changes_between(revisions_, current_revisions)
                : SceneChange::All;
            revisions_ = current_revisions;
            has_revisions_ = true;
        }
        if (interop_->state() != CudaOpenGlInteropState::Fallback &&
            interop_->state() != CudaOpenGlInteropState::Unavailable) {
            const CudaStreamHandle stream = session_.cuda_stream_handle();
            CudaSurfaceHandle surface = 0;
            if (interop_->begin_frame(
                    settings.width,
                    settings.height,
                    stream,
                    surface)) {
                try {
                    session_.render_next_frame_to_cuda_surface(
                        snapshot,
                        camera,
                        settings,
                        effective_frame_state,
                        surface);
                } catch (...) {
                    interop_->cancel_frame();
                    throw;
                }
                if (interop_->end_frame(stream)) {
                    session_.set_cuda_presentation_state(true, false);
                    output_ = OpenGlTextureHandle{
                        interop_->texture(),
                        interop_->width(),
                        interop_->height(),
                        true,
                        interop_};
                    return output_;
                }
                session_.download_current_cuda_frame(*framebuffer_);
                session_.set_cuda_presentation_state(false, true);
                output_ = HostFrameHandle{framebuffer_};
                return output_;
            }
        }

        session_.render_next_frame(
            snapshot,
            camera,
            settings,
            effective_frame_state,
            *framebuffer_);
        session_.set_cuda_presentation_state(false, true);
        output_ = HostFrameHandle{framebuffer_};
        return output_;
    }

    const RenderFrameOutput& output() const override {
        return output_;
    }

    ViewerRenderBackendStatistics statistics() const override {
        CudaPathViewerStatistics result;
        result.accumulated_samples = session_.accumulated_samples();
        result.interop_status = interop_state_name(interop_->state());
        result.interop_detail = interop_->reason();
        if (const CudaPathStatistics* cuda = session_.cuda_statistics()) {
            result.cuda = *cuda;
        }
        return result;
    }

private:
    PathInteractiveSession session_;
    std::shared_ptr<CudaOpenGlInteropTexture> interop_ =
        std::make_shared<CudaOpenGlInteropTexture>();
    std::shared_ptr<Framebuffer> framebuffer_ =
        std::make_shared<Framebuffer>(1, 1);
    RenderFrameOutput output_;
    SceneRevisions revisions_;
    bool has_revisions_ = false;
};

}  // namespace

OpenGlShaderControl* open_gl_shader_control(ViewerRenderBackend& backend) {
    return dynamic_cast<OpenGlShaderControl*>(&backend);
}

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
