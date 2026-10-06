#include "render/interactive/viewer_render_backend.h"
#include "render/dxr/dxr_renderer.h"
#include <glad/gl.h>

#include "platform/opengl/cuda_opengl_interop.h"
#include "render/optix/optix_realtime_renderer.h"
#include "render/opengl/opengl_raster_renderer.h"
#include "render/opengl/opengl_ddgi_pass.h"

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
        std::filesystem::path fragment_shader_path, bool disable_interop)
        : renderer_(std::make_shared<OpenGlRasterRenderer>(
              std::move(vertex_shader_path),
              std::move(fragment_shader_path))), ddgi_(disable_interop) {}

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
        source_id_ = snapshot.source_id;
        revisions_ = snapshot.revisions;
        has_snapshot_ = true;
        update_output();
    }

    const RenderFrameOutput& render(
        const RenderSceneSnapshot& snapshot,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state) override {
        const SceneChangeSet changes = scene_changes_for_snapshot(
            source_id_,
            revisions_,
            has_snapshot_,
            snapshot,
            frame_state.scene_changes);
        source_id_ = snapshot.source_id;
        revisions_ = snapshot.revisions;
        has_snapshot_ = true;
        if (changes != SceneChange::None) {
            render_scene_ = flatten_render_scene_snapshot(snapshot);
            renderer_->sync_scene(render_scene_, changes);
        }
        InteractiveFrameState effective_frame_state = frame_state;
        effective_frame_state.scene_changes = changes;
        const auto& probes = ddgi_.update(snapshot, settings, effective_frame_state);
        renderer_->render(
            render_scene_,
            camera,
            settings,
            effective_frame_state, &probes);
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
        result.techniques = renderer_->technique_diagnostics();
        result.ddgi = ddgi_.statistics();
        result.ddgi.gather_ms = result.techniques.ddgi_gather_ms;
        if(result.ddgi.active && !result.techniques.ddgi_active) {
            result.ddgi.active = false;
            result.ddgi.status = "updating (query bypassed)";
        }
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
    OpenGlDdgiPass ddgi_;
    Scene render_scene_;
    RenderFrameOutput output_;
    std::uint64_t source_id_ = 0;
    SceneRevisions revisions_;
    bool has_snapshot_ = false;

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

class RealtimeViewerRenderBackend final : public ViewerRenderBackend {
public:
    explicit RealtimeViewerRenderBackend(bool disable_interop):disable_interop_(disable_interop) {}
    InteractiveRenderMode mode() const override {
        return InteractiveRenderMode::Rtrt;
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
                "CUDA RTRT cannot use the current OpenGL context: " +
                selection_reason);
        }
        renderer_ = std::make_unique<OptixRealtimeRenderer>(*device_context);
        renderer_->reset(snapshot, settings);
        framebuffer_->resize(settings.width, settings.height);
        output_ = HostFrameHandle{framebuffer_};
        interop_->initialize(*device_context);
        if (disable_interop_) interop_->disable("CUDA/OpenGL interop disabled by command line");
    }

    const RenderFrameOutput& render(
        const RenderSceneSnapshot& snapshot,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state) override {
        // Scene-change detection is owned by OptixRealtimeRenderer,
        // which diffs revisions against its uploaded snapshot. The caller's
        // hint flows through unchanged.
        if (interop_->state() != CudaOpenGlInteropState::Fallback &&
            interop_->state() != CudaOpenGlInteropState::Unavailable) {
            const CudaStreamHandle stream = renderer_->stream_handle();
            CudaSurfaceHandle surface = 0;
            if (interop_->begin_frame(
                    settings.width,
                    settings.height,
                    stream,
                    surface)) {
                try {
                    renderer_->render_next_frame_to_surface(
                        snapshot,
                        camera,
                        settings,
                        frame_state,
                        surface);
                } catch (...) {
                    interop_->cancel_frame();
                    throw;
                }
                if (interop_->end_frame(stream)) {
                    renderer_->set_presentation_state(true, false);
                    output_ = OpenGlTextureHandle{
                        interop_->texture(),
                        interop_->width(),
                        interop_->height(),
                        true,
                        interop_};
                    return output_;
                }
                renderer_->download_current_frame(*framebuffer_);
                renderer_->set_presentation_state(false, true);
                output_ = HostFrameHandle{framebuffer_};
                return output_;
            }
        }

        renderer_->render_next_frame(
            snapshot,
            camera,
            settings,
            frame_state,
            *framebuffer_);
        renderer_->set_presentation_state(false, true);
        output_ = HostFrameHandle{framebuffer_};
        return output_;
    }

    const RenderFrameOutput& output() const override {
        return output_;
    }

    ViewerRenderBackendStatistics statistics() const override {
        CudaPathViewerStatistics result;

        result.interop_status = interop_state_name(interop_->state());
        result.interop_detail = interop_->reason();
        if (renderer_) {
            renderer_->refresh_statistics();
            result.cuda = renderer_->statistics();
        }
        return result;
    }

private:
    bool disable_interop_ = false;
    std::unique_ptr<OptixRealtimeRenderer> renderer_;
    std::shared_ptr<CudaOpenGlInteropTexture> interop_ =
        std::make_shared<CudaOpenGlInteropTexture>();
    std::shared_ptr<Framebuffer> framebuffer_ =
        std::make_shared<Framebuffer>(1, 1);
    RenderFrameOutput output_;
};

class DxrViewerRenderBackend final : public ViewerRenderBackend {
public:
    explicit DxrViewerRenderBackend(std::shared_ptr<D3d12Context> context):renderer_(std::move(context)){}
    InteractiveRenderMode mode() const override{return InteractiveRenderMode::Dxr;}
    RenderModeCapability capabilities() const override{return RenderModeCapability::Temporal;}
    void reset(const RenderSceneSnapshot& scene,const RenderSettings& settings) override{renderer_.reset(scene,settings);}
    const RenderFrameOutput& render(const RenderSceneSnapshot& scene,const Camera& camera,const RenderSettings& settings,const InteractiveFrameState& frame) override{return renderer_.render(scene,camera,settings,frame);}
    const RenderFrameOutput& output() const override{return renderer_.output();}
    ViewerRenderBackendStatistics statistics() const override{return renderer_.statistics();}
    void readback(Framebuffer& destination) override{renderer_.readback(destination);}
private:
    DxrRenderer renderer_;
};
}  // namespace

void ViewerRenderBackend::readback(Framebuffer& destination) {
    const auto& frame=output();
    if(const auto* host=std::get_if<HostFrameHandle>(&frame);host && host->framebuffer){destination=*host->framebuffer;return;}
    if(const auto* texture=std::get_if<OpenGlTextureHandle>(&frame);texture && texture->texture) {
        std::vector<float> pixels(std::size_t(texture->width)*texture->height*4);glBindTexture(GL_TEXTURE_2D,texture->texture);glGetTexImage(GL_TEXTURE_2D,0,GL_RGBA,GL_FLOAT,pixels.data());
        destination.resize(texture->width,texture->height);
        for(int y=0;y<texture->height;++y)for(int x=0;x<texture->width;++x){const int source_y=texture->flip_y?y:texture->height-1-y;const auto index=(std::size_t(source_y)*texture->width+x)*4;destination.set_pixel(x,y,Color(pixels[index],pixels[index+1],pixels[index+2]));}return;
    }
    throw std::runtime_error("No completed frame to capture");
}

OpenGlShaderControl* open_gl_shader_control(ViewerRenderBackend& backend) {
    return dynamic_cast<OpenGlShaderControl*>(&backend);
}

std::unique_ptr<ViewerRenderBackend> make_viewer_render_backend(
    InteractiveRenderMode mode,
    const std::filesystem::path& vertex_shader_path,
    const std::filesystem::path& fragment_shader_path,
    bool disable_cuda_interop,
    std::shared_ptr<D3d12Context> dxr_context) {
    if(mode==InteractiveRenderMode::Dxr)return std::make_unique<DxrViewerRenderBackend>(std::move(dxr_context));
    if (mode == InteractiveRenderMode::OpenGl) {
        return std::make_unique<OpenGlViewerRenderBackend>(
            vertex_shader_path,
            fragment_shader_path, disable_cuda_interop);
    }
    if (mode == InteractiveRenderMode::Rtrt) {
        return std::make_unique<RealtimeViewerRenderBackend>(disable_cuda_interop);
    }
    throw std::invalid_argument("unsupported interactive render mode");
}

}  // namespace renderer
